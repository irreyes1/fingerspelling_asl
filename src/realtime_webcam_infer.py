import argparse
import json
import math
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.model_loader import load_model_from_checkpoint

CTC_BLANK_ID = 0
MP_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def load_vocab(vocab_json_path: str) -> Tuple[Dict[str, int], Dict[int, str], int]:
    with open(vocab_json_path, "r", encoding="utf-8") as f:
        base_char_to_idx = {k: int(v) for k, v in json.load(f).items()}

    if "<blank>" in base_char_to_idx:
        blank_id = int(base_char_to_idx["<blank>"])
        char_to_idx = base_char_to_idx
    else:
        blank_id = CTC_BLANK_ID
        char_to_idx = {k: v + 1 for k, v in base_char_to_idx.items()}

    idx2char = {int(v): k for k, v in char_to_idx.items()}
    return char_to_idx, idx2char, blank_id


def ctc_decode_text(log_probs: torch.Tensor, idx2char: Dict[int, str], blank_id: int) -> str:
    pred_ids = torch.argmax(log_probs, dim=2)[:, 0].tolist()
    collapsed = []
    prev = None
    for token in pred_ids:
        if token != prev and token != blank_id:
            collapsed.append(token)
        prev = token
    return "".join(idx2char.get(int(t), "") for t in collapsed)


def sanitize_decoded_text(text: str, lowercase_phrases: bool, letters_only: bool) -> str:
    out = str(text)
    if lowercase_phrases:
        out = out.lower()
    if letters_only:
        out = re.sub(r"[^a-z]", "", out)
    return out


def find_right_hand(result) -> Tuple[Optional[List], Optional[List]]:
    if not result.hand_landmarks:
        return None, None

    best_idx = None
    if result.handedness:
        for i, handed in enumerate(result.handedness):
            if not handed:
                continue
            cat = handed[0]
            label = (cat.category_name or "").lower()
            if label == "right":
                best_idx = i
                break

    if best_idx is None:
        best_idx = 0

    hand = result.hand_landmarks[best_idx]
    handed = result.handedness[best_idx] if result.handedness and len(result.handedness) > best_idx else None
    return hand, handed


def hand_to_vec63(hand_landmarks) -> np.ndarray:
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)  # (21,3)
    wrist = pts[0:1, :]
    pts = pts - wrist

    # Scale by median distance to stabilize person-camera distance.
    d = np.linalg.norm(pts[:, :2], axis=1)
    ref = float(np.median(d[d > 1e-6])) if np.any(d > 1e-6) else 1.0
    pts = pts / max(ref, 1e-6)

    return np.nan_to_num(pts.reshape(-1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)  # 63


def adapt_feature_dim(vec63: np.ndarray, prev_vec63: np.ndarray, expected_dim: int) -> np.ndarray:
    if expected_dim == 63:
        return vec63
    if expected_dim == 126:
        delta = vec63 - prev_vec63
        return np.nan_to_num(np.concatenate([vec63, delta], axis=0).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Fallback: pad/truncate to checkpoint input dim.
    out = np.zeros((expected_dim,), dtype=np.float32)
    m = min(expected_dim, vec63.shape[0])
    out[:m] = vec63[:m]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def overlay_text(frame, lines: List[str], x: int = 16, y: int = 28):
    for i, line in enumerate(lines):
        yy = y + i * 28
        cv2.putText(frame, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 255, 40), 2, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description="Realtime webcam letter inference")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pt)")
    p.add_argument("--vocab_json", type=str, default=None, help="Path to character_to_prediction_index.json")
    p.add_argument("--hand_model", type=str, default=None, help="Path to hand_landmarker.task")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=160)
    p.add_argument("--min_frames", type=int, default=20)
    p.add_argument("--infer_every", type=int, default=2, help="Run model every N frames")
    p.add_argument("--letter_conf_threshold", type=float, default=0.35)
    p.add_argument("--stable_required", type=int, default=4)
    p.add_argument("--pause_frames", type=int, default=16, help="No-letter frames to commit word")
    p.add_argument("--vote_window", type=int, default=32, help="Frames for temporal vote on non-blank argmax")
    p.add_argument("--release_frames", type=int, default=12, help="Frames with no valid letter required before accepting next letter")
    p.add_argument("--min_vote_conf", type=float, default=0.45, help="Minimum vote confidence to consider a valid letter")
    p.add_argument("--min_margin", type=float, default=0.01, help="Minimum top1-top2 margin to reduce ambiguous letters")
    p.add_argument(
        "--blank_skip_threshold",
        type=float,
        default=1.0,
        help="Reject letter if blank_prob >= threshold * best_nonblank_prob",
    )
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    root = here.parent

    hand_model_path = Path(args.hand_model) if args.hand_model else root / "artifacts" / "models" / "hand_landmarker.task"
    if not hand_model_path.exists():
        raise FileNotFoundError(f"Missing MediaPipe model: {hand_model_path}")

    vocab_path = Path(args.vocab_json) if args.vocab_json else root / "data" / "asl-fingerspelling" / "character_to_prediction_index.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab json: {vocab_path}")

    _, idx2char, blank_id = load_vocab(str(vocab_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_model_from_checkpoint(args.ckpt, device=device)
    model = loaded.model
    expected_dim = int(loaded.input_dim)
    ckpt_cfg = loaded.config or {}
    cfg_letters_only = bool(ckpt_cfg.get("letters_only", False))
    cfg_lowercase = bool(ckpt_cfg.get("lowercase_phrases", False))
    cfg_max_frames = int(ckpt_cfg.get("max_frames", args.max_frames))
    if int(args.max_frames) != int(cfg_max_frames):
        print(f"Warning: --max_frames={args.max_frames} differs from checkpoint max_frames={cfg_max_frames}")
    if "decode_blank_skip_threshold" in ckpt_cfg and float(args.blank_skip_threshold) == 1.0:
        args.blank_skip_threshold = float(ckpt_cfg["decode_blank_skip_threshold"])

    print(f"Model loaded. input_dim={expected_dim} output_dim={loaded.output_dim} device={device}")
    print(f"Checkpoint text filters: lowercase={cfg_lowercase} letters_only={cfg_letters_only}")
    print(f"Blank gating threshold: {args.blank_skip_threshold:.3f}")

    base_options = python.BaseOptions(model_asset_path=str(hand_model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam (prueba índice 1 o permisos)")

    print("Webcam inferencia activa. ESC para salir.")

    # Store only frames with detected right hand; we pad at inference time to match training layout.
    buffer = deque(maxlen=args.max_frames)
    prev_vec63 = np.zeros((63,), dtype=np.float32)
    frame_id = 0
    fps = 0.0
    t_prev = time.perf_counter()
    t0 = time.perf_counter()

    stable_letter = ""
    stable_count = 0
    no_letter_count = 0
    armed_for_new_letter = True
    live_text = ""
    committed_words: List[str] = []

    top3_text = "-"
    conf = 0.0
    vote_conf = 0.0
    margin = 0.0
    blank_prob = 0.0
    entropy = 0.0
    candidate_letter = ""
    candidate_effective_conf = 0.0
    candidate_valid = False
    raw_ctc_text = ""
    status_text = "SPACE: capturar letra | c: limpiar | ESC: salir"
    status_until = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.perf_counter()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - t0) * 1000)
        result = detector.detect_for_video(mp_image, ts_ms)

        right_hand, right_handed = find_right_hand(result)
        hand_label = "-"
        if right_handed and len(right_handed) > 0:
            h = right_handed[0]
            hand_label = f"{h.category_name}:{h.score:.2f}"

        if right_hand is not None:
            vec63 = hand_to_vec63(right_hand)
            feat = adapt_feature_dim(vec63, prev_vec63, expected_dim)
            prev_vec63 = vec63
            buffer.append(feat)

            # Draw hand
            h, w = frame.shape[:2]
            pts = []
            for lm in right_hand:
                x = int(lm.x * w)
                y = int(lm.y * h)
                pts.append((x, y))
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            for a, b in MP_HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        else:
            # Keep temporal state from real hand frames only; no-hand frames are handled by gating logic.
            no_letter_count += 1
            stable_count = 0
            stable_letter = ""
            candidate_valid = False
            candidate_letter = ""
            candidate_effective_conf = 0.0

        # Inference loop
        if len(buffer) >= args.min_frames and (frame_id % args.infer_every == 0):
            hist = list(buffer)
            valid_t = min(len(hist), args.max_frames)
            x_np = np.zeros((args.max_frames, expected_dim), dtype=np.float32)
            if valid_t > 0:
                x_np[:valid_t] = np.stack(hist[-valid_t:], axis=0).astype(np.float32)
            x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1,T,D)

            with torch.no_grad():
                log_probs = model(x)  # (T,1,C)
                valid_log_probs = log_probs[:valid_t, :, :] if valid_t > 0 else log_probs[:1, :, :]
                probs = torch.exp(valid_log_probs[:, 0, :])  # (T_valid,C)
                mean_probs = probs.mean(dim=0)  # (C,)

            nonblank = mean_probs.clone()
            if 0 <= blank_id < nonblank.shape[0]:
                nonblank[blank_id] = 0.0
            vals, idxs = torch.topk(nonblank, k=min(3, nonblank.shape[0]))
            vals_np = vals.detach().cpu().numpy().tolist()
            idx_np = idxs.detach().cpu().numpy().tolist()
            top3 = [(idx2char.get(int(i), f"#{i}"), float(v)) for i, v in zip(idx_np, vals_np)]
            top3_text = " | ".join([f"{c}:{v:.2f}" for c, v in top3])

            if len(vals_np) >= 1:
                conf = float(vals_np[0])
            if len(vals_np) >= 2:
                margin = float(vals_np[0] - vals_np[1])
            blank_prob = float(mean_probs[blank_id].item()) if 0 <= blank_id < mean_probs.shape[0] else 0.0

            p = mean_probs.detach().cpu().numpy()
            p = p / max(p.sum(), 1e-8)
            entropy = float(-(p * np.log(p + 1e-12)).sum() / math.log(max(len(p), 2)))
            raw_ctc_text = sanitize_decoded_text(
                ctc_decode_text(valid_log_probs, idx2char, blank_id),
                cfg_lowercase,
                cfg_letters_only,
            )

            # Robust real-time letter for CTC: majority vote over recent non-blank framewise argmax.
            frame_ids = torch.argmax(valid_log_probs[:, 0, :], dim=1).detach().cpu().numpy().tolist()
            if 0 < args.vote_window < len(frame_ids):
                frame_ids = frame_ids[-args.vote_window:]
            nonblank_ids = [int(i) for i in frame_ids if int(i) != blank_id]
            pred_letter = ""
            vote_conf = 0.0
            if nonblank_ids:
                vals_u, cnts = np.unique(np.array(nonblank_ids, dtype=np.int64), return_counts=True)
                best_pos = int(np.argmax(cnts))
                best_id = int(vals_u[best_pos])
                best_count = int(cnts[best_pos])
                pred_letter = idx2char.get(best_id, "")
                pred_letter = sanitize_decoded_text(pred_letter, cfg_lowercase, cfg_letters_only)
                vote_conf = best_count / max(len(nonblank_ids), 1)

            effective_conf = max(float(conf), float(vote_conf))
            candidate_letter = pred_letter
            candidate_effective_conf = effective_conf
            blank_dominant = blank_prob >= (args.blank_skip_threshold * max(conf, 1e-8))

            is_valid_letter = (
                bool(pred_letter)
                and (effective_conf >= args.letter_conf_threshold)
                and (vote_conf >= args.min_vote_conf)
                and (margin >= args.min_margin)
                and (not blank_dominant)
            )
            candidate_valid = is_valid_letter

            if is_valid_letter:
                no_letter_count = 0
                if pred_letter == stable_letter:
                    stable_count += 1
                else:
                    stable_letter = pred_letter
                    stable_count = 1

                if armed_for_new_letter and stable_count >= args.stable_required:
                    if not live_text or live_text[-1] != pred_letter:
                        live_text += pred_letter
                        live_text = sanitize_decoded_text(live_text, cfg_lowercase, cfg_letters_only)
                    armed_for_new_letter = False
            else:
                no_letter_count += 1
                stable_count = 0
                stable_letter = ""
                if no_letter_count >= args.release_frames:
                    armed_for_new_letter = True

            if no_letter_count >= args.pause_frames and len(live_text) > 0:
                committed_words.append(sanitize_decoded_text(live_text, cfg_lowercase, cfg_letters_only))
                if len(committed_words) > 8:
                    committed_words = committed_words[-8:]
                live_text = ""
                no_letter_count = 0

        lines = [
            "ASL Webcam Inference",
            f"FPS: {fps:.1f} | Frames: {len(buffer)}/{args.max_frames}",
            f"Hand: {hand_label}",
            f"Top3: {top3_text}",
            f"Conf: {conf:.2f} | Vote: {vote_conf:.2f} | Blank: {blank_prob:.2f} | Margin: {margin:.2f} | Entropy: {entropy:.2f}",
            f"State: {'ARMED' if armed_for_new_letter else 'LOCKED'}",
            f"Raw CTC: {raw_ctc_text if raw_ctc_text else '-'}",
            f"Live letters: {live_text if live_text else '-'}",
            f"Words: {' '.join(committed_words[-4:]) if committed_words else '-'}",
            "SPACE: capturar letra (modo guiado) | c: limpiar | ESC: salir",
        ]
        if time.perf_counter() < status_until:
            lines.append(status_text)
        overlay_text(frame, lines)

        cv2.imshow("Fingerspelling Webcam Inference", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            if candidate_valid and candidate_letter:
                live_text += candidate_letter
                status_text = f"Capturada: '{candidate_letter}' (conf={candidate_effective_conf:.2f})"
            else:
                status_text = "Sin captura valida: ajusta gesto/iluminacion o espera mejor confianza"
            status_until = time.perf_counter() + 1.6
        if key == ord("c"):
            live_text = ""
            committed_words = []

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
