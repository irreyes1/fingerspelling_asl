"""
Evaluate a trained ASL model on the supplemental dataset (held-out test set).

The supplemental set contains participants not present in train.csv, making it
a zero-leakage test set. It must NOT have been used during training
(i.e. --use_supplemental must be False in the training run).

Usage:
    python -m src.evaluate \
        --ckpt artifacts/models/run_best.pt \
        --data_dir src/data/asl-fingerspelling
"""

import argparse
import os
import re

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate
from tqdm import tqdm

from src.data.dataset import (
    ASLRightHandDataset,
    collate_fn,
    count_valid_frames,
    read_right_hand_sequence,
)
from src.data.vocab import build_ctc_vocab, encode_phrase
from src.model_loader import load_model_from_checkpoint
from src.utils.metrics import (
    _collect_predictions_and_targets,
    _compute_average_edit_distance,
    _compute_wer,
)


def _existing_file_ids(landmarks_dir: str) -> set:
    if not os.path.isdir(landmarks_dir):
        return set()
    out = set()
    for fn in os.listdir(landmarks_dir):
        if fn.endswith(".parquet"):
            try:
                out.add(int(os.path.splitext(fn)[0]))
            except ValueError:
                pass
    return out


def main():
    p = argparse.ArgumentParser(
        description="Evaluate a trained ASL model on the (rerun) test set"
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    p.add_argument(
        "--data_dir",
        type=str,
        default="src/data/asl-fingerspelling",
        help="Root data directory (must contain labels.pq and test_landmarks/)",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--n_examples", type=int, default=10, help="Number of GT/PRED examples to print"
    )
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="If set, randomly subsample this many rows before the landmark check (speeds up quick runs)"
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load checkpoint & read training config ---
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)

    ckpt_config = ckpt.get("config", {})
    ckpt_epoch = ckpt.get("epoch", "?")

    max_frames = ckpt_config.get("max_frames", 160)
    print(f"Checkpoint: epoch={ckpt_epoch}, max_frames={max_frames}")

    # --- Paths ---
    label_pq = os.path.join(args.data_dir, "labels.pq")
    test_landmarks = os.path.join(args.data_dir, "test_landmarks")
    vocab_json = os.path.join(args.data_dir, "character_to_prediction_index.json")

    for path in [label_pq, test_landmarks, vocab_json]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required path: {path}")

    # --- Vocab ---
    letter_to_int, int_to_letter, blank_id = build_ctc_vocab(vocab_json)

    # --- Load & filter supplemental metadata ---
    df = pd.read_parquet(label_pq)
    required_cols = {"file_id", "sequence_id", "phrase"} # "participant_id" not in test set
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"labels.pq is missing columns: {missing}")

    have_ids = _existing_file_ids(test_landmarks)
    if not have_ids:
        raise ValueError(f"No parquet files found in {test_landmarks}")
    df = df[df["file_id"].isin(have_ids)].copy()
    print(f"Rows after filtering to available parquets ({len(have_ids)} files): {len(df)}")

    _clean_re = re.compile(r"^[a-z ]+$")
    df["phrase"] = df["phrase"].astype(str).str.lower().str.strip()
    df = df[df["phrase"].apply(lambda x: bool(_clean_re.match(x)) and len(x) > 0)].copy()
    print(f"Rows after filtering to letters-only phrases: {len(df)}")

    if args.max_samples is not None and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows (--max_samples)")

    print("Pre-filtering sequences with no right-hand landmarks...")
    valid_mask = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking landmarks", leave=False):
        ppath = os.path.join(test_landmarks, f"{int(row['file_id'])}.parquet")
        if not os.path.exists(ppath):
            valid_mask.append(False)
            continue
        X_raw = read_right_hand_sequence(ppath, int(row["sequence_id"]))
        valid_mask.append(count_valid_frames(X_raw) > 0)
    df = df[valid_mask].copy()
    print(f"Rows after filtering no-data sequences: {len(df)}")

    df["encoded"] = df["phrase"].apply(lambda x: encode_phrase(str(x), letter_to_int))
    print(f"Test samples: {len(df)}\n")

    # --- Dataset & loader (no augmentation) ---
    test_ds = ASLRightHandDataset(
        df,
        landmarks_dir=test_landmarks,
        max_frames=max_frames,
        training=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    # --- Load model ---
    print("Loading model from checkpoint...")
    loaded = load_model_from_checkpoint(ckpt_path=args.ckpt, device=device)
    model = loaded.model

    # --- Run inference (single pass) ---
    print("Running evaluation...")
    preds, targets, _ = _collect_predictions_and_targets(
        model, test_loader, int_to_letter, device, blank_id
    )

    if len(preds) == 0:
        print("ERROR: no valid samples found in the test set.")
        return

    # --- Compute metrics ---
    cer = float(CharErrorRate()(preds, targets).item())
    wer = _compute_wer(preds, targets)
    seq_acc = float(sum(1 for p, t in zip(preds, targets) if p == t) / len(preds))
    avg_edit = _compute_average_edit_distance(preds, targets)

    print("\n========== TEST RESULTS ==========")
    print(f"  Samples      : {len(preds)}")
    print(f"  CER          : {cer:.4f}")
    print(f"  WER          : {wer:.4f}")
    print(f"  Exact Match  : {seq_acc:.4f}  ({seq_acc * 100:.1f}%)")
    print(f"  Avg Edit Dist: {avg_edit:.4f}")
    print("===================================\n")

    if args.n_examples > 0:
        cer_fn = CharErrorRate()
        sorted_pairs = sorted(
            zip(targets, preds),
            key=lambda x: float(cer_fn([x[1]], [x[0]]).item()),
        )
        n = min(args.n_examples, len(sorted_pairs))
        print(f"Sample predictions ({n} best CER, ascending):")
        for gt, pred in sorted_pairs[:n]:
            sample_cer = float(cer_fn([pred], [gt]).item())
            status = "✓" if gt == pred else "✗"
            print(f"  [{status}] CER={sample_cer:.2f}  GT: {gt!r:30s}  PRED: {pred!r}")


if __name__ == "__main__":
    main()
