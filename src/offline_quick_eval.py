import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate

from src.data.dataset import (
    ASLRightHandDataset,
    collate_fn,
    count_valid_frames,
    read_right_hand_sequence,
)
from src.model_loader import load_model_from_checkpoint
from src.utils.metrics import (
    _collect_predictions_and_targets,
    _compute_average_edit_distance,
    _compute_wer,
    _levenshtein_distance,
)


def _build_compact_letter_vocab() -> tuple[Dict[str, int], Dict[int, str], int]:
    alphabet = [" "] + [chr(ord("a") + i) for i in range(26)]
    blank_id = 0
    char_to_idx = {c: i + 1 for i, c in enumerate(alphabet)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    idx_to_char[blank_id] = ""
    return char_to_idx, idx_to_char, blank_id


def _encode_phrase(text: str, char_to_idx: Dict[str, int]) -> list[int]:
    return [char_to_idx[ch] for ch in str(text) if ch in char_to_idx]


def run_quick_supplemental_test(
    ckpt_path: Path,
    data_dir: Path,
    output_dir: Path,
    max_samples: int = 40,
    batch_size: int = 16,
    max_files: int = 3,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_model_from_checkpoint(str(ckpt_path), device=device)
    char_to_idx, idx_to_char, blank_id = _build_compact_letter_vocab()

    supp_csv = data_dir / "supplemental_metadata.csv"
    supp_landmarks = data_dir / "supplemental_landmarks"

    df = pd.read_csv(supp_csv).copy()
    have_ids = sorted(int(p.stem) for p in supp_landmarks.glob("*.parquet"))[: max(1, int(max_files))]
    df = df[df["file_id"].isin(set(have_ids))].copy()

    clean_re = re.compile(r"^[a-z ]+$")
    df["phrase"] = df["phrase"].astype(str).str.lower().str.strip()
    df = df[df["phrase"].apply(lambda x: bool(clean_re.match(x)) and len(x) > 0)].copy()

    if int(max_samples) < len(df):
        df = df.sample(n=int(max_samples), random_state=42).reset_index(drop=True)

    valid_mask = []
    unreadable_files = set()
    for _, row in df.iterrows():
        ppath = supp_landmarks / f"{int(row['file_id'])}.parquet"
        if not ppath.exists():
            valid_mask.append(False)
            continue
        try:
            x_raw = read_right_hand_sequence(str(ppath), int(row["sequence_id"]))
            valid_mask.append(count_valid_frames(x_raw) > 0)
        except Exception:
            unreadable_files.add(int(row["file_id"]))
            valid_mask.append(False)

    df = df[valid_mask].copy()
    df["encoded"] = df["phrase"].apply(lambda x: _encode_phrase(str(x), char_to_idx))
    df["landmarks_subdir"] = "supplemental_landmarks"

    dataset = ASLRightHandDataset(
        df,
        landmarks_dir=str(data_dir),
        max_frames=int((loaded.config or {}).get("max_frames", 160)),
        use_delta_features=bool((loaded.config or {}).get("use_delta_features", loaded.input_dim > 63)),
        normalize_landmarks=not bool((loaded.config or {}).get("disable_landmark_centering", False)),
        landmark_scale_mode=str((loaded.config or {}).get("landmark_scale_mode", "median_radius")),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    preds, targets = _collect_predictions_and_targets(
        loaded.model,
        loader,
        idx_to_char,
        device,
        blank_id,
    )
    cer_metric = CharErrorRate()

    rows = []
    for gt, pred in zip(targets, preds):
        rows.append(
            {
                "gt": gt,
                "pred": pred,
                "cer": float(cer_metric([pred], [gt]).item()) if gt else float("nan"),
                "edit_distance": int(_levenshtein_distance(pred, gt)),
                "gt_len": int(len(gt)),
                "pred_len": int(len(pred)),
                "exact_match": bool(pred == gt),
            }
        )
    cases = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.csv"
    cases.to_csv(cases_path, index=False)

    summary = {
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "coverage": {
            "available_parquet_files_local": len(have_ids),
            "rows_after_filtering": int(len(df)),
            "unreadable_files": sorted(unreadable_files),
        },
        "split": {
            "val_cases_scored": int(len(cases)),
        },
        "metrics": {
            "cer": float(cases["cer"].mean()) if len(cases) else float("nan"),
            "wer": _compute_wer(preds, targets) if len(cases) else float("nan"),
            "sequence_accuracy": float(cases["exact_match"].mean()) if len(cases) else float("nan"),
            "avg_edit_distance": _compute_average_edit_distance(preds, targets) if len(cases) else float("nan"),
        },
        "best_examples": cases.sort_values(["cer", "edit_distance", "gt_len"], ascending=[True, True, True]).head(10).to_dict(orient="records"),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["artifacts"] = {
        "summary": str(summary_path),
        "cases": str(cases_path),
    }
    return summary
