import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate

from src.data.dataset import ASLRightHandDataset, collate_fn
from src.model_loader import load_model_from_checkpoint
from src.utils.metrics import _compute_wer, _levenshtein_distance, _collect_predictions_and_targets


@dataclass
class TestArtifacts:
    summary_path: Path
    cases_path: Path
    metrics_plot_path: Path
    cer_hist_path: Path
    edit_scatter_path: Path


def load_ctc_vocab(vocab_json_path: Path) -> tuple[Dict[str, int], Dict[int, str], int]:
    raw = json.loads(vocab_json_path.read_text(encoding="utf-8"))
    char_to_idx = {str(k): int(v) + 1 for k, v in raw.items()}
    idx_to_char = {int(v): str(k) for k, v in char_to_idx.items()}
    blank_id = 0
    idx_to_char[blank_id] = ""
    return char_to_idx, idx_to_char, blank_id


def build_compact_letter_vocab() -> tuple[Dict[str, int], Dict[int, str], int]:
    alphabet = [" "] + [chr(ord("a") + i) for i in range(26)]
    blank_id = 0
    char_to_idx = {c: i + 1 for i, c in enumerate(alphabet)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    idx_to_char[blank_id] = ""
    return char_to_idx, idx_to_char, blank_id


def sanitize_phrase(text: str, lowercase: bool, letters_only: bool) -> str:
    out = str(text)
    if lowercase:
        out = out.lower()
    if letters_only:
        out = re.sub(r"[^a-z ]", "", out)
        out = re.sub(r" +", " ", out).strip()
    return out


def encode_phrase(text: str, char_to_idx: Dict[str, int], lowercase: bool, letters_only: bool) -> List[int]:
    out = sanitize_phrase(text, lowercase=lowercase, letters_only=letters_only)
    return [char_to_idx[ch] for ch in out if ch in char_to_idx]


def split_by_participant(df: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = df["participant_id"].drop_duplicates().tolist()
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(participants), generator=generator).tolist()
    n_val = max(1, int(len(participants) * float(val_ratio)))
    val_participants = {participants[i] for i in perm[:n_val]}
    train_df = df[~df["participant_id"].isin(val_participants)].copy()
    val_df = df[df["participant_id"].isin(val_participants)].copy()
    return train_df, val_df


def collect_case_rows(model, dataloader, idx_to_char: Dict[int, str], device: torch.device, blank_id: int) -> pd.DataFrame:
    preds, targets = _collect_predictions_and_targets(
        model=model,
        dataloader=dataloader,
        int_to_letter=idx_to_char,
        device=device,
        blank_id=blank_id,
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
    return pd.DataFrame(rows)


def build_local_eval_split(
    data_dir: Path,
    char_to_idx: Dict[str, int],
    lowercase_phrases: bool,
    letters_only: bool,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    train_csv = data_dir / "train.csv"
    train_landmarks = data_dir / "train_landmarks"
    df = pd.read_csv(train_csv).copy()
    total_rows = int(len(df))
    available_parquets = {int(p.stem) for p in train_landmarks.glob("*.parquet")}
    df["landmarks_subdir"] = "train_landmarks"
    df = df[df["file_id"].isin(available_parquets)].copy()
    rows_with_local_parquet = int(len(df))
    df["phrase"] = df["phrase"].astype(str).apply(
        lambda s: sanitize_phrase(s, lowercase=lowercase_phrases, letters_only=letters_only)
    )
    df = df[df["phrase"].str.len() > 0].copy()
    df["encoded"] = df["phrase"].astype(str).apply(
        lambda s: encode_phrase(s, char_to_idx, lowercase=False, letters_only=False)
    )
    df = df[df["encoded"].map(len) > 0].copy()
    train_df, val_df = split_by_participant(df, val_ratio=val_ratio, seed=seed)
    stats = {
        "train_csv_rows_total": total_rows,
        "available_parquet_files_local": int(len(available_parquets)),
        "rows_with_local_parquet": rows_with_local_parquet,
        "rows_after_encoding": int(len(df)),
        "coverage_rows_pct": (rows_with_local_parquet / total_rows * 100.0) if total_rows else 0.0,
        "participant_count_local": int(df["participant_id"].nunique()),
    }
    return train_df, val_df, stats


def load_exact_eval_split(
    split_csv: Path,
    char_to_idx: Dict[str, int],
    lowercase_phrases: bool,
    letters_only: bool,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    df = pd.read_csv(split_csv).copy()
    total_rows = int(len(df))
    if "landmarks_subdir" not in df.columns:
        df["landmarks_subdir"] = "train_landmarks"
    df["phrase"] = df["phrase"].astype(str).apply(
        lambda s: sanitize_phrase(s, lowercase=lowercase_phrases, letters_only=letters_only)
    )
    df = df[df["phrase"].str.len() > 0].copy()
    df["encoded"] = df["phrase"].astype(str).apply(
        lambda s: encode_phrase(s, char_to_idx, lowercase=False, letters_only=False)
    )
    df = df[df["encoded"].map(len) > 0].copy()
    stats = {
        "split_csv_rows_total": total_rows,
        "rows_after_encoding": int(len(df)),
        "participant_count": int(df["participant_id"].nunique()) if "participant_id" in df.columns else 0,
    }
    return df.reset_index(drop=True), stats


def maybe_check_webcam(camera_index: int) -> Dict[str, object]:
    if cv2 is None:
        return {
            "camera_index": int(camera_index),
            "opened": False,
            "error": "opencv_not_installed",
        }
    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    ok = bool(cap.isOpened())
    if ok:
        ok, _ = cap.read()
    cap.release()
    return {
        "camera_index": int(camera_index),
        "opened": bool(ok),
    }


def save_plots(cases: pd.DataFrame, output_dir: Path) -> tuple[Path, Path, Path]:
    if plt is None:
        missing = output_dir / "plotting_unavailable.txt"
        missing.write_text("matplotlib_not_installed\n", encoding="utf-8")
        return missing, missing, missing
    metrics = {
        "CER": float(cases["cer"].mean()) if len(cases) else float("nan"),
        "AvgEdit": float(cases["edit_distance"].mean()) if len(cases) else float("nan"),
        "ExactMatch": float(cases["exact_match"].mean()) if len(cases) else float("nan"),
        "GTLen": float(cases["gt_len"].mean()) if len(cases) else float("nan"),
    }

    metrics_plot_path = output_dir / "metrics_overview.png"
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(list(metrics.keys()), list(metrics.values()), color=["#194b7a", "#d06b2f", "#4a8f43", "#7a6a19"])
    plt.title("Portable Test Metrics")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, metrics.values()):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(metrics_plot_path, dpi=160)
    plt.close()

    cer_hist_path = output_dir / "cer_histogram.png"
    plt.figure(figsize=(8, 4.5))
    plt.hist(cases["cer"], bins=min(12, max(4, len(cases))), color="#194b7a", edgecolor="white")
    plt.title("CER Distribution Across Local Validation Cases")
    plt.xlabel("CER")
    plt.ylabel("Cases")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(cer_hist_path, dpi=160)
    plt.close()

    edit_scatter_path = output_dir / "edit_distance_vs_gt_len.png"
    plt.figure(figsize=(8, 4.5))
    plt.scatter(cases["gt_len"], cases["edit_distance"], c=cases["cer"], cmap="viridis", s=40)
    plt.title("Edit Distance vs Ground-Truth Length")
    plt.xlabel("GT length")
    plt.ylabel("Edit distance")
    plt.grid(alpha=0.25)
    plt.colorbar(label="CER")
    plt.tight_layout()
    plt.savefig(edit_scatter_path, dpi=160)
    plt.close()

    return metrics_plot_path, cer_hist_path, edit_scatter_path


def run_test(
    ckpt_path: Path,
    data_dir: Path,
    hand_model_path: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    batch_size: int,
    run_webcam_check: bool,
    camera_index: int,
    split_csv: Optional[Path] = None,
    max_eval_rows: int = 0,
) -> TestArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)

    char_to_idx, idx_to_char, blank_id = load_ctc_vocab(data_dir / "character_to_prediction_index.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_model_from_checkpoint(str(ckpt_path), device=device)
    ckpt_cfg = loaded.config or {}
    expected_dim = int(loaded.input_dim)
    expected_output_dim = int(loaded.output_dim)
    use_delta_features = bool(ckpt_cfg.get("use_delta_features", expected_dim > 63))
    normalize_landmarks = not bool(ckpt_cfg.get("disable_landmark_centering", False))
    inferred_letters_only = expected_output_dim == 28
    lowercase_phrases = bool(ckpt_cfg.get("lowercase_phrases", inferred_letters_only))
    letters_only = bool(ckpt_cfg.get("letters_only", inferred_letters_only))
    if inferred_letters_only:
        char_to_idx, idx_to_char, blank_id = build_compact_letter_vocab()

    available_parquets = {int(p.stem) for p in (data_dir / "train_landmarks").glob("*.parquet")}
    if split_csv is not None:
        exact_val_df, split_stats = load_exact_eval_split(
            split_csv=split_csv,
            char_to_idx=char_to_idx,
            lowercase_phrases=lowercase_phrases,
            letters_only=letters_only,
        )
        val_df = exact_val_df[exact_val_df["file_id"].isin(available_parquets)].copy().reset_index(drop=True)
        train_df = pd.DataFrame(columns=val_df.columns)
        coverage_stats = {
            "available_parquet_files_local": int(len(available_parquets)),
            "split_csv_rows_total": int(split_stats["split_csv_rows_total"]),
            "rows_after_encoding_in_split": int(split_stats["rows_after_encoding"]),
            "rows_with_local_parquet": int(len(val_df)),
            "coverage_rows_pct": (len(val_df) / split_stats["split_csv_rows_total"] * 100.0)
            if split_stats["split_csv_rows_total"]
            else 0.0,
            "participant_count_local": int(val_df["participant_id"].nunique()) if len(val_df) else 0,
        }
    else:
        train_df, val_df, coverage_stats = build_local_eval_split(
            data_dir=data_dir,
            char_to_idx=char_to_idx,
            lowercase_phrases=lowercase_phrases,
            letters_only=letters_only,
            val_ratio=val_ratio,
            seed=seed,
        )

    if int(max_eval_rows) > 0 and len(val_df) > int(max_eval_rows):
        val_df = val_df.head(int(max_eval_rows)).copy().reset_index(drop=True)
        coverage_stats["rows_used_for_eval"] = int(len(val_df))

    val_ds = ASLRightHandDataset(
        val_df,
        landmarks_dir=str(data_dir),
        max_frames=int(ckpt_cfg.get("max_frames", 160)),
        use_delta_features=use_delta_features,
        normalize_landmarks=normalize_landmarks,
        landmark_scale_mode=str(ckpt_cfg.get("landmark_scale_mode", "median_radius")),
    )
    try:
        validity = val_ds.summarize_validity()
    except Exception as exc:
        validity = {
            "validity_summary_failed": 1,
            "error": str(exc),
            "scanned": int(len(val_ds)),
        }
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    cases = collect_case_rows(loaded.model, val_loader, idx_to_char=idx_to_char, device=device, blank_id=blank_id)
    metrics = {
        "cer": float(cases["cer"].mean()) if len(cases) else float("nan"),
        "wer": _compute_wer(cases["pred"].tolist(), cases["gt"].tolist()) if len(cases) else float("nan"),
        "sequence_accuracy": float(cases["exact_match"].mean()) if len(cases) else float("nan"),
        "avg_edit_distance": float(cases["edit_distance"].mean()) if len(cases) else float("nan"),
    }

    cases_path = output_dir / "cases.csv"
    cases.to_csv(cases_path, index=False)
    metrics_plot_path, cer_hist_path, edit_scatter_path = save_plots(cases, output_dir)

    webcam_check = maybe_check_webcam(camera_index) if run_webcam_check else {"skipped": True, "camera_index": int(camera_index)}
    summary = {
        "checkpoint": str(ckpt_path),
        "hand_model": str(hand_model_path),
        "device": str(device),
        "expected_input_dim": expected_dim,
        "expected_output_dim": expected_output_dim,
        "checkpoint_config": ckpt_cfg,
        "preprocessing": {
            "use_delta_features": use_delta_features,
            "normalize_landmarks": normalize_landmarks,
            "lowercase_phrases": lowercase_phrases,
            "letters_only": letters_only,
            "landmark_scale_mode": str(ckpt_cfg.get("landmark_scale_mode", "median_radius")),
        },
        "coverage": coverage_stats,
        "split": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "val_validity": validity,
            "val_cases_scored": int(len(cases)),
            "split_csv": str(split_csv) if split_csv is not None else None,
        },
        "metrics": metrics,
        "webcam_check": webcam_check,
        "best_examples": cases.sort_values(["cer", "edit_distance", "gt_len"], ascending=[True, True, True]).head(10).to_dict(orient="records"),
        "worst_examples": cases.sort_values(["cer", "edit_distance", "gt_len"], ascending=[False, False, False]).head(10).to_dict(orient="records"),
        "plots": {
            "metrics_overview": str(metrics_plot_path),
            "cer_histogram": str(cer_hist_path),
            "edit_distance_vs_gt_len": str(edit_scatter_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TestArtifacts(
        summary_path=summary_path,
        cases_path=cases_path,
        metrics_plot_path=metrics_plot_path,
        cer_hist_path=cer_hist_path,
        edit_scatter_path=edit_scatter_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Portable TEST phase for webcam inference repo")
    parser.add_argument("--ckpt", type=str, default="artifacts/models/run_20260312_231052_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/asl-fingerspelling")
    parser.add_argument("--hand_model", type=str, default="artifacts/models/hand_landmarker.task")
    parser.add_argument("--output_dir", type=str, default="artifacts/eval/latest_portable_test")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--run_webcam_check", action="store_true")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--split_csv", type=str, default=None)
    parser.add_argument("--max_eval_rows", type=int, default=0)
    args = parser.parse_args()

    artifacts = run_test(
        ckpt_path=Path(args.ckpt),
        data_dir=Path(args.data_dir),
        hand_model_path=Path(args.hand_model),
        output_dir=Path(args.output_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        run_webcam_check=bool(args.run_webcam_check),
        camera_index=args.camera_index,
        split_csv=Path(args.split_csv) if args.split_csv else None,
        max_eval_rows=args.max_eval_rows,
    )
    print(json.dumps({"summary": str(artifacts.summary_path), "cases": str(artifacts.cases_path)}, indent=2))


if __name__ == "__main__":
    main()
