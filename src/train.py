import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from src.data.dataset import ASLRightHandDataset, collate_fn
from src.models.embedded_rnn import EmbeddedRNN
from src.models.tcn_bilstm import TCNBiRNN
from src.utils.metrics import ctc_greedy_decode, evaluate_metrics

CTC_BLANK_ID = 0


def encode_phrase(phrase: str, letter_to_int: Dict[str, int]) -> List[int]:
    return [letter_to_int[c] for c in phrase if c in letter_to_int]


def build_ctc_vocab(vocab_json_path: str):
    with open(vocab_json_path, "r", encoding="utf-8") as f:
        base_char_to_idx = {k: int(v) for k, v in json.load(f).items()}

    if "<blank>" in base_char_to_idx:
        blank_id = int(base_char_to_idx["<blank>"])
        char_to_idx = base_char_to_idx
    else:
        blank_id = CTC_BLANK_ID
        char_to_idx = {k: v + 1 for k, v in base_char_to_idx.items()}

    idx_to_char = {int(v): k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char, blank_id


def split_by_participant(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42):
    participants = df["participant_id"].unique().tolist()
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(participants), generator=rng).tolist()
    n_val = max(1, int(len(participants) * val_ratio))

    val_participants = set(participants[i] for i in perm[:n_val])
    train_df = df[~df["participant_id"].isin(val_participants)].copy()
    val_df = df[df["participant_id"].isin(val_participants)].copy()
    return train_df, val_df


def existing_file_ids(landmarks_dir: str):
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


def parse_wandb_tags(tags_raw: str):
    if not tags_raw:
        return None
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    return tags if tags else None


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def collect_gt_pred_examples(
    model,
    dataloader,
    int_to_letter,
    device,
    blank_id,
    n_examples: int = 5,
) -> List[Tuple[str, str]]:
    model.eval()
    examples: List[Tuple[str, str]] = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            X, Y, input_lens, target_lens = batch
            X = X.to(device)
            outputs = model(X)  # (T, B, C)
            model_input_lens = project_input_lengths(model, input_lens)
            batch_size = outputs.shape[1]
            y_list = Y.detach().cpu().tolist()

            start = 0
            for i in range(batch_size):
                valid_t = int(model_input_lens[i].item())
                pred_text = ctc_greedy_decode(outputs[:valid_t, i, :], int_to_letter, blank_id)
                target_len = int(target_lens[i].item())
                tgt_ids = y_list[start:start + target_len]
                start += target_len
                tgt_text = "".join(int_to_letter.get(int(t), "") for t in tgt_ids if int(t) != blank_id)
                examples.append((tgt_text, pred_text))
                if len(examples) >= n_examples:
                    return examples

    return examples


def log_examples_to_wandb(
    model,
    dataloader,
    int_to_letter,
    device,
    blank_id,
    global_step,
    split_name: str = "val",
    n_examples: int = 5,
):
    examples = collect_gt_pred_examples(
        model=model,
        dataloader=dataloader,
        int_to_letter=int_to_letter,
        device=device,
        blank_id=blank_id,
        n_examples=n_examples,
    )
    if len(examples) == 0:
        raise RuntimeError("Could not collect any GT/PRED examples.")
    if len(examples) < n_examples:
        base = list(examples)
        while len(examples) < n_examples:
            examples.append(base[(len(examples) - len(base)) % len(base)])

    print(f"Logging {n_examples} GT/PRED examples ({split_name}):")
    for i, (gt, pred) in enumerate(examples, start=1):
        print(f"[{i}] GT: {gt}")
        print(f"    PRED: {pred}")

    table = wandb.Table(columns=["split", "idx", "gt", "pred"])
    for i, (gt, pred) in enumerate(examples, start=1):
        table.add_data(split_name, i, gt, pred)
    wandb.log({"examples/gt_pred": table, "global_step": global_step}, step=global_step)


def build_dataframes(args, train_csv: str, supplemental_csv: str):
    df = pd.read_csv(train_csv).copy()
    df["landmarks_subdir"] = "train_landmarks"

    if args.use_supplemental_data:
        if not os.path.exists(supplemental_csv):
            raise FileNotFoundError(f"Missing supplemental csv: {supplemental_csv}")
        sup = pd.read_csv(supplemental_csv).copy()
        sup["landmarks_subdir"] = "supplemental_landmarks"
        df = pd.concat([df, sup], ignore_index=True)

    required_cols = {"file_id", "sequence_id", "participant_id", "phrase", "landmarks_subdir"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSVs are missing columns: {missing}")

    return df


def filter_to_available_files(df: pd.DataFrame, data_dir: str):
    keep = []
    for subdir, group in df.groupby("landmarks_subdir"):
        folder = os.path.join(data_dir, subdir)
        have_ids = existing_file_ids(folder)
        if not have_ids:
            print(f"Warning: no parquet files in {folder}")
            continue
        keep.append(group[group["file_id"].isin(have_ids)])
    if not keep:
        return df.iloc[0:0].copy()
    return pd.concat(keep, ignore_index=True)


def parse_kernel_list(raw: str) -> Tuple[int, ...]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) == 0:
        return (3, 3, 3)
    return tuple(vals)


def create_model(args, input_dim: int, output_dim: int):
    subsampling_hidden_dim = None
    if int(args.temporal_subsampling_hidden_dim) > 0:
        subsampling_hidden_dim = int(args.temporal_subsampling_hidden_dim)

    if args.arch == "embedded_rnn":
        model = EmbeddedRNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            enable_temporal_subsampling=args.enable_temporal_subsampling,
            temporal_subsampling_type=args.temporal_subsampling_type,
            temporal_subsampling_factor=args.temporal_subsampling_factor,
            temporal_subsampling_hidden_dim=subsampling_hidden_dim,
        )
    elif args.arch == "tcn_bilstm":
        model = TCNBiRNN(
            input_dim=input_dim,
            proj_dim=args.tcn_proj_dim,
            tcn_kernels=parse_kernel_list(args.tcn_kernels),
            rnn_hidden=args.hidden_dim,
            rnn_layers=args.num_layers,
            rnn_type=args.rnn_type.lower(),
            output_dim=output_dim,
            bidirectional=args.bidirectional,
            enable_temporal_subsampling=args.enable_temporal_subsampling,
            temporal_subsampling_type=args.temporal_subsampling_type,
            temporal_subsampling_factor=args.temporal_subsampling_factor,
            temporal_subsampling_hidden_dim=subsampling_hidden_dim,
        )
    else:
        raise ValueError(f"Unsupported arch: {args.arch}")
    return model


def project_input_lengths(model: nn.Module, input_lens: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "transform_input_lengths"):
        return model.transform_input_lengths(input_lens)
    return input_lens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/asl-fingerspelling")
    p.add_argument("--train_csv", type=str, default="train.csv")
    p.add_argument("--supplemental_csv", type=str, default="supplemental_metadata.csv")
    p.add_argument("--use_supplemental_data", action="store_true")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logdir", type=str, default="artifacts/logs")
    p.add_argument("--max_frames", type=int, default=160)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip_norm", type=float, default=0.0)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--arch", type=str, default="embedded_rnn", choices=["embedded_rnn", "tcn_bilstm"])
    p.add_argument("--rnn_type", type=str, default="lstm", choices=["rnn", "gru", "lstm"])
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--tcn_proj_dim", type=int, default=256)
    p.add_argument("--tcn_kernels", type=str, default="3,3,3")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_size", type=int, default=200)
    p.add_argument("--val_size", type=int, default=200)
    p.add_argument("--max_phrase_len", type=int, default=0)
    p.add_argument("--overfit_subset", type=int, default=0)
    p.add_argument("--eval_train_metrics", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--disable_pin_memory", action="store_true")
    p.add_argument("--use_delta_features", action="store_true")
    p.add_argument("--disable_landmark_centering", action="store_true")
    p.add_argument("--landmark_scale_mode", type=str, default="median_radius")
    p.add_argument("--early_stopping_patience", type=int, default=0)
    p.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    p.add_argument("--save_best_only", action="store_true")

    # Optional W&B
    p.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb_project", type=str, default="fingerspelling_asl")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_tags", type=str, default="", help="Comma-separated tags for W&B")

    # Compatibility-only args to avoid parser mismatch with old run configs.
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--conformer_d_model", type=int, default=256)
    p.add_argument("--conformer_layers", type=int, default=8)
    p.add_argument("--conformer_heads", type=int, default=4)
    p.add_argument("--conformer_ff_expansion", type=int, default=4)
    p.add_argument("--conformer_conv_kernel", type=int, default=15)
    p.add_argument("--conformer_dropout", type=float, default=0.1)
    p.add_argument("--conformer_subsample_stride", type=int, default=1)
    p.add_argument("--augment_prob", type=float, default=0.0)
    p.add_argument("--augment_train", action="store_true")
    p.add_argument("--aug_landmark_drop", type=float, default=0.0)
    p.add_argument("--aug_frame_drop", type=float, default=0.0)
    p.add_argument("--aug_shift", type=float, default=0.0)
    p.add_argument("--aug_scale", type=float, default=0.0)
    p.add_argument("--aug_rot_deg", type=float, default=0.0)
    p.add_argument("--disable_feature_norm", action="store_true")
    p.add_argument("--disable_scale_normalization", action="store_true")
    p.add_argument("--decode_blank_skip_threshold", type=float, default=1.0)
    p.add_argument("--beam_size", type=int, default=1)
    p.add_argument("--representative_proxy", action="store_true")
    p.add_argument("--proxy_len_bins", type=int, default=4)
    p.add_argument("--letters_only", action="store_true")
    p.add_argument("--lowercase_phrases", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--enable_temporal_subsampling", action="store_true")
    p.add_argument("--temporal_subsampling_type", type=str, default="conv", choices=["conv", "avgpool", "maxpool"])
    p.add_argument("--temporal_subsampling_factor", type=int, default=2, choices=[2, 4])
    p.add_argument("--temporal_subsampling_hidden_dim", type=int, default=0)

    args = p.parse_args()

    train_csv = args.train_csv if os.path.isabs(args.train_csv) else os.path.join(args.data_dir, args.train_csv)
    supplemental_csv = (
        args.supplemental_csv
        if os.path.isabs(args.supplemental_csv)
        else os.path.join(args.data_dir, args.supplemental_csv)
    )
    vocab_json = os.path.join(args.data_dir, "character_to_prediction_index.json")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing {train_csv}")
    if not os.path.exists(vocab_json):
        raise FileNotFoundError(f"Missing {vocab_json}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    letter_to_int, int_to_letter, blank_id = build_ctc_vocab(vocab_json)
    df = build_dataframes(args, train_csv=train_csv, supplemental_csv=supplemental_csv)
    df = filter_to_available_files(df, data_dir=args.data_dir)
    if len(df) == 0:
        raise ValueError("No rows left after filtering by available parquet files.")

    if args.lowercase_phrases:
        df["phrase"] = df["phrase"].astype(str).str.lower()
    if args.letters_only:
        df["phrase"] = df["phrase"].astype(str).str.replace(r"[^a-z]", "", regex=True)

    if args.max_phrase_len > 0:
        df = df[df["phrase"].astype(str).str.len() <= args.max_phrase_len].copy()
        if len(df) == 0:
            raise ValueError("No rows left after max_phrase_len filtering.")

    if args.overfit_subset > 0:
        n_subset = min(args.overfit_subset, len(df))
        overfit_df = df.sample(n=n_subset, random_state=args.seed).copy()
        overfit_df["encoded"] = overfit_df["phrase"].apply(lambda x: encode_phrase(str(x), letter_to_int))
        train_df = overfit_df.copy()
        val_df = overfit_df.copy()
        print(f"Overfit mode enabled: using same {n_subset} samples for train and val")
    else:
        train_df, val_df = split_by_participant(df, val_ratio=args.val_ratio, seed=args.seed)
        train_df["encoded"] = train_df["phrase"].apply(lambda x: encode_phrase(str(x), letter_to_int))
        val_df["encoded"] = val_df["phrase"].apply(lambda x: encode_phrase(str(x), letter_to_int))

        if args.train_size and args.train_size > 0 and args.train_size < len(train_df):
            train_df = train_df.sample(args.train_size, random_state=args.seed)
        if args.val_size and args.val_size > 0 and args.val_size < len(val_df):
            val_df = val_df.sample(args.val_size, random_state=args.seed)

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    train_ds = ASLRightHandDataset(
        train_df,
        landmarks_dir=args.data_dir,
        max_frames=args.max_frames,
        use_delta_features=args.use_delta_features,
        normalize_landmarks=(not args.disable_landmark_centering),
        landmark_scale_mode=args.landmark_scale_mode,
    )
    val_ds = ASLRightHandDataset(
        val_df,
        landmarks_dir=args.data_dir,
        max_frames=args.max_frames,
        use_delta_features=args.use_delta_features,
        normalize_landmarks=(not args.disable_landmark_centering),
        landmark_scale_mode=args.landmark_scale_mode,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": not args.disable_pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Infer input dim from first valid batch.
    first_batch = None
    for b in train_loader:
        if b is not None:
            first_batch = b
            break
    if first_batch is None:
        raise RuntimeError("No valid training batch found. Check data filters and phrase encoding.")

    X0, _, _, _ = first_batch
    input_dim = int(X0.shape[2])
    output_dim = max(int_to_letter.keys()) + 1

    model = create_model(args, input_dim=input_dim, output_dim=output_dim).to(device)
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_path = os.path.join(args.logdir, run_name)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    print(f"TensorBoard logdir: {log_path}")

    wandb_enabled = args.use_wandb and args.wandb_mode != "disabled"
    if args.use_wandb and wandb is None:
        raise ImportError("wandb is not installed. Run: pip install wandb")

    if wandb_enabled:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or run_name,
            config=vars(args),
            mode=args.wandb_mode,
            tags=parse_wandb_tags(args.wandb_tags),
        )

    best_cer = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    global_step = 0
    history_rows = []
    t_train_start = time.time()

    best_ckpt_path = os.path.join("artifacts", "models", f"{run_name}_best.pt")
    os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)

    for epoch in range(args.epochs):
        t_epoch_start = time.time()
        model.train()
        losses = []
        blank_ratios = []
        in_tar_ratios = []
        in_lens_before_means = []
        in_lens_after_means = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)

        for batch in pbar:
            if batch is None:
                continue
            X, Y, in_lens, tar_lens = batch
            X = X.to(device, non_blocking=True)
            model_in_lens = project_input_lengths(model, in_lens)

            optimizer.zero_grad()
            log_probs = model(X)  # (T, B, C)
            loss = criterion(log_probs, Y, model_in_lens, tar_lens)
            loss.backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            with torch.no_grad():
                pred_ids = torch.argmax(log_probs, dim=2)  # (T, B)
                blank_mask = (pred_ids == blank_id).float()
                blank_ratios.append(float(blank_mask.mean().item()))
                ratio_vals = (model_in_lens.float() / tar_lens.float().clamp_min(1.0)).detach().cpu()
                in_tar_ratios.append(float(ratio_vals.mean().item()))
                in_lens_before_means.append(float(in_lens.float().mean().item()))
                in_lens_after_means.append(float(model_in_lens.float().mean().item()))

            loss_val = float(loss.item())
            losses.append(loss_val)
            writer.add_scalar("loss/train_step", loss_val, global_step)
            if wandb_enabled:
                wandb.log({"loss/train_step": loss_val, "global_step": global_step}, step=global_step)
            global_step += 1
            pbar.set_postfix(loss=loss_val)

        mean_loss = float(sum(losses) / max(1, len(losses)))
        mean_blank_ratio = float(sum(blank_ratios) / max(1, len(blank_ratios)))
        mean_in_tar_ratio = float(sum(in_tar_ratios) / max(1, len(in_tar_ratios)))
        mean_input_len_before = float(sum(in_lens_before_means) / max(1, len(in_lens_before_means)))
        mean_input_len_after = float(sum(in_lens_after_means) / max(1, len(in_lens_after_means)))
        writer.add_scalar("loss/train", mean_loss, epoch)
        writer.add_scalar("train/loss", mean_loss, epoch)
        writer.add_scalar("diag/blank_ratio_pred", mean_blank_ratio, epoch)
        writer.add_scalar("diag/input_target_len_ratio", mean_in_tar_ratio, epoch)
        writer.add_scalar("diag/input_len_mean_before", mean_input_len_before, epoch)
        writer.add_scalar("diag/input_len_mean_after", mean_input_len_after, epoch)
        print(f"Epoch {epoch + 1}: train loss={mean_loss:.4f}")

        train_metrics = None
        if args.eval_train_metrics:
            train_metrics = evaluate_metrics(model, train_loader, int_to_letter=int_to_letter, device=device, blank_id=blank_id)
            writer.add_scalar("cer/train", train_metrics["cer"], epoch)
            writer.add_scalar("wer/train", train_metrics["wer"], epoch)
            writer.add_scalar("sequence_accuracy/train", train_metrics["sequence_accuracy"], epoch)
            writer.add_scalar("avg_edit_distance/train", train_metrics["avg_edit_distance"], epoch)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                Xv, Yv, in_lens_v, tar_lens_v = batch
                Xv = Xv.to(device, non_blocking=True)
                model_in_lens_v = project_input_lengths(model, in_lens_v)
                log_probs_v = model(Xv)
                val_losses.append(float(criterion(log_probs_v, Yv, model_in_lens_v, tar_lens_v).item()))

        val_loss = float(sum(val_losses) / max(1, len(val_losses)))
        metrics = evaluate_metrics(model, val_loader, int_to_letter=int_to_letter, device=device, blank_id=blank_id)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("cer/val", metrics["cer"], epoch)
        writer.add_scalar("val/CER", metrics["cer"], epoch)
        writer.add_scalar("wer/val", metrics["wer"], epoch)
        writer.add_scalar("sequence_accuracy/val", metrics["sequence_accuracy"], epoch)
        writer.add_scalar("val/sequence_accuracy", metrics["sequence_accuracy"], epoch)
        writer.add_scalar("avg_edit_distance/val", metrics["avg_edit_distance"], epoch)

        if wandb_enabled:
            payload = {
                "epoch": epoch + 1,
                "loss/train": mean_loss,
                "train/loss": mean_loss,
                "loss/val": val_loss,
                "val/loss": val_loss,
                "diag/blank_ratio_pred": mean_blank_ratio,
                "diag/input_target_len_ratio": mean_in_tar_ratio,
                "diag/input_len_mean_before": mean_input_len_before,
                "diag/input_len_mean_after": mean_input_len_after,
                "cer/val": metrics["cer"],
                "val/CER": metrics["cer"],
                "wer/val": metrics["wer"],
                "sequence_accuracy/val": metrics["sequence_accuracy"],
                "val/sequence_accuracy": metrics["sequence_accuracy"],
                "avg_edit_distance/val": metrics["avg_edit_distance"],
                "global_step": global_step,
            }
            if train_metrics is not None:
                payload.update(
                    {
                        "cer/train": train_metrics["cer"],
                        "wer/train": train_metrics["wer"],
                        "sequence_accuracy/train": train_metrics["sequence_accuracy"],
                        "avg_edit_distance/train": train_metrics["avg_edit_distance"],
                    }
                )
            wandb.log(payload, step=global_step)

        print(
            f"Epoch {epoch + 1}: val CER={metrics['cer']:.4f} | "
            f"WER={metrics['wer']:.4f} | ExactMatch={metrics['sequence_accuracy']:.4f} | "
            f"AvgEditDist={metrics['avg_edit_distance']:.4f}"
        )
        epoch_time_sec = float(time.time() - t_epoch_start)
        print(f"Epoch {epoch + 1}: time={epoch_time_sec:.1f}s")

        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": mean_loss,
                "blank_ratio_pred": mean_blank_ratio,
                "input_target_len_ratio": mean_in_tar_ratio,
                "input_len_mean_before": mean_input_len_before,
                "input_len_mean_after": mean_input_len_after,
                "val_loss": val_loss,
                "val_cer": metrics["cer"],
                "val_wer": metrics["wer"],
                "val_exact_match": metrics["sequence_accuracy"],
                "val_avg_edit_distance": metrics["avg_edit_distance"],
                "epoch_time_sec": epoch_time_sec,
                "global_step": global_step,
            }
        )

        # Save epoch checkpoint unless best-only mode.
        if not args.save_best_only:
            ckpt_path = os.path.join("artifacts", "models", f"{run_name}_epoch{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": vars(args),
                },
                ckpt_path,
            )

        # Best checkpoint and early stopping
        val_cer = metrics["cer"]
        if val_cer + args.early_stopping_min_delta < best_cer:
            best_cer = val_cer
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": vars(args),
                },
                best_ckpt_path,
            )
            print(f"Epoch {epoch + 1}: new best val CER={best_cer:.4f} (saved best checkpoint: {best_ckpt_path})")
        else:
            epochs_no_improve += 1

        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch + 1}: "
                f"no val CER improvement for {epochs_no_improve} epochs. "
                f"Best={best_cer:.4f} at epoch {best_epoch}."
            )
            break

    if wandb_enabled:
        log_examples_to_wandb(
            model=model,
            dataloader=val_loader,
            int_to_letter=int_to_letter,
            device=device,
            blank_id=blank_id,
            global_step=global_step,
            split_name="val",
            n_examples=5,
        )

    # Persist epoch history as CSV for quick analysis/review.
    hist_csv = os.path.join(log_path, "metrics_history.csv")
    pd.DataFrame(history_rows).to_csv(hist_csv, index=False)

    # Persist run manifest for reproducibility and MLOps traceability.
    total_train_time_sec = float(time.time() - t_train_start)
    manifest = {
        "run_name": run_name,
        "git_commit": get_git_commit(),
        "best_val_cer": best_cer,
        "best_epoch": best_epoch,
        "epochs_requested": args.epochs,
        "epochs_executed": len(history_rows),
        "total_train_time_sec": total_train_time_sec,
        "best_checkpoint_path": best_ckpt_path,
        "logdir": log_path,
        "config": vars(args),
    }
    manifest_path = os.path.join(log_path, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    writer.close()
    if wandb_enabled:
        wandb.finish()

    print("Done.")


if __name__ == "__main__":
    main()
