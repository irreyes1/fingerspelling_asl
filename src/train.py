import argparse
import json
import os
import re
from datetime import datetime
from typing import List, Tuple

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

try:
    from clearml import Task
except ImportError:
    Task = None

from src.data.dataset import ASLRightHandDataset, collate_fn
from src.data.vocab import build_ctc_vocab, encode_phrase
from src.models.embedded_rnn import EmbeddedRNN
from src.models.tcn_bilstm import TCNBiRNN
from src.utils.metrics import ctc_greedy_decode, evaluate_metrics


def split_by_participant(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42):
    # Split by participant_id to avoid leakage
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
            outputs = model(X, input_lens)  # (T, B, C)
            batch_size = outputs.shape[1]
            y_list = Y.detach().cpu().tolist()

            start = 0
            for i in range(batch_size):
                valid_t = int(input_lens[i].item())
                pred_text = ctc_greedy_decode(
                    outputs[:valid_t, i, :], int_to_letter, blank_id
                )
                target_len = int(target_lens[i].item())
                tgt_ids = y_list[start : start + target_len]
                start += target_len
                tgt_text = "".join(
                    int_to_letter.get(int(t), "") for t in tgt_ids if int(t) != blank_id
                )
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
        # Keep the persistent rule of logging 5 rows even on tiny subsets.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/asl-fingerspelling")
    p.add_argument("--train_csv", type=str, default="train.csv")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--logdir", type=str, default="artifacts/logs")
    p.add_argument("--max_frames", type=int, default=160)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="L2 regularization (Adam weight_decay)",
    )
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for model")
    p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Stop if val CER doesn't improve for N epochs (0=disabled)",
    )
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument(
        "--tcn_kernels",
        type=str,
        default="3,3,3",
        help="Comma-separated kernel sizes for TCN blocks",
    )
    p.add_argument("--rnn_layers", type=int, default=2)
    p.add_argument(
        "--rnn_type", type=str, default="lstm", choices=["lstm", "gru", "rnn"]
    )
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes for parallel data loading")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_size", type=int, default=200)  # small by default
    p.add_argument("--val_size", type=int, default=200)
    p.add_argument(
        "--use_supplemental",
        action="store_true",
        help="Also load supplemental_metadata.csv + supplemental_landmarks/.",
    )
    p.add_argument(
        "--max_phrase_len",
        type=int,
        default=0,
        help="If >0, keep only samples with phrase length <= this value.",
    )
    p.add_argument(
        "--overfit_subset",
        type=int,
        default=0,
        help="If >0, sample this many rows and use same subset for train/val.",
    )
    p.add_argument(
        "--eval_train_metrics",
        action="store_true",
        help="Also compute CER/WER/ExactMatch/AvgEditDist on train split.",
    )

    # Optional Weights & Biases tracking
    p.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb_project", type=str, default="fingerspelling_asl")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    p.add_argument(
        "--wandb_tags", type=str, default="", help="Comma-separated tags for W&B"
    )

    args = p.parse_args()

    # ClearML experiment tracking & queue integration
    # Appends short task ID to run_name to prevent checkpoint overwrites across runs
    if Task is not None:
        _task = Task.init(
            project_name="fingerspelling_asl",
            task_name=args.run_name or "train",
            task_type=Task.TaskTypes.training,
        )
        args.run_name = f"{args.run_name or 'train'}_{_task.id[:8]}"

    train_csv = args.train_csv
    if not os.path.isabs(train_csv):
        train_csv = os.path.join(args.data_dir, train_csv)
    vocab_json = os.path.join(args.data_dir, "character_to_prediction_index.json")
    landmarks_dir = os.path.join(args.data_dir, "train_landmarks")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing {train_csv}")
    if not os.path.exists(vocab_json):
        raise FileNotFoundError(f"Missing {vocab_json}")
    if not os.path.isdir(landmarks_dir):
        raise FileNotFoundError(f"Missing folder {landmarks_dir}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load vocab mapping (char -> id) in CTC-compatible form
    letter_to_int, int_to_letter, blank_id = build_ctc_vocab(vocab_json)

    # Load train.csv
    df = pd.read_csv(train_csv)
    required_cols = {"file_id", "sequence_id", "participant_id", "phrase"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {missing}")

    # Filter by parquets you actually downloaded
    have_ids = existing_file_ids(landmarks_dir)
    if not have_ids:
        raise ValueError(
            f"No parquet files found in {landmarks_dir}. "
            f"Download a few like 0.parquet, 1.parquet, etc."
        )
    df = df[df["file_id"].isin(have_ids)].copy()
    print(
        f"Rows after filtering to available parquets ({len(have_ids)} file_ids): {len(df)}"
    )

    if args.use_supplemental:
        supp_csv = os.path.join(args.data_dir, "supplemental_metadata.csv")
        supp_landmarks = os.path.join(args.data_dir, "supplemental_landmarks")
        if os.path.exists(supp_csv) and os.path.isdir(supp_landmarks):
            supp_df = pd.read_csv(supp_csv)
            supp_have = existing_file_ids(supp_landmarks)
            supp_df = supp_df[supp_df["file_id"].isin(supp_have)].copy()
            supp_df["_landmarks_dir"] = supp_landmarks
            print(f"Supplemental rows: {len(supp_df)} ({len(supp_have)} file_ids)")
            df["_landmarks_dir"] = landmarks_dir
            df = pd.concat([df, supp_df], ignore_index=True)
            print(f"Combined total rows: {len(df)}")
        else:
            print(f"Warning: supplemental data not found at {supp_csv}, skipping")

    _clean_re = re.compile(r"^[a-z ]+$")
    df["phrase"] = df["phrase"].astype(str).str.lower().str.strip()
    df = df[
        df["phrase"].apply(lambda x: bool(_clean_re.match(x)) and len(x) > 0)
    ].copy()
    print(f"Rows after filtering to letters-only phrases: {len(df)}")

    # Pre-filter: remove sequences with no right-hand data (all NaN).
    # These are left-hand signers or detection failures — waste of training time.
    from src.data.dataset import count_valid_frames, read_right_hand_sequence

    print("Pre-filtering sequences with no right-hand landmarks...")
    valid_mask = []
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Checking landmarks", leave=False
    ):
        fid = int(row["file_id"])
        sid = int(row["sequence_id"])
        lm_dir = landmarks_dir
        if "_landmarks_dir" in row.index:
            lm_dir = row["_landmarks_dir"]
        ppath = os.path.join(lm_dir, f"{fid}.parquet")
        if not os.path.exists(ppath):
            valid_mask.append(False)
            continue
        X_raw = read_right_hand_sequence(ppath, sid)
        n_valid = count_valid_frames(X_raw)
        valid_mask.append(n_valid > 0)
    df = df[valid_mask].copy()
    print(f"Rows after filtering no-data sequences: {len(df)}")

    if args.max_phrase_len > 0:
        df = df[df["phrase"].astype(str).str.len() <= args.max_phrase_len].copy()
        print(
            f"Rows after filtering by max_phrase_len={args.max_phrase_len}: {len(df)}"
        )
        if len(df) == 0:
            raise ValueError("No rows left after max_phrase_len filtering.")

    if args.overfit_subset > 0:
        n_subset = min(args.overfit_subset, len(df))
        overfit_df = df.sample(n=n_subset, random_state=args.seed).copy()
        overfit_df["encoded"] = overfit_df["phrase"].apply(
            lambda x: encode_phrase(str(x), letter_to_int)
        )
        train_df = overfit_df.copy()
        val_df = overfit_df.copy()
        print(f"Overfit mode enabled: using same {n_subset} samples for train and val")
    else:
        # Split by participant_id
        train_df, val_df = split_by_participant(
            df, val_ratio=args.val_ratio, seed=args.seed
        )

        # Encode targets
        train_df["encoded"] = train_df["phrase"].apply(
            lambda x: encode_phrase(str(x), letter_to_int)
        )
        val_df["encoded"] = val_df["phrase"].apply(
            lambda x: encode_phrase(str(x), letter_to_int)
        )

        # Sample small subsets for local dev
        if args.train_size and args.train_size < len(train_df):
            train_df = train_df.sample(args.train_size, random_state=args.seed)
        if args.val_size and args.val_size < len(val_df):
            val_df = val_df.sample(args.val_size, random_state=args.seed)

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    # Datasets / loaders
    train_ds = ASLRightHandDataset(
        train_df,
        landmarks_dir=landmarks_dir,
        max_frames=args.max_frames,
        use_per_row_dir=args.use_supplemental,
        training=True,
    )
    val_ds = ASLRightHandDataset(
        val_df,
        landmarks_dir=landmarks_dir,
        max_frames=args.max_frames,
        use_per_row_dir=args.use_supplemental,
        training=False,
    )

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=use_cuda, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=use_cuda, persistent_workers=args.num_workers > 0)

    # Model — 63 landmarks + 63 delta features = 126
    input_dim = 126
    output_dim = max(int_to_letter.keys()) + 1

    model = EmbeddedRNN(
        input_dim, args.hidden_dim, output_dim, dropout=args.dropout
    ).to(device)

    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    # Tracking setup
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    # TensorBoard
    log_path = os.path.join(args.logdir, run_name)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    print(f"TensorBoard logdir: {log_path}")

    # Weights & Biases
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

    global_step = 0
    best_val_cer = float("inf")
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        blank_ratios = []
        in_tar_ratios = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)

        for batch in pbar:
            if batch is None:
                continue
            X, Y, in_lens, tar_lens = batch
            X = X.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_cuda):
                log_probs = model(X, in_lens)  # (T, B, C)
                loss = criterion(log_probs, Y, in_lens, tar_lens)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: skipping batch with loss={loss.item()}")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            # Simple diagnostics: blank-token dominance and input/target length ratio.
            with torch.no_grad():
                pred_ids = torch.argmax(log_probs, dim=2)  # (T, B)
                blank_mask = (pred_ids == blank_id).float()
                blank_ratios.append(float(blank_mask.mean().item()))
                ratio_vals = (
                    (in_lens.float() / tar_lens.float().clamp_min(1.0)).detach().cpu()
                )
                in_tar_ratios.append(float(ratio_vals.mean().item()))

            loss_val = float(loss.item())
            losses.append(loss_val)

            writer.add_scalar("loss/train_step", loss_val, global_step)
            if wandb_enabled:
                wandb.log(
                    {"loss/train_step": loss_val, "global_step": global_step},
                    step=global_step,
                )

            global_step += 1
            pbar.set_postfix(loss=loss_val)

        mean_train_loss = float(sum(losses) / max(1, len(losses)))
        mean_blank_ratio = float(sum(blank_ratios) / max(1, len(blank_ratios)))
        mean_in_tar_ratio = float(sum(in_tar_ratios) / max(1, len(in_tar_ratios)))
        writer.add_scalar("loss/train", mean_train_loss, epoch)
        writer.add_scalar("diag/blank_ratio_pred", mean_blank_ratio, epoch)
        writer.add_scalar("diag/input_target_len_ratio", mean_in_tar_ratio, epoch)
        print(f"Epoch {epoch + 1}: train loss={mean_train_loss:.4f}")

        metrics_train = None
        if args.eval_train_metrics:
            metrics_train = evaluate_metrics(
                model,
                train_loader,
                int_to_letter=int_to_letter,
                device=device,
                blank_id=blank_id,
            )
            writer.add_scalar("cer/train", metrics_train["cer"], epoch)
            writer.add_scalar("wer/train", metrics_train["wer"], epoch)
            writer.add_scalar(
                "sequence_accuracy/train", metrics_train["sequence_accuracy"], epoch
            )
            writer.add_scalar(
                "avg_edit_distance/train", metrics_train["avg_edit_distance"], epoch
            )

        # Validation metrics
        metrics_val = evaluate_metrics(
            model,
            val_loader,
            int_to_letter=int_to_letter,
            device=device,
            blank_id=blank_id,
            loss_fn=criterion,
        )
        writer.add_scalar("loss/val", metrics_val["loss"], epoch)
        writer.add_scalar("cer/val", metrics_val["cer"], epoch)
        writer.add_scalar("wer/val", metrics_val["wer"], epoch)
        writer.add_scalar(
            "sequence_accuracy/val", metrics_val["sequence_accuracy"], epoch
        )
        writer.add_scalar(
            "avg_edit_distance/val", metrics_val["avg_edit_distance"], epoch
        )

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("learning_rate", current_lr, epoch)
        scheduler.step(metrics_val["cer"])

        if wandb_enabled:
            payload = {
                "epoch": epoch + 1,
                "loss/train": mean_train_loss,
                "loss/val": metrics_val["loss"],
                "diag/blank_ratio_pred": mean_blank_ratio,
                "diag/input_target_len_ratio": mean_in_tar_ratio,
                "cer/val": metrics_val["cer"],
                "wer/val": metrics_val["wer"],
                "sequence_accuracy/val": metrics_val["sequence_accuracy"],
                "avg_edit_distance/val": metrics_val["avg_edit_distance"],
                "global_step": global_step,
                "lr": current_lr,
            }
            if metrics_train is not None:
                payload.update(
                    {
                        "cer/train": metrics_train["cer"],
                        "wer/train": metrics_train["wer"],
                        "sequence_accuracy/train": metrics_train["sequence_accuracy"],
                        "avg_edit_distance/train": metrics_train["avg_edit_distance"],
                    }
                )
            wandb.log(payload, step=global_step)

        if metrics_train is not None:
            print(
                f"Epoch {epoch + 1}: "
                f"train CER={metrics_train['cer']:.4f} | "
                f"WER={metrics_train['wer']:.4f} | "
                f"ExactMatch={metrics_train['sequence_accuracy']:.4f} | "
                f"AvgEditDist={metrics_train['avg_edit_distance']:.4f}"
            )
        print(
            f"Epoch {epoch + 1}: "
            f"lr={current_lr:.4f} | "
            f"diag blank_ratio_pred={mean_blank_ratio:.4f} | "
            f"input/target ratio={mean_in_tar_ratio:.2f}"
        )

        print(
            f"Epoch {epoch + 1}: "
            f"val loss={metrics_val['loss']:.4f} | "
            f"val CER={metrics_val['cer']:.4f} | "
            f"WER={metrics_val['wer']:.4f} | "
            f"ExactMatch={metrics_val['sequence_accuracy']:.4f} | "
            f"AvgEditDist={metrics_val['avg_edit_distance']:.4f}"
        )

        # Save best checkpoint (by val CER)
        if metrics_val["cer"] < best_val_cer:
            best_val_cer = metrics_val["cer"]
            epochs_without_improvement = 0
            ckpt_path = os.path.join("artifacts", "models", f"{run_name}_best.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": vars(args),
                },
                ckpt_path,
            )
            print(f"  -> Saved best model (val CER={best_val_cer:.4f})")
        else:
            epochs_without_improvement += 1

        # Save latest checkpoint (for resuming)
        ckpt_path = os.path.join("artifacts", "models", f"{run_name}_latest.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args),
            },
            ckpt_path,
        )

        # Early stopping
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping: val CER did not improve for {args.early_stopping_patience} epochs. "
                f"Best val CER={best_val_cer:.4f}"
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

    writer.close()
    if wandb_enabled:
        wandb.finish()

    print("Done.")


if __name__ == "__main__":
    main()
