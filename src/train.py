import os
import re
import json
import argparse
from fsspec.implementations.local import LocalFileSystem 
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

from src.models.embedded_rnn import EmbeddedRNN
from src.data.dataset import ASLRightHandDataset, collate_fn
from src.utils.metrics import evaluate_metrics, ctc_greedy_decode
from src.utils.filesystem import get_filesystem, join_path

CTC_BLANK_ID = 0


def encode_phrase(phrase: str, letter_to_int: dict) -> list:
    return [letter_to_int[c] for c in phrase if c in letter_to_int]


def build_ctc_vocab(vocab_json_path: str, fs=LocalFileSystem()) -> Tuple[dict, dict, int]:
    with fs.open(vocab_json_path, "r") as f:
        base_char_to_idx = {k: int(v) for k, v in json.load(f).items()}

    if "<blank>" in base_char_to_idx:
        blank_id = int(base_char_to_idx["<blank>"])
        char_to_idx = base_char_to_idx
    else:
        blank_id = CTC_BLANK_ID
        # Reserve 0 for CTC blank and shift labels by +1
        char_to_idx = {k: v + 1 for k, v in base_char_to_idx.items()}

    idx_to_char = {int(v): k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char, blank_id


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


def existing_file_ids(landmarks_dir: str, fs=LocalFileSystem()):
    if not fs.isdir(landmarks_dir):
        return set()
    out = set()
    filenames = [f['name'] for f in fs.listdir(landmarks_dir)]
    matches = [re.search(r'(\d+).parquet', fn) for fn in filenames]
    out = set(int(m.group(1)) for m in matches if m is not None)
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
            outputs = model(X)  # (T, B, C)
            batch_size = outputs.shape[1]
            y_list = Y.detach().cpu().tolist()

            start = 0
            for i in range(batch_size):
                valid_t = int(input_lens[i].item())
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
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_size", type=int, default=200)  # small by default
    p.add_argument("--val_size", type=int, default=200)
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
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_tags", type=str, default="", help="Comma-separated tags for W&B")

    args = p.parse_args()

    fs = get_filesystem(args.data_dir)
    train_csv = args.train_csv
    
    if not os.path.isabs(train_csv):
        train_csv = join_path(fs, args.data_dir, train_csv)
    vocab_json =join_path(fs,args.data_dir, "character_to_prediction_index.json")
    landmarks_dir = join_path(fs, args.data_dir, "train_landmarks")

    if not fs.exists(train_csv):
        raise FileNotFoundError(f"Missing {train_csv}")
    if not fs.exists(vocab_json):
        raise FileNotFoundError(f"Missing {vocab_json}")
    if not fs.isdir(landmarks_dir):
        raise FileNotFoundError(f"Missing folder {landmarks_dir}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load vocab mapping (char -> id) in CTC-compatible form
    letter_to_int, int_to_letter, blank_id = build_ctc_vocab(vocab_json, fs)

    # Load train.csv
    df = pd.read_csv(train_csv)
    required_cols = {"file_id", "sequence_id", "participant_id", "phrase"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {missing}")

    # Filter by parquets you actually downloaded
    have_ids = existing_file_ids(landmarks_dir, fs)
    if not have_ids:
        raise ValueError(
            f"No parquet files found in {landmarks_dir}. "
            f"Download a few like 0.parquet, 1.parquet, etc."
        )
    df = df[df["file_id"].isin(have_ids)].copy()
    print(f"Rows after filtering to available parquets ({len(have_ids)} file_ids): {len(df)}")
    if args.max_phrase_len > 0:
        df = df[df["phrase"].astype(str).str.len() <= args.max_phrase_len].copy()
        print(f"Rows after filtering by max_phrase_len={args.max_phrase_len}: {len(df)}")
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
        # Split by participant_id
        train_df, val_df = split_by_participant(df, val_ratio=args.val_ratio, seed=args.seed)

        # Encode targets
        train_df["encoded"] = train_df["phrase"].apply(lambda x: encode_phrase(str(x), letter_to_int))
        val_df["encoded"] = val_df["phrase"].apply(lambda x: encode_phrase(str(x), letter_to_int))

        # Sample small subsets for local dev
        if args.train_size and args.train_size < len(train_df):
            train_df = train_df.sample(args.train_size, random_state=args.seed)
        if args.val_size and args.val_size < len(val_df):
            val_df = val_df.sample(args.val_size, random_state=args.seed)

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    # Datasets / loaders
    train_ds = ASLRightHandDataset(train_df, landmarks_dir=landmarks_dir, max_frames=args.max_frames)
    val_ds = ASLRightHandDataset(val_df, landmarks_dir=landmarks_dir, max_frames=args.max_frames)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Model
    # Right-hand features are typically 21 landmarks * 3 coords = 63
    input_dim = 63
    hidden_dim = args.hidden_dim
    # output_dim must match max id + 1 (including blank=0)
    output_dim = max(int_to_letter.keys()) + 1

    model = EmbeddedRNN(input_dim, hidden_dim, output_dim).to(device)

    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
            log_probs = model(X)  # (T, B, C)

            loss = criterion(log_probs, Y, in_lens, tar_lens)
            loss.backward()
            optimizer.step()

            # Simple diagnostics: blank-token dominance and input/target length ratio.
            with torch.no_grad():
                pred_ids = torch.argmax(log_probs, dim=2)  # (T, B)
                blank_mask = (pred_ids == blank_id).float()
                blank_ratios.append(float(blank_mask.mean().item()))
                ratio_vals = (in_lens.float() / tar_lens.float().clamp_min(1.0)).detach().cpu()
                in_tar_ratios.append(float(ratio_vals.mean().item()))

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
        writer.add_scalar("loss/train", mean_loss, epoch)
        writer.add_scalar("diag/blank_ratio_pred", mean_blank_ratio, epoch)
        writer.add_scalar("diag/input_target_len_ratio", mean_in_tar_ratio, epoch)
        print(f"Epoch {epoch + 1}: train loss={mean_loss:.4f}")

        train_metrics = None
        if args.eval_train_metrics:
            train_metrics = evaluate_metrics(
                model,
                train_loader,
                int_to_letter=int_to_letter,
                device=device,
                blank_id=blank_id,
            )
            writer.add_scalar("cer/train", train_metrics["cer"], epoch)
            writer.add_scalar("wer/train", train_metrics["wer"], epoch)
            writer.add_scalar("sequence_accuracy/train", train_metrics["sequence_accuracy"], epoch)
            writer.add_scalar("avg_edit_distance/train", train_metrics["avg_edit_distance"], epoch)

        # Validation metrics
        metrics = evaluate_metrics(
            model,
            val_loader,
            int_to_letter=int_to_letter,
            device=device,
            blank_id=blank_id,
        )
        writer.add_scalar("cer/val", metrics["cer"], epoch)
        writer.add_scalar("wer/val", metrics["wer"], epoch)
        writer.add_scalar("sequence_accuracy/val", metrics["sequence_accuracy"], epoch)
        writer.add_scalar("avg_edit_distance/val", metrics["avg_edit_distance"], epoch)

        if wandb_enabled:
            payload = {
                "epoch": epoch + 1,
                "loss/train": mean_loss,
                "diag/blank_ratio_pred": mean_blank_ratio,
                "diag/input_target_len_ratio": mean_in_tar_ratio,
                "cer/val": metrics["cer"],
                "wer/val": metrics["wer"],
                "sequence_accuracy/val": metrics["sequence_accuracy"],
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

        if train_metrics is not None:
            print(
                f"Epoch {epoch + 1}: "
                f"train CER={train_metrics['cer']:.4f} | "
                f"WER={train_metrics['wer']:.4f} | "
                f"ExactMatch={train_metrics['sequence_accuracy']:.4f} | "
                f"AvgEditDist={train_metrics['avg_edit_distance']:.4f}"
            )
        print(
            f"Epoch {epoch + 1}: "
            f"diag blank_ratio_pred={mean_blank_ratio:.4f} | "
            f"input/target ratio={mean_in_tar_ratio:.2f}"
        )

        print(
            f"Epoch {epoch + 1}: "
            f"val CER={metrics['cer']:.4f} | "
            f"WER={metrics['wer']:.4f} | "
            f"ExactMatch={metrics['sequence_accuracy']:.4f} | "
            f"AvgEditDist={metrics['avg_edit_distance']:.4f}"
        )

        # Save checkpoint
        ckpt_path = os.path.join("artifacts", "models", f"{run_name}_epoch{epoch + 1}.pt")
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
