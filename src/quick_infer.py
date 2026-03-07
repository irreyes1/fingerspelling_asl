import os
import ast
import json
import argparse
import warnings

import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import ASLRightHandDataset, collate_fn
from src.model_loader import load_model_from_checkpoint

CTC_BLANK_ID = 0


# ---------------------------------------------------------
# Utility: greedy CTC decode for (T, B, C)
# ---------------------------------------------------------
def greedy_decode_batch(log_probs, idx2char, blank_id=0, input_lens=None):
    """
    log_probs: (T, B, C)
    returns: List[str] (size B)
    """
    preds = torch.argmax(log_probs, dim=2)  # (T, B)

    decoded_strings = []

    T, B = preds.shape

    for b in range(B):
        valid_t = int(input_lens[b]) if input_lens is not None else T
        seq = preds[:valid_t, b].tolist()

        collapsed = []
        prev = None
        for token in seq:
            if token != prev:
                if token != blank_id:
                    collapsed.append(token)
            prev = token

        text = "".join(idx2char[t] for t in collapsed if t in idx2char)
        decoded_strings.append(text)

    return decoded_strings


def encode_phrase(phrase, char_to_idx):
    phrase_str = str(phrase)
    encoded = []
    for c in phrase_str:
        if c in char_to_idx:
            encoded.append(char_to_idx[c])
        elif not c.isspace():
            warnings.warn(f"Caracter '{c}' no encontrado en vocabulario, omitido en: {phrase_str!r}")
    return encoded


def parse_encoded(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Valor 'encoded' invalido: {value!r}") from e
        if isinstance(parsed, list):
            return [int(v) for v in parsed]
    raise ValueError(f"Formato no soportado para 'encoded': {type(value)}")


def load_vocab(args):
    if args.vocab_json is not None:
        vocab_path = args.vocab_json
    else:
        vocab_path = os.path.join(
            os.path.dirname(args.csv), "character_to_prediction_index.json"
        )

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"No encuentro vocabulario JSON: {vocab_path}. Usa --vocab_json para indicarlo."
        )

    with open(vocab_path, "r", encoding="utf-8") as f:
        base_char_to_idx = {k: int(v) for k, v in json.load(f).items()}

    if "<blank>" in base_char_to_idx:
        blank_id = int(base_char_to_idx["<blank>"])
        char_to_idx = base_char_to_idx
    else:
        blank_id = CTC_BLANK_ID
        # Reserve 0 for CTC blank and shift labels by +1
        char_to_idx = {k: v + 1 for k, v in base_char_to_idx.items()}

    idx2char = {int(v): k for k, v in char_to_idx.items()}
    return char_to_idx, idx2char, blank_id


def _project_root():
    """Ruta del proyecto (raiz, donde esta data/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------
# Build dataset like train.py
# ---------------------------------------------------------
def build_dataset(args, char_to_idx, lowercase_phrases: bool = False, letters_only: bool = False):
    root = _project_root()
    if args.csv is None:
        args.csv = os.path.join(root, "data", "asl-fingerspelling", "train.csv")

    if args.landmarks_dir is None:
        args.landmarks_dir = os.path.join(
            root, "data", "asl-fingerspelling", "train_landmarks"
        )

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"No encuentro CSV: {args.csv}")

    if not os.path.isdir(args.landmarks_dir):
        raise FileNotFoundError(f"No encuentro carpeta parquets: {args.landmarks_dir}")

    df = pd.read_csv(args.csv)

    if "phrase" in df.columns:
        if lowercase_phrases:
            df["phrase"] = df["phrase"].astype(str).str.lower()
        if letters_only:
            df["phrase"] = df["phrase"].astype(str).str.replace(r"[^a-z]", "", regex=True)
        df = df[df["phrase"].astype(str).str.len() > 0].copy()

    # Re-encode from phrase when text filters are active to keep consistency with training.
    if "encoded" not in df.columns or lowercase_phrases or letters_only:
        if "phrase" not in df.columns:
            raise ValueError("El CSV no tiene ni 'encoded' ni 'phrase'.")
        df["encoded"] = df["phrase"].apply(lambda x: encode_phrase(x, char_to_idx))
    else:
        df["encoded"] = df["encoded"].apply(parse_encoded)

    if args.n is not None and args.n > 0:
        df = df.head(args.n).reset_index(drop=True)

    dataset = ASLRightHandDataset(
        df=df,
        landmarks_dir=args.landmarks_dir,
        max_frames=args.max_frames,
    )

    return dataset


# ---------------------------------------------------------
# Load checkpoint safely
# ---------------------------------------------------------
# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(description="Quick inference ASL")

    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (.pt)")
    parser.add_argument("--n", type=int, default=16,
                        help="Number of samples to use")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to train.csv")
    parser.add_argument("--landmarks_dir", type=str, default=None,
                        help="Folder containing parquet files")
    parser.add_argument("--max_frames", type=int, default=160,
                        help="Max frames (must match training)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vocab_json", type=str, default=None,
                        help="Path to character_to_prediction_index.json")
    parser.add_argument(
        "--letters_only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force letters_only filter; by default follows checkpoint config if available",
    )
    parser.add_argument(
        "--lowercase_phrases",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force lowercase filter; by default follows checkpoint config if available",
    )

    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"No encuentro checkpoint: {args.ckpt}")

    root = _project_root()
    if args.csv is None:
        args.csv = os.path.join(root, "data", "asl-fingerspelling", "train.csv")

    char_to_idx, idx2char, blank_id = load_vocab(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    loaded = load_model_from_checkpoint(ckpt_path=args.ckpt, device=device)
    model = loaded.model

    ckpt_cfg = loaded.config or {}
    lowercase_phrases = bool(ckpt_cfg.get("lowercase_phrases", False)) if args.lowercase_phrases is None else bool(args.lowercase_phrases)
    letters_only = bool(ckpt_cfg.get("letters_only", False)) if args.letters_only is None else bool(args.letters_only)
    print(f"Text filters: lowercase={lowercase_phrases} letters_only={letters_only}")

    print("Building dataset...")
    dataset = build_dataset(args, char_to_idx, lowercase_phrases=lowercase_phrases, letters_only=letters_only)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Buscar primer batch valido
    batch = None
    for b in loader:
        if b is not None:
            batch = b
            break

    if batch is None:
        raise RuntimeError("No se encontro ningun batch valido.")

    X, Y, input_lens, target_lens = batch

    X = X.to(device)

    if int(loaded.input_dim) != int(X.shape[2]):
        raise ValueError(
            f"Input dim mismatch: model expects {loaded.input_dim}, dataset has {X.shape[2]}. "
            "Use data/preprocessing compatible with this checkpoint."
        )

    print("Running inference...")
    with torch.no_grad():
        log_probs = model(X)  # (T, B, C)
    model_input_lens = input_lens
    if hasattr(model, "transform_input_lengths"):
        model_input_lens = model.transform_input_lengths(input_lens)

    preds = greedy_decode_batch(
        log_probs,
        idx2char=idx2char,
        blank_id=blank_id,
        input_lens=model_input_lens,
    )

    # reconstruir GT desde Y concatenado
    gt_texts = []
    offset = 0
    for length in target_lens:
        length_int = int(length)
        tokens = Y[offset: offset + length_int].tolist()
        text = "".join(idx2char.get(t, "") for t in tokens if t != blank_id)
        gt_texts.append(text)
        offset += length_int

    print("\n========== RESULTADOS ==========")
    for i in range(len(preds)):
        print(f"[{i}]")
        print("GT   :", gt_texts[i])
        print("PRED :", preds[i])
        print("-" * 40)


if __name__ == "__main__":
    main()
