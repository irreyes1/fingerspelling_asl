import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import pyarrow.parquet as pq
import pyarrow as pa

MAX_FRAMES_DEFAULT = 160

_RIGHT_HAND_COLS: Optional[List[str]] = None

def _get_right_hand_cols(parquet_path: str) -> List[str]:
    """
    Detect columns containing 'right_hand' from a parquet schema (cached globally).
    Kaggle ASL Fingerspelling parquets store flattened landmarks per frame as columns.
    """
    global _RIGHT_HAND_COLS
    if _RIGHT_HAND_COLS is None:
        pq_file = pq.ParquetFile(parquet_path)
        _RIGHT_HAND_COLS = [c for c in pq_file.schema.names if "right_hand" in c]
        if not _RIGHT_HAND_COLS:
            raise ValueError(
                f"No columns containing 'right_hand' found in parquet schema: {parquet_path}"
            )
    return _RIGHT_HAND_COLS

def read_right_hand_sequence(parquet_path: str, sequence_id: int) -> np.ndarray:
    cols = _get_right_hand_cols(parquet_path)
    table = pq.read_table(
        parquet_path,
        filters=[("sequence_id", "=", sequence_id)],
        columns=cols,
    )
    X = table.to_pandas().values.astype(np.float32)  # (T, D)
    return X

def compute_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute distances between fingertips. X shape: (T, 63) -> returns (T, 10)."""
    T = X.shape[0]
    pts = X.reshape(T, 21, 3)
    tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
    dists = []
    for i in range(len(tips)):
      for j in range(i + 1, len(tips)):
          d = np.linalg.norm(pts[:, tips[i], :] - pts[:, tips[j], :], axis=1)
          dists.append(d)
    return np.stack(dists, axis=1).astype(np.float32)  # (T, 10)

def count_valid_frames(X: np.ndarray) -> int:
    # A frame is valid if not all values are NaN
    return int(np.sum(~np.all(np.isnan(X), axis=1)))

def normalize_frames(X: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Pad/truncate to fixed length, keeping NaNs as-is for valid-frame counting.
    Pads with zeros.
    """
    T, D = X.shape
    if T > max_frames:
        return X[:max_frames]
    if T < max_frames:
        pad = np.zeros((max_frames - T, D), dtype=np.float32)
        return np.vstack([X, pad])
    return X

class ASLRightHandDataset(Dataset):
      """
      Each item:
        X: (max_frames, D) float32
        Y: (U,) long (targets)
        input_len: int (valid frames before padding/trunc)
        target_len: int (U)
      """
      def __init__(
          self,
          df: pd.DataFrame,
          landmarks_dir: str,
          max_frames: int = MAX_FRAMES_DEFAULT,
          use_per_row_dir: bool = False,
          training: bool = False,
      ):
          self.df = df.reset_index(drop=True)
          self.landmarks_dir = landmarks_dir
          self.max_frames = max_frames
          self.use_per_row_dir = use_per_row_dir
          self.training = training

      def __len__(self) -> int:
          return len(self.df)

      def __getitem__(self, idx: int):
          row = self.df.iloc[idx]
          file_id = int(row["file_id"])
          sequence_id = int(row["sequence_id"])

          if self.use_per_row_dir and "_landmarks_dir" in row.index:
              lm_dir = row["_landmarks_dir"]
          else:
              lm_dir = self.landmarks_dir

          parquet_path = os.path.join(lm_dir, f"{file_id}.parquet")
          if not os.path.exists(parquet_path):
              return None

          X_raw = read_right_hand_sequence(parquet_path, sequence_id)  # (T, D)

          # Drop frames where ALL landmarks are NaN (hand not detected).
          valid_mask = ~np.all(np.isnan(X_raw), axis=1)
          X_clean = X_raw[valid_mask]

          if len(X_clean) == 0:
              return None

          # Zero remaining per-landmark NaN (rare)
          X_clean = np.nan_to_num(X_clean, nan=0.0)

          # Augmentation (training only, before normalize so shifts are in raw coords)
          if self.training:
              X_clean = augment(X_clean)

          input_len = min(len(X_clean), self.max_frames)

          X = normalize_frames(X_clean, self.max_frames)
          X = normalize_landmarks(X)

          # Delta features: frame-to-frame velocity
          deltas = compute_deltas(X)
          X = np.concatenate([X, deltas], axis=1)  # (T, 126)

          Y = torch.tensor(row["encoded"], dtype=torch.long)
          target_len = int(len(Y))

          if input_len < target_len or target_len == 0:
              return None

          X = torch.tensor(X, dtype=torch.float32)

          return X, Y, int(min(input_len, self.max_frames)), target_len

def augment(X: np.ndarray) -> np.ndarray:
    """Data augmentation for training. Applied BEFORE normalize_landmarks."""
    # Temporal resample: stretch/compress (simulates different signing speeds)
    if np.random.random() < 0.5:
        T = X.shape[0]
        scale = np.random.uniform(0.8, 1.2)
        new_T = max(1, int(T * scale))
        indices = np.linspace(0, T - 1, new_T).astype(int)
        X = X[indices]

    # Random spatial shift (simulates hand position variation)
    if np.random.random() < 0.5:
        shift = np.random.normal(0, 0.02, size=(1, X.shape[1])).astype(np.float32)
        X = X + shift

    # Random scale jitter (simulates hand size / distance variation)
    if np.random.random() < 0.5:
        scale = np.random.uniform(0.9, 1.1)
        X = X * scale

    return X


def compute_deltas(X: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame velocity (delta features).

    X shape: (T, D) -> returns (T, D) with delta[t] = X[t] - X[t-1], delta[0] = 0.
    """
    deltas = np.zeros_like(X)
    deltas[1:] = X[1:] - X[:-1]
    return deltas

def normalize_landmarks(X: np.ndarray) -> np.ndarray:
    """Center landmarks on wrist and scale by max absolute x/y value.

    X shape: (T, 63) — columns are laid out as:
      x_hand_0..x_hand_20, y_hand_0..y_hand_20, z_hand_0..z_hand_20
    (all x first, then all y, then all z).
    """
    T = X.shape[0]
    xs = X[:, :21].copy()     # (T, 21)
    ys = X[:, 21:42].copy()   # (T, 21)
    zs = X[:, 42:63].copy()   # (T, 21)

    # Center: subtract wrist (landmark 0) per frame
    xs -= xs[:, 0:1]
    ys -= ys[:, 0:1]
    zs -= zs[:, 0:1]

    # Scale: normalize by max absolute x/y across all frames & landmarks
    max_val = max(np.abs(xs).max(), np.abs(ys).max())
    if max_val > 1e-6:
        xs /= max_val
        ys /= max_val
        zs /= max_val

    return np.concatenate([xs, ys, zs], axis=1)  # (T, 63)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    Xs, Ys, in_lens, tar_lens = [], [], [], []
    for X, Y, in_len, tar_len in batch:
        Xs.append(X)
        Ys.append(Y)
        in_lens.append(in_len)
        tar_lens.append(tar_len)

    Xs = torch.stack(Xs)               # (B, T, D)
    Ys = torch.cat(Ys)                 # (sum_U,)
    in_lens = torch.tensor(in_lens, dtype=torch.long)
    tar_lens = torch.tensor(tar_lens, dtype=torch.long)

    return Xs, Ys, in_lens, tar_lens
