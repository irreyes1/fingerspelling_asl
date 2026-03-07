from typing import Dict, List, Sequence

import torch
from torchmetrics.text import CharErrorRate


def ctc_greedy_decode(log_probs: torch.Tensor, int_to_letter: Dict[int, str], blank_id: int = 0) -> str:
    """
    log_probs: (T, C) for a single sample.
    Greedy decode with CTC collapsing + blank removal.
    """
    pred_indices = torch.argmax(log_probs, dim=-1).detach().cpu().tolist()
    decoded_chars: List[str] = []
    prev = None
    for idx in pred_indices:
        if idx == prev or idx == blank_id:
            prev = idx
            continue
        decoded_chars.append(int_to_letter.get(int(idx), ""))
        prev = idx
    return "".join(decoded_chars)


def _levenshtein_distance(a: Sequence, b: Sequence) -> int:
    """Classic Levenshtein distance for generic sequences."""
    n = len(a)
    m = len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            cost = 0 if ai == bj else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost # substitution / match
            )
        prev, curr = curr, prev
    return prev[m]


def _collect_predictions_and_targets(
    model,
    dataloader,
    int_to_letter: Dict[int, str],
    device: torch.device,
    blank_id: int = 0,
) -> tuple[List[str], List[str]]:
    preds: List[str] = []
    targets: List[str] = []

    for batch in dataloader:
        if batch is None:
            continue
        X, Y, input_lens, target_lens = batch
        X = X.to(device)
        model_input_lens = input_lens
        if hasattr(model, "transform_input_lengths"):
            model_input_lens = model.transform_input_lengths(input_lens)

        outputs = model(X)  # (T, B, C)
        B = outputs.shape[1]
        Y_list = Y.detach().cpu().tolist()

        start = 0
        for i in range(B):
            # Decode prediction
            valid_t = int(model_input_lens[i].item())
            pred_text = ctc_greedy_decode(outputs[:valid_t, i, :], int_to_letter, blank_id)

            # Decode target
            tl = int(target_lens[i].item())
            tgt_ids = Y_list[start:start + tl]
            start += tl
            tgt_text = "".join(int_to_letter[int(t)] for t in tgt_ids)

            preds.append(pred_text)
            targets.append(tgt_text)

    return preds, targets


def _compute_wer(preds: List[str], targets: List[str]) -> float:
    total_word_edits = 0
    total_target_words = 0

    for pred, target in zip(preds, targets):
        pred_words = pred.split()
        target_words = target.split()
        total_word_edits += _levenshtein_distance(pred_words, target_words)
        total_target_words += len(target_words)

    if total_target_words == 0:
        return float("nan")
    return float(total_word_edits / total_target_words)


def _compute_average_edit_distance(preds: List[str], targets: List[str]) -> float:
    if len(preds) == 0:
        return float("nan")
    distances = [_levenshtein_distance(pred, target) for pred, target in zip(preds, targets)]
    return float(sum(distances) / len(distances))


@torch.no_grad()
def evaluate_cer(model, dataloader, int_to_letter: Dict[int, str], device: torch.device, blank_id: int = 0) -> float:
    model.eval()
    cer = CharErrorRate()
    preds, targets = _collect_predictions_and_targets(
        model=model,
        dataloader=dataloader,
        int_to_letter=int_to_letter,
        device=device,
        blank_id=blank_id,
    )

    if len(preds) == 0:
        return float("nan")
    return float(cer(preds, targets).item())


@torch.no_grad()
def evaluate_metrics(
    model,
    dataloader,
    int_to_letter: Dict[int, str],
    device: torch.device,
    blank_id: int = 0,
) -> Dict[str, float]:
    model.eval()
    cer_metric = CharErrorRate()

    preds, targets = _collect_predictions_and_targets(
        model=model,
        dataloader=dataloader,
        int_to_letter=int_to_letter,
        device=device,
        blank_id=blank_id,
    )

    if len(preds) == 0:
        return {
            "cer": float("nan"),
            "wer": float("nan"),
            "sequence_accuracy": float("nan"),
            "avg_edit_distance": float("nan"),
        }

    cer = float(cer_metric(preds, targets).item())
    wer = _compute_wer(preds, targets)
    seq_acc = float(sum(1 for p, t in zip(preds, targets) if p == t) / len(preds))
    avg_edit_distance = _compute_average_edit_distance(preds, targets)

    return {
        "cer": cer,
        "wer": wer,
        "sequence_accuracy": seq_acc,
        "avg_edit_distance": avg_edit_distance,
    }
