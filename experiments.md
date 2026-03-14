# ASL Fingerspelling — Experiment Tracking

| Run Name | Date | Model | Epochs | Batch Size | LR | Hidden Dim | Proj Dim | RNN Layers | Dropout | Train Size | Val Size | Early Stop | Exec Time (h) | Best val CER | Best val WER | Epoch @ Best | Notes | W&B Run Name |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clearml-l4-merge | 2026-03-13 | bilstm | 45 | 256 | 1e-3 | 512 | 256 | 3 | 0.5 | 50000 | 50000 | no | — | 1.0 | 1.0 | 1 | train_size cap limited real data to ~8k rows; LR too aggressive; Merge latest changes from team | clearml-l4-merge_76fe7a1e |
| clearml-l4-full-data-45epochs | 2026-03-13 | bilstm | 45 | 64 | 1e-4 | 512 | 256 | 3 | 0.3 | 0 (all) | 0 (all) | yes (p=10) | 2.6 | 0.612 | 1.014 | 45 | Full data, 45 epochs; CER plateaued, WER >1; params revised | clearml-l4-merge-params-fix-45epochs_e8cef390 |
| clearml-l4-merge-full-data-75epochs | 2026-03-13 | bilstm | 75 | 64 | 1e-4 | 512 | 256 | 2 | 0.3 | 0 (all) | 0 (all) | yes (p=10) | — | — | — | — | Full dataset, Improved FP16 on GPU | clearml-l4-full-data-75epochs |
| clearml-l4-full-data-100epochs-nodrop | 2026-03-14 | bilstm | 100 | 64 | 1e-4 | 512 | 128 | 2 | 0 | 0 (all) | 0 (all) | yes (p=15) | — | — | — | — | No dropout, full data; scheduler on cer; in progress | clearml-l4-full-data-100epochs-nodrop |
| clearml-l4-weight-decay-1e4 | — | bilstm | 100 | 64 | 1e-4 | 512 | 256 | 3 | 0 | 0 (all) | 0 (all) | yes (p=15) | — | — | — | — | L2 weight_decay=1e-4, no dropout; first regularization test | PENDING |

---

## Notes / Suggestions

---

### Mixed Precision Training (`torch.amp`)

_Suggestion Implemented in experiment #3 onwards..._

Adding mixed precision (FP16) to the training loop would yield ~2x speedup on the NVIDIA L4 with no impact on model quality. This is a code change, not a hyperparameter change, so it applies to all future runs. Implementation:

```python
scaler = torch.cuda.amp.GradScaler()

with torch.autocast(device_type="cuda"):
    output = model(x)
    loss = criterion(output, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Expected result:** This would halve experiment execution time (from ~2.6h to ~1.3h), allowing faster iteration across runs.

**Actual result:** Training 75 epochs took ~4hs

---

### Scheduler now monitors val CER (commit `8d60549`, 2026-03-14)

_Experiments #4 and onwards will run with this change._

Merged from `irreyes1/main` (author: Pau Vila). The intended change was:

- **`scheduler.step(metrics_val["cer"])`** replaces `scheduler.step(metrics_val["loss"])` — `ReduceLROnPlateau` now reduces LR when val CER stops improving instead of val loss. This is a better signal since CER is the actual evaluation metric.

**Expected result:** CTC loss can decrease while CER stays flat or worsens — the model may be learning to distribute probability mass more smoothly without actually producing better character sequences. Monitoring CER directly ensures the LR is reduced when what we actually care about stops improving.

---

### L2 Regularization via `--weight_decay` (experiment `clearml-l4-weight-decay-1e4`)

`weight_decay` is fully implemented in `train.py:387` and passed to Adam. However, all previous runs used `weight_decay=0`, so regularization has never been tested.

**How it works:** L2 adds a penalty proportional to the squared magnitude of all weights, nudging each parameter slightly toward zero on every update. In Adam this is equivalent to multiplying each weight by `(1 - lr * weight_decay)` before the gradient step.

**Why it may help here:** past runs show CER plateauing early (0.612 at epoch 45). Large LSTM gate weights can saturate tanh/sigmoid activations → vanishing gradients → training stalls. Weight decay keeps weights small and gradients flowing.

**CLI flag:**
```bash
--weight_decay 1e-4
```

**Expected result:** Lower final CER compared to `weight_decay=0` baseline (exp #4), especially in later epochs where CER was flat. Typical safe range: `1e-5` to `1e-3`.

---

### Model Selection via `--model` flag (TODO)

Currently `train.py` always uses `EmbeddedRNN` (hardcoded BiLSTM). `TCNBiRNN` exists in `src/models/tcn_bilstm.py` but is never instantiated. The args `--rnn_type`, `--rnn_layers`, `--proj_dim`, and `--tcn_kernels` are parsed but have no effect.

**Changes needed:**

1. **`src/models/tcn_bilstm.py`** — add `input_lengths=None` to `forward()` and use `pack_padded_sequence` (same pattern as `EmbeddedRNN`) so it's compatible with the training loop.

2. **`src/train.py`** — add `--model` arg and wire up selection:
```python
p.add_argument("--model", type=str, default="embedded_rnn", choices=["embedded_rnn", "tcn_bilstm"])
```
```python
if args.model == "tcn_bilstm":
    tcn_kernels = tuple(int(k) for k in args.tcn_kernels.split(","))
    model = TCNBiRNN(
        input_dim, args.proj_dim, tcn_kernels,
        args.hidden_dim, args.rnn_layers, args.rnn_type, output_dim
    ).to(device)
else:
    model = EmbeddedRNN(
        input_dim, args.hidden_dim, output_dim, dropout=args.dropout
    ).to(device)
```

Once implemented, a natural next experiment would be comparing `--model tcn_bilstm --rnn_type lstm` vs the current `embedded_rnn` baseline on the same data split.
