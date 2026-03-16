# ASL Fingerspelling — Experiment Tracking

| Run Name | Date | Model | Epochs | Batch Size | LR | Hidden Dim | Dropout | Train Size | Val Size | Early Stop | Exec Time (h) | Best val CER | Best val WER | Epoch @ Best | Notes | W&B Run Name |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clearml-l4-merge | 2026-03-13 | bilstm | 45 | 256 | 1e-3 | 512 | 0.5 | 50000 | 50000 | no | — | 1.0 | 1.0 | 1 | train_size cap limited real data to ~8k rows; LR too aggressive; Merge latest changes from team | clearml-l4-merge_76fe7a1e |
| clearml-l4-full-data-45epochs | 2026-03-13 | bilstm | 45 | 64 | 1e-4 | 512 | 0.3 | 0 (all) | 0 (all) | yes (p=10) | 2.6 | 0.612 | 1.014 | 45 | Full data, 45 epochs; CER plateaued, WER >1; params revised | clearml-l4-merge-params-fix-45epochs_e8cef390 |
| clearml-l4-merge-full-data-75epochs | 2026-03-13 | bilstm | 75 | 64 | 1e-4 | 512 | 0.3 | 0 (all) | 0 (all) | yes (p=10) | — | — | — | — | Full dataset, Improved FP16 on GPU | clearml-l4-full-data-75epochs |
| clearml-l4-full-data-100epochs-nodrop | 2026-03-14 | bilstm | 100 | 64 | 1e-4 | 512 | 0 | 0 (all) | 0 (all) | yes (p=15) | 10.6 | 0.471 | 0.985 | 100 | No dropout, full data; scheduler on CER; LR decayed to 3.125e-6 (5× reductions); CER improved vs dropout run | clearml-l4-full-data-100epochs-nodrop_202e8769 |
| clearml-l4-weight-decay-1e4 | 2026-03-15 | bilstm | 100 | 128 | 1e-4 | 512 | 0 | 0 (all) | 0 (all) | yes (p=15) | 11.0 | 0.812 | 1.032 | 100 | L2 weight_decay=1e-4, 3 LSTM layers, batch=128; CER much worse than prev run (0.471); LR decayed to 7.8125e-7 (7× reductions); | clearml-l4-weight-decay-1e4_de9e11a2 |
| clearml-l4-weight-decay-1e4-lrate-1e3 | 2026-03-16 | bilstm | 100 | 128 | 1e-3 | 512 | 0 | 0 (all) | 0 (all) | yes (p=15) | 5.8 | 0.402 | 0.917 | 53 | Best val CER across all runs; early stopping at epoch 53; LR decayed to 6.25e-5 (4× reductions); significant overfitting (train CER 0.117 vs val 0.402) | clearml-l4-weight-decay-1e4-lrate-1e3_8564e777 |
| clearml-l4-golden-arch-100 | 2026-03-16 | bilstm | 100 | 64 | 1e-3 | 256 | 0 | 50000 | 50000 | yes (p=15) | 8.2 | 0.421 | 0.954 | 77 | train_size capped at 50k; early stopping at epoch 77; LR decayed to 7.8125e-6 (7× reductions); less overfitting than weight-decay's best run (train CER 0.235 vs val 0.421, gap=0.186) | clearml-l4-golden-arch-100_45f8e478 |
| clearml-l4-best-config-dropout-03 | — | bilstm | 100 | 128 | 1e-3 | 512 | 0.3 | 0 (all) | 0 (all) | yes (p=15) | — | — | — | — | PENDING #1. dropout=0.3 on best config (hidden_dim=512, batch=128, weight_decay=1e-4); first fair test of dropout at this hyperparameter setting; expected train/val gap < 0.285 | — |
| clearml-l4-golden-arch-full-data | — | bilstm | 100 | 64 | 1e-3 | 256 | 0 | 0 (all) | 0 (all) | yes (p=15) | — | — | — | — | PENDING #2. Full data with golden-run arch (hidden_dim=256 + proj_dim=128); tests whether reduced overfitting in golden run is architecture-driven; expected val CER < 0.402 | — |

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