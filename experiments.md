# ASL Fingerspelling — Experiment Tracking

| Run Name | Date | Model | Epochs | Batch Size | LR | Hidden Dim | Dropout | Train Size | Val Size | Early Stop | Exec Time (h) | Best val CER | Best val WER | Epoch @ Best | Notes | W&B Run Name |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clearml-l4-full-data-45epochs | 2026-03-13 | bilstm | 45 | 64 | 1e-4 | 512 | 0.3 | 0 (all) | 0 (all) | yes (p=10) | 2.6 | 0.612 | 1.014 | 45 | Full data, 45 epochs; CER plateaued, WER >1; params revised | clearml-l4-merge-params-fix-45epochs_e8cef390 |
| clearml-l4-merge-full-data-75epochs | 2026-03-13 | bilstm | 75 | 64 | 1e-4 | 512 | 0.3 | 0 (all) | 0 (all) | yes (p=10) | 7.8 | 0.441 | 0.964 | 75 | Full dataset, improved FP16 on GPU; ran all 75 epochs without early stopping; LR decayed to 2.5e-5 (2× reductions); train CER 0.329 vs val 0.441 (gap=0.112) | clearml-l4-full-data-75epochs |
| clearml-l4-full-data-100epochs-nodrop | 2026-03-14 | bilstm | 100 | 64 | 1e-4 | 512 | 0 | 0 (all) | 0 (all) | yes (p=15) | 10.6 | 0.471 | 0.985 | 100 | No dropout, full data; scheduler on CER; LR decayed to 3.125e-6 (5× reductions); CER improved vs dropout run | clearml-l4-full-data-100epochs-nodrop_202e8769 |
| clearml-l4-weight-decay-1e4 | 2026-03-15 | bilstm | 100 | 128 | 1e-4 | 512 | 0 | 0 (all) | 0 (all) | yes (p=15) | 11.0 | 0.812 | 1.032 | 100 | L2 weight_decay=1e-4, 3 LSTM layers, batch=128; CER much worse than prev run (0.471); LR decayed to 7.8125e-7 (7× reductions); | clearml-l4-weight-decay-1e4_de9e11a2 |
| clearml-l4-weight-decay-1e4-lrate-1e3 | 2026-03-16 | bilstm | 100 | 128 | 1e-3 | 512 | 0 | 0 (all) | 0 (all) | yes (p=15) | 5.8 | 0.402 | 0.917 | 53 | Best val CER across all runs; early stopping at epoch 53; LR decayed to 6.25e-5 (4× reductions); significant overfitting (train CER 0.117 vs val 0.402) | clearml-l4-weight-decay-1e4-lrate-1e3_8564e777 |
| clearml-l4-golden-arch-100 | 2026-03-16 | bilstm | 100 | 64 | 1e-3 | 256 | 0 | 50000 | 50000 | yes (p=15) | 8.2 | 0.421 | 0.954 | 77 | train_size capped at 50k; early stopping at epoch 77; LR decayed to 7.8125e-6 (7× reductions); less overfitting than weight-decay's best run (train CER 0.235 vs val 0.421, gap=0.186) | clearml-l4-golden-arch-100_45f8e478 |
| clearml-l4-best-config-dropout-03 | 2026-03-16 | bilstm | 100 | 128 | 1e-3 | 512 | 0.3 | 0 (all) | 0 (all) | yes (p=25) | 8.5 | 0.394 | 0.903 | 79 | New best val CER; dropout=0.3 reduced overfitting (train CER 0.175 vs val 0.394, gap=0.219 vs 0.285 in best run); LR decayed to 3.90625e-6 (8× reductions);  | clearml-l4-best-config-dropout-03_ee0f4f0f |
| clearml-l4-golden-arch-full-data | 2026-03-17 | bilstm | 100 | 64 | 1e-3 | 256 | 0 | 0 (all) | 0 (all) | yes (p=15) | 3.2 | 0.511 | 0.998 | 30 | Worse than golden-arch-100 (0.421); blank_ratio=0.924 — near CTC collapse; train/val gap only 0.020 (both ~0.5) — model undertrained; only 1 LR reduction to 5e-4| clearml-l4-golden-arch-full-data_e98d4852 |
| clearml-l4-best-config-batch-64 | 2026-03-17 | bilstm | 100 | 64 | 1e-3 | 512 | 0.3 | 0 (all) | 0 (all) | yes (p=25) | 6.9 | 0.386 | 0.902 | 65 | New best val CER; batch 128→64 improved val CER (0.394→0.386) but widened overfitting gap (0.219→0.284, train CER 0.102 vs val 0.386); LR decayed to 7.8125e-6 (7× reductions) | clearml-l4-best-config-batch-64_c83d55ef |
| clearml-l4-golden-arch-hidden-256 | 2026-03-17 | bilstm | 100 | 64 | 1e-3 | 256 | 0.3 | 0 (all) | 0 (all) | yes (p=25) | 9.5 | 0.383 | 0.907 | 89 | New best val CER and best gap; hidden=256 + dropout=0.3 delivered both targets — val CER 0.383 (vs 0.386) and gap 0.165 (vs 0.284); LR decayed to 4.88e-7 (10× reductions); train CER 0.218 vs val 0.383 | clearml-l4-golden-arch-hidden-256_a2613536 |
| clearml-l4-best-config-batch-32 | — | bilstm | 100 | 32 | 1e-3 | 512 | 0.3 | 0 (all) | 0 (all) | yes (p=25) | — | — | — | — | RUNNING. Batch size exploration: batch 64→32 on best config (hidden=512, dropout=0.3, weight_decay=1e-4); continues batch size reduction trend (128→64→32); expected val CER < 0.386 | — |
| clearml-l4-best-config-dropout-04 | — | bilstm | 100 | 64 | 1e-3 | 512 | 0.4 | 0 (all) | 0 (all) | yes (p=25) | — | — | — | — | SCHEDULED. dropout 0.3→0.4 on best config (batch=64, hidden=512); addresses widened gap (0.284) in best run; expected gap < 0.284 and val CER ≤ 0.386 | — |

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

**Actual result:** Training 75 epochs took 7.8h — speedup was less than expected.

---

### Scheduler now monitors val CER (commit `8d60549`, 2026-03-14)

_Implemented from experiment #4 (`clearml-l4-full-data-100epochs-nodrop`) onwards._

Merged from `irreyes1/main` (author: Pau Vila). The change was:

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

**Actual result:** val CER 0.812 — significantly worse than the no-decay baseline (0.471). The poor result was likely caused by lr=1e-4 being too low for this config (LR decayed to 7.8125e-7, effectively zero), not weight decay itself.

---

### Reintroduce dropout with best config (experiment `clearml-l4-best-config-dropout-03`)

_COMPLETED_

The best config (lr=1e-3, batch=128, hidden=512, weight_decay=1e-4) has never been tested with dropout. The earlier dropout run (`clearml-l4-full-data-45epochs`, dropout=0.3) used very different hyperparameters, so no fair comparison exists. `clearml-l4-weight-decay-1e4-lrate-1e3` shows significant overfitting (train CER 0.117 vs val 0.402, gap=0.285) — dropout is the most direct tool to close it.

**Expected result:** train/val CER gap below 0.285; val CER below 0.402 if overfitting was the main bottleneck.

**Actual result:** val CER 0.394 (new best), gap=0.219 — both targets met. Run deviated from spec: patience=25 (params confirmed not in use).

---

### Full data with golden-run architecture (experiment `clearml-l4-golden-arch-full-data`)

_COMPLETED_

`clearml-l4-golden-arch-100` (hidden_dim=256) showed better generalisation (gap=0.186) than the best run (gap=0.285), but train_size was capped at 50k. The reduced overfitting likely comes from the smaller architecture, not the data cap — so adding full data should bring val CER down without widening the gap.

**Expected result:** val CER below 0.421 (`clearml-l4-golden-arch-100`) and potentially below 0.402 (best run), with the train/val gap remaining contained.

**Actual result:** val CER 0.511 — worse than the capped-data run (0.421); blank_ratio=0.924 suggesting near CTC collapse; model stopped at epoch 30 with near-zero train/val gap (both ~0.5). The reduced overfitting in `clearml-l4-golden-arch-100` was likely due to the data cap, not the architecture alone.

---
