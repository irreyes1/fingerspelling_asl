# ASL Fingerspelling — Experiment Tracking

| Run Name | Date | Model | Epochs | Batch Size | LR | Hidden Dim | Proj Dim | RNN Layers | Dropout | Train Size | Val Size | Early Stop | Exec Time (h) | Best val CER | Best val WER | Epoch @ Best | Notes | W&B Run Name |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clearml-l4-merge | 2026-03-13 | lstm | 45 | 256 | 1e-3 | 512 | 256 | 3 | 0.5 | 50000 | 50000 | no | — | 1.0 | 1.0 | 1 | train_size cap limited real data to ~8k rows; LR too aggressive; Merge latest changes from team | clearml-l4-merge_76fe7a1e |
| clearml-l4-full-data-45epochs | 2026-03-13 | lstm | 45 | 64 | 1e-4 | 512 | 256 | 3 | 0.3 | 0 (all) | 0 (all) | yes (p=10) | 2.6 | 0.612 | 1.014 | 45 | Full data, 45 epochs; CER plateaued, WER >1; params revised | clearml-l4-merge-params-fix-45epochs_e8cef390 |
| clearml-l4-merge-full-data-75epochs | 2026-03-13 | lstm | 75 | 64 | 1e-4 | 512 | 256 | 2 | 0.3 | 0 (all) | 0 (all) | yes (p=10) | — | — | — | — | Full dataset, in progress; Improved FP16 on GPU | clearml-l4-full-data-75epochs |
| bilstm-supplemental | — | bilstm | 150 | 64 | 1e-3 | 512 | 256 | 3 | 0.3 | 0 (all) | 0 (all) | yes (p=20) | — | — | — | — | BiLSTM + supplemental data; suggestions #2 and #4 | — |

---

## Notes / Suggestions

### Mixed Precision Training (`torch.amp`) — Suggestion #6
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

This would halve experiment execution time (from ~2.6h to ~1.3h), allowing faster iteration across runs.
