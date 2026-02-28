# zh_en epoch3 pilot: baseline vs TMMEA-DA v1_best (seed=42)

- baseline_run: `20260301-002341-MEAformer-epoch3-DBP15K-zh_en-s42`
- method_run: `20260301-005700-TMMEA-DA-v1-best-epoch3-DBP15K-zh_en-s42`

| metric | baseline_epoch3 | tmmeada_v1_best_epoch3 | delta(method-baseline) |
|---|---:|---:|---:|
| l2r Hits@1 | 0.6272 | 0.6278 | +0.0006 |
| l2r Hits@10 | 0.8970 | 0.8969 | -0.0001 |
| l2r MRR | 0.7190 | 0.7190 | +0.0000 |
| r2l Hits@1 | 0.6262 | 0.6261 | -0.0001 |
| r2l Hits@10 | 0.8952 | 0.8952 | +0.0000 |
| r2l MRR | 0.7170 | 0.7170 | +0.0000 |

Notes:
- This is a single-seed pilot under epoch=3, used for training-budget trend check only.
- Next formal step should be multi-seed runs with the same epoch budget.