# zh_en epoch3 multi-seed: baseline vs TMMEA-DA v1_best

- baseline num_runs: `5`
- method num_runs: `5`

| metric | baseline_epoch3 | tmmeada_v1_best_epoch3 | delta(method-baseline) |
|---|---:|---:|---:|
| l2r Hits@1 | 0.6233 +/- 0.0085 | 0.6233 +/- 0.0085 | +0.0000 |
| l2r Hits@10 | 0.8926 +/- 0.0054 | 0.8926 +/- 0.0055 | +0.0000 |
| l2r MRR | 0.7146 +/- 0.0070 | 0.7146 +/- 0.0070 | +0.0000 |
| r2l Hits@1 | 0.6233 +/- 0.0065 | 0.6234 +/- 0.0065 | +0.0001 |
| r2l Hits@10 | 0.8925 +/- 0.0026 | 0.8924 +/- 0.0026 | -0.0001 |
| r2l MRR | 0.7148 +/- 0.0054 | 0.7150 +/- 0.0054 | +0.0002 |

Notes:
- Both methods share the same epoch=3 training budget and seed set.