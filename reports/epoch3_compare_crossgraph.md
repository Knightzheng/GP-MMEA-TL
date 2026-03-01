# Cross-graph epoch3: baseline vs TMMEA-DA v1_best

| dataset | baseline_runs | method_runs | l2r H@1 delta | l2r H@10 delta | l2r MRR delta | r2l H@1 delta | r2l H@10 delta | r2l MRR delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FBDB15K | 1 | 1 | +0.0003 | +0.0010 | +0.0000 | +0.0000 | +0.0006 | +0.0010 |
| FBYG15K | 1 | 1 | +0.0001 | -0.0002 | +0.0000 | -0.0001 | +0.0000 | +0.0000 |

Notes:
- run counts (baseline/method): FBDB15K=1/1, FBYG15K=1/1
- current stage includes pilot comparisons (fewer than 5 seeds).