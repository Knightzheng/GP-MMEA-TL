# Transfer Adapt Error-Bucket Summary

| bucket_type | bucket_name | n_targets | mean delta H@1 | mean delta H@10 | mean delta MRR | mean delta MR |
|---|---|---:|---:|---:|---:|---:|
| scenario | cross_graph | 2 | +0.000605 | +0.002025 | +0.000950 | -12.279450 |
| scenario | cross_lingual | 2 | +0.010540 | +0.015150 | +0.012100 | -8.535150 |
| confidence_level | formal_5seed | 4 | +0.005572 | +0.008587 | +0.006525 | -10.407300 |
| difficulty_bucket | easy | 2 | +0.010540 | +0.015150 | +0.012100 | -8.535150 |
| difficulty_bucket | hard | 1 | +0.001000 | +0.002100 | +0.001100 | -12.504400 |
| difficulty_bucket | very_hard | 1 | +0.000210 | +0.001950 | +0.000800 | -12.054500 |

## Interpretation

- Positive `delta_avg_mrr_mean` indicates transfer gain over baseline.
- Negative `delta_avg_mr_mean` indicates lower mean rank (better).
- Bucket views are intended for report-side error analysis summarization.
