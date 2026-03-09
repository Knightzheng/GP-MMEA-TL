# Transfer Adapt Error-Bucket Summary

| bucket_type | bucket_name | n_targets | mean delta H@1 | mean delta H@10 | mean delta MRR | mean delta MR |
|---|---|---:|---:|---:|---:|---:|
| scenario | cross_graph | 2 | +0.000937 | +0.001763 | +0.000925 | -13.654700 |
| scenario | cross_lingual | 2 | +0.005483 | +0.009000 | +0.006425 | -4.505900 |
| confidence_level | formal_5seed | 2 | +0.005570 | +0.009150 | +0.006600 | -10.157100 |
| confidence_level | pilot_2seed | 2 | +0.000850 | +0.001613 | +0.000750 | -8.003500 |
| difficulty_bucket | easy | 2 | +0.005483 | +0.009000 | +0.006425 | -4.505900 |
| difficulty_bucket | hard | 1 | +0.001000 | +0.002100 | +0.001100 | -12.504400 |
| difficulty_bucket | very_hard | 1 | +0.000875 | +0.001425 | +0.000750 | -14.805000 |

## Interpretation

- Positive `delta_avg_mrr_mean` indicates transfer gain over baseline.
- Negative `delta_avg_mr_mean` indicates lower mean rank (better).
- Bucket views are intended for report-side error analysis summarization.
