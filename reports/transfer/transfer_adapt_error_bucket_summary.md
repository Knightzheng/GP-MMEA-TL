# Transfer Adapt Error-Bucket Summary

| bucket_type | bucket_name | n_targets | mean delta H@1 | mean delta H@10 | mean delta MRR | mean delta MR |
|---|---|---:|---:|---:|---:|---:|
| scenario | cross_graph | 2 | +0.002975 | +0.008805 | +0.004950 | -121.331950 |
| scenario | cross_lingual | 2 | +0.010540 | +0.015150 | +0.012100 | -8.535150 |
| confidence_level | formal_5seed | 4 | +0.006757 | +0.011978 | +0.008525 | -64.933550 |
| difficulty_bucket | easy | 2 | +0.010540 | +0.015150 | +0.012100 | -8.535150 |
| difficulty_bucket | hard | 1 | +0.001410 | +0.001930 | +0.001600 | -35.847200 |
| difficulty_bucket | very_hard | 1 | +0.004540 | +0.015680 | +0.008300 | -206.816700 |

## Interpretation

- Positive `delta_avg_mrr_mean` indicates transfer gain over baseline.
- Negative `delta_avg_mr_mean` indicates lower mean rank (better).
- Bucket views are intended for report-side error analysis summarization.
