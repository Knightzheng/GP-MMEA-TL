# Epoch10 v2 tuned Decision

- generated_at: `2026-03-04T03:39:24`
- threshold: `delta_avg_mrr >= 0.003` + all required seeds positive
- decision: `continue_tuning_or_error_analysis`

| dataset | delta_avg_mrr | seed_deltas | consistent_positive | pass_threshold |
|---|---:|---|---:|---:|
| zh_en | -0.0005 | s42:-0.0015, s3407:0.0005 | False | False |
| FBDB15K | -0.0008 | s42:-0.0020, s3407:0.0005 | False | False |
