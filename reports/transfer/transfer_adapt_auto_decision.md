# Transfer Adapt Auto Decision

- timestamp: `2026-03-05 02:27:24`
- threshold: `delta_avg_mrr_mean >= 0.001` and seed-wise non-negative
- required_targets: `['ja_en', 'FBDB15K']`
- reference_seeds: `[42, 3407]`
- decision: `run_tmmeada_tuned_lite_on_ja_fbdb`
- next_action: `run_transfer_adapt_tuned_queue`

| target | delta_avg_mrr_mean | consistent_positive | enough_runs | pass_threshold |
|---|---:|---:|---:|---:|
| ja_en | +0.000750 | True | True | False |
| FBDB15K | -0.000250 | False | True | False |
