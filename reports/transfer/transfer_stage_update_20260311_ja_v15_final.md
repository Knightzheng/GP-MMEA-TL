# Transfer-Adapt Stage Update (2026-03-11): ja_en v15 Full5 Finalized

## Summary

- Completed `ja_en v15` transfer-adapt full 5-seed evaluation:
  - seeds: `42, 3407, 2026, 7, 123`
- Final compare file:
  - `reports/transfer/transfer_adapt_ja_v15_expand5_compare_vs_baseline.csv`
- Final `ja_en` result:
  - `delta_avg_hits@1_mean = +0.01094`
  - `delta_avg_hits@10_mean = +0.01410`
  - `delta_avg_mrr_mean = +0.01210`
  - `delta_avg_mr_mean = -9.26050`

## Technical Fixes Applied During Recovery

- Tightened transfer summary completeness checks:
  - only runs with `[DONE] return_code=0` are treated as completed
- Fixed resumable selection logic:
  - compare reports now use matched completed seed sets only
- Added reusable helper module:
  - `scripts/transfer_adapt_utils.py`
- Added `ja_en v15` resumable controls:
  - `scripts/run_transfer_adapt_ja_v15_pilot.py`
  - `scripts/run_transfer_adapt_ja_v15_iter_queue.py`

## Result Positioning

- `ja_en v15` is now the best formal `ja_en` transfer-adapt result in this repository.
- Compared with the previous `ja_en` main-table choice (`v6_mixed`, `delta_avg_mrr_mean = -0.01630` under current regenerated source table), `v15` provides a clear improvement.
- After refreshing the 4-target main table, all targets remain positive:
  - `ja_en`: `+0.01210`
  - `FBDB15K`: `+0.00080`
  - `fr_en`: `+0.01210`
  - `FBYG15K`: `+0.00110`

## Updated Outputs

- Main table:
  - `reports/transfer/transfer_adapt_main_results_4target.csv`
  - `reports/transfer/transfer_adapt_main_results_4target.md`
- Error bucket summary:
  - `reports/transfer/transfer_adapt_error_bucket_summary.csv`
  - `reports/transfer/transfer_adapt_error_bucket_summary.md`
- ja_en decision snapshot:
  - `reports/transfer/transfer_adapt_ja_v15_iter_decision.json`
  - `reports/transfer/transfer_adapt_ja_v15_iter_decision.md`

## Conclusion

- `ja_en` has been upgraded from an older weak-gain/negative-gain branch to a stable formal 5-seed positive result.
- The project now has a unified 4-target formal 5-seed transfer-adapt main table ready for thesis main-result usage.
