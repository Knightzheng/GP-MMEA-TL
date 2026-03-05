# Transfer Adapt Auto Branch Plan (2026-03-05)

## Trigger condition
- Wait for current background queue completion:
  - seed: `3407`
  - targets: `ja_en`, `FBDB15K`
  - both baseline and TMMEA-DA must finish.

## Decision rule
- Input: `reports/transfer/transfer_adapt_pilot_compare_tmmeada_vs_baseline.csv`
- For each required target:
  - `delta_avg_mrr_mean >= 0.001`
  - seed-wise deltas (`42`, `3407`) are non-negative
  - run count is sufficient (`>=2` per side)
- If all required targets pass -> expand branch.
- Otherwise -> non-expand branch.

## Branch A (expand)
- Script: `scripts/run_transfer_adapt_expand_queue.py`
- Targets: `fr_en`, `FBYG15K`
- Seeds: `42,3407`
- Output compare:
  - `reports/transfer/transfer_adapt_expand_compare_tmmeada_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_expand_compare_tmmeada_vs_baseline.md`

## Branch B (non-expand)
- Script: `scripts/run_transfer_adapt_tuned_queue.py`
- Method: `TMMEA-DA tuned_lite` (lower aux weights)
- Targets: `ja_en`, `FBDB15K`
- Seeds: `42,3407`
- Output compare:
  - `reports/transfer/transfer_adapt_tuned_lite_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_tuned_lite_compare_vs_baseline.md`

## Current running auto orchestrator
- Script: `scripts/auto_after_transfer_adapt_queue.py`
- Log:
  - `runs/transfer/transfer_adapt_auto/auto_after_queue_20260305-004521.out.log`
