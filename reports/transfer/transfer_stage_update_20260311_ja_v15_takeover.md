# 2026-03-11 ja_en v15 Takeover Note

## Current State

- Mainline project status remains the 4-target transfer-adapt result set finalized on 2026-03-09.
- Current uncommitted work is a new `ja_en` transfer-adapt branch around `v15`.
- Verified completed pilot seeds for `v15`: `42`, `2026`.
- Verified pilot result on matched seeds `[42, 2026]`:
  - `delta_avg_mrr_mean = +0.013250`
  - compare file: `reports/transfer/transfer_adapt_ja_v15_pilot2seed_compare_vs_baseline.csv`

## Crash Recovery Findings

- No running `python` training process was found during takeover.
- A `seed=3407` run exists under `runs/transfer/transfer_adapt_ja_v15_pilot/target_eval/`, but its log ends with:
  - `[DONE] return_code=1073807364`
- That run is interrupted and must not be treated as a completed result.
- After tightening completion checks, the current `v15` full-progress status is:
  - available seeds: `[42, 2026]`
  - missing seeds: `[3407, 7, 123]`

## Code/Workflow Fixes Applied

- Added `scripts/transfer_adapt_utils.py` for reusable transfer-adapt run selection helpers.
- Updated `scripts/run_transfer_adapt_ja_v15_pilot.py`:
  - support `--run-missing 0/1`
  - skip already completed seeds
  - summarize from merged unique completed runs only
- Updated `scripts/run_transfer_adapt_ja_v15_iter_queue.py`:
  - support `--run-missing 0/1`
  - compare only on matched completed seed sets
  - record missing seeds in decision outputs
- Updated `scripts/summarize_transfer_formal.py` and expand5 resume scripts so only logs with `[DONE] return_code=0` are counted as complete.

## Files Regenerated During Takeover

- `reports/transfer/transfer_adapt_ja_v15_iter_decision.json`
- `reports/transfer/transfer_adapt_ja_v15_iter_decision.md`
- `reports/transfer/transfer_adapt_ja_v15_expand5_compare_vs_baseline.csv`
- `reports/transfer/transfer_adapt_ja_v15_pilot2seed_compare_vs_baseline.csv`

## Recommended Next Step

1. Re-run missing `v15` seeds: `3407`, `7`, `123`.
2. Rebuild the decision report and verify whether `+0.013250` still holds under expanded seeds.
3. Only after `v15` full5 is stable, decide whether `v15a/v15b/v15c` pilot variants are still worth launching.
