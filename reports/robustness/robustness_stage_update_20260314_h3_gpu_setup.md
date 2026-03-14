# H3 / GPU Support Setup Update

- date: `2026-03-14`
- thread role: `optimization / evidence-strengthening`
- status: `infrastructure ready, formal results pending`

## What Was Added

1. Minimal missing-modality pressure-test support for `MEAformer`.
2. Peak GPU memory logging at the end of each run.
3. A fixed runner script for the minimal H3 matrix.
4. A fixed summarizer that can aggregate both accuracy metrics and GPU peak memory.

## Files

- code:
  - `baselines/MEAformer/config.py`
  - `baselines/MEAformer/src/data.py`
  - `baselines/MEAformer/main.py`
  - `scripts/run_meaformer.py`
  - `scripts/run_h3_missing_modality_minimal.py`
  - `scripts/summarize_h3_missing_modality.py`
- report skeleton:
  - `reports/robustness/h3_missing_modality_minimal_summary.md`
  - `reports/robustness/h3_missing_modality_minimal_summary.csv`
  - `reports/robustness/h3_missing_modality_minimal_per_run.csv`

## Current Interpretation

- This step does **not** mean H3 has already been experimentally verified.
- What is now verified is the execution chain:
  - missing-image injection parameters can be passed from config to runtime
  - runtime can log GPU peak allocated/reserved memory
  - the H3 matrix can be launched from a single script
  - a summary table can be generated even before formal runs exist

## Recommended Next Run

Use the smallest still-useful setup first:

- dataset: `zh_en`
- variants: `baseline`, `v1_full`, `wo_missing_gate`
- drop rates: `0.0`, `0.3`, `0.6`
- seeds: `42`, `2026`

If compute is tight, reduce in this order:

1. keep `v1_full` and `wo_missing_gate`, drop `baseline`
2. keep `0.0` and `0.6`, drop `0.3`
3. keep `seed=42` only and mark results explicitly as pilot

## Thesis Usage Boundary

- The current files can already support a sentence such as:
  - `the infrastructure for a controlled missing-modality pressure test has been prepared and validated at the script level`
- They cannot yet support:
  - `H3 is experimentally confirmed`
  - any quantitative robustness conclusion
