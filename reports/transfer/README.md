# Transfer Reports Guide

## 1. Official Mainline Entry Points

If you only care about the current project mainline, start with these files:

1. `transfer_adapt_main_results_4target.md`
   - Formal 4-target summary table.
2. `transfer_adapt_main_results_4target.csv`
   - Structured data closest to the thesis main table.
3. `transfer_adapt_significance_summary.md`
   - Significance and stability support for the 4-target `5-seed` package.
4. `transfer_case_analysis_examples.md`
   - Current formal case-analysis examples.
5. `transfer_case_analysis_thesis_sync_20260315.md`
   - Notes how the case package already expanded from `6` to `8` formal samples.
6. `transfer_case_pattern_summary_20260316.md`
   - Grouped success/failure patterns built from the current `8` formal cases.
7. `transfer_efficiency_summary.md`
   - Wall-clock efficiency summary.
8. `transfer_gpu_peak_minimal_summary.md`
   - Minimal formal GPU-peak-memory supplement.
9. `transfer_gpu_peak_minimal_thesis_sync_20260315.md`
   - Thesis-ready wording and usage boundary for the GPU supplement.
10. `transfer_extra_baseline_limitation_writeup.md`
   - Conservative wording for why extra baselines were not expanded further.

## 2. Formal Compare Files for the Four Targets

- `ja_en`
  - `transfer_adapt_ja_v15_expand5_compare_vs_baseline.csv`
- `fr_en`
  - `transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv`
- `FBDB15K`
  - `transfer_adapt_v18_fbdb_v18c_expand5_compare_vs_baseline.csv`
- `FBYG15K`
  - `transfer_adapt_v24_fbyg_v24b_expand5_compare_vs_baseline.csv`

These files are among the direct sources of `transfer_adapt_main_results_4target.*`.

## 3. Mainline Support Files

- `transfer_adapt_significance_writeup.md`
  - Thesis-ready significance wording.
- `transfer_case_analysis_examples.csv`
  - Structured data for the current case package.
- `transfer_case_pattern_summary_20260316.csv`
  - Structured data for the grouped case-pattern summary.
- `transfer_efficiency_summary.csv`
  - Structured data for wall-clock analysis.
- `transfer_gpu_peak_minimal_summary.csv`
  - Structured data for the minimal GPU supplement.
- `transfer_gpu_peak_minimal_chart_ready.csv`
  - Long-form chart-ready data for defense figures.

## 4. How to Read Historical Stage Files

There are many `transfer_stage_update_*.md` and `transfer_adapt_v*.md/csv/json` files in this directory.

Recommended usage:

1. Use `transfer_stage_update_20260311_ja_v15_final.md` to trace how `ja_en` was finalized.
2. Use `transfer_stage_update_20260312_v18_fbdb_bipartite_full5.md` to trace why `FBDB15K` was fixed at `v18c`.
3. Use `transfer_stage_update_20260314_fbyg_v24_strict_source_full5.md` to trace why `FBYG15K` was fixed at `v24b`.
4. Treat most other `v*` files as exploration history rather than direct final-evidence entry points.

## 5. Current Boundary

1. This folder no longer contains `H3` materials.
2. GPU peak memory is already supplemented, but still only as an auxiliary minimal check.
3. If you only need the mainline closure evidence chain, go back to `../notes/mainline_traceability_matrix_20260315.md`.
4. The case-analysis package currently contains `8` formal samples. If the thesis main text keeps only `6`, the remaining `2` are best placed in appendix or defense materials.
5. `transfer_case_pattern_summary_20260316.*` reorganizes existing cases only; it is not a new statistical experiment.
