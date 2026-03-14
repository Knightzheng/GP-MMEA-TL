# Transfer Reports Guide

## 1. 主线正式结果入口

如果只关心当前项目主线，请优先阅读以下文件：

1. `transfer_adapt_main_results_4target.md`
   - 当前 4 个目标域正式主表说明。
2. `transfer_adapt_main_results_4target.csv`
   - 与论文主表、答辩表格最接近的结构化数据。
3. `transfer_adapt_significance_summary.md`
   - 4 个目标域 `5-seed` 正增益的显著性与稳定性补强。
4. `transfer_case_analysis_examples.md`
   - 当前可直接入文的代表性案例。
5. `transfer_efficiency_summary.md`
   - 当前已完成的 wall-clock 效率补证。
6. `transfer_gpu_peak_minimal_summary.md`
   - 当前已完成的最小正式 GPU 峰值显存补测。

## 2. 4 个目标域的正式对比文件

- `ja_en`
  - `transfer_adapt_ja_v15_expand5_compare_vs_baseline.csv`
- `fr_en`
  - `transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv`
- `FBDB15K`
  - `transfer_adapt_v18_fbdb_v18c_expand5_compare_vs_baseline.csv`
- `FBYG15K`
  - `transfer_adapt_v24_fbyg_v24b_expand5_compare_vs_baseline.csv`

这些文件是 `transfer_adapt_main_results_4target.*` 的直接来源之一。

## 3. 支撑主线的辅助分析文件

- `transfer_adapt_error_bucket_summary.md`
  - 用于解释不同目标域与不同难度分桶下的表现差异。
- `transfer_adapt_significance_writeup.md`
  - 论文可直接吸收的显著性分析文字版。
- `transfer_case_analysis_examples.csv`
  - 案例分析对应的结构化数据。
- `transfer_efficiency_summary.csv`
  - 效率分析对应的结构化数据。
- `transfer_gpu_peak_minimal_summary.csv`
  - GPU 峰值显存最小正式补测的结构化数据。

## 4. 阶段报告的正确使用方式

本目录中有大量 `transfer_stage_update_*.md`、`transfer_adapt_v*.md/csv/json` 文件，它们主要用于保留方法迭代过程。

推荐使用方式：

1. 用 `transfer_stage_update_20260311_ja_v15_final.md` 查看 `ja_en` 主表版本是如何收口的。
2. 用 `transfer_stage_update_20260312_v18_fbdb_bipartite_full5.md` 查看 `FBDB15K` 主表版本如何收口到 `v18c`。
3. 用 `transfer_stage_update_20260314_fbyg_v24_strict_source_full5.md` 查看 `FBYG15K` 主表版本为何固定为 `v24b`。
4. 其他 `v*` 文件更多用于追踪探索过程，而不是作为最终主线证据直接引用。

## 5. 当前边界

1. 本目录当前不再包含 `H3` 相关材料。
2. GPU 峰值显存当前已经补出最小正式结果，但仍只属于辅助支撑，不得写成主线完成前提。
3. 若只需要项目主线闭环证据，请优先回到：
   - `../notes/mainline_traceability_matrix_20260315.md`
