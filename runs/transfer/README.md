# Transfer Runs Guide

## 1. 如何快速定位主线正式 run

如果只关心当前 4 个目标域的正式主线结果，请优先看以下目录：

| target | baseline formal runs | method formal runs |
| --- | --- | --- |
| `ja_en` | `transfer_adapt_ja_v15_full_baseline_matched_ref/target_eval/` | `transfer_adapt_ja_v15_full_ref/target_eval/` |
| `fr_en` | `transfer_adapt_v14_fren_expand5_merged_baseline/target_eval/` | `transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval/` |
| `FBDB15K` | `transfer_adapt_v18_fbdb_v18c_expand5_baseline_matched_ref/target_eval/` | `transfer_adapt_v18_fbdb_v18c_expand5_ref/target_eval/` |
| `FBYG15K` | `transfer_adapt_v24_fbyg_v24b_expand5_baseline_matched_ref/target_eval/` | `transfer_adapt_v24_fbyg_v24b_expand5_ref/target_eval/` |

这些目录对应的结果会被汇总进：

- `reports/transfer/transfer_adapt_main_results_4target.*`
- `reports/transfer/transfer_adapt_significance_summary.*`
- `reports/transfer/transfer_efficiency_summary.*`

## 2. 正式 run 的识别规则

当前仓库中，能够直接支撑主线结论的 transfer run 通常满足：

1. 位于本 README 第 1 节列出的 8 个目录之一。
2. 目标目录下保留正式 `log.txt / config / run_card / artifact_manifest`。
3. 对应 `5-seed` 正式汇总，而不是单 seed pilot。

## 3. 历史探索目录如何看待

本目录下还保留大量以下类型目录：

- `transfer_adapt_v*_pilot_*`
- `transfer_adapt_v*_queue`
- `transfer_adapt_*_baseline_ref`
- `transfer_adapt_*_auto`

这些目录主要用于：

1. 回溯方法迭代过程。
2. 支撑阶段决策与失败原因分析。
3. 保留从 pilot 到 final 的路径证据。

但它们通常不应被直接当作当前主线正式结果引用，除非在项目级总表或正式阶段报告中被明确点名。

## 4. 当前推荐阅读顺序

1. 先看 `../../reports/notes/mainline_traceability_matrix_20260315.md`
2. 再看 `../../reports/transfer/README.md`
3. 最后进入本目录中的正式 `target_eval/` 子目录逐一核对 run 级留痕

## 5. 当前边界

1. 本目录当前不再保留 `H3` 相关 run。
2. GPU 峰值显存最小正式补测已经在 `../experiments/gpu_peak_minimal/` 中完成，但它仍只属于辅助支撑，不构成主线完成前提。
