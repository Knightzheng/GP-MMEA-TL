# 主线闭环一页式说明（2026-03-15）

## 1. 当前一句话结论

对照任务书、开题报告与当前仓库正式结果，项目主线已经完成闭环：当前已经形成统一的 `source-train -> target-adapt -> target-eval` 迁移实验链路，并在 `ja_en / fr_en / FBDB15K / FBYG15K` 四个目标域上取得 `5-seed` 稳定正增益。后续需要继续保守处理的内容主要是辅助支撑项，而不是主线是否成立。

## 2. 任务书 / 开题要求与当前正式支撑

| 核心要求 | 当前状态 | 直接支撑文件 |
| --- | --- | --- |
| 复现基线模型，形成可靠对照 | 已完成 | `reports/baseline/baseline_epoch3_results_summary.csv`, `reports/baseline/baseline_epoch3_crossgraph_results_summary.csv` |
| 建立可迁移的统一实验链路 | 已完成 | `reports/transfer/transfer_adapt_main_results_4target.md`, `reports/transfer/transfer_stage_update_20260311_ja_v15_final.md`, `reports/transfer/transfer_stage_update_20260312_v18_fbdb_bipartite_full5.md`, `reports/transfer/transfer_stage_update_20260314_fbyg_v24_strict_source_full5.md` |
| 在不同目标域验证迁移有效性 | 已完成 | `reports/transfer/transfer_adapt_main_results_4target.csv`, `reports/transfer/transfer_adapt_*_compare_vs_baseline.csv` |
| 分析关键设计对迁移性能的作用 | 已完成主线所需最小支撑 | `reports/epoch3/epoch3_ablation_zh_en_multiseed.md`, `reports/transfer/transfer_stage_update_20260312_v18_fbdb_bipartite_full5.md`, `reports/transfer/transfer_stage_update_20260314_fbyg_v24_strict_source_full5.md` |
| 给出统计、案例与效率分析 | 已完成 | `reports/transfer/transfer_adapt_significance_summary.md`, `reports/transfer/transfer_case_analysis_examples.md`, `reports/transfer/transfer_efficiency_summary.md`, `reports/transfer/transfer_gpu_peak_minimal_summary.md` |
| 过程留痕、结果可追溯、可复现 | 已完成 | `reports/notes/mainline_traceability_matrix_20260315.md`, `README.md`, `PROCESS_LOG.md`, `PROJECT_OPERATION_RECORD.md`, `runs/transfer/README.md` |

## 3. 当前已经完成的部分

1. `MEAformer` baseline 已复现完成，并形成统一对照口径。
2. 统一迁移链路已经固定，正式主表已经收口到 `4` 个目标域 `5-seed` 结果。
3. 主线结论已经有统计显著性、案例分析和 wall-clock 效率三类补证支撑。
4. GPU 峰值显存已经补出代表性最小正式结果，可作为辅助开销说明。
5. 项目当前已经具备“要求 -> 结果 -> 脚本 -> run 目录”的可追溯映射。

## 4. 当前仍属于辅助支撑的部分

1. GPU 峰值显存仍只是代表性最小补测，不是全目标域、全 seed 的统一显存统计。
2. 额外 baseline 尚未补入，因此公平比较结论仍应保守限制在当前 `MEAformer-based transfer setting` 内。
3. `H3` 已延期，不属于当前主线闭环必要条件，也不应进入当前论文主体表述。

## 5. 面向答辩 / 验收的直接表述

本项目已经完成任务书与开题报告要求的主线工作：首先复现 `MEAformer` 基线并建立统一的 `source-train -> target-adapt -> target-eval` 迁移链路，随后在 `ja_en`、`fr_en`、`FBDB15K` 和 `FBYG15K` 四个目标域上完成 `5-seed` 正式验证，结果均相对匹配 baseline 呈现稳定正增益。在此基础上，项目还补充了显著性分析、案例分析、wall-clock 效率统计和代表性 GPU 峰值显存补测，因此当前论文主体已经有足够的项目证据支撑。仍需保守说明的是，GPU 与额外 baseline 属于辅助支撑，当前结论不应外推为“对所有骨干模型和所有开销指标都已完成全面验证”。
