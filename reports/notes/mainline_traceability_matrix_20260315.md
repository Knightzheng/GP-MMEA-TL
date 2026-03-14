# 主线复现与追溯总表（2026-03-15）

## 1. 目的与当前口径

本文件用于把“任务书 / 开题报告要求”与“当前项目中的正式结果、脚本入口、run 目录、过程记录”建立一一对应关系，方便后续：

1. 验收项目主线是否已经闭环。
2. 后续继续做 README / 报告导航收口。
3. 让论文线程和后来者能快速定位正式证据。

当前统一口径：

- 主线已围绕“统一迁移实验链路 + 4 个目标域 5-seed 正增益 + 目标域自适应 / 伪标签质量控制具有积极作用”闭环。
- `H3` 相关内容已从当前仓库移除，不纳入本总表。
- GPU 峰值显存仍属于辅助支撑，但当前已补出代表性最小正式结果。

## 2. 任务书 / 开题要求对齐矩阵

| 任务书 / 开题要求 | 当前状态 | 正式支撑文件 | 脚本入口 / 运行入口 | 正式 run / 留痕目录 | 当前边界 |
| --- | --- | --- | --- | --- | --- |
| 动机实验：跨语言与跨图谱迁移评测 | 已完成 | `reports/transfer/transfer_adapt_main_results_4target.md` | `scripts/run_transfer_train_eval.py`, `scripts/make_transfer_main_and_bucket_report.py`, `scripts/summarize_transfer_formal.py` | `runs/transfer/transfer_adapt_ja_v15_full_baseline_matched_ref/target_eval/`, `runs/transfer/transfer_adapt_ja_v15_full_ref/target_eval/`, `runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval/`, `runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval/`, `runs/transfer/transfer_adapt_v18_fbdb_v18c_expand5_baseline_matched_ref/target_eval/`, `runs/transfer/transfer_adapt_v18_fbdb_v18c_expand5_ref/target_eval/`, `runs/transfer/transfer_adapt_v24_fbyg_v24b_expand5_baseline_matched_ref/target_eval/`, `runs/transfer/transfer_adapt_v24_fbyg_v24b_expand5_ref/target_eval/` | 已形成统一 4 目标域 `5-seed` 主表，不再需要继续追加主线 rerun 才能证明主线成立 |
| 设计可迁移的多模态实体对齐模型 | 已完成阶段版并固定主表口径 | `reports/transfer/transfer_stage_update_20260311_ja_v15_final.md`, `reports/transfer/transfer_stage_update_20260312_v18_fbdb_bipartite_full5.md`, `reports/transfer/transfer_stage_update_20260314_fbyg_v24_strict_source_full5.md` | `scripts/run_transfer_adapt_ja_v15_iter_queue.py`, `scripts/run_transfer_adapt_v18_fbdb_iter_queue.py`, `scripts/run_transfer_adapt_v24_fbyg_iter_queue.py` | 对应 `runs/transfer/transfer_adapt_*_ref/target_eval/` 正式目录 | 当前主结论建立在“整体链路有效”而非“每个局部模块都被单独严格证明”之上 |
| 基线模型复现 | 已完成 | `reports/baseline/baseline_epoch3_results_summary.csv`, `reports/baseline/baseline_epoch3_crossgraph_results_summary.csv` | `scripts/train_baseline.py`, `scripts/run_meaformer_multiseed.py`, `scripts/run_meaformer_crossgraph_multiseed.py`, `scripts/aggregate_meaformer_results.py` | `runs/experiments/baseline/baseline_epoch3/`, `runs/experiments/baseline/baseline_epoch3_crossgraph/` | 基线主线复现已落盘，当前主要缺口不在结果，而在导航集中度 |
| 方法主线复现 | 已完成 | `reports/tmmeada/tmmeada_v1_best_epoch3_results_summary.csv`, `reports/tmmeada/tmmeada_v1_best_epoch3_crossgraph_results_summary.csv` | `scripts/run_tmmeada_multiseed.py`, `scripts/make_tmmeada_baseline_compare_all.py` | `runs/experiments/tmmeada/tmmeada_v1_best_epoch3/`, `runs/experiments/tmmeada/tmmeada_v1_best_epoch3_crossgraph/` | 当前 `epoch=3` 结果更适合作为主线受控对比与消融支撑，而不是最终迁移主表 |
| 主实验：验证跨数据集迁移整体表现 | 已完成 | `reports/transfer/transfer_adapt_main_results_4target.csv`, `reports/transfer/transfer_adapt_main_results_4target.md` | `scripts/make_transfer_main_and_bucket_report.py`, `scripts/summarize_transfer_formal.py` | 见本表“动机实验”一行所列 8 组正式 baseline / method 目录 | 目前已满足主实验与论文主表需求 |
| 对比实验：与现有模型比较 | 已完成当前主线口径 | `reports/transfer/transfer_adapt_*_compare_vs_baseline.csv`, `reports/transfer/transfer_formal_compare_tmmeada_vs_baseline.md` | `scripts/compare_transfer_summaries.py`, `scripts/make_tmmeada_baseline_compare_all.py` | baseline 与 method 对应正式目录按 seed 一一匹配 | 当前额外 baseline 仍未补，但不构成主线未完成 |
| 消融实验：分析关键设计必要性 | 已完成主线所需最小闭环 | `reports/epoch3/epoch3_ablation_zh_en_multiseed.md`, `reports/epoch3/epoch3_ablation_zh_en_multiseed.csv` | `scripts/summarize_epoch3_ablation_zh_en_multiseed.py`, `scripts/make_epoch3_ablation_zh_en.py` | `runs/experiments/tmmeada/tmmeada_v1_ablation_epoch3/`, `runs/experiments/tmmeada/tmmeada_v1_best_epoch3/`, `runs/experiments/baseline/baseline_epoch3/` | 可支撑“关键机制参与主线表现”，但不宜夸大为每个模块都已被强证 |
| 结果记录与分析：系统分析不同目标域性能变化 | 已完成 | `reports/transfer/transfer_adapt_error_bucket_summary.md`, `reports/transfer/transfer_stage_update_20260309_main_table_bucket.md` | `scripts/make_transfer_main_and_bucket_report.py` | 依赖 4 目标域正式结果目录 | 已可支撑任务书中“分析不同迁移场景”的要求 |
| 稳定性与显著性 | 已完成 | `reports/transfer/transfer_adapt_significance_summary.md`, `reports/transfer/transfer_adapt_significance_writeup.md` | `scripts/analyze_transfer_significance.py` | 基于 4 个目标域 `5-seed` 正式结果汇总生成 | 当前可写“4/4 目标域 5/5 seed 正增益 + CI 下界大于 0” |
| 案例分析 | 已完成 | `reports/transfer/transfer_case_analysis_examples.md`, `reports/transfer/transfer_case_analysis_examples.csv` | `scripts/build_transfer_case_analysis.py` | 基于正式目标域日志生成 | 当前样本量足以入文，但仍可在主线完成后继续扩展示例数量 |
| 效率分析 | 已完成 wall-clock 版本，并补出最小 GPU 辅助结果 | `reports/transfer/transfer_efficiency_summary.md`, `reports/transfer/transfer_efficiency_summary.csv`, `reports/transfer/transfer_gpu_peak_minimal_summary.md` | `scripts/summarize_transfer_efficiency.py`, `scripts/summarize_gpu_peak_minimal.py` | 基于正式 `log.txt` 聚合与最小正式 GPU 补测生成 | GPU 结果当前只覆盖代表性目标域与 `seed=42`，不能替代完整 wall-clock 主表 |
| 过程记录、可复现、可追溯 | 已完成并仍在增强 | `README.md`, `PROCESS_LOG.md`, `PROJECT_OPERATION_RECORD.md`, `reports/notes/thread_sync_shared.md` | 项目级记录文件持续同步 | run 级 `config/log/run_card/artifact_manifest` 已保留在正式目录中 | 当前最值得继续优化的是导航集中度与主线入口清晰度 |

## 3. 主线正式结果与运行入口矩阵

### 3.1 受控复现与消融

| 主题 | 正式结果文件 | 运行脚本 | 正式 run 目录 |
| --- | --- | --- | --- |
| baseline epoch3（DBP15K） | `reports/baseline/baseline_epoch3_results_summary.csv` | `scripts/run_meaformer_multiseed.py`, `scripts/aggregate_meaformer_results.py` | `runs/experiments/baseline/baseline_epoch3/` |
| baseline epoch3（跨图谱） | `reports/baseline/baseline_epoch3_crossgraph_results_summary.csv` | `scripts/run_meaformer_crossgraph_multiseed.py`, `scripts/aggregate_meaformer_results.py` | `runs/experiments/baseline/baseline_epoch3_crossgraph/` |
| TMMEA-DA v1 best epoch3（DBP15K） | `reports/tmmeada/tmmeada_v1_best_epoch3_results_summary.csv` | `scripts/run_tmmeada_multiseed.py`, `scripts/make_tmmeada_baseline_compare_all.py` | `runs/experiments/tmmeada/tmmeada_v1_best_epoch3/` |
| TMMEA-DA v1 best epoch3（跨图谱） | `reports/tmmeada/tmmeada_v1_best_epoch3_crossgraph_results_summary.csv` | `scripts/run_tmmeada_multiseed.py`, `scripts/make_tmmeada_baseline_compare_all.py` | `runs/experiments/tmmeada/tmmeada_v1_best_epoch3_crossgraph/` |
| zh_en 多 seed 消融 | `reports/epoch3/epoch3_ablation_zh_en_multiseed.csv` | `scripts/summarize_epoch3_ablation_zh_en_multiseed.py`, `scripts/make_epoch3_ablation_zh_en.py` | `runs/experiments/tmmeada/tmmeada_v1_ablation_epoch3/` |

### 3.2 迁移主表 4 目标域

| 目标域 | baseline 正式目录 | method 正式目录 | 主结果汇总 | 说明 |
| --- | --- | --- | --- | --- |
| `ja_en` | `runs/transfer/transfer_adapt_ja_v15_full_baseline_matched_ref/target_eval/` | `runs/transfer/transfer_adapt_ja_v15_full_ref/target_eval/` | `reports/transfer/transfer_adapt_ja_v15_expand5_compare_vs_baseline.csv` | 当前主表方法版本：`v15_refresh4_da0025_expand5` |
| `fr_en` | `runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval/` | `runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval/` | `reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv` | 当前主表方法版本：`v14b_refresh4_da0025_expand5` |
| `FBDB15K` | `runs/transfer/transfer_adapt_v18_fbdb_v18c_expand5_baseline_matched_ref/target_eval/` | `runs/transfer/transfer_adapt_v18_fbdb_v18c_expand5_ref/target_eval/` | `reports/transfer/transfer_adapt_v18_fbdb_v18c_expand5_compare_vs_baseline.csv` | 当前主表方法版本：`v18c_bipartite_late_il_skiprel_expand5` |
| `FBYG15K` | `runs/transfer/transfer_adapt_v24_fbyg_v24b_expand5_baseline_matched_ref/target_eval/` | `runs/transfer/transfer_adapt_v24_fbyg_v24b_expand5_ref/target_eval/` | `reports/transfer/transfer_adapt_v24_fbyg_v24b_expand5_compare_vs_baseline.csv` | 当前主表方法版本：`v24b_strictsrc_staged_fresh_il_top400_expand5` |

总汇总文件：

- `reports/transfer/transfer_adapt_main_results_4target.csv`
- `reports/transfer/transfer_adapt_main_results_4target.md`
- `reports/transfer/transfer_adapt_significance_summary.csv`
- `reports/transfer/transfer_adapt_significance_summary.md`

## 4. 当前仍存在的主线外缺口

1. GPU 峰值显存已补出最小正式结果。
   - 相关入口：`scripts/run_gpu_peak_minimal.py`, `scripts/summarize_gpu_peak_minimal.py`
   - 当前结果文件：`reports/transfer/transfer_gpu_peak_minimal_summary.md`
   - 但它仍只覆盖 `seed=42` 与代表性目标域，因此只能写成辅助支撑补测。
2. README / reports / runs 导航仍可进一步收口。
   - 当前已经能找到，但还不够“一眼看清”。
3. 历史探索性材料较多。
   - 已不影响主线成立，但还可进一步减少后来者在目录间跳转成本。

## 5. 直接结论

1. 任务书与开题报告要求的主线项目任务已经具备明确的正式支撑链。
2. 当前最值得继续做的优化，不是再补新的主线实验，而是：
   - 继续把 README / reports / runs 的导航收口到本总表周围；
   - 仅在必要时继续做更小范围的辅助支撑整理。
3. 论文线程现阶段最应吸收的是：
   - 本文件中的“任务要求 -> 正式证据 -> 脚本入口 -> run 目录”映射；
   - 而不是再等待新的主线 rerun。
