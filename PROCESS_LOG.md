# 阶段执行日志

本文件不再按“每天/每次更新”滚动追记，而是按项目阶段记录“这一阶段实际执行了什么”。旧的时间线细节已归档至：

- [PROCESS_LOG_chronological_pre_stage_reorg_20260319.md](/d:/code/codes/cursor/BYSJ_zyf/reports/notes/archive/PROCESS_LOG_chronological_pre_stage_reorg_20260319.md)

## S0 基础环境与数据准备

### 本阶段做了什么

1. 建立 `env/` 中的环境快照、依赖锁文件和硬件信息。
2. 完成 DBP15K 的预处理、切分与统一导出。
3. 准备 MEAformer 所需的数据布局，并同步官方数据清单。
4. 固化日志、目录与指标规范。

### 关键入口

- `scripts/preprocess_dbp15k.py`
- `scripts/prepare_meaformer_data.py`
- `scripts/sync_official_meaformer_data.py`

## S1 baseline 复现

### 本阶段做了什么

1. 完成 `DBP15K` 的 `zh_en / ja_en / fr_en` baseline 多 seed 运行。
2. 完成 `FBDB15K / FBYG15K` baseline 多 seed 运行。
3. 将 baseline 结果整理到 `reports/baseline/`，形成后续统一对照口径。

### 关键入口

- `configs/baselines/`
- `scripts/run_meaformer.py`
- `scripts/run_meaformer_multiseed.py`
- `scripts/run_meaformer_crossgraph_multiseed.py`

## S2 TMMEA-DA 受控开发

### 本阶段做了什么

1. 在 `MEAformer` 上接入 `domain_align / source_select / missing_gate` 等机制。
2. 完成 `epoch=3` 受控对照。
3. 完成 `zh_en` 多 seed 消融。
4. 形成 `reports/tmmeada/`、`reports/epoch3/` 与 `reports/compare/` 的阶段成果。

### 关键入口

- `configs/tmmeada/`
- `scripts/run_tmmeada_multiseed.py`
- `scripts/run_tmmeada_v1_weight_sweep.py`
- `scripts/make_epoch3_*.py`

## S3 epoch10 pilot 与阶段决策

### 本阶段做了什么

1. 完成 `epoch10` pilot、`v2` 系列变体与阶段自动决策材料。
2. 将“哪些方向继续推进、哪些方向停止投入”的判断沉淀到 `reports/epoch10/` 与 `reports/planning/`。

### 关键入口

- `reports/epoch10/`
- `reports/planning/`
- `scripts/auto_decide_after_epoch10.py`
- `scripts/compare_epoch10_v2_tuned_vs_baseline.py`

## S4 transfer 主线搭建

### 本阶段做了什么

1. 固化 `source-train -> target-eval` 的正式配置族。
2. 完成 smoke、formal 与早期 `adapt` 链路。
3. 将 `v3-v8` 早期迁移自适应实验写入 `runs/transfer/` 与 `reports/transfer/`。

### 关键入口

- `scripts/run_transfer_train_eval.py`
- `scripts/run_transfer_formal_queue.py`
- `scripts/run_transfer_adapt_v3_queue.py`
- `scripts/run_transfer_adapt_v8_expand_queue.py`

## S5 目标域分支优化与主表收口

### 本阶段做了什么

1. `ja_en / fr_en` 分支推进到 `v9-v15`。
2. `FBDB15K` 分支推进到 `v16-v18`。
3. `FBYG15K` 分支推进到 `v19-v25`。
4. 四个目标域正式主表收口为当前 `5-seed` 版本。

### 关键入口

- `scripts/run_transfer_adapt_v9_fren_auto.py` 至 `scripts/run_transfer_adapt_v14_fren_auto.py`
- `scripts/run_transfer_adapt_ja_v15_iter_queue.py`
- `scripts/run_transfer_adapt_v16_fbdb_iter_queue.py`
- `scripts/run_transfer_adapt_v19_fbyg_iter_queue.py` 至 `scripts/run_transfer_adapt_v25_fbyg_iter_queue.py`
- `scripts/summarize_transfer_formal.py`
- `scripts/make_transfer_main_and_bucket_report.py`

## S6 主线收口与论文/答辩支撑

### 本阶段做了什么

1. 完成显著性统计、案例分析、效率汇总与 GPU 最小补测。
2. 完成主线追溯总表、完整性校验、一页式闭环说明和答辩材料包。
3. 将论文/答辩所需的浓缩材料统一沉淀到 `reports/notes/` 与 `reports/transfer/`。

### 关键入口

- `scripts/analyze_transfer_significance.py`
- `scripts/build_transfer_case_analysis.py`
- `scripts/summarize_transfer_efficiency.py`
- `scripts/run_gpu_peak_minimal.py`
- `scripts/summarize_gpu_peak_minimal.py`
- `scripts/verify_mainline_artifacts.py`

## S7 中期与提交材料

### 本阶段做了什么

1. 清理旧线程留下的中期自动稿痕迹。
2. 基于当前真实项目状态重写中期正文。
3. 保留学校模板适配所需的脚本与目录说明。

### 关键入口

- `reports/midterm/README.md`
- `reports/midterm/`
- `scripts/render_midterm_report_template.py`

## 当前使用建议

1. 想理解项目主线，先看 [PROJECT_OPERATION_RECORD.md](/d:/code/codes/cursor/BYSJ_zyf/PROJECT_OPERATION_RECORD.md)。
2. 想理解目录结构，先看 [PROJECT_STAGE_TREE.md](/d:/code/codes/cursor/BYSJ_zyf/PROJECT_STAGE_TREE.md) 和各目录下的 `by_stage/README.md`。
3. 想追旧时间线，再进入归档目录。
