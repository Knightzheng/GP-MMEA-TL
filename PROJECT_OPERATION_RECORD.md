# 项目阶段总记录

本文件是当前项目的主记录文件，按研究阶段组织，不再按日期滚动堆叠。旧的时间线版本已归档到：

- [PROJECT_OPERATION_RECORD_chronological_pre_stage_reorg_20260319.md](/d:/code/codes/cursor/BYSJ_zyf/reports/notes/archive/PROJECT_OPERATION_RECORD_chronological_pre_stage_reorg_20260319.md)

## 项目目标

项目围绕“多模态实体对齐模型的可迁移能力研究”展开，核心问题不是单一 benchmark 上的最高分，而是跨语言、跨图谱、跨数据集场景下，模型如何从源域迁移到目标域，并尽量减少对目标域人工标注的依赖。

## S0 基础环境与数据准备

### 这一阶段解决的问题

1. 如何建立可复现实验基础。
2. 如何统一数据准备、日志留痕与指标口径。

### 主要产物

- `env/`
- `data/`
- `00_requirements.md`
- `EXPERIMENT_LOGGING.md`
- `metrics_spec.md`

### 阶段结论

这一阶段完成后，项目具备了“配置、运行、日志、结果、汇总”可追溯的基础框架。

## S1 baseline 复现

### 这一阶段解决的问题

1. 没有可靠 baseline，就无法判断后续改造是否有效。
2. 需要在多数据集、多 seed 下形成统一对照。

### 主要产物

- `configs/baselines/`
- `runs/experiments/baseline/`
- `reports/baseline/`

### 阶段结论

`MEAformer` baseline 已在 `DBP15K` 与跨图谱数据集上复现完成，后续所有方法比较都建立在这一基线上。

## S2 TMMEA-DA 受控开发

### 这一阶段解决的问题

1. 最小侵入式方法改造是否可运行。
2. 在统一预算下，方法模块是否相对 baseline 呈现可信变化。

### 主要产物

- `configs/tmmeada/`
- `runs/experiments/tmmeada/`
- `reports/tmmeada/`
- `reports/epoch3/`
- `reports/compare/`

### 阶段结论

这一阶段的价值不在于得到最终最强结果，而在于回答“方法是否值得继续深入”。当前结果表明：在公平预算下，方法与 baseline 总体接近，单个模块贡献不够硬，因此后续优化不能继续停留在“堆模块”上，而必须转向目标域导向的迁移链路设计。

## S3 epoch10 pilot 与阶段决策

### 这一阶段解决的问题

1. 更高训练预算下是否存在值得继续扩展的方法方向。
2. 哪些方向应继续，哪些方向应停止投入。

### 主要产物

- `reports/epoch10/`
- `reports/planning/`
- `configs/tmmeada/` 中的 `epoch10` / `v2` 相关变体

### 阶段结论

这一阶段帮助项目完成了“是否继续在受控设置中深挖”的判断，为后续把重心转到 transfer 主线提供了依据。

## S4 transfer 主线搭建

### 这一阶段解决的问题

1. 如何从单数据集对照过渡到跨数据集迁移主线。
2. 如何把 `source train`、`target eval` 和 `target adapt` 串成统一流程。

### 主要产物

- `configs/transfer/`
- `configs/transfer_adapt/` 的早期版本
- `runs/transfer/transfer_smoke*`
- `runs/transfer/transfer_formal*`
- `runs/transfer/transfer_adapt_v3 ~ v8*`
- `reports/transfer/` 中的 bootstrap 结果与阶段报告

### 阶段结论

这一阶段完成后，项目从“受控实验项目”转变为“迁移实验项目”，开始真正回答开题报告中的跨数据集迁移问题。

## S5 目标域分支优化与主表收口

### 这一阶段解决的问题

1. 不同目标域的失败模式不同，不能继续混合推进。
2. 需要分别为 `ja_en`、`fr_en`、`FBDB15K`、`FBYG15K` 收口正式版本。

### 子分支

#### S5-A `ja_en / fr_en`

- 主要目录：
  - `runs/transfer/transfer_adapt_v9 ~ v15*`
  - `reports/transfer/transfer_adapt_*fren*`
  - `reports/transfer/transfer_adapt_*ja*`
- 关键结论：
  - `fr_en` 最终收口到 `v14b_refresh4_da0025_expand5`
  - `ja_en` 最终收口到 `v15_refresh4_da0025_expand5`

#### S5-B `FBDB15K`

- 主要目录：
  - `runs/transfer/transfer_adapt_v16 ~ v18*`
  - `reports/transfer/transfer_adapt_*fbdb*`
- 关键结论：
  - `FBDB15K` 主表收口到 `v18c_bipartite_late_il_skiprel_expand5`

#### S5-C `FBYG15K`

- 主要目录：
  - `runs/transfer/transfer_adapt_v19 ~ v25*`
  - `reports/transfer/transfer_adapt_*fbyg*`
- 关键结论：
  - `FBYG15K` 主表收口到 `v24b_strictsrc_staged_fresh_il_top400_expand5`
  - `v25` 作为机制验证保留，但不切换主表

### 阶段总结论

这一阶段结束后，项目已经形成四目标域统一主表：

- `ja_en`
- `fr_en`
- `FBDB15K`
- `FBYG15K`

对应主文件：

- `reports/transfer/transfer_adapt_main_results_4target.md`
- `reports/transfer/transfer_adapt_main_results_4target.csv`

## S6 主线收口与论文/答辩支撑

### 这一阶段解决的问题

1. 主表之外还需要哪些支撑，才能让结论更可信。
2. 论文和答辩如何直接吸收当前主线证据。

### 主要产物

- `reports/transfer/transfer_adapt_significance_summary.*`
- `reports/transfer/transfer_case_analysis_examples.*`
- `reports/transfer/transfer_efficiency_summary.*`
- `reports/transfer/transfer_gpu_peak_minimal_summary.*`
- `reports/notes/mainline_traceability_matrix_20260315.md`
- `reports/notes/mainline_closure_onepage_20260315.md`
- `reports/notes/mainline_artifact_integrity_20260315.md`
- `reports/notes/four_target_evidence_map_20260316.md`
- `reports/notes/defense_qa_packet_20260316.md`

### 阶段结论

项目主线已经闭环。后续高价值工作主要是材料组织、结论边界表达和答辩复用，而不是继续盲目追加主线 rerun。

## S7 中期与提交材料

### 这一阶段解决的问题

1. 如何把当前真实项目状态转换成学校要求的中期材料。
2. 如何避免继续沿用旧线程遗留的自动生成稿。

### 主要产物

- `reports/midterm/README.md`
- `reports/midterm/` 中的正文与模板适配材料
- `scripts/render_midterm_report_template.py`

### 阶段结论

中期材料已切换到“模板 + 开题报告 + 当前真实结果”的写法，不再依赖旧自动稿。

## 当前结构调整策略

本轮结构整理采用“逻辑结构重建、真实路径保留”的策略：

1. 新增 `by_stage/` 导航层，让 `reports/`、`runs/`、`scripts/`、`configs/` 都能按阶段阅读。
2. 重写顶层 README 与主记录文件，改为按阶段组织。
3. 将旧的时间线式大记录归档，而不是继续让它们充当主导航。

## 当前边界

1. `runs/`、`configs/`、`scripts/` 的真实路径没有做高风险整体搬迁。
2. 若后续真的要物理迁移这些目录，必须先统一修改所有硬编码路径和历史引用，再做第二轮结构工程。
3. 当前这一轮调整的目标是先解决“看不清、找不到、记不住”的问题，而不是为了目录美观冒复现链路断裂的风险。
