# 项目全流程操作记录（完整版）

- 项目：`GP-MMEA-TL`（多模态实体对齐迁移实验）
- 记录范围：`2026-02-28` 至 `2026-03-09`
- 当前分支：`sort`
- 当前提交：`1ea10a0`
- 记录时间：`2026-03-09`

## 1. 记录目的

本文件用于完整留痕以下内容，供中期/最终报告直接引用：

1. 从项目启动至当前阶段的关键操作与处理过程。
2. 核心代码改造点与原因。
3. 新建文件/目录的分类清单与作用。
4. 阶段性结果与当前状态快照。

## 2. 仓库状态快照（写入本记录时）

- 提交数：`41`（`git rev-list --count HEAD`）
- 文件总数：`8465`
- 关键目录文件数：
  - `scripts/`：`58` 个 `.py`（含 `__pycache__` 总文件 `126`）
  - `configs/`：`93` 个配置文件
  - `reports/`：`406` 个报告文件
  - `runs/`：`1714` 个文件；命名为时间戳实验目录的 run 数量 `419`
- 迁移阶段报告文件（`reports/transfer/`）：`303` 个

## 3. 全流程时间线（做了什么操作、做了什么处理）

### 阶段 A：项目初始化与实验规范落地（2026-02-28 上午）

主要操作：

1. 初始化仓库与基础实验流水线。
2. 建立“可复现”最小闭环：需求-指标-运行-留痕。
3. 固化环境描述与硬件快照。

主要处理：

1. 解决环境执行细节问题（`conda run`、pip 安装参数等）。
2. 建立统一 run 产物结构（`run_card/config/metrics/log/artifact_manifest`）。

新增/落地关键文件：

- `00_requirements.md`：任务约束、阶段目标、待完成项。
- `metrics_spec.md`：统一指标口径（Hits@1/Hits@10/MRR）。
- `project_charter.yaml`：项目章程与里程碑。
- `EXPERIMENT_LOGGING.md`：实验留痕规范。
- `PROCESS_LOG.md`：过程日志（后续持续追加）。
- `env/conda-pytorch.yaml`、`env/requirements.lock.txt`、`env/hardware_snapshot.txt`：环境证据。

### 阶段 B：数据接入与 baseline 复现（2026-02-28）

主要操作：

1. 接入 DBP15K 预处理流水线。
2. 接入官方 MEAformer 代码与数据结构。
3. 跑通 baseline 首次可用结果，并扩展多语种、多图谱。

主要处理：

1. 修复缺失依赖（如 `scipy`）导致的运行失败。
2. 修复 `ill_ent_ids` 格式差异导致的数据读取异常。
3. 在显存受限条件下切换到可运行安全配置（`*_rtx3060_safe.yaml`）。

新增/落地关键文件：

- `scripts/preprocess_dbp15k.py`：DBP15K 下载、切分、导出统一 TSV。
- `scripts/prepare_meaformer_data.py`：转换为 MEAformer 所需目录。
- `scripts/run_meaformer.py`：统一单次训练/评测入口并自动写 run 证据。
- `scripts/run_meaformer_multiseed.py`：DBP 多 seed。
- `scripts/run_meaformer_crossgraph_multiseed.py`：跨图谱多 seed。
- `scripts/collect_meaformer_results.py`、`scripts/aggregate_meaformer_results.py`：汇总统计。
- `configs/baselines/*.yaml`：baseline 配置族（DBP+FBDB+FBYG、epoch 变体）。
- `reports/baseline/*`、`reports/meaformer_results_*`：baseline 汇总与均值方差。

### 阶段 C：TMMEA-DA MVP 与 v1 模块化改造（2026-02-28 至 2026-03-02）

主要操作：

1. 在 MEAformer 上实现 TMMEA-DA MVP（Domain Align）。
2. 扩展 v1：`source_select` + `missing_gate` 两个可开关模块。
3. 完成 DBP15K（`zh_en/ja_en/fr_en`）与跨图谱（`FBDB15K/FBYG15K`）多 seed 对比。
4. 完成 `zh_en epoch3` 的消融实验（含 5-seed）。

主要处理：

1. 把方法改造做成“配置开关”，避免硬编码导致实验不可比。
2. 用多 seed 替代单次运行，降低偶然性。
3. 修复报告文字与真实 seed 状态不一致问题（脚本改为自动读取 `num_runs`）。

核心改造文件：

- `baselines/MEAformer/config.py`
- `baselines/MEAformer/model/MEAformer.py`
- `baselines/MEAformer/src/data.py`
- `scripts/run_meaformer.py`

新增/落地关键文件：

- `scripts/run_tmmeada_multiseed.py`
- `scripts/run_tmmeada_v1_weight_sweep.py`
- `scripts/summarize_tmmeada_v1_sweep.py`
- `scripts/make_tmmeada_v1_compare_zh_en.py`
- `scripts/make_tmmeada_v1_best_compare_zh_en.py`
- `scripts/make_tmmeada_baseline_compare*.py`
- `scripts/make_epoch3_*.py`
- `scripts/summarize_epoch3_ablation_zh_en_multiseed.py`
- `configs/tmmeada/*.yaml`（MVP/v1/ablation/v2 pilot 系列）
- `reports/tmmeada/*`、`reports/epoch3/*`、`reports/compare/*`

### 阶段 D：v2 调优与自动决策流水线（2026-03-03 至 2026-03-04）

主要操作：

1. 建立 `epoch10` pilot 自动比较与自动决策。
2. 实现 v2a/v2b/v2c 分支试跑与自动接续。

主要处理：

1. 在夜间连续运行中加入“自动下一步”分派，减少手动盯进程。
2. 将试跑结果自动输出决策文件（`.md + .json`），便于报告追溯。

新增/落地关键文件：

- `scripts/auto_decide_after_epoch10.py`
- `scripts/auto_decide_next_stage.py`
- `scripts/auto_next_after_v2b.py`
- `scripts/auto_compare_v2_tuned.py`
- `scripts/run_next_stage_pilot_queue.py`
- `scripts/compare_epoch10_v2_tuned_vs_baseline.py`
- `reports/epoch10/*`

### 阶段 E：source->target 正式迁移链路（2026-03-04）

主要操作：

1. 建立 source 训练 + target 评估完整流水线。
2. 跑通 smoke 与 formal 两套迁移流程（baseline 和 tmmeada）。
3. 进行一次目录重构，清理 `runs/reports` 的混放问题。

主要处理：

1. 统一迁移实验命名与目录，确保后续自动化脚本可识别。
2. 形成 `source_train` / `target_eval` 双段式可追溯结构。

新增/落地关键文件：

- `scripts/run_transfer_train_eval.py`
- `scripts/run_transfer_formal_queue.py`
- `scripts/summarize_transfer_formal.py`
- `configs/transfer/*.yaml`
- `runs/transfer/transfer_smoke*`
- `runs/transfer/transfer_formal*`
- `reports/transfer/transfer_smoke_*`

### 阶段 F：transfer-adapt v3~v14 连续优化（2026-03-05 至 2026-03-08）

主要操作：

1. v3-v7：针对 `ja_en`、`FBDB15K` 做策略迭代。
2. v8：扩展到 `fr_en`、`FBYG15K`。
3. v9-v13：持续优化 `fr_en`（过滤、恢复、轻量模块）。
4. v14：引入 IL 刷新频率控制，形成明显正增益。

主要处理：

1. 采用“pilot -> 自动选优 -> formal -> 2-seed 汇总”固定流程。
2. 出现断电后执行断点恢复，保证队列可续跑。
3. 将自动决策、对比表、阶段更新统一沉淀到 `reports/transfer/`。

核心改造文件：

- `baselines/MEAformer/config.py`：新增 `--il_refresh_interval`。
- `baselines/MEAformer/main.py`：IL 刷新逻辑改为可配置间隔。
- `scripts/run_meaformer.py`：支持透传 `il_refresh_interval` 及更多方法参数。

新增/落地关键文件（部分）：

- `scripts/run_transfer_adapt_v3_queue.py`
- `scripts/run_transfer_adapt_v4_queue.py`
- `scripts/run_transfer_adapt_v5_queue.py`
- `scripts/run_transfer_adapt_v6_mixed_queue.py`
- `scripts/run_transfer_adapt_v7_fbdb_auto.py`
- `scripts/run_transfer_adapt_v8_expand_queue.py`
- `scripts/run_transfer_adapt_v9_fren_auto.py`
- `scripts/run_transfer_adapt_v10_fren_auto.py`
- `scripts/run_transfer_adapt_v11_fren_auto.py`
- `scripts/run_transfer_adapt_v12_fren_auto.py`
- `scripts/run_transfer_adapt_v13_fren_auto.py`
- `scripts/run_transfer_adapt_v14_fren_auto.py`
- `configs/transfer_adapt/*.yaml`（v3-v14 变体）
- `reports/transfer/transfer_adapt_v*_*.{csv,md,json}`

### 阶段 G：5-seed 扩展收尾、自动监控、同步闭环（2026-03-08 至 2026-03-09）

主要操作：

1. 完成 `fr_en v14` 的 5-seed 扩展与最终汇总。
2. 启动并完成 `FBYG15K expand5`（补齐 `2026/7/123`）。
3. 加入每小时进度上报脚本并在收尾后停止。
4. 完成 `sort` 分支同步提交。

主要处理：

1. 自动 finalize 脚本修复为“仅提交已暂存文件”。
2. 断点续跑脚本自动识别已完成 seed，避免重复计算。
3. 形成最终状态文件用于报告引用。

新增/落地关键文件：

- `scripts/run_transfer_adapt_v14_fren_expand5_resume.py`
- `scripts/run_transfer_adapt_fbyg_expand5_resume.py`
- `scripts/auto_after_v14_expand5_then_next.py`
- `scripts/auto_after_transfer_adapt_queue.py`
- `scripts/hourly_progress_reporter.py`
- `reports/transfer/transfer_adapt_v14_fren_expand5_status.{json,md}`
- `reports/transfer/transfer_adapt_fbyg_expand5_status.{json,md}`
- `reports/transfer/transfer_adapt_fbyg_expand5_progress_compare_vs_baseline.{csv,md}`
- `reports/transfer/hourly_progress.md`

## 4. 核心代码改造明细（用于论文“方法实现”章节）

### 4.1 参数层（`baselines/MEAformer/config.py`）

新增可控参数：

- `use_domain_align`, `domain_align_weight`
- `use_source_select`, `source_select_weight`, `source_select_temp`
- `use_missing_gate`, `missing_align_weight`
- `il_refresh_interval`

作用：

1. 将方法模块显式参数化，支持公平消融。
2. 支持迁移自适应阶段的 IL 刷新节奏调优。

### 4.2 损失与训练逻辑（`baselines/MEAformer/model/MEAformer.py`）

新增/扩展逻辑：

1. `domain_align`（含正样本与 hard 负样本项）。
2. `source_select` 模态源选择辅助项。
3. `missing_align` 缺失感知图像对齐项。
4. loss 字典暴露新增项，便于 TensorBoard 和日志分析。

作用：

1. 让 TMMEA-DA 与 baseline 在同一主干上可直接切换。
2. 支持后续按模块做消融与版本迭代。

### 4.3 IL 刷新控制（`baselines/MEAformer/main.py`）

新增逻辑：

- 将原固定刷新节奏改为 `semi_learn_step * il_refresh_interval`。

作用：

- 在 `fr_en` 上显著改善后期噪声积累问题（v14）。

### 4.4 统一运行入口（`scripts/run_meaformer.py`）

新增能力：

1. `meta.stage`、`meta.model_tag` 路径组织。
2. 方法参数透传（domain/source_select/missing/il_refresh 等）。
3. 自动生成 `run_card.md / config.yaml / log.txt / artifact_manifest.json`。

作用：

- 形成“配置驱动 + 自动留痕 + 可复现”的统一执行接口。

## 5. 新建文件清单与作用（分类）

### 5.1 根目录治理文件

- `00_requirements.md`：任务边界、里程碑、交付要求。
- `metrics_spec.md`：统一指标定义与比较口径。
- `project_charter.yaml`：项目章程。
- `EXPERIMENT_LOGGING.md`：实验留痕规范模板。
- `PROCESS_LOG.md`：过程日志。
- `README.md`：项目说明、阶段结果摘要、运行方式。
- `base.py`：本机环境初始化/安装入口。
- `test.py`：Python/CUDA/Anaconda 可用性检测。

### 5.2 环境文件（`env/`）

- `conda-pytorch.yaml`：Conda 环境定义。
- `requirements.lock.txt`：依赖锁定快照。
- `hardware_snapshot.txt`：硬件与驱动快照证据。

### 5.3 核心脚本文件（`scripts/`，58 个）

#### A. 数据与基础运行

- `preprocess_dbp15k.py`：DBP15K 预处理与切分导出。
- `prepare_meaformer_data.py`：转换为 MEAformer 数据组织。
- `sync_official_meaformer_data.py`：同步官方数据并产生日志。
- `train_baseline.py`：早期 baseline 占位训练脚本。
- `run_meaformer.py`：统一训练/评测总入口。
- `run_meaformer_multiseed.py`：DBP 多 seed 执行。
- `run_meaformer_crossgraph_multiseed.py`：跨图谱多 seed。
- `run_tmmeada_multiseed.py`：TMMEA-DA 多 seed。
- `run_from_base_config_multiseed.py`：按 base config 批量改 seed 执行。

#### B. 汇总与对比

- `collect_meaformer_results.py`：收集 run 结果到 summary。
- `aggregate_meaformer_results.py`：汇总均值/标准差。
- `summarize_transfer_formal.py`：formal transfer 汇总。
- `compare_transfer_summaries.py`：迁移阶段对比。
- `make_tmmeada_baseline_compare.py`：v0 vs baseline（zh_en）对比。
- `make_tmmeada_baseline_compare_dbp15k.py`：DBP 多语种对比。
- `make_tmmeada_baseline_compare_all.py`：全数据集对比。
- `make_tmmeada_v1_compare_zh_en.py`：v1 vs baseline 对比。
- `make_tmmeada_v1_best_compare_zh_en.py`：v1_best 对比。
- `compare_epoch10_v2_tuned_vs_baseline.py`：epoch10 pilot 对比。
- `summarize_tmmeada_v1_sweep.py`：权重搜索汇总。
- `make_epoch3_pilot_compare_zh_en.py`：epoch3 zh_en 试跑对比。
- `make_epoch3_multiseed_compare_zh_en.py`：epoch3 zh_en 5-seed 对比。
- `make_epoch3_compare_dbp15k.py`：epoch3 DBP 全语种对比。
- `make_epoch3_compare_crossgraph.py`：epoch3 跨图谱对比。
- `make_epoch3_ablation_zh_en.py`：zh_en 消融对比。
- `summarize_epoch3_ablation_zh_en_multiseed.py`：消融 5-seed 汇总。

#### C. epoch10/v2 自动决策链

- `run_next_stage_pilot_queue.py`：下一阶段 pilot 队列。
- `auto_decide_after_epoch10.py`：epoch10 后自动决策。
- `auto_decide_next_stage.py`：下一阶段自动选择。
- `auto_next_after_v2b.py`：v2b 后自动接续。
- `auto_compare_v2_tuned.py`：v2 tuned 自动对比。
- `run_tmmeada_v1_weight_sweep.py`：v1 权重搜索执行。

#### D. source->target 迁移与 transfer-adapt 队列

- `run_transfer_train_eval.py`：source 训练 + target 评测。
- `run_transfer_formal_queue.py`：formal 队列。
- `run_transfer_adapt_pilot_queue.py`：adapt pilot 队列。
- `run_transfer_adapt_tuned_queue.py`：adapt tuned 队列。
- `run_transfer_adapt_expand_queue.py`：扩展队列。
- `run_transfer_adapt_v3_queue.py`：v3 队列。
- `run_transfer_adapt_v4_queue.py`：v4 队列。
- `run_transfer_adapt_v5_queue.py`：v5 队列。
- `run_transfer_adapt_v6_mixed_queue.py`：v6 混合策略队列。
- `run_transfer_adapt_v7_fbdb_auto.py`：v7 FBDB 自动选优。
- `run_transfer_adapt_v8_expand_queue.py`：v8 扩展队列。
- `run_transfer_adapt_v9_fren_auto.py`：v9 fr_en 自动流程。
- `run_transfer_adapt_v10_fren_auto.py`：v10 fr_en 自动流程。
- `run_transfer_adapt_v11_fren_auto.py`：v11 fr_en 自动流程。
- `run_transfer_adapt_v12_fren_auto.py`：v12 fr_en 自动流程。
- `run_transfer_adapt_v13_fren_auto.py`：v13 fr_en 自动流程。
- `run_transfer_adapt_v14_fren_auto.py`：v14 fr_en 自动流程。
- `run_transfer_adapt_v14_fren_expand5_resume.py`：v14 5-seed 断点续跑。
- `run_transfer_adapt_fbyg_expand5_resume.py`：FBYG 5-seed 断点续跑。

#### E. 自动监控与自动收尾

- `auto_after_transfer_adapt_queue.py`：adapt 队列收尾自动化。
- `auto_after_transfer_adapt_v3.py`：v3 后处理。
- `auto_after_transfer_adapt_v4.py`：v4 后处理。
- `auto_after_transfer_adapt_v5.py`：v5 后处理。
- `auto_after_transfer_adapt_v6_mixed.py`：v6 后处理。
- `auto_after_v14_expand5_then_next.py`：v14 expand5 完成后自动衔接下一阶段。
- `hourly_progress_reporter.py`：每小时进度汇总到报告文件。

### 5.4 配置文件（`configs/`，93 个）

按用途分为 4 组：

1. `configs/baselines/`（16 个）：baseline 在不同数据集、epoch、显存约束下的配置。
2. `configs/tmmeada/`（29 个）：MVP/v1/ablation/v2 pilot 等方法配置。
3. `configs/transfer/`（10 个）：source->target 正式迁移（train/eval）配置。
4. `configs/transfer_adapt/`（38 个）：v3-v14 迁移自适应策略配置。

命名规则说明：

- `*_epoch3.yaml` / `*_epoch8_pilot.yaml` / `*_epoch10_pilot.yaml`：训练预算阶段。
- `v9/v10/v11/.../v14`：优化迭代代号。
- `unsup_il`：无标注目标域 + IL 流程。

### 5.5 报告与运行产物（`reports/`、`runs/`）

`reports/` 子目录及作用：

- `reports/baseline/`：baseline 汇总。
- `reports/tmmeada/`：方法汇总、对比、消融。
- `reports/epoch3/`、`reports/epoch10/`：分预算阶段分析。
- `reports/compare/`：跨方法聚合对比。
- `reports/transfer/`：迁移与自适应主报告（当前核心证据目录）。
- `reports/midterm/`：中期报告草稿与章节。
- `reports/planning/`：任务书对照与计划状态。

`runs/` 结构及作用：

- `runs/experiments/baseline/*`：baseline 训练 run。
- `runs/experiments/tmmeada/*`：方法训练 run。
- `runs/transfer/*`：迁移相关 run（smoke/formal/adapt/expand）。
- 每个 run 目录统一包含：
  - `run_card.md`
  - `config.yaml`
  - `log.txt`
  - `artifact_manifest.json`

## 6. 阶段结果快照（写入本记录时）

- `fr_en`（v14 expand5）：
  - 文件：`reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv`
  - 5-seed 结果：`delta_avg_mrr_mean = +0.0121`
- `FBYG15K`（expand5）：
  - 文件：`reports/transfer/transfer_adapt_fbyg_expand5_progress_compare_vs_baseline.csv`
  - 5-seed 结果：`delta_avg_mrr_mean = +0.0011`
- 完成状态文件：
  - `reports/transfer/transfer_adapt_v14_fren_expand5_status.md`
  - `reports/transfer/transfer_adapt_fbyg_expand5_status.md`

## 7. 异常与恢复记录（关键）

1. 出现过多次断电/自动关机场景。
2. 已实现断点续跑脚本（v14 与 FBYG expand5），并完成缺失 seeds 自动补跑。
3. 自动 finalize 流程曾修复“提交范围控制”问题，最终版本仅提交已暂存文件。
4. 小时级进度上报脚本已在阶段收尾后停止，避免后台进程残留。

## 8. 当前可直接用于报告的证据路径

1. 方法改造说明：`baselines/MEAformer/config.py`, `baselines/MEAformer/model/MEAformer.py`, `baselines/MEAformer/main.py`
2. 统一运行入口：`scripts/run_meaformer.py`
3. 迁移阶段主证据：`reports/transfer/`
4. 运行原始证据：`runs/transfer/`, `runs/experiments/`
5. 中期素材：`reports/midterm/`

## 9. 提交历史摘要（按时间顺序）

以下为从项目启动到当前的提交主线（共 41 次）：

1. `e971759` initialize baseline pipeline
2. `5cd7487` tmmeada zh_en 5-seed
3. `8b4362c` 扩展 ja/fr 5-seed + README
4. `3f35509` 跨图谱 5-seed
5. `317d834` v1 source-select/missing-gate smoke
6. `ee26899` v1 zh_en 5-seed
7. `74f763c` v1 sweep + epoch3 pilot
8. `5ff6169` epoch3 zh_en 5-seed
9. `023ed98` epoch3 ja/fr pilot
10. `5aefef5` 断电恢复 ja_en
11. `68debe0` fr_en epoch3 5-seed 完成
12. `77658a8` 跨图谱 epoch3 pilot
13. `7a7c19d` 跨图谱 epoch3 5-seed
14. `84f3532` zh_en 消融 pilot
15. `1302f4d` zh_en 消融 5-seed
16. `bff76d0` v2 tuning pipeline
17. `c4945b2` 记录 v2a 并启动 v2b
18. `cea1564` v2b 后自动分派
19. `1d3d571` 夜间 v2b-v2c 汇总
20. `c3fb3b3` v2b/v2c 对比结果
21. `c2c2a6f` source->target transfer pipeline
22. `08197f7` formal transfer queue
23. `18bed28` 刷新 v2 决策临时数据
24. `1d5d302` runs/reports 结构重构
25. `b0a72ca` transfer adapt v3-v7
26. `c834d6f` README 同步 v7 状态
27. `a56326a` v8 扩展 s42
28. `b9ef820` v8 2-seed 完成
29. `826eeff` v9 fr_en 自动优化
30. `35f8d87` v10 fr_en 自动优化
31. `ed8643a` v11 置信过滤管线
32. `c559cdb` v12 恢复优化
33. `eedc7fd` v13 轻量模块优化
34. `d03e6a9` v14 fr_en 断电恢复并收敛
35. `d2b5e2b` 启动 v14 expand5
36. `bca9e94` 自动监控/收尾与 FBYG resume
37. `ac29723` 修复自动收尾提交范围
38. `e174cdc` 纳入 v14 run-card
39. `47b5e7f` 完成 v14 fr_en expand5
40. `66f40e4` 启动 FBYG expand5
41. `1ea10a0` 完成 FBYG expand5 并停止小时上报

## 10. 说明

- 本文件为“阶段全流程总记录”，后续如继续实验，建议以追加方式更新本文件底部，不覆盖历史内容。
- 若论文附录需要“按实验编号逐条列出”，可基于 `runs/**/run_card.md` 自动生成二级附录。

## 11. 2026-03-09 追加记录（程序结束后的文档收口）

本次追加操作（对应“程序结束后更新记录与 README”）：

1. 核对运行状态：确认项目下无进行中的 `python.exe` 训练进程。
2. 确认 `FBYG expand5` 已完整结束：
   - `reports/transfer/transfer_adapt_fbyg_expand5_status.md` 显示 `final_missing_seeds=[]`。
   - `reports/transfer/transfer_adapt_fbyg_expand5_progress_compare_vs_baseline.csv` 已产出 5-seed 对比结果。
3. 新增阶段完成报告：
   - `reports/transfer/transfer_stage_update_20260309_fbyg_expand5_final.md`。
4. 更新 `README.md`：
   - “最新迁移阶段报告”指向 `transfer_stage_update_20260309_fbyg_expand5_final.md`；
   - “当前阶段结论”改为使用 `fr_en / FBYG15K` 的 5-seed 正式结果；
   - 追加“阶段更新（2026-03-09）”说明 expand5 收官状态与关键结果。

本次追加的直接作用：

- 让仓库首页结论与最新正式结果保持一致；
- 让中期/终稿引用路径从“启动状态”切换到“完成状态”；
- 补齐 `FBYG15K` 收官报告，形成完整证据链闭环。

## 12. 2026-03-09 追加记录（主结果表与误差分桶分析）

本次追加操作（对应“继续下一步：主表与分析”）：

1. 新增自动生成脚本：
   - `scripts/make_transfer_main_and_bucket_report.py`
2. 生成 4 目标统一主结果表：
   - `reports/transfer/transfer_adapt_main_results_4target.csv`
   - `reports/transfer/transfer_adapt_main_results_4target.md`
3. 生成分桶分析结果：
   - `reports/transfer/transfer_adapt_error_bucket_summary.csv`
   - `reports/transfer/transfer_adapt_error_bucket_summary.md`
4. 新增阶段报告：
   - `reports/transfer/transfer_stage_update_20260309_main_table_bucket.md`
5. 更新 `README.md`：
   - 最新阶段报告链接切换到 `transfer_stage_update_20260309_main_table_bucket.md`；
   - 当前阶段结论增加“2-seed/5-seed 置信度说明”；
   - 追加“阶段更新（2026-03-09）”记录主表与分桶分析产物。

本次追加的直接作用：

- 把零散的 transfer-adapt 结果收敛为“可直接进论文主表”的统一格式；
- 给出可追溯、可复算的误差分桶统计口径（场景/置信度/难度）；
- 为下一步“补齐 ja_en/FBDB15K 的 5-seed”提供明确基线。

## 13. 2026-03-09 追加记录（ja_en + FBDB15K expand5 自动收口）

本次追加操作：

1. 恢复并完成 `ja_en + FBDB15K` 的 expand5 队列（缺失 seed 自动补跑）。
2. 自动刷新 4目标统一主结果表与误差分桶分析。
3. 自动更新 README 与阶段报告链接。
4. 将本阶段结果与脚本改动提交并同步到 GitHub `sort` 分支。

新增阶段报告：

- `reports/transfer/transfer_stage_update_20260309_ja_fbdb_expand5_final.md`

## 14. 2026-03-11 追加记录（ja_en v15 正式收口）

本次追加操作：

1. 接管并核查 `ja_en v15` 未提交实验分支，确认已有 `s42/s2026` 完成、`s3407` 中断。
2. 修复 transfer-adapt 恢复/汇总口径：
   - 仅将 `[DONE] return_code=0` 的 run 视为完成；
   - 续跑与汇总时只选择“按 seed 去重后的最新完整 run”。
3. 补跑 `ja_en v15` 缺失 seeds：
   - `3407`
   - `7`
   - `123`
4. 刷新 `ja_en v15` 决策与正式 compare 文件：
   - `reports/transfer/transfer_adapt_ja_v15_iter_decision.{md,json}`
   - `reports/transfer/transfer_adapt_ja_v15_expand5_compare_vs_baseline.{csv,md}`
5. 更新 4目标统一主结果表与误差分桶分析，将 `ja_en` 主表条目切换为 `v15_refresh4_da0025_expand5`。
6. 更新 `README.md`、`PROCESS_LOG.md` 与本记录，并准备同步 GitHub。

关键结果：

- `ja_en v15`（5-seed）：
  - `delta_avg_hits@1_mean = +0.01094`
  - `delta_avg_hits@10_mean = +0.01410`
  - `delta_avg_mrr_mean = +0.01210`
- 当前 4目标主表均为 `5-seed` 正式正增益。

新增阶段报告：

- `reports/transfer/transfer_stage_update_20260311_ja_v15_takeover.md`
- `reports/transfer/transfer_stage_update_20260311_ja_v15_final.md`

## 15. 2026-03-12 追加记录（FBDB15K v17 噪声控制 pilot）

本次追加操作：

1. 重新评估 `FBDB15K` 的优化方向，确认不再继续围绕 `domain_align_weight` 做微调。
2. 新增 `v17` 三个 `2-seed pilot` 变体并完成自动汇总：
   - `v17a_no_il_balanced`
   - `v17b_late_il_strict`
   - `v17c_late_il_skiprel`
3. 新增自动脚本：
   - `scripts/run_transfer_adapt_v17_fbdb_iter_queue.py`
4. 生成 `v17` 决策与对比文件：
   - `reports/transfer/transfer_adapt_v17_fbdb_iter_decision.{md,json}`
   - `reports/transfer/transfer_adapt_v17_fbdb_pilot_v17a_compare_vs_baseline.{csv,md}`
   - `reports/transfer/transfer_adapt_v17_fbdb_pilot_v17b_compare_vs_baseline.{csv,md}`
   - `reports/transfer/transfer_adapt_v17_fbdb_pilot_v17c_compare_vs_baseline.{csv,md}`
5. 更新 `README.md`、`PROCESS_LOG.md` 与阶段报告链接，记录 `v17` 结论。

关键结果：

- 参考主表版本：`FBDB15K v7b`（`5-seed delta_avg_mrr_mean = +0.0008`）
- `v17` pilot（`2-seed`, vs baseline）：
  - `v17a = -0.00800`
  - `v17b = -0.00850`
  - `v17c = -0.00775`
- `v17b/v17c` 在严格晚启 IL 设置下均未产生有效新伪链接（日志为 `il_filter raw=0 kept=0`）。

本次追加的直接作用：

- 用一次完整的 `P0` 验证，排除了“继续压缩 IL 注入量即可修复 FBDB”的路径；
- 明确将下一步优化切换到 `P1`：修改 `visual_pivot_induction` 的选种机制；
- 避免继续在 `FBDB15K` 上消耗时间做低收益的 `DA weight` 与轻量配置搜索。

新增阶段报告：

- `reports/transfer/transfer_stage_update_20260312_v17_fbdb_noise_control.md`

## 16. 2026-03-12 追加记录（FBDB15K v18 bipartite seeds 正式收口）

本次追加操作：

1. 在 `baselines/MEAformer/src/data.py` 中重写 `visual_pivot_induction` 的可选分支，引入：
   - `mutual nearest` 过滤
   - `margin` 过滤
   - `unsup_no_fallback`
   - `unsup_k_max`
2. 在 `baselines/MEAformer/config.py` 中新增相应命令行参数，并在 `scripts/run_meaformer.py` 中补齐透传。
3. 新增 `v18` 三个 `FBDB15K` pilot 配置：
   - `v18a_bipartite_no_il`
   - `v18b_bipartite_late_il`
   - `v18c_bipartite_late_il_skiprel`
4. 新增自动脚本：
   - `scripts/run_transfer_adapt_v18_fbdb_iter_queue.py`
5. 完成 `2-seed pilot -> 自动选优 -> 5-seed expand` 全流程。
6. 生成 `v18` 决策、pilot compare、formal compare 文件，并将 `FBDB15K` 主表版本切换为 `v18c`。
7. 刷新 4目标统一主结果表与误差分桶分析。

关键结果：

- `v18` 初始 visual seeds 真值率约为 `15.67%`，明显高于 `v17` 的约 `5.67%`。
- `v18` pilot（`2-seed`, vs baseline）：
  - `v18a = +0.00750`
  - `v18b = +0.00700`
  - `v18c = +0.00800`
- `v18c` 正式 `5-seed`（vs baseline）：
  - `delta_avg_hits@1_mean = +0.00454`
  - `delta_avg_hits@10_mean = +0.01568`
  - `delta_avg_mrr_mean = +0.00830`
  - `delta_avg_mr_mean = -206.81670`

本次追加的直接作用：

- 验证 `P1` 路线正确，将 `FBDB15K` 从“边际小正增益”提升为“稳定明显正增益”；
- 将 `FBDB15K` 主表版本从 `v7b` 切换为 `v18c`；
- 证明 `FBDB15K` 的主要瓶颈在伪种子生成，而不是继续做 `DA weight` 或轻量 IL 调参。

新增阶段报告：

- `reports/transfer/transfer_stage_update_20260312_v18_fbdb_bipartite_full5.md`

## 17. 2026-03-12 追加记录（FBYG15K v19/v20 pilot，主表保持 v8）

本次追加操作：

1. 在迁移加载链路中新增前缀级过滤能力：
   - `baselines/MEAformer/config.py`
   - `baselines/MEAformer/main.py`
   - `scripts/run_meaformer.py`
   - `scripts/run_transfer_train_eval.py`
2. 补强源模型解析逻辑，使 `scripts/transfer_adapt_utils.py` 可以回收已有 `transfer_adapt_*` 源检查点。
3. 新增 `FBYG15K v19` 三个 pilot 配置与自动脚本：
   - `tmmeada_target_fbyg15k_v19a_late_il_strict`
   - `tmmeada_target_fbyg15k_v19b_late_il_skiprel`
   - `tmmeada_target_fbyg15k_v19c_late_il_skiprel_skipfusion`
   - `scripts/run_transfer_adapt_v19_fbyg_iter_queue.py`
4. 发现 `v19` 的 `il_start=8` 与当前 fresh-proposal 周期错位，导致 IL 实际接近关闭。
5. 在此基础上新增 `FBYG15K v20` 两个对齐周期的 pilot 配置与自动脚本：
   - `tmmeada_target_fbyg15k_v20a_aligned_il_skiprel_skipfusion`
   - `tmmeada_target_fbyg15k_v20b_aligned_il_q90_skiprel_skipfusion`
   - `scripts/run_transfer_adapt_v20_fbyg_iter_queue.py`
6. 完成 `v19` 与 `v20` 的 `2-seed pilot` 全流程，并生成决策与 compare 文件。
7. 更新 `README.md`、`PROCESS_LOG.md` 与最新阶段报告链接，但保持主结果表不切换。

关键结果：

- 当前参考主表版本：`FBYG15K v8_mild_da_expand5`
  - `5-seed delta_avg_mrr_mean = +0.00110`
- `v19` pilot（`2-seed`, vs baseline）：
  - `v19a = -0.00225`
  - `v19b = -0.00250`
  - `v19c = +0.00100`
- `v20` pilot（`2-seed`, vs baseline）：
  - `v20a = +0.00050`
  - `v20b = +0.00050`

关键诊断：

- `v19` 主要验证了“更保守的迁移加载”，因为其 `late IL` 与 fresh-proposal 周期错位；
- `v20` 对齐周期后，`epoch 5` 虽出现大量 IL 候选，但到 `epoch 9` 实际注入只剩 `1` 条链接，且真值率为 `0.0%`；
- 这表明 `FBYG15K` 的下一步瓶颈在 IL 生成/刷新机制本身，而不是继续压 `quantile` 或继续追加 `skip keys/prefixes`。

本次追加的直接作用：

- 排除了 `FBYG15K` 上继续做轻量 `IL schedule / transfer-skip` 搜索的必要性；
- 保留了当前最优正式版本 `v8`，避免以 pilot 偶然波动替换主表；
- 为后续是否继续做 `FBYG15K` 方法优化，提供了明确而可复现的负结果证据链。

新增阶段报告：

- `reports/transfer/transfer_stage_update_20260312_fbyg_v19_v20_pilot.md`

## 18. 2026-03-12 追加记录（FBYG15K v21 fresh-IL full5，主表切换）

本次追加操作：

1. 基于 `v19/v20` 的诊断，转向验证 `FBYG15K` 上的 fresh-IL 立即注入路线，目标是修复“候选在注入前塌缩”的问题。
2. 新增 `FBYG15K v21` 三个 pilot 配置与自动脚本：
   - `tmmeada_target_fbyg15k_v21a_fresh_il_q80_skiprel_skipfusion`
   - `tmmeada_target_fbyg15k_v21b_fresh_il_q90_skiprel_skipfusion`
   - `tmmeada_target_fbyg15k_v21c_fresh_il_q95_skiprel_skipfusion`
   - `scripts/run_transfer_adapt_v21_fbyg_iter_queue.py`
3. 完成 `2-seed pilot -> 自动选优 -> 5-seed expand` 全流程。
4. 新增 `v21` 决策、pilot compare、formal compare 与 run-card 文件。
5. 刷新 `scripts/make_transfer_main_and_bucket_report.py` 的 `FBYG15K` 主表入口，将主结果切换到 `v21a full5`。
6. 更新 `README.md`、`PROCESS_LOG.md` 与最新阶段报告链接。

关键结果：

- 当前旧参考主表版本：`FBYG15K v8_mild_da_expand5`
  - `5-seed delta_avg_mrr_mean = +0.00110`
- `v21` pilot（`2-seed`, vs baseline）：
  - `v21a = +0.00200`
  - `v21b = +0.00100`
  - `v21c = +0.00100`
- 自动决策：
  - `best_variant_pilot = v21a`
  - `improve_over_current_ref = +0.00090`
  - 达到扩展阈值后自动扩展到 `5-seed`
- `v21a` 正式 `5-seed`（vs baseline）：
  - `delta_avg_hits@1_mean = +0.00141`
  - `delta_avg_hits@10_mean = +0.00193`
  - `delta_avg_mrr_mean = +0.00160`
  - `delta_avg_mr_mean = -35.84720`

关键诊断：

- `v21a` 的 fresh-IL 立即注入，避免了 `v20` 中“候选到最终只剩 1 条链接”的塌缩问题；
- `5-seed` 日志中，新增链接规模稳定在 `397 ~ 450` 条，真值率约 `1.8% ~ 2.5%`；
- 这说明 `FBYG15K` 的小幅正式增益可以通过“及时注入 fresh proposals”获得，但后续若继续优化，重点仍应放在候选质量而非继续做晚启 IL 网格搜索。

本次追加的直接作用：

- 将 `FBYG15K` 主表版本从 `v8` 切换为 `v21a`；
- 把 `FBYG15K` 的 `5-seed delta_avg_mrr_mean` 从 `+0.00110` 提升到 `+0.00160`；
- 保持统一 4 目标主结果表继续为 `5-seed` 全正增益状态。

新增阶段报告：

- `reports/transfer/transfer_stage_update_20260312_fbyg_v21_fresh_il_full5.md`
