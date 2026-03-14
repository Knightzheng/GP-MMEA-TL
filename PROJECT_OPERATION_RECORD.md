# 项目全流程操作记录（完整版）

- 项目：`GP-MMEA-TL`（多模态实体对齐迁移实验）
- 记录范围：`2026-02-28` 至 `2026-03-14`
- 当前分支：`main`
- 当前提交（写入前快照）：`a5eb168`
- 记录时间：`2026-03-14`

## 导读（给第一次看这个项目的人）

这份文件不是简单的“实验日志堆叠”，而是整个毕业设计的过程总记录。它回答的是下面几个问题：

1. 这个项目到底在研究什么。
2. 我为什么要把实验拆成这么多阶段和版本。
3. 每个阶段具体尝试了什么，哪些方向有效，哪些方向无效。
4. 截至目前，项目已经形成了什么正式结论。

如果用更容易理解的话来说，这个项目在做的是：给两个知识图谱里的实体“找同一个现实世界对象”。例如两个图谱里都在写同一部电影、同一个人物、同一个地点，但名称、语言、属性和图片不完全一样，模型需要自动把它们对应起来。

我做的工作不是只跑一个模型分数，而是把整个研究链路逐步搭出来：

- 先把官方 baseline 复现出来，确保对照组可靠。
- 再把自己的方法模块加进去，判断有没有真实提升。
- 然后把任务推进到 `source -> target` 迁移场景，研究模型能否从一个数据集迁移到另一个数据集。
- 最后对不同目标任务持续做版本优化，并把结果整理成正式主表、阶段报告和可复现证据链。

## 常见术语（外行版）

| 术语 | 通俗解释 |
|---|---|
| `baseline` | 官方原始模型，作为公平对照组 |
| `TMMEA-DA` | 我在 baseline 基础上扩展的方法原型 |
| `seed` | 随机种子。换不同 seed 重复运行，是为了避免“只碰巧跑好一次” |
| `pilot` | 小规模试跑，通常先跑 1 个或 2 个 seed，用来快速筛方向 |
| `expand5` / `full5` | 把有希望的方案扩展到 5 个 seed，作为正式结果 |
| `source -> target` | 先在一个数据集上学到可迁移知识，再迁移到另一个目标数据集 |
| `IL` | 迭代式伪链接生成。模型先自己猜一批链接，再拿这些猜测继续训练 |
| `伪种子` / `伪标签` | 模型自动生成、但不一定完全正确的训练信号 |
| `strict-source` | 每个 seed 只允许使用严格对应的 source checkpoint，不混用旧模型 |
| `delta_avg_mrr_mean` | 方法相对 baseline 的平均提升值。大于 0 就表示方法更好 |

## 阶段总览（外行版）

| 阶段 | 时间 | 主要尝试 | 想解决什么问题 | 阶段结论 |
|---|---|---|---|---|
| A. 初始化与规范 | 2026-02-28 | 建环境、定指标、统一实验留痕 | 没有规范就无法做后续可复现实验 | 建立了完整实验骨架 |
| B. Baseline 复现 | 2026-02-28 | 跑通 `MEAformer`，覆盖 `DBP15K` 和跨图谱数据 | 没有可靠对照组，后续改进无法判断 | baseline 在 5 个目标数据集上复现完成 |
| C. TMMEA-DA MVP / v1 | 2026-02-28 至 2026-03-02 | 加入 `domain align`、`source_select`、`missing_gate`，并做多 seed、消融、epoch3 比较 | 自己的方法模块是否真的有用 | 早期模块在公平预算下大多与 baseline 接近 |
| D. 迁移实验链路 | 2026-03-03 至 2026-03-04 | 建立 `source_train -> target_eval` 流程 | 模型是否具备跨数据集迁移能力 | 迁移实验从 smoke 走到了 formal |
| E. 跨语言持续优化 | 2026-03-05 至 2026-03-11 | 围绕 `ja_en`、`fr_en` 多轮迭代自适应策略 | 跨语言迁移能否形成稳定正增益 | `ja_en` 和 `fr_en` 最终都得到了明显正增益 |
| F. `FBDB15K` 攻坚 | 2026-03-11 至 2026-03-12 | 从调权重转向改伪种子质量 | 跨图谱噪声是否才是主要瓶颈 | `v18` 证明更干净的伪种子是关键突破点 |
| G. `FBYG15K` 攻坚 | 2026-03-12 至 2026-03-14 | 从晚启 IL、静态过滤一路试到 staged fresh-IL、strict-source 与 adaptive top-k | 怎样在高噪声跨图谱场景里稳定提升 | `v24` 证明主线有效，`v25` 说明单纯 adaptive top-k 还不够 |
| H. 主表与收口 | 持续进行 | 统一 4 目标主表、误差分桶、阶段报告 | 如何把结果整理成能直接写进论文的正式结论 | 当前 4 个目标均为 `5-seed` 正增益 |

截至 `2026-03-14`，当前 4 目标统一主表已经全部是 `5-seed` 正式口径，分别为：

- `ja_en`: `delta_avg_mrr_mean = +0.01210`
- `FBDB15K`: `delta_avg_mrr_mean = +0.00830`
- `fr_en`: `delta_avg_mrr_mean = +0.01210`
- `FBYG15K`: `delta_avg_mrr_mean = +0.00280`

下面从第 `1` 节开始进入更偏技术细节的完整留痕。

## 1. 记录目的

本文件用于完整留痕以下内容，供中期/最终报告直接引用：

1. 从项目启动至当前阶段的关键操作与处理过程。
2. 核心代码改造点与原因。
3. 新建文件/目录的分类清单与作用。
4. 阶段性结果与当前状态快照。

## 2. 仓库状态快照（写入本记录时）

- 提交数：`57`（`git rev-list --count HEAD`）
- 文件总数：`12728`
- 关键目录文件数：
  - `scripts/`：`77` 个 `.py`
  - `configs/`：`126` 个配置文件
  - `reports/`：`690` 个报告文件
  - `runs/`：`3213` 个文件；命名为时间戳实验目录的 run 数量 `833`
- 迁移阶段报告文件（`reports/transfer/`）：`587` 个

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

- 当前最重要的统一主表：
  - `reports/transfer/transfer_adapt_main_results_4target.md`
  - `reports/transfer/transfer_adapt_main_results_4target.csv`
- 当前 4 个目标任务均为 `5-seed` 正式结果，且相对 baseline 为正增益：
  - `ja_en`：`delta_avg_mrr_mean = +0.01210`
  - `FBDB15K`：`delta_avg_mrr_mean = +0.00830`
  - `fr_en`：`delta_avg_mrr_mean = +0.01210`
  - `FBYG15K`：`delta_avg_mrr_mean = +0.00280`
- 当前 4 目标平均提升：
  - `delta_avg_hits@1_mean = +0.006897`
  - `delta_avg_hits@10_mean = +0.012650`
  - `delta_avg_mrr_mean = +0.008825`
  - `delta_avg_mr_mean = -66.674325`
- 最新阶段收口报告：
  - `reports/transfer/transfer_stage_update_20260314_fbyg_v25_adaptive_topk_pilot.md`
- 最新优化判断：
  - `FBYG15K v25` 已验证 `adaptive top-k` 机制确实工作，但最优 pilot 仍未超过 `v24b`，因此主表保持不变。

## 7. 异常与恢复记录（关键）

1. 出现过多次断电/自动关机场景。
2. 已实现断点续跑脚本（v14 与 FBYG expand5），并完成缺失 seeds 自动补跑。
3. 自动 finalize 流程曾修复“提交范围控制”问题，最终版本仅提交已暂存文件。
4. 小时级进度上报脚本已在阶段收尾后停止，避免后台进程残留。

## 8. 当前可直接用于报告的证据路径

1. 首页与项目总览：`README.md`
2. 过程总记录：`PROJECT_OPERATION_RECORD.md`
3. 方法改造说明：`baselines/MEAformer/config.py`, `baselines/MEAformer/model/MEAformer.py`, `baselines/MEAformer/main.py`
4. 统一运行入口：`scripts/run_meaformer.py`
5. 当前统一主结果表：`reports/transfer/transfer_adapt_main_results_4target.md`
6. 迁移阶段主证据目录：`reports/transfer/`
7. 运行原始证据：`runs/transfer/`, `runs/experiments/`
8. 中期素材：`reports/midterm/`

## 9. 提交历史摘要（按时间顺序）

以下为从项目启动到 `2026-03-09` 的首批主线提交（前 `41` 次）。`2026-03-11` 之后的新提交、对应实验与收口动作，已经在本文件后续“追加记录”中继续补充说明。

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

## 19. 2026-03-13 追加记录（FBYG15K v22 quality-filter pilot，主表保持 v21）

本次追加操作：

1. 基于 `v21` 结果继续优化 `FBYG15K`，把方向从“更快注入 fresh-IL”推进到“提高 fresh-IL 候选质量”。
2. 在 `baselines/MEAformer` 中新增 `IL` 质量优先过滤参数：
   - `il_margin_min`
   - `il_quality_quantile`
   - `il_topk_max`
   - `il_margin_weight`
3. 在 `baselines/MEAformer/model/MEAformer.py` 中重写 `Iter_new_links` 的过滤逻辑：
   - 为互选候选计算 `confidence / confidence margin / quality`
   - 支持按 `margin + quality + topk cap` 过滤
4. 新增 `FBYG15K v22` 三个 pilot 配置与自动脚本：
   - `tmmeada_target_fbyg15k_v22a_fresh_il_quality_top200`
   - `tmmeada_target_fbyg15k_v22b_fresh_il_quality_top100`
   - `tmmeada_target_fbyg15k_v22c_fresh_il_quality_top300`
   - `scripts/run_transfer_adapt_v22_fbyg_iter_queue.py`
5. 完成 `2-seed pilot -> 自动选优 -> 是否扩展 full5` 全流程。
6. 生成 `v22` 决策、compare 与 run-card 文件。
7. 更新 `README.md`、`PROCESS_LOG.md` 与最新阶段报告链接，但保持主结果表不切换。

关键结果：

- 当前参考主表版本：`FBYG15K v21a_fresh_il_q80_skiprel_skipfusion_expand5`
  - `5-seed delta_avg_mrr_mean = +0.00160`
- `v22` pilot（`2-seed`, vs baseline）：
  - `v22a = +0.00050`
  - `v22b = +0.00125`
  - `v22c = +0.00125`
- 自动决策：
  - `best_variant_pilot = v22b`
  - `improve_over_current_ref = -0.00035`
  - 未达到扩展阈值，不扩展到 `5-seed`

关键诊断：

- `v22a`：`kept=200`，真值率 `3.5% / 1.5%`
- `v22b`：`kept=100`，真值率 `6.0% / 1.0%`
- `v22c`：`kept=300`，真值率 `2.7% / 1.0%`

这表明：

- 静态质量过滤确实能显著提高个别 seed 的伪链接精度；
- 但跨 seed 稳定性不够，精度提升没有稳定转化为更好的最终 `MRR`；
- `FBYG15K` 的后续优化不应再继续做静态 `filter/cap` 网格，而应转向分阶段或自适应注入。

本次追加的直接作用：

- 排除了 `FBYG15K` 上继续做静态 `IL quality threshold / topk cap` 搜索的必要性；
- 保留 `v21` 作为当前最优正式主表版本；
- 为后续如果继续优化 `FBYG15K`，提供了明确的“下一步不该怎么做”的负结果证据链。

新增阶段报告：

- `reports/transfer/transfer_stage_update_20260313_fbyg_v22_quality_pilot.md`

## 20. 2026-03-13 追加记录（FBYG15K v23 staged fresh-IL full5，主表切换）

本次追加操作：

1. 基于 `v22` 的负结果，正式把 `FBYG15K` 的后续优化方向切到 staged fresh-IL，而不是继续做静态质量阈值搜索。
2. 在 `MEAformer` 中新增多轮 fresh proposal 与分阶段过滤能力：
   - `il_fresh_epochs`
   - `il_confidence_min_schedule`
   - `il_confidence_quantile_schedule`
   - `il_confidence_keep_min_schedule`
   - `il_margin_min_schedule`
   - `il_quality_quantile_schedule`
   - `il_topk_max_schedule`
3. 在 `baselines/MEAformer/model/MEAformer.py` 中加入 phase-aware IL 过滤逻辑，并在日志中输出 `phase/fresh` 标记。
4. 新增 `FBYG15K v23` 三个 pilot 配置与自动脚本：
   - `tmmeada_target_fbyg15k_v23a_staged_fresh_il_top250`
   - `tmmeada_target_fbyg15k_v23b_staged_fresh_il_top400`
   - `tmmeada_target_fbyg15k_v23c_staged_fresh_il_epoch8_top250`
   - `scripts/run_transfer_adapt_v23_fbyg_iter_queue.py`
5. 完成 `2-seed pilot -> 自动选优 -> 5-seed expand` 全流程。
6. 刷新 `scripts/make_transfer_main_and_bucket_report.py` 的 `FBYG15K` 主表入口，将主结果切换到 `v23b full5`。
7. 更新 `README.md`、`PROCESS_LOG.md`、主结果表与最新阶段报告链接。

关键结果：

- 当前旧参考主表版本：`FBYG15K v21a_fresh_il_q80_skiprel_skipfusion_expand5`
  - `5-seed delta_avg_mrr_mean = +0.00160`
- `v23` pilot（`2-seed`, vs baseline）：
  - `v23a = +0.00225`
  - `v23b = +0.00300`
  - `v23c = +0.00200`
- 自动决策：
  - `best_variant_pilot = v23b`
  - `improve_over_current_ref = +0.00140`
  - 达到扩展阈值并自动扩展到 `5-seed`
- `v23b` 正式 `5-seed`（vs baseline）：
  - `delta_avg_hits@1_mean = +0.00186`
  - `delta_avg_hits@10_mean = +0.00460`
  - `delta_avg_mrr_mean = +0.00270`
  - `delta_avg_mr_mean = -43.13610`

关键诊断：

- `v23b` 采用两阶段 fresh-IL：
  - `phase 0 (epoch 5)`: 先注入 `100` 条高精度候选
  - `phase 1 (epoch 7)`: 再补充 `400` 条更大规模候选
- 5 个 seed 的日志都出现了稳定的两阶段注入模式；
- 与 `v21` 的单次大注入相比，`v23b` 更好地平衡了“先稳住训练”与“后续再补规模”这两个目标；
- 这说明 `FBYG15K` 的最优下一步不是继续调单次过滤阈值，而是把候选质量和注入时机拆开处理。

本次追加的直接作用：

- 将 `FBYG15K` 主表版本从 `v21a` 切换为 `v23b`；
- 把 `FBYG15K` 的 `5-seed delta_avg_mrr_mean` 从 `+0.00160` 提升到 `+0.00270`；
- 使统一 4 目标主结果表中的 `FBYG15K` 一项进一步稳定上升。

新增阶段报告：

- `reports/transfer/transfer_stage_update_20260313_fbyg_v23_staged_fresh_il_full5.md`

## 21. 2026-03-14 追加记录（FBYG15K v24 strict-source staged fresh-IL full5，主表切换）

本次追加操作：

1. 先暂停继续追加 `FBYG` 新技巧，优先修复 `source checkpoint` 口径不一致问题。
2. 确认只有 `seed=42/3407` 存在 exact `zh_en baseline transfer formal` source checkpoint，而 `2026/7/123` 缺失。
3. 新增 `scripts/ensure_transfer_source_formal.py`，用于自动补齐 exact source formal checkpoint。
4. 补齐以下 baseline source formal：
   - `seed=2026`
   - `seed=7`
   - `seed=123`
5. 修改 `scripts/transfer_adapt_utils.py`：
   - `resolve_source_model_name` 默认只接受 exact formal-source
   - 不再静默回退到旧的 `transfer_adapt` checkpoint
6. 在此基础上新增 `FBYG15K v24` 三个 strict-source staged fresh-IL 变体与自动脚本：
   - `tmmeada_target_fbyg15k_v24a_strictsrc_staged_fresh_il_top250`
   - `tmmeada_target_fbyg15k_v24b_strictsrc_staged_fresh_il_top400`
   - `tmmeada_target_fbyg15k_v24c_strictsrc_staged_fresh_il_epoch8_top250`
   - `scripts/run_transfer_adapt_v24_fbyg_iter_queue.py`
7. 完成 `2-seed pilot -> 自动选优 -> 5-seed expand` 全流程。
8. 刷新 `scripts/make_transfer_main_and_bucket_report.py` 的 `FBYG15K` 主表入口，将主结果切换到 `v24b full5`。
9. 更新 `README.md`、`PROCESS_LOG.md`、主结果表与最新阶段报告链接。

关键结果：

- 当前旧参考主表版本：`FBYG15K v23b_staged_fresh_il_top400_expand5`
  - `5-seed delta_avg_mrr_mean = +0.00270`
- `v24` pilot（`2-seed`, vs baseline）：
  - `v24a = +0.00200`
  - `v24b = +0.00300`
  - `v24c = +0.00200`
- 自动决策：
  - `best_variant_pilot = v24b`
  - `improve_over_current_ref = +0.00030`
  - 达到扩展阈值后自动扩展到 `5-seed`
- `v24b` 正式 `5-seed`（vs baseline）：
  - `delta_avg_hits@1_mean = +0.00197`
  - `delta_avg_hits@10_mean = +0.00462`
  - `delta_avg_mrr_mean = +0.00280`
  - `delta_avg_mr_mean = -42.81030`

关键诊断：

- 本轮最大的价值首先是“把 source 口径清洗干净”，其次才是数值再抬高一点；
- `v24b` 的全部 5 个 seed 都明确加载了 exact `baseline_transfer_formal` source model；
- staged fresh-IL 的正增益在 strict-source 条件下仍然成立，说明 `FBYG15K` 当前主结论具备更强的可复现性与可解释性。

本次追加的直接作用：

- 将 `FBYG15K` 主表版本从 `v23b` 切换为 `v24b`；
- 把 `FBYG15K` 的 `5-seed delta_avg_mrr_mean` 从 `+0.00270` 提升到 `+0.00280`；
- 同时把 `FBYG15K` 当前主结果提升为 strict formal-source 口径的正式结果。

新增阶段报告：

- `reports/transfer/transfer_stage_update_20260314_fbyg_v24_strict_source_full5.md`

## 22. 2026-03-14 追加记录（FBYG15K v25 adaptive top-k pilot，主表保持 v24）

本次追加操作：

1. 选取 `FBYG15K` 的最优下一步方向，基于当前主表 `v24b` 继续验证 `phase-2 adaptive top-k`，而不是回到静态 `filter/cap` 搜索。
2. 在 `baselines/MEAformer/config.py` 中新增分阶段自适应 `top-k` 参数：
   - `il_adaptive_topk`
   - `il_adaptive_topk_scale`
   - `il_adaptive_topk_min`
   - `il_adaptive_topk_scale_schedule`
   - `il_adaptive_topk_min_schedule`
3. 在 `baselines/MEAformer/model/MEAformer.py` 中加入 phase-aware `pre_topk_count` 记录与 `effective_topk` 自适应逻辑。
4. 在 `baselines/MEAformer/main.py` 中补充 IL 日志输出，记录 `pre_topk/effective_topk/prev_pre_topk` 等信息。
5. 在 `scripts/run_meaformer.py` 中补齐上述参数透传。
6. 新增 `FBYG15K v25` 三个 pilot 配置与自动脚本：
   - `tmmeada_target_fbyg15k_v25a_strictsrc_staged_adaptivetopk_s100`
   - `tmmeada_target_fbyg15k_v25b_strictsrc_staged_adaptivetopk_s125`
   - `tmmeada_target_fbyg15k_v25c_strictsrc_staged_adaptivetopk_s100_min300`
   - `scripts/run_transfer_adapt_v25_fbyg_iter_queue.py`
7. 完成 `2-seed pilot -> 自动选优 -> 是否扩展 full5` 全流程。
8. 新增 `v25` 决策、compare、run-card 与阶段报告文件。
9. 更新 `README.md`、`PROCESS_LOG.md` 与最新阶段报告链接，但保持主结果表不切换。

关键结果：

- 当前参考主表版本：`FBYG15K v24b_strictsrc_staged_fresh_il_top400_expand5`
  - `5-seed delta_avg_mrr_mean = +0.00280`
- `v25` pilot（`2-seed`, vs baseline）：
  - `v25a = +0.00200`
  - `v25b = +0.00200`
  - `v25c = +0.00250`
- 自动决策：
  - `best_variant_pilot = v25c`
  - `improve_over_current_ref = -0.00030`
  - 未达到扩展阈值，不扩展到 `5-seed`

关键诊断：

- 这轮的正面结论不是“最终赢了”，而是 `adaptive top-k` 已在日志中明确生效；
- `phase 1` 的 `effective_topk` 会根据 `phase 0` 的 `pre_topk_count` 自动变化，例如：
  - `v25a/s2026`: `prev_pre_topk=200 -> effective_topk=250`
  - `v25b/s42`: `prev_pre_topk=233 -> effective_topk=291`
  - `v25c/s42`: `prev_pre_topk=233 -> effective_topk=300`
  - `v25c/s2026`: `prev_pre_topk=200 -> effective_topk=300`
- 但最终 `phase 1` 新增链接真值率仍偏低，说明当前瓶颈更可能是“第二阶段候选一致性不足”，而不只是固定 `top-k` 设置过硬。

本次追加的直接作用：

- 验证了 `FBYG15K` 上 `adaptive top-k` 这条机制路线确实可运行、可追溯，不是无效实现；
- 同时排除了“继续单独扫 adaptive top-k 数值就能稳定超过 `v24b`”的可能性；
- 为后续如果继续优化 `FBYG15K`，把下一步更明确地推进到 `phase-wise consistency constraints`。

新增阶段报告：

- `reports/transfer/transfer_stage_update_20260314_fbyg_v25_adaptive_topk_pilot.md`

## 23. 2026-03-14 追加记录（论文/答辩支撑材料开始收口）

本次追加操作：

1. 明确将当前线程定位为“优化线程”，不再以论文全文撰写为主，而是专门补强实验、证据链与答辩说服力。
2. 新增显著性分析脚本：
   - `scripts/analyze_transfer_significance.py`
3. 基于当前统一 4 目标 `5-seed` 正式主表，输出显著性分析材料：
   - `reports/transfer/transfer_adapt_significance_per_seed.csv`
   - `reports/transfer/transfer_adapt_significance_summary.csv`
   - `reports/transfer/transfer_adapt_significance_summary.md`
   - `reports/transfer/transfer_adapt_significance_writeup.md`
4. 新增案例分析脚本：
   - `scripts/build_transfer_case_analysis.py`
5. 利用当前已保留的 `pred.txt` 文件，直接抽取案例级成功/失败样本，而不额外重跑正式实验：
   - `reports/transfer/transfer_case_analysis_examples.csv`
   - `reports/transfer/transfer_case_analysis_examples.md`
6. 新增效率汇总脚本：
   - `scripts/summarize_transfer_efficiency.py`
7. 基于当前 formal 日志补出 wall-clock 汇总：
   - `reports/transfer/transfer_efficiency_per_run.csv`
   - `reports/transfer/transfer_efficiency_summary.csv`
   - `reports/transfer/transfer_efficiency_summary.md`
8. 更新 `README.md` 与 `PROCESS_LOG.md` 中的支撑材料入口。

关键结论：

- 显著性部分：
  - 当前结果组织形式最适合做“配对 seed 统计”，而不是把 5 个 seed 当成互相独立的大样本；
  - 推荐口径：
    - 主体：paired bootstrap `95% CI`
    - 小样本显著性：exact one-sided sign test
    - 辅助：exact one-sided Wilcoxon signed-rank test
  - 当前 4 个目标域在 `avg MRR` 上均满足：
    - `5/5 seed` 正增益
    - bootstrap `95% CI` 下界 `> 0`
    - sign test / Wilcoxon `p = 0.03125`
- 案例部分：
  - 已先补出 `6` 个代表性案例：
    - `ja_en`：2 个失败/边界案例
    - `FBDB15K`：2 个成功大幅纠错案例
    - `FBYG15K`：2 个成功大幅纠错案例
  - 这批案例的作用是：
    - 证明跨图谱增益不仅存在于均值上，也体现在严重误排样本的 rank recovery 上；
    - 同时保留 `ja_en` 的失败样本，避免把方法结论表述得过满。
- 效率部分：
  - 现有材料已足以汇总 wall-clock 时间；
  - GPU 峰值显存尚未在全部正式 run 中统一记录，因此若论文需要完整“时间+显存”表，还需补做一次最小代价测量。

当前判断：

- 这批新增材料比继续做一轮轻量调参更直接服务于论文终稿与答辩；
- 下一轮若继续优化，应优先考虑：
  - `H3` 的缺失率压力测试是否值得做最小版；
  - `FBYG15K` 是否进入 `phase-wise consistency constraints`；
  - 额外 baseline 是否真的值得投入。

## 24. 2026-03-15 追加记录（项目主线闭环检查与辅助支撑状态重整）

本次追加操作：

1. 按“项目优化线程”定位重新接管当前仓库，通读任务书、需求冻结、共享交接、论文初稿、过程记录、阶段报告、主结果文件、鲁棒性材料和关键脚本。
2. 新增项目级差距评估文件：
   - `reports/notes/taskbook_gap_assessment_20260315.md`
3. 明确给出当前判断：
   - 任务书 / 开题报告主线已经基本闭环；
   - 当前最高优先级不再是继续追加主线 rerun，而是把主线材料同步清楚、保守写清辅助项边界。
4. 将以下内容补入项目导航与目录说明：
   - `reports/robustness/`
   - `reports/robustness/h3_missing_modality_minimal_summary.md`
   - `reports/notes/taskbook_gap_assessment_20260315.md`
   - `runs/experiments/h3_missing_modality_minimal/`
   - `runs/experiments/gpu_peak_minimal/`（注明当前仍未形成正式可用结果）
5. 核对 `H3` 当前真实状态：
   - 已存在 `zh_en`
   - 已存在 `seed=42`
   - 已存在 `drop_rate={0.0, 0.6}`
   - 已存在 `v1_full / wo_missing_gate` 的最小单 seed summary
   - 但仍未包含 baseline / 多 seed / 多目标域，不能作为 H3 正式验证
6. 核对 GPU 峰值显存当前真实状态：
   - 代码能力与脚本入口已接入
   - 当前仓库中仍以 dry-run 和失败尝试为主
   - 一次实际尝试出现 `AssertionError: self.args.il_start < self.args.epoch`
   - 因此当前仍没有可直接入文的正式 GPU 峰值汇总表
7. 修正 GPU 最小补测脚本：
   - `scripts/run_gpu_peak_minimal.py`
   - 当 transfer config 使用 `il` 且 `epoch <= il_start` 时，自动把最小 rerun epoch 抬到 `il_start + 1`
   - 目的不是夸大“GPU 已完成”，而是避免后续继续生成无效最小补测配置

关键判断：

1. 主线闭环：
   - `baseline 复现`
   - `统一迁移链路`
   - `4 目标 5-seed 正式主表`
   - `核心消融`
   - `显著性 / 案例 / wall-clock`
   - `过程记录`
   以上均已具备论文与答辩所需的基本支撑。
2. 主线剩余缺口主要在材料组织，不在实验结果本身。
3. `H3 / GPU / 扩展案例 / 额外 baseline` 统一下调为辅助支撑项，不得再被写成任务书主线完成标志。

本次追加的直接作用：

1. 让后来者能更清楚地区分“主线已完成”与“辅助项仍待补强”。
2. 避免论文线程继续把 GPU 峰值显存误读为“已经有正式结果”。
3. 将 `H3` 当前最小单 seed pilot 的真实边界写清楚，防止其被过度外推。

新增 / 更新文件：

- `reports/notes/taskbook_gap_assessment_20260315.md`
- `README.md`
- `reports/README.md`
- `runs/README.md`
- `PROCESS_LOG.md`
- `PROJECT_OPERATION_RECORD.md`
- `reports/notes/thread_sync_shared.md`
- `scripts/run_gpu_peak_minimal.py`

## 25. 2026-03-15 追加记录（深度仓库整理：H3 暂停并移出当前项目树）

本次追加操作：

1. 删除 `H3` 相关结果文件、汇总表、运行目录与脚本入口，避免其继续干扰主线项目管理。
2. 从 `MEAformer` 当前训练链路中移除人工图像缺失注入参数，恢复主线代码入口的简洁状态。
3. 更新项目导航、过程日志、共享同步板与差距评估口径，统一说明：
   - `H3` 已从当前仓库移除；
   - 后续仅在主线完整结束后再重新尝试；
   - 论文线程不应再从当前仓库引用旧的 `H3` 留痕。

本次追加的直接作用：

1. 让当前仓库只保留主线必须的结果、代码与记录。
2. 避免后续误把 `H3` 试验能力当成仍在维护的主线入口。
3. 为接下来继续做主线材料整理、GPU 最小补测或正式代码优化留出更清晰的仓库结构。

## 26. 2026-03-15 追加记录（阶段1：主线复现与追溯总表）

本次追加操作：

1. 新增项目级总表文件：
   - `reports/notes/mainline_traceability_matrix_20260315.md`
2. 将任务书 / 开题报告中的主线要求，与当前仓库中的以下内容建立对应关系：
   - 正式结果文件；
   - 关键脚本入口；
   - 正式 run 目录；
   - 当前边界说明。
3. 将该总表同步回项目导航与共享文件，确保后续线程不再需要在多个目录之间反复跳转确认主线证据。

本次追加的直接作用：

1. 让“项目是否真正完成任务书主线”这件事可以被快速核对。
2. 让后续 README / reports / runs 的整理有了统一锚点。
3. 让论文线程可以直接吸收“要求 -> 证据 -> 脚本 -> run”的闭环映射，而不是只拿到零散结果文件。

## 27. 2026-03-15 追加记录（阶段2：主线导航再收口）

本次追加操作：

1. 新增：
   - `reports/transfer/README.md`
   - `runs/transfer/README.md`
2. 将 transfer 相关材料拆成两层导航：
   - 结果文件入口导航；
   - 正式 run 目录导航。
3. 明确区分：
   - 哪些是当前主线正式结果；
   - 哪些只是历史探索、阶段迭代或队列留痕。

本次追加的直接作用：

1. 降低后来者进入 `transfer/` 目录后的理解成本。
2. 避免把探索性 `v*` 文件误当作当前正式主表证据。
3. 为阶段3 的 GPU 最小正式补测留出更干净的项目导航结构。
