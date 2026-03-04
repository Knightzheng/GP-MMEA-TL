# 毕业设计中期报告（可提交版）

> 版本：`2026-03-02`  
> 项目：`GP-MMEA-TL`（多模态实体对齐跨域迁移）  
> 说明：本文档已对齐当前仓库最新正式实验结果（`epoch=3`、`5-seed`）。

## 1. 课题背景与研究目标

本课题聚焦多模态实体对齐（Multimodal Entity Alignment, MMEA）的跨域可迁移性问题。  
核心目标是在跨语言（DBP15K）与跨图谱（FB15K-DB15K、FB15K-YAGO15K）设置下，建立统一可复现实验流程，并验证所提出的 TMMEA-DA 模块是否能稳定提升迁移性能。

对应研究问题（RQ）与假设（H）如下：

- RQ1：跨语言与跨图谱场景中，哪些域差异是性能退化主因？
- RQ2：域对齐与多源选择模块能否稳定降低负迁移？
- RQ3：模态缺失/噪声下，鲁棒融合是否仍保持优势？
- H1：域对齐损失应提升 Hits@1 / MRR。
- H2：多源选择应提升稳定性（更低方差）。
- H3：缺失感知策略应降低高缺失场景的性能退化。

## 2. 数据集、指标与实验口径

### 2.1 数据集

- 跨语言：DBP15K（`zh_en`, `ja_en`, `fr_en`，`train_ratio=0.3`）
- 跨图谱：`FBDB15K`、`FBYG15K`

### 2.2 指标

- 主指标：`Hits@1`、`Hits@10`、`MRR`
- 统计口径：关键实验均采用 `5 seeds = {42, 3407, 2026, 7, 123}`，报告 `mean ± std`
- 评测方向：`l2r` 与 `r2l` 双向结果同时报告

### 2.3 可复现设置

- 硬件：RTX 3060 Laptop GPU（6GB）
- 环境：`conda bysj-main`（调度）+ `conda bysj-meaformer`（训练）
- 统一训练预算：当前正式结果采用 `epoch=3` 安全配置

## 3. 阶段工作完成情况

截至目前，已完成以下可复现实验闭环：

1. 基线 MEAformer 在 DBP15K 三语种 `epoch=3` 正式 `5-seed` 复现。
2. TMMEA-DA `v1_best` 在 DBP15K 三语种 `epoch=3` 正式 `5-seed` 对比。
3. 基线与 TMMEA-DA `v1_best` 在跨图谱（FBDB15K、FBYG15K）`epoch=3` 正式 `5-seed` 对比。
4. `zh_en` 上三组模块消融（`wo_domain_align`、`wo_source_select`、`wo_missing_gate`）`epoch=3` 正式 `5-seed` 完成。
5. 全流程记录与产物留痕：配置文件、运行日志、汇总 CSV、对比 Markdown、过程日志均已归档。

## 4. 核心实验结果

结果来源文件：

- `reports/epoch3_compare_dbp15k.csv`
- `reports/epoch3_compare_crossgraph.csv`
- `reports/epoch3_ablation_zh_en_multiseed.csv`

### 4.1 DBP15K（epoch=3，5-seed）Baseline vs TMMEA-DA v1_best

#### l2r 方向

| 语言对 | Baseline Hits@1 | Method Hits@1 | ΔHits@1 | Baseline MRR | Method MRR | ΔMRR |
|---|---:|---:|---:|---:|---:|---:|
| zh_en | 0.6233 | 0.6233 | +0.0000 | 0.7146 | 0.7146 | +0.0000 |
| ja_en | 0.6014 | 0.6014 | +0.0000 | 0.6956 | 0.6958 | +0.0002 |
| fr_en | 0.6026 | 0.6027 | +0.0001 | 0.6994 | 0.6996 | +0.0002 |

#### r2l 方向

| 语言对 | Baseline Hits@1 | Method Hits@1 | ΔHits@1 | Baseline MRR | Method MRR | ΔMRR |
|---|---:|---:|---:|---:|---:|---:|
| zh_en | 0.6233 | 0.6234 | +0.0001 | 0.7148 | 0.7150 | +0.0002 |
| ja_en | 0.5997 | 0.5998 | +0.0001 | 0.6944 | 0.6944 | +0.0000 |
| fr_en | 0.6000 | 0.5999 | -0.0001 | 0.6974 | 0.6972 | -0.0002 |

### 4.2 跨图谱（epoch=3，5-seed）Baseline vs TMMEA-DA v1_best

#### l2r 方向

| 数据集 | Baseline Hits@1 | Method Hits@1 | ΔHits@1 | Baseline MRR | Method MRR | ΔMRR |
|---|---:|---:|---:|---:|---:|---:|
| FBDB15K | 0.1902 | 0.1903 | +0.0001 | 0.2882 | 0.2882 | +0.0000 |
| FBYG15K | 0.1612 | 0.1614 | +0.0002 | 0.2454 | 0.2458 | +0.0004 |

#### r2l 方向

| 数据集 | Baseline Hits@1 | Method Hits@1 | ΔHits@1 | Baseline MRR | Method MRR | ΔMRR |
|---|---:|---:|---:|---:|---:|---:|
| FBDB15K | 0.1953 | 0.1954 | +0.0001 | 0.2930 | 0.2934 | +0.0004 |
| FBYG15K | 0.1617 | 0.1617 | +0.0000 | 0.2460 | 0.2460 | +0.0000 |

### 4.3 zh_en 消融（epoch=3，5-seed）

| 变体 | l2r Hits@1 | l2r MRR | r2l Hits@1 | r2l MRR | 相对 full 结论 |
|---|---:|---:|---:|---:|---|
| v1_best_full | 0.6233 | 0.7146 | 0.6234 | 0.7150 | 参考 |
| wo_domain_align | 0.6233 | 0.7146 | 0.6234 | 0.7150 | 几乎一致 |
| wo_source_select | 0.6233 | 0.7146 | 0.6233 | 0.7148 | 极小幅回落 |
| wo_missing_gate | 0.6233 | 0.7146 | 0.6234 | 0.7150 | 几乎一致 |

## 5. 阶段性结论

1. 当前项目已达到“可复现、可统计、可追溯”的中期交付要求：数据、配置、运行、评测、汇总与日志链路完整。  
2. 从正式 `5-seed` 结果看，TMMEA-DA v1_best 相对 baseline 的增益量级在 `10^-4`，显著小于当前实验波动（std 量级 `10^-3` 到 `10^-2`），尚不能支撑“显著优于基线”的结论。  
3. 消融结果显示三类模块在当前预算下贡献不明显，说明下一阶段重点应从“继续堆叠模块”转向“训练策略与预算重设计 + 有针对性的参数搜索”。  
4. 本阶段可形成论文中的“负结果与稳定性分析”证据链，为后续方法改进提供可靠基线。

## 6. 中期后下一阶段计划

### 6.1 目标

在有限算力下优先验证“是否存在可复现的有效增益区间”，避免盲目全量重跑。

### 6.2 执行路径

1. 进行小规模 pilot：`zh_en` + `FBDB15K`，`epoch=8/10`，先用 `seed=42,3407`。  
2. 仅调两项关键超参：`domain_align_weight`、`source_select_weight`。  
3. 若任一数据集出现稳定提升（建议门槛：`ΔMRR >= +0.003`），再扩展到正式 `5-seed` 与全数据集。  
4. 若未达门槛，则将论文主线转为“系统复现 + 负结果分析 + 机制解释”，并补充误差分桶图与案例分析。

### 6.3 时间预算

- pilot 阶段：约 `0.5~1` 天  
- 成功分支（扩展正式实验）：约 `1.5~2.5` 天  
- 收口写作与图表：约 `0.5~1` 天

## 7. 提交材料清单（本仓库可直接引用）

- 实验需求冻结：`00_requirements.md`
- 指标口径：`metrics_spec.md`
- 全流程日志：`PROCESS_LOG.md`
- 中期报告正文（本文件）：`reports/midterm_report_submission.md`
- 关键结果表：
  - `reports/epoch3_compare_dbp15k.csv`
  - `reports/epoch3_compare_crossgraph.csv`
  - `reports/epoch3_ablation_zh_en_multiseed.csv`


## 18. 核心代码改造记录（2026-03-03，供报告引用）

为提升 `TMMEA-DA` 在正式口径下的增益可信度，本阶段完成了“最小侵入式”核心代码改造：不改变主干编码器结构，仅增强辅助损失机制与训练调度。

- 改造入口：`baselines/MEAformer/config.py`
  - 新增参数：`aux_start_epoch`、`aux_ramp_epochs`、`domain_align_margin`、`domain_align_neg_weight`。
- 损失改造：`baselines/MEAformer/model/MEAformer.py`
  - 域对齐从“正样本 MSE”扩展为“正样本 + hard-negative hinge”；
  - 引入 `aux_scale`（按 epoch 分阶段启用辅助损失）；
  - 增加诊断日志字段：`aux_scale`、`domain_align_pos`、`domain_align_hard`、`domain_align_term`、`missing_align_term`。
- 训练循环改造：`baselines/MEAformer/main.py`
  - 训练时传入 `current_epoch`，确保分阶段策略生效。
- 运行脚本改造：`scripts/run_meaformer.py`
  - 完成新增参数从 yaml 到训练进程的透传。
- 调优配置：`configs/tmmeada/meaformer_zh_en_tmmeada_v2_tuned_epoch10_pilot.yaml`

完整过程记录与可复现实验命令见：`reports/core_code_refactor_20260303.md`。
