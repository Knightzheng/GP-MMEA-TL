# 毕业设计中期报告（可提交版）

> 更新日期：2026-03-10  
> 项目：GP-MMEA-TL（多模态实体对齐迁移学习）  
> 说明：本文档对齐截至 2026-03-10 的仓库实际运行结果与实验记录。

## 1. 课题背景与研究目标

实体对齐是知识图谱融合中的关键任务。传统方法通常依赖目标数据集中的标注对齐种子，但真实场景中高质量标注成本高，导致模型跨数据集迁移时性能下降。  
本课题聚焦“多模态实体对齐的可迁移能力”，核心目标是：

1. 在统一实验口径下复现并评估主流模型的跨域表现。  
2. 设计并实现可迁移增强方法 TMMEA-DA。  
3. 在跨语言与跨图谱设置下，系统分析迁移增益与失效原因。  

## 2. 研究问题与假设

- RQ1：跨语言与跨图谱场景中，哪些域差异导致迁移性能退化。  
- RQ2：域对齐与多源选择是否能稳定降低负迁移。  
- RQ3：在模态缺失与噪声条件下，方法是否仍具鲁棒性。  

- H1：引入域对齐损失可提升目标域 Hits@1 与 MRR。  
- H2：多源选择可降低多次运行方差。  
- H3：缺失感知机制可降低高缺失场景下性能退化。  

## 3. 实验设置与复现规范

### 3.1 数据集

- DBP15K：`zh_en`、`ja_en`、`fr_en`  
- 跨图谱：`FBDB15K`、`FBYG15K`  

### 3.2 指标口径

- 主指标：`Hits@1`、`Hits@10`、`MRR`  
- 辅助指标：`MR`  
- 统计方式：关键结论均采用 `5-seed`（`42, 3407, 2026, 7, 123`）均值统计。  

### 3.3 环境与记录

- 硬件：RTX 3060 Laptop GPU  
- 训练环境：`bysj-meaformer`；调度与汇总环境：`bysj-main`  
- 记录规范：每次实验保留 `run_card/config/log/metrics/artifact_manifest`，保证可追溯复现。  

## 4. 本阶段已完成工作

1. 完成 MEAformer 基线在 DBP15K 与跨图谱数据上的多 seed 复现。  
2. 完成 TMMEA-DA 在同口径条件下的实现、训练与对比评估。  
3. 完成 `zh_en` 的模块消融（`wo_domain_align`、`wo_source_select`、`wo_missing_gate`，5-seed）。  
4. 完成 source->target 的迁移链路与目标域自适应实验。  
5. 完成 `ja_en / FBDB15K / fr_en / FBYG15K` 四目标统一主表（均为 5-seed）。  
6. 完成阶段报告、主表、分桶统计、自动化脚本与运行日志沉淀。  

## 5. 阶段性结果

### 5.1 受控设置（epoch=3）结果

在 `epoch=3` 的统一预算下，TMMEA-DA 与 baseline 在 DBP15K 与跨图谱上的差异总体很小：

- DBP15K 三语种中，MRR 变化约在 `-0.0002 ~ +0.0002`。  
- FBDB15K/FBYG15K 中，MRR 最大提升约 `+0.0004`。  
- `zh_en` 消融中，三个模块开关对结果影响均较弱。  

结论：在较小训练预算下，方法有效性尚未形成显著统计优势，但复现实验链路稳定可靠。

### 5.2 迁移主结果（source=zh_en，4 目标 5-seed）

| 目标域 | baseline runs | tmmeada runs | delta MRR |
|---|---:|---:|---:|
| ja_en | 5 | 5 | -0.0163 |
| FBDB15K | 5 | 5 | +0.0008 |
| fr_en | 5 | 5 | +0.0121 |
| FBYG15K | 5 | 5 | +0.0011 |

关键观察：

1. 四个目标域中，`3/4` 为正增益，`ja_en` 出现明显负迁移。  
2. 跨图谱目标（FBDB15K、FBYG15K）整体呈小幅稳定正向变化。  
3. `fr_en` 目标在 v14b 配置下有较明显提升，是当前阶段最强正例。  

## 6. 结果分析与中期结论

1. 本课题已完成“可复现工程闭环 + 多场景迁移评估”两项中期核心目标。  
2. TMMEA-DA 在部分目标域有效，但跨域稳定性不足，存在目标域敏感问题。  
3. 当前阶段结论应表述为：  
   - 方法具备提升潜力；  
   - 但尚未在全部目标域稳定优于 baseline；  
   - 需要针对负迁移场景做定向改进与误差约束。  

## 7. 与任务书对齐情况

| 任务书要求 | 当前状态 | 说明 |
|---|---|---|
| 动机实验：跨语言与跨图谱迁移评测 | 已完成 | 已形成 4 目标统一 5-seed 主表 |
| 设计可迁移多模态 EA 模型并验证 | 已完成阶段版 | TMMEA-DA 已实现并完成多轮实验 |
| 消融实验验证模块必要性 | 已完成 | zh_en 5-seed 消融已产出 |
| 过程记录与可复现性 | 已完成 | 运行日志、配置与汇总文件齐全 |
| 文献综述与论文化梳理 | 进行中 | 已有框架，需补强系统综述表达 |

## 8. 下一阶段计划（中期后）

1. 围绕 `ja_en` 负迁移做定向改造，优先控制伪标签噪声与域偏移误差。  
2. 完成误差分桶与典型案例分析，增强“为何有效/为何失效”的解释链路。  
3. 固化终稿主表、附录与复现实验脚本，准备论文终稿与答辩材料。  

## 9. 可直接引用的证据文件

- 需求与指标：`00_requirements.md`、`metrics_spec.md`  
- 过程日志：`PROCESS_LOG.md`、`PROJECT_OPERATION_RECORD.md`  
- 受控实验主表：  
  - `reports/epoch3/epoch3_compare_dbp15k.csv`  
  - `reports/epoch3/epoch3_compare_crossgraph.csv`  
  - `reports/epoch3/epoch3_ablation_zh_en_multiseed.csv`  
- 迁移主表与分桶：  
  - `reports/transfer/transfer_adapt_main_results_4target.csv`  
  - `reports/transfer/transfer_adapt_error_bucket_summary.csv`  
  - `reports/transfer/transfer_stage_update_20260309_ja_fbdb_expand5_final.md`  

