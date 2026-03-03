# 中期报告实验章节草稿（最新版）

## 4. 实验设计与实现细节

### 4.1 实验目标
本阶段目标是完成“可复现、可统计、可追溯”的多模态实体对齐实验闭环，并在统一预算下验证 TMMEA-DA 对比 baseline 的真实增益。

### 4.2 基线与方法
- 基线：`MEAformer`
- 方法：`TMMEA-DA v1_best`（domain align + source select + missing gate）
- 对比原则：同数据、同预算、同随机种子、同评测口径

### 4.3 数据与设置
- DBP15K：`zh_en`, `ja_en`, `fr_en`（`train_ratio=0.3`）
- 跨图谱：`FBDB15K`, `FBYG15K`
- 训练预算：`epoch=3`
- 随机种子：`42, 3407, 2026, 7, 123`

### 4.4 评测口径
- 指标：`Hits@1`, `Hits@10`, `MRR`（`l2r/r2l`）
- 汇总方式：`mean ± std`
- 结果文件：
  - `reports/epoch3_compare_dbp15k.csv`
  - `reports/epoch3_compare_crossgraph.csv`
  - `reports/epoch3_ablation_zh_en_multiseed.csv`

## 5. 实验结果与分析

### 5.1 DBP15K 正式 5-seed 对比结果

- 三语种上，方法与基线基本持平，增益量级约 `10^-4`。
- `zh_en`：`l2r/r2l` 的 Hits@1 与 MRR 均几乎一致。
- `ja_en`：有极小正向变化，但未达到可视为显著提升的量级。
- `fr_en`：个别方向出现 `-0.0001~-0.0002` 级别波动，整体仍可视为持平。

### 5.2 跨图谱正式 5-seed 对比结果

- `FBDB15K`、`FBYG15K` 上方法相对基线均为极小正向或持平变化（`0~0.0004` 量级）。
- 该结果说明跨图谱链路已稳定复现，但当前方法增益尚弱。

### 5.3 zh_en 消融（正式 5-seed）

- `wo_domain_align` 与 full 基本一致；
- `wo_missing_gate` 与 full 基本一致；
- `wo_source_select` 仅出现极小回落（约 `r2l Hits@1 -0.0001` 量级）。

结论：当前预算下各模块贡献差异不显著。

## 6. 阶段性结论

1. 已完成中期所需的工程与实验闭环，可支撑“可复现研究过程”提交。  
2. 方法在正式 5-seed 口径下尚未体现统计上有说服力的提升，需在后续阶段优化训练策略与超参。  
3. 中期文本中建议如实呈现：流程与证据链完整，方法增益暂不显著，并给出针对性的下一步实验计划。

## 7. 下一阶段计划

1. 先执行小规模 pilot（`zh_en` + `FBDB15K`，`epoch=8/10`，2-seed）验证是否存在有效增益区间。  
2. 若 pilot 达到提升门槛（如 `ΔMRR >= +0.003`），再扩展到全数据与 5-seed 正式跑。  
3. 若未达门槛，转为“负结果 + 误差分析 + 机制解释”的收口方案，保证论文可交付质量。

> 完整可提交正文见：`reports/midterm_report_submission.md`

## 附：核心代码改造过程（2026-03-03）

本阶段已将核心代码改造形成可复现记录，覆盖：
- 参数层新增（aux 启动与升温、domain hard-negative）
- 损失函数改造（domain align + missing/source 的分阶段调度）
- 训练循环 epoch 感知改造
- 运行脚本参数透传改造
- 配套 tuned pilot 配置

详见：`reports/core_code_refactor_20260303.md`。
