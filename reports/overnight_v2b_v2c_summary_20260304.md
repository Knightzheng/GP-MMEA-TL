# 夜间自动实验汇总（2026-03-04）

## 1. 运行完整性检查

本次夜间链路包含两段：
1) `v2b_lite_hardneg` 2-seed pilot（`zh_en` + `FBDB15K`）  
2) 基于 `v2b` 决策自动触发 `v2c_source_only` 2-seed pilot（`zh_en` + `FBDB15K`）

检查结论：
- 当前无训练进程在运行（Python 训练进程已全部退出）。
- `v2b` 与 `v2c` 对应四组有效 run（两数据集 x 两seed）均为 `DONE`。
- 报告文件已自动生成且时间戳完整。

说明：
- 各 stage 中有一小部分更早的历史目录 `DONE=False`（旧尝试/中断 run），自动汇总脚本已按“同 seed 取最新 run_id”策略过滤，不影响最终统计。

## 2. 结果文件清单

### 2.1 v2b
- `reports/epoch10_compare_v2b_lite_hardneg_pilot.csv`
- `reports/epoch10_compare_v2b_lite_hardneg_pilot.md`
- `reports/epoch10_v2b_lite_hardneg_decision.json`
- `reports/epoch10_v2b_lite_hardneg_decision.md`

### 2.2 v2c（由自动分支触发）
- `reports/epoch10_compare_v2c_source_only_pilot.csv`
- `reports/epoch10_compare_v2c_source_only_pilot.md`
- `reports/epoch10_v2c_source_only_decision.json`
- `reports/epoch10_v2c_source_only_decision.md`

## 3. 核心结果摘录

对比对象统一为：`baseline epoch10 pilot`（2-seed: `42,3407`）。

### 3.1 v2b_lite_hardneg
- `zh_en`：
  - `l2r_mrr: 0.7500 -> 0.7500 (Δ +0.0000)`
  - `r2l_mrr: 0.7505 -> 0.7495 (Δ -0.0010)`
  - `Δavg_mrr = -0.0005`
- `FBDB15K`：
  - `l2r_mrr: 0.3205 -> 0.3200 (Δ -0.0005)`
  - `r2l_mrr: 0.3205 -> 0.3195 (Δ -0.0010)`
  - `Δavg_mrr = -0.0008`
- 自动决策：`continue_tuning_or_error_analysis`

### 3.2 v2c_source_only
- `zh_en`：
  - `l2r_mrr: 0.7500 -> 0.7500 (Δ +0.0000)`
  - `r2l_mrr: 0.7505 -> 0.7505 (Δ +0.0000)`
  - `Δavg_mrr = +0.0000`
- `FBDB15K`：
  - `l2r_mrr: 0.3205 -> 0.3205 (Δ +0.0000)`
  - `r2l_mrr: 0.3205 -> 0.3205 (Δ +0.0000)`
  - `Δavg_mrr = +0.0000`
- 自动决策：`continue_tuning_or_error_analysis`

## 4. 分析结论

1. `v2b`（轻量 hard-negative）仍出现“均值不升反降”的趋势，说明 hard-negative 机制在当前实现与预算下仍未形成稳定正增益。  
2. `v2c`（仅 source-select）表现为“几乎完全持平 baseline”，说明该模块当前更多是“稳定但弱贡献”，还不足以单独拉开迁移性能。  
3. 到 `epoch10 + 2-seed` 这一档，现有 v2 系列改造尚未达到预设门槛（`Δavg_mrr >= +0.003` 且双seed同向为正）。

## 5. 建议的下一步（面向任务书）

结合任务书“可迁移能力”目标，下一阶段不建议继续在当前 loss 组合上做小步微调，建议转向：
- 明确构建“迁移矩阵主表”：
  - `ZH-EN -> JA-EN/FR-EN`
  - `DBP15K -> FBDB15K/FBYG15K`
- 引入“目标域伪标签自训练”与“迁移能力估计权重”两条高收益机制（可分别做开关消融）。
- 以“缩小跨域性能落差”为主要贡献表达，而不是仅追求同域分数绝对值提升。

