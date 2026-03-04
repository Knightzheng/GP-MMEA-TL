# Epoch10 v2 tuned 结果复盘（2026-03-03）

## 1. 结论摘要
- `TMMEA-DA v2 tuned` 在 `2-seed (42,3407)` pilot 下未超过 baseline。
- 自动决策结果：`continue_tuning_or_error_analysis`。
- 当前不建议直接扩展到 5-seed 正式跑，应先做小规模定向调参。

## 2. 关键结果（来自自动汇总）
- 文件：
  - `reports/epoch10_compare_v2_tuned_pilot.csv`
  - `reports/epoch10_v2_tuned_decision.json`

### 2.1 zh_en
- baseline vs v2 tuned（mean, 2-seed）：
  - `l2r_mrr: 0.7500 -> 0.7500 (Δ +0.0000)`
  - `r2l_mrr: 0.7505 -> 0.7495 (Δ -0.0010)`
- 平均 MRR 差值：`Δavg_mrr = -0.0005`
- seed 级差值（method - baseline, avg_mrr）：
  - `s42: -0.0015`
  - `s3407: +0.0005`

### 2.2 FBDB15K
- baseline vs v2 tuned（mean, 2-seed）：
  - `l2r_mrr: 0.3205 -> 0.3200 (Δ -0.0005)`
  - `r2l_mrr: 0.3205 -> 0.3195 (Δ -0.0010)`
- 平均 MRR 差值：`Δavg_mrr = -0.00075`
- seed 级差值（method - baseline, avg_mrr）：
  - `s42: -0.0020`
  - `s3407: +0.0005`

## 3. 现象解释
- `s42` 在两个数据集都明显变差，而 `s3407` 小幅变好，说明当前 v2 参数组合引入了更强的 seed 敏感性（方差变大）。
- v2 相比 v1 的主要变化是“更强辅助损失 + hard-negative + 更早介入”，综合强度偏大，可能在部分 seed 上对主对比学习造成干扰。

## 4. 下一步改进方案（已落地配置）
目标：先判断“hard-negative 是否是主要负面来源”，再决定是否保留。

### 4.1 方案A：v2a_no_hardneg（先稳住）
- 策略：
  - 关闭 hard-negative（`domain_align_margin=0`, `domain_align_neg_weight=0`）
  - 保留分阶段辅助调度（`aux_start_epoch=4`, `aux_ramp_epochs=3`）
  - 辅助权重回落到 v1 量级（`domain/source/missing = 0.1/0.05/0.1`）
- 配置：
  - `configs/tmmeada/meaformer_zh_en_tmmeada_v2a_no_hardneg_epoch10_pilot.yaml`
  - `configs/tmmeada/meaformer_fbdb15k_tmmeada_v2a_no_hardneg_epoch10_pilot.yaml`

### 4.2 方案B：v2b_lite_hardneg（轻量硬负样本）
- 策略：
  - 保留 hard-negative，但显著减弱（`margin=0.2`, `neg_weight=0.25`）
  - 辅助项同样延后启用（`aux_start_epoch=4`, `aux_ramp_epochs=3`）
- 配置：
  - `configs/tmmeada/meaformer_zh_en_tmmeada_v2b_lite_hardneg_epoch10_pilot.yaml`
  - `configs/tmmeada/meaformer_fbdb15k_tmmeada_v2b_lite_hardneg_epoch10_pilot.yaml`

## 5. 执行顺序建议（节省时间）
1. 先跑 `v2a_no_hardneg` 的 `2-seed`（zh_en + FBDB15K）。  
2. 若 `Δavg_mrr >= +0.001` 且两 seed 同向，再跑 `v2b_lite_hardneg`。  
3. 只有当任一方案达到 `Δavg_mrr >= +0.003` 且两数据集都同向时，才扩展 5-seed 正式实验。
