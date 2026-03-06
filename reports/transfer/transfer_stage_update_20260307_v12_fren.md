# 迁移实验阶段报告（2026-03-07，v12 fr_en）

## 1. 本阶段目标
- 在 `v11` 大幅退化后，快速恢复 `fr_en` 迁移性能并验证“温和过滤”是否有增益。
- 执行自动流程：`pilot(3变体) -> 选优 -> formal -> 2-seed汇总`。

## 2. 本阶段改动
- 新增配置：
  - `configs/transfer_adapt/tmmeada_target_fr_en_v12a_recover_v10.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v12b_mild_filter_highkeep.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v12c_mild_filter_da03.yaml`
- 新增脚本：
  - `scripts/run_transfer_adapt_v12_fren_auto.py`
- 策略说明：
  - `v12a`：回稳对照（等价于 v10 逻辑，过滤基本关闭）；
  - `v12b/v12c`：高保留率的温和置信过滤（避免 v11 的过度筛除）。

## 3. 自动决策结果
- pilot seed：`42`
- formal seed：`3407`
- pilot 对比（vs baseline，`delta_avg_mrr_mean`）：
  - `v12a_recover_v10`: `-0.00100`
  - `v12b_mild_filter_highkeep`: `-0.01700`
  - `v12c_mild_filter_da03`: `-0.02200`
- 选优分支：`v12a_recover_v10`

决策文件：
- `reports/transfer/transfer_adapt_v12_fren_decision.md`
- `reports/transfer/transfer_adapt_v12_fren_decision.json`

## 4. 正式 2-seed 结果（fr_en）
来源：
- `reports/transfer/transfer_adapt_v12_fren_2seed_compare_vs_baseline.csv`
- `reports/transfer/transfer_adapt_v12_fren_2seed_compare_vs_v10.csv`

关键结论：
- vs baseline：`delta_avg_mrr_mean = -0.00025`
- vs v10：`delta_avg_mrr_mean = 0.00000`（完全持平）

## 5. 结论
- `v12` 成功把 `v11` 的异常退化拉回到 `v10` 稳定水平。
- 在当前实现下，温和置信过滤（v12b/v12c）仍未优于 `v10` 对照分支。

## 6. 下一步建议
1. 固定 `v12a/v10` 作为当前最优对照，避免继续在过滤强度上消耗预算。
2. 将优化重点切到“结构级改动”（例如 source_select/missing_gate 的低权重联动），而不是继续筛阈值。
3. 在 `fr_en` 与 `FBYG15K` 上做同口径试验矩阵，优先寻找可迁移的统一策略。
