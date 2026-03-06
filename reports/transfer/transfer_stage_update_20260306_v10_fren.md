# 迁移实验阶段报告（2026-03-06，v10 fr_en）

## 1. 本阶段目标
- 针对 `fr_en` 在 transfer-adapt 中仍轻微落后 baseline 的问题，继续做自动化小范围参数优化。
- 保持与既有流程一致：`pilot -> auto select -> formal -> 2-seed summary`。

## 2. 本阶段新增内容
- 新增配置：
  - `configs/transfer_adapt/tmmeada_target_fr_en_v10a_unsup900.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v10b_da0025.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v10c_da0035.yaml`
- 新增自动脚本：
  - `scripts/run_transfer_adapt_v10_fren_auto.py`

## 3. 自动流程与决策
- pilot seed：`42`
- formal seed：`3407`
- pilot 变体与结果（vs baseline，`delta_avg_mrr_mean`）：
  - `v10a_unsup900`: `-0.01050`
  - `v10b_da0025`: `-0.00100`
  - `v10c_da0035`: `-0.00100`
- 自动选优：`v10b_da0025`

决策文件：
- `reports/transfer/transfer_adapt_v10_fren_decision.md`
- `reports/transfer/transfer_adapt_v10_fren_decision.json`

## 4. 最终 2-seed 结果（fr_en）
来源文件：
- `reports/transfer/transfer_adapt_v10_fren_2seed_compare_vs_baseline.csv`
- `reports/transfer/transfer_adapt_v10_fren_2seed_compare_vs_v9.csv`

关键指标：
- vs baseline：
  - `delta_avg_hits@1_mean = -0.000175`
  - `delta_avg_hits@10_mean = -0.000175`
  - `delta_avg_mrr_mean = -0.00025`
- vs v9：
  - `delta_avg_mrr_mean = 0.00000`（持平）

## 5. 结论
- v10 在 `fr_en` 上没有产生新的正增益，最终与 v9 持平。
- 当前 `fr_en` 仍是“接近 baseline 但略低”的状态，差值已非常小（`MRR -0.00025` 量级）。
- 下一步优化应转向“策略级改动”而非继续在同一超参邻域微调。

## 6. 建议下一步（用于后续自动脚本）
1. 做“低置信伪标签过滤 + 动态阈值”的一轮 pilot（优先检查是否能提升 `fr_en` 的稳定性）。
2. 将 `fr_en` 与 `FBYG15K` 放入同一批自动流程，统一输出误差对比与可视化。
3. 若仍无增益，进入 `5-seed` 正式统计并转写“负迁移边界分析”章节，保证论文说服力。
