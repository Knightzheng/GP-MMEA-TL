# 迁移实验阶段报告（v17 FBDB 噪声控制 pilot）

- 时间戳：`2026-03-12 01:51`
- 阶段目标：在 `FBDB15K` 上优先验证“减少伪标签注入”是否优于继续微调 `domain_align_weight`。

## 本阶段执行内容

1. 新增 `v17` 三个 `2-seed pilot` 变体（`42, 2026`）：
   - `v17a_no_il_balanced`
   - `v17b_late_il_strict`
   - `v17c_late_il_skiprel`
2. 新增自动迭代脚本：
   - `scripts/run_transfer_adapt_v17_fbdb_iter_queue.py`
3. 新增配置：
   - `configs/transfer_adapt/tmmeada_target_fbdb15k_v17a_no_il_balanced.yaml`
   - `configs/transfer_adapt/tmmeada_target_fbdb15k_v17b_late_il_strict.yaml`
   - `configs/transfer_adapt/tmmeada_target_fbdb15k_v17c_late_il_skiprel.yaml`
4. 统一与 matched baseline 做 `2-seed` compare，并按 `delta_avg_mrr_mean` 自动决策是否扩展到 `5-seed`。

## 关键结果

参考主表版本：

- `v7b_expand5`：`delta_avg_mrr_mean = +0.0008`

本轮 pilot 结果（vs matched baseline）：

| variant | delta_avg_mrr_mean | delta_avg_hits@1_mean | 结论 |
|---|---:|---:|---|
| v17a | -0.00800 | -0.00435 | 关闭 IL 后仍明显退化 |
| v17b | -0.00850 | -0.00445 | 晚启 + 严格 IL 更差 |
| v17c | -0.00775 | -0.00455 | 三者中最好，但仍明显负增益 |

自动决策：

- `best_variant_pilot = v17c`
- `improve_over_current_ref = -0.00855`
- `expanded_variant_to_full5 = None`

## 诊断结论

1. `P0` 的方向判断只成立了一半：
   - 的确不应继续围绕 `domain_align_weight` 做小步调参；
   - 但仅靠“减少伪标签注入”还不够，当前 `FBDB15K` 的初始无监督 seeds 本身就已经太脏。
2. 初始 visual seeds 的真值率虽然从 `v16a s42` 的 `3.78%` 提升到了 `v17 s42` 的 `5.67%`，但仍然远低于可支撑稳定迁移增益的水平。
3. `v17b / v17c` 在 `il_start=8` 且 `il_confidence_quantile=0.8` 的严格设置下，日志中 `epoch 8/9` 都是：
   - `il_filter raw=0 kept=0`
   - 说明严格 IL 已经基本“不开火”，瓶颈并不在 IL 调度，而在 IL 之前的初始 seeds。
4. `v17c` 跳过 `rel_fc` 迁移后只有极小改善（`-0.00775` vs `-0.00850`），说明更保守的 transfer load 有帮助，但不足以扭转当前噪声底座。

## 下一步

下一步不再继续做 `P0` 风格的纯配置搜索，也不再继续调 `domain_align_weight`。应直接进入 `P1`：

1. 修改 `baselines/MEAformer/src/data.py` 中的 `visual_pivot_induction`。
2. 引入 `mutual nearest + margin` 过滤。
3. 增加 `unsup_no_fallback`，禁用“阈值不够时按 rank 填满”的回填逻辑。
4. 增加 `unsup_k_max`，改为“质量优先而非数量优先”。
5. 先在 `FBDB15K` 做 `2-seed` 验证，再决定是否扩展到 `5-seed`。

## 相关文件

- 决策：
  - `reports/transfer/transfer_adapt_v17_fbdb_iter_decision.json`
  - `reports/transfer/transfer_adapt_v17_fbdb_iter_decision.md`
- compare：
  - `reports/transfer/transfer_adapt_v17_fbdb_pilot_v17a_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v17_fbdb_pilot_v17b_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v17_fbdb_pilot_v17c_compare_vs_baseline.csv`
- 运行脚本：
  - `scripts/run_transfer_adapt_v17_fbdb_iter_queue.py`
