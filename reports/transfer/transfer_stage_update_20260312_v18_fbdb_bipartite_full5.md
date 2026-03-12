# 迁移实验阶段报告（v18 FBDB bipartite seeds 正式收口）

- 时间戳：`2026-03-12 04:50`
- 阶段目标：将 `FBDB15K` 的优化方向从 `P0` 配置调参切换到 `P1` 伪种子质量改造，并验证能否稳定超过当前主表版本 `v7b`。

## 本阶段代码改造

1. 修改 `baselines/MEAformer/src/data.py`：
   - 为 `visual_pivot_induction` 增加 `mutual nearest + margin` 过滤分支
   - 支持 `unsup_no_fallback`
   - 支持 `unsup_k_max`
   - 将空伪种子结果固定为 `(0, 2)` 形状，避免严格过滤时后续流程报错
2. 修改 `baselines/MEAformer/config.py`：
   - 新增 `unsup_k_max`
   - 新增 `unsup_use_bipartite_filter`
   - 新增 `unsup_margin_min`
   - 新增 `unsup_no_fallback`
3. 修改 `scripts/run_meaformer.py`：
   - 新增上述 `unsup_*` 参数透传

## 本阶段实验配置

新增 `v18` 三个变体（`2-seed pilot: 42, 2026`）：

- `v18a_bipartite_no_il`
- `v18b_bipartite_late_il`
- `v18c_bipartite_late_il_skiprel`

新增自动脚本：

- `scripts/run_transfer_adapt_v18_fbdb_iter_queue.py`

新增配置：

- `configs/transfer_adapt/tmmeada_target_fbdb15k_v18a_bipartite_no_il.yaml`
- `configs/transfer_adapt/tmmeada_target_fbdb15k_v18b_bipartite_late_il.yaml`
- `configs/transfer_adapt/tmmeada_target_fbdb15k_v18c_bipartite_late_il_skiprel.yaml`

## 关键诊断

在 `FBDB15K` 上，新选种器的初始 visual seeds 质量出现了明显提升：

- `v17` 初始 seeds 真值率：约 `5.67%`
- `v18` 初始 seeds 真值率：约 `15.67%`

这说明瓶颈确实主要在 `visual_pivot_induction` 的伪种子生成机制，而不是 `domain_align_weight` 或 IL 调度本身。

## Pilot 结果（2-seed）

参考主表版本：

- `v7b_expand5`：`delta_avg_mrr_mean = +0.0008`

本轮 `2-seed pilot`（vs matched baseline）：

| variant | delta_avg_mrr_mean | 结论 |
|---|---:|---|
| v18a | +0.00750 | 显著优于当前主表版本 |
| v18b | +0.00700 | 显著优于当前主表版本 |
| v18c | +0.00800 | 最优，自动扩展到 `5-seed` |

自动决策：

- `best_variant_pilot = v18c`
- `improve_over_current_ref = +0.00720`
- `expanded_variant_to_full5 = v18c`

## 正式结果（5-seed）

正式扩展 seeds：

- `42`
- `3407`
- `2026`
- `7`
- `123`

`v18c` 正式 `5-seed`（vs baseline）：

- `delta_avg_hits@1_mean = +0.00454`
- `delta_avg_hits@10_mean = +0.01568`
- `delta_avg_mrr_mean = +0.00830`
- `delta_avg_mr_mean = -206.81670`

## 结论

1. `P1` 路线验证成功，`FBDB15K` 已从 `v7b` 的 `+0.0008` 提升到 `v18c` 的 `+0.0083`。
2. 当前 `FBDB15K` 的主要收益来自“更干净的初始 visual seeds”，而不是继续压缩 IL 或微调 `DA weight`。
3. `v18c` 可作为新的 `FBDB15K` 主表版本。
4. 当前 4 目标主结果表中，`ja_en / FBDB15K / fr_en / FBYG15K` 仍全部为 `5-seed` 正增益，其中 `FBDB15K` 已不再是“边际小正增益”。

## 相关文件

- 决策：
  - `reports/transfer/transfer_adapt_v18_fbdb_iter_decision.json`
  - `reports/transfer/transfer_adapt_v18_fbdb_iter_decision.md`
- pilot compare：
  - `reports/transfer/transfer_adapt_v18_fbdb_pilot_v18a_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v18_fbdb_pilot_v18b_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v18_fbdb_pilot_v18c_compare_vs_baseline.csv`
- formal compare：
  - `reports/transfer/transfer_adapt_v18_fbdb_v18c_expand5_compare_vs_baseline.csv`
