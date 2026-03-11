# 迁移实验阶段报告（v16 FBDB 试验收尾）

- 时间戳：`2026-03-11 23:19`
- 阶段目标：在 `FBDB15K` 上尝试超过当前主表版本 `v7b`（5-seed）的迁移增益。

## 本阶段执行内容

1. 新增 3 个 `v16` 变体配置并完成 `2-seed pilot`（`42, 2026`）：
   - `v16a_refresh4_balanced`
   - `v16b_refresh4_strict`
   - `v16c_refresh5_srcsel`
2. 统一使用 matched baseline 做对比汇总。
3. 自动生成决策文件，并按阈值判断是否扩展 `5-seed`。

## 关键结果

参考版本（当前主表）：
- `v7_expand5`：`delta_avg_mrr_mean = +0.0008`

本轮 pilot 结果（vs baseline）：

| variant | delta_avg_mrr_mean | 结论 |
|---|---:|---|
| v16a | -0.00175 | 三者中最好，但为负增益 |
| v16b | -0.00200 | 负增益 |
| v16c | -0.00275 | 负增益且最差 |

自动决策结果：
- `best_variant_pilot = v16a`
- `improve_over_current_ref = -0.00255`
- `expanded_variant_to_full5 = None`（未触发扩展）

## 产物清单

- 决策文件：
  - `reports/transfer/transfer_adapt_v16_fbdb_iter_decision.json`
  - `reports/transfer/transfer_adapt_v16_fbdb_iter_decision.md`
- 对比结果：
  - `reports/transfer/transfer_adapt_v16_fbdb_pilot_v16a_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v16_fbdb_pilot_v16b_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v16_fbdb_pilot_v16c_compare_vs_baseline.csv`
- 脚本与配置：
  - `scripts/run_transfer_adapt_v16_fbdb_iter_queue.py`
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v16a_refresh4_balanced.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v16b_refresh4_strict.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v16c_refresh5_srcsel.yaml`

## 结论

1. `v16` 本轮未超过 `v7`，且在 pilot 口径下整体退化。
2. 当前 `FBDB15K` 仍保留 `v7b (5-seed)` 作为主结果版本。
3. 后续如继续优化，建议优先调整“跨图谱特征尺度差异/迁移加载错配”方向，而不是继续加大伪标签过滤强度。
