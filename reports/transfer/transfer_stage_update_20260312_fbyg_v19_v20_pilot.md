# 迁移实验阶段报告（FBYG15K v19/v20 pilot）

- 时间戳：`2026-03-12 13:45`
- 目标域：`FBYG15K`
- 参考主表版本：`v8_mild_da_expand5`
- 参考正式结果：`delta_avg_mrr_mean = +0.00110`（5-seed）

## 本阶段目标

在 `FBYG15K` 上继续做针对性优化，重点验证两条路线是否能够超过当前主表版本：

1. 抑制后期 IL 伪链接噪声；
2. 进一步保守化迁移加载，避免跨图谱负迁移。

## 代码与脚本改动

1. 迁移加载新增前缀级跳过能力：
   - `baselines/MEAformer/config.py`
   - `baselines/MEAformer/main.py`
   - `scripts/run_meaformer.py`
   - `scripts/run_transfer_train_eval.py`
2. 源模型解析补强，允许复用已有 `transfer_adapt_*` 源检查点：
   - `scripts/transfer_adapt_utils.py`
3. 新增 `FBYG15K` 自动迭代脚本：
   - `scripts/run_transfer_adapt_v19_fbyg_iter_queue.py`
   - `scripts/run_transfer_adapt_v20_fbyg_iter_queue.py`

## v19：strict late-IL + conservative transfer

新增配置：

- `configs/transfer_adapt/tmmeada_target_fbyg15k_v19a_late_il_strict.yaml`
- `configs/transfer_adapt/tmmeada_target_fbyg15k_v19b_late_il_skiprel.yaml`
- `configs/transfer_adapt/tmmeada_target_fbyg15k_v19c_late_il_skiprel_skipfusion.yaml`

pilot（2-seed：`42, 2026`）结果：

| variant | delta_avg_mrr_mean | 结论 |
|---|---:|---|
| v19a | -0.00225 | 严格晚启 IL 退化明显 |
| v19b | -0.00250 | 仅 skip `rel_fc` 仍退化 |
| v19c | +0.00100 | 保守迁移后恢复为小正增益，但仍低于 `v8` |

关键诊断：

- `v19` 的 `il_start=8` 与当前 `Iter_new_links` 的 5-epoch fresh-proposal 节奏错位；
- 实际日志显示 `il_filter raw=0 kept=0`，说明 `v19a/v19b/v19c` 基本等价于“关闭 IL”；
- `v19c` 虽未超过 `v8`，但说明 `skip rel_fc + skip fusion` 能显著降低负迁移。

## v20：对齐 IL 刷新周期后的复验

为排除 `v19` 的调度错位影响，新增对齐刷新周期的两组配置：

- `configs/transfer_adapt/tmmeada_target_fbyg15k_v20a_aligned_il_skiprel_skipfusion.yaml`
- `configs/transfer_adapt/tmmeada_target_fbyg15k_v20b_aligned_il_q90_skiprel_skipfusion.yaml`

设置特点：

- 保留 `skip rel_fc + skip fusion`
- 将 `il_start` 改为 `5`，与 fresh-proposal 周期对齐
- 比较 `q=0.8` 与 `q=0.9`

pilot（2-seed：`42, 2026`）结果：

| variant | delta_avg_mrr_mean | 结论 |
|---|---:|---|
| v20a | +0.00050 | 低于当前主表版本 |
| v20b | +0.00050 | 与 `v20a` 基本相同，仍低于当前主表版本 |

关键诊断：

- `epoch 5` 初始 IL 候选量很大：
  - `v20a`: `raw=2247/2139`, `kept=450/428`
  - `v20b`: `raw=2247/2139`, `kept=225/214`
- 但候选在后续 epoch 快速塌缩，到 `epoch 9` 只剩 `1` 条注入链接；
- 最终注入链接真值率为 `0.0%`（`seed=42` 与 `2026` 均为 `#true_links: 0`）。

这说明问题不在“是否足够严格”，而在当前 IL 机制的候选稳定性与置信度排序本身。

## 结论

1. `FBYG15K` 当前主表版本保持不变：
   - `v8_mild_da_expand5`
   - `delta_avg_mrr_mean = +0.00110`（5-seed）
2. `v19/v20` 均未达到扩展到 `5-seed` 的门槛，因此不替换主表。
3. `skip rel_fc + skip fusion` 是有价值的保守迁移方向，但单靠“压 IL 注入量”或“调整 IL 启动时机”不足以超过 `v8`。
4. 若继续优化 `FBYG15K`，下一步应改 IL 生成/刷新机制本身，而不是继续做 `il_start / quantile / skip keys` 级别的轻量搜索。

## 相关文件

- 决策：
  - `reports/transfer/transfer_adapt_v19_fbyg_iter_decision.json`
  - `reports/transfer/transfer_adapt_v19_fbyg_iter_decision.md`
  - `reports/transfer/transfer_adapt_v20_fbyg_iter_decision.json`
  - `reports/transfer/transfer_adapt_v20_fbyg_iter_decision.md`
- pilot compare：
  - `reports/transfer/transfer_adapt_v19_fbyg_pilot_v19a_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v19_fbyg_pilot_v19b_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v19_fbyg_pilot_v19c_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v20_fbyg_pilot_v20a_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v20_fbyg_pilot_v20b_compare_vs_baseline.csv`
