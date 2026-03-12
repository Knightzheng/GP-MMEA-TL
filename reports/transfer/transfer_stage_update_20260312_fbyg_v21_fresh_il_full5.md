# 迁移实验阶段报告（FBYG15K v21 fresh-IL full5）

- 日期：`2026-03-12`
- 目标域：`FBYG15K`
- 背景参考：`v19/v20` 已证明“晚启 IL + 更严格过滤”无法超过当前主表版本 `v8`，且存在候选在注入前塌缩为 `1` 条的失败模式。

## 1. 本轮目标

验证 `FBYG15K` 上的关键瓶颈是否在“IL 候选生成后没有及时注入”，并测试更保守迁移下的 fresh-IL 立即注入策略能否稳定超过当前主表版本：

- 当前参考版本：`v8_mild_da_expand5`
- 参考结果：`delta_avg_mrr_mean = +0.00110`

## 2. 本轮设置

新增 `v21` 三个 `2-seed pilot` 变体：

- `v21a_fresh_il_q80_skiprel_skipfusion`
- `v21b_fresh_il_q90_skiprel_skipfusion`
- `v21c_fresh_il_q95_skiprel_skipfusion`

共同设置：

- `il_start = 5`
- `il_refresh_interval = 1`
- `transfer_skip_keys = multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias`
- `transfer_skip_prefixes = multimodal_encoder.fusion.`

新增产物：

- 配置：
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v21a_fresh_il_q80_skiprel_skipfusion.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v21b_fresh_il_q90_skiprel_skipfusion.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v21c_fresh_il_q95_skiprel_skipfusion.yaml`
- 自动脚本：
  - `scripts/run_transfer_adapt_v21_fbyg_iter_queue.py`
- 决策文件：
  - `reports/transfer/transfer_adapt_v21_fbyg_iter_decision.md`
  - `reports/transfer/transfer_adapt_v21_fbyg_iter_decision.json`

## 3. Pilot 结果

`pilot_seeds = [42, 2026]`

| 变体 | delta_avg_mrr_mean | 结论 |
|---|---:|---|
| v21a | +0.00200 | 最优，超过当前 `v8` |
| v21b | +0.00100 | 未超过 `v8` |
| v21c | +0.00100 | 未超过 `v8` |

自动决策结果：

- `best_variant_pilot = v21a`
- `improve_over_current_ref = +0.00090`
- 达到扩展阈值 `+0.00050`
- 自动扩展到 `5-seed`

## 4. Full-5 正式结果

正式 `5-seed` 比较文件：

- `reports/transfer/transfer_adapt_v21_fbyg_v21a_expand5_compare_vs_baseline.csv`
- `reports/transfer/transfer_adapt_v21_fbyg_v21a_expand5_compare_vs_baseline.md`

正式结果（vs baseline）：

- `delta_avg_hits@1_mean = +0.00141`
- `delta_avg_hits@10_mean = +0.00193`
- `delta_avg_mrr_mean = +0.00160`
- `delta_avg_mr_mean = -35.84720`

相对旧主表版本 `v8`：

- `delta_avg_mrr_mean`: `+0.00110 -> +0.00160`
- 提升幅度：`+0.00050`

## 5. 关键诊断

`v21a` 的 fresh-IL 立即注入确实修复了 `v20` 的“候选在注入前塌缩”问题。`5-seed` 日志显示：

- `seed=42`: `#new_links_select=450`, `#true_links=8`, `true link ratio=1.8%`
- `seed=2026`: `#new_links_select=428`, `#true_links=8`, `true link ratio=1.9%`
- `seed=3407`: `#new_links_select=424`, `#true_links=8`, `true link ratio=1.9%`
- `seed=7`: `#new_links_select=397`, `#true_links=10`, `true link ratio=2.5%`
- `seed=123`: `#new_links_select=436`, `#true_links=9`, `true link ratio=2.1%`

这说明：

1. `FBYG15K` 的增益来源并不是把伪链接“变得很准”，而是避免了 `v20` 那种最终只剩 `1` 条链接的极端塌缩；
2. 当前 fresh-IL 路线已经能带来可复现的小幅正式增益；
3. 但伪链接真值率仍只有约 `2%`，后续若继续优化，重点仍应放在 fresh proposal 的质量提升，而不是再回到 `late IL / 更高 quantile` 的轻量搜索。

## 6. 结论

本轮 `v21` 验证成功：

- `FBYG15K` 主表版本切换为 `v21a_fresh_il_q80_skiprel_skipfusion_expand5`
- `FBYG15K` 的 `5-seed delta_avg_mrr_mean` 从 `+0.00110` 提升到 `+0.00160`
- 当前统一 4 目标主表继续保持全部正增益

已同步刷新：

- `reports/transfer/transfer_adapt_main_results_4target.csv`
- `reports/transfer/transfer_adapt_main_results_4target.md`
- `reports/transfer/transfer_adapt_error_bucket_summary.csv`
- `reports/transfer/transfer_adapt_error_bucket_summary.md`
