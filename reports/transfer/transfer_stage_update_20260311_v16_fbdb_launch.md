# 迁移实验阶段报告（v16 FBDB 优化启动）

- 时间戳：`2026-03-11 22:11`
- 目标：在 `FBDB15K` 上继续提升迁移自适应效果（当前主表为 `v7b`, `delta_avg_mrr_mean=+0.0008`）。

## 本次新增

- 配置：
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v16a_refresh4_balanced.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v16b_refresh4_strict.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v16c_refresh5_srcsel.yaml`
- 自动队列脚本：
  - `scripts/run_transfer_adapt_v16_fbdb_iter_queue.py`

## 运行策略

1. 先跑 `2-seed pilot`（`42, 2026`）对比 `v16a/v16b/v16c`。
2. 与 matched baseline 自动汇总并按 `delta_avg_mrr_mean` 选优。
3. 仅当 pilot 最优结果超过当前 `v7 expand5` 参考值且增益超过阈值（默认 `0.0005`）时，自动扩展到 `5-seed`。

## 队列状态

- 队列日志：
  - `runs/transfer/iter_queue/fbdb_v16_iter_20260311-221101.out.log`
  - `runs/transfer/iter_queue/fbdb_v16_iter_20260311-221101.err.log`
- 当前已启动阶段：
  - `runs/transfer/transfer_adapt_v16_fbdb_pilot_v16a/target_eval/`

## 下一步

- 队列结束后自动生成：
  - `reports/transfer/transfer_adapt_v16_fbdb_iter_decision.json`
  - `reports/transfer/transfer_adapt_v16_fbdb_iter_decision.md`
- 若触发扩展，将额外生成：
  - `reports/transfer/transfer_adapt_v16_fbdb_<best_variant>_expand5_compare_vs_baseline.csv`
