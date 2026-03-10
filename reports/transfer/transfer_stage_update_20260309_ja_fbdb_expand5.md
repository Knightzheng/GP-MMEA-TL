# 迁移实验阶段报告（ja_en + FBDB15K 扩展到 5-seed）

- 时间戳: `20260310-225922`
- 覆盖目标: `ja_en`, `FBDB15K`
- 目标: 将两目标从 2-seed 扩展到 5-seed，并刷新 4目标主结果表。

## 输出文件

- `reports/transfer/transfer_adapt_ja_expand5_status.{md,json}`
- `reports/transfer/transfer_adapt_fbdb_expand5_status.{md,json}`
- `reports/transfer/transfer_adapt_v6_mixed_ja_expand5_compare_vs_baseline.{csv,md}`
- `reports/transfer/transfer_adapt_v7_fbdb_expand5_compare_vs_baseline.{csv,md}`
- `reports/transfer/transfer_adapt_main_results_4target.{csv,md}`
- `reports/transfer/transfer_adapt_error_bucket_summary.{csv,md}`

## ja_en（5-seed）

- runs(b/m): `5/5`
- `delta_avg_hits@1_mean = -0.013560`
- `delta_avg_hits@10_mean = -0.020770`
- `delta_avg_mrr_mean = -0.016300`
- `delta_avg_mr_mean = +17.290800`

## FBDB15K（5-seed）

- runs(b/m): `5/5`
- `delta_avg_hits@1_mean = +0.000210`
- `delta_avg_hits@10_mean = +0.001950`
- `delta_avg_mrr_mean = +0.000800`
- `delta_avg_mr_mean = -12.054500`

## 结论

1. 两目标已具备 5-seed 正式口径后，4目标主表可用于论文主结果。
2. 若某目标仍无显著提升，可继续做目标域伪标签策略微调（仅在该目标域单独推进）。
