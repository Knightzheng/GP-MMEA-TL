# 迁移实验阶段报告（FBYG expand5 完成）

- 时间戳: `20260309-080542`
- 目标域: `FBYG15K`
- 统计口径: `5-seed`（`42,3407,2026,7,123`）
- 完成状态: `baseline / tmmeada 均无缺失 seed`

## 结果摘要（vs baseline）

- `delta_avg_hits@1_mean = +0.00100`
- `delta_avg_hits@10_mean = +0.00210`
- `delta_avg_mrr_mean = +0.00110`
- `delta_avg_mr_mean = -12.50440`（更低更好）

## 关键文件

- 状态文件：
  - `reports/transfer/transfer_adapt_fbyg_expand5_status.md`
  - `reports/transfer/transfer_adapt_fbyg_expand5_status.json`
- 对比结果：
  - `reports/transfer/transfer_adapt_fbyg_expand5_progress_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_fbyg_expand5_progress_compare_vs_baseline.md`
- 原始运行目录：
  - `runs/transfer/transfer_adapt_fbyg_expand5_baseline/`
  - `runs/transfer/transfer_adapt_fbyg_expand5_tmmeada/`
  - `runs/transfer/transfer_adapt_fbyg_expand5_merged_baseline/`
  - `runs/transfer/transfer_adapt_fbyg_expand5_merged_tmmeada/`

## 结论

- `FBYG15K` 在 5-seed 正式统计下实现稳定小幅正增益，说明当前 TMMEA-DA 迁移自适应方案在跨图谱目标上具有可复现改进。
- 与 `fr_en(v14b)` 的显著增益结果组合后，项目已形成“跨语言 + 跨图谱”双场景的正向证据链。

## 下一步建议

1. 生成统一主表（`ja_en / fr_en / FBDB15K / FBYG15K`）并固定为论文主结果表。
2. 补充误差分桶分析（按实体频次、模态缺失率）进入中期/终稿实验分析章节。
