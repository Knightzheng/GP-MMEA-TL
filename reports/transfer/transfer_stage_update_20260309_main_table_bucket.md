# 迁移实验阶段报告（4目标主结果表 + 分桶分析）

- 时间戳: `20260309`
- 阶段目标: 将 `ja_en / FBDB15K / fr_en / FBYG15K` 汇总为统一主结果表，并补充可直接入文的误差分桶分析。

## 本阶段新增产物

- 主结果表：
  - `reports/transfer/transfer_adapt_main_results_4target.csv`
  - `reports/transfer/transfer_adapt_main_results_4target.md`
- 分桶分析：
  - `reports/transfer/transfer_adapt_error_bucket_summary.csv`
  - `reports/transfer/transfer_adapt_error_bucket_summary.md`
- 生成脚本：
  - `scripts/make_transfer_main_and_bucket_report.py`

## 主结果（当前最佳变体）

| target | variant | runs(b/m) | delta MRR |
|---|---|---:|---:|
| ja_en | v6_mixed | 2/2 | +0.00075 |
| FBDB15K | v7b_formal | 2/2 | +0.00075 |
| fr_en | v14b_refresh4_da0025_expand5 | 5/5 | +0.01210 |
| FBYG15K | v8_mild_da_expand5 | 5/5 | +0.00110 |

整体平均增益（4 目标，非加权）：

- `delta_avg_hits@1_mean = +0.003210`
- `delta_avg_hits@10_mean = +0.005381`
- `delta_avg_mrr_mean = +0.003675`
- `delta_avg_mr_mean = -9.080300`（更低更好）

## 分桶观察（用于实验分析章节）

- 按场景分桶：
  - `cross_lingual` 平均 `delta MRR = +0.006425`
  - `cross_graph` 平均 `delta MRR = +0.000925`
- 按置信度分桶：
  - `formal_5seed` 平均 `delta MRR = +0.006600`
  - `pilot_2seed` 平均 `delta MRR = +0.000750`
- 按难度分桶（baseline MRR）：
  - `easy` 平均 `delta MRR = +0.006425`
  - `hard` 平均 `delta MRR = +0.001100`
  - `very_hard` 平均 `delta MRR = +0.000750`

## 结论

1. 当前 4 个目标域都保持了相对 baseline 的正向改进。
2. 增益主要由 `fr_en(v14b)` 拉动，`FBYG15K` 与 `ja_en/FBDB15K` 为小幅稳定提升。
3. 下一步重点是把 `ja_en` 与 `FBDB15K` 从 2-seed 扩展到 5-seed，统一主表口径后进入终稿主结果表。
