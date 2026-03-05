# 任务书对齐状态更新（2026-03-05）

## 已完成

- 动机实验链路已从纯 `only_test` 迁移到 `target adapt (unsup + IL)`，并稳定运行。
- 已完成 `source=zh_en` 到目标域的 2-seed 对比：
  - `ja_en`
  - `FBDB15K`
- 已完成多轮优化（v3-v7）并产出自动决策报告：
  - `reports/transfer/transfer_adapt_v7_fbdb_decision.md`
  - `reports/transfer/transfer_adapt_v7_fbdb_compare_vs_baseline.csv`

## 当前结果摘要

- `ja_en`（v6 mixed）：
  - `delta_avg_mrr_mean = +0.00075`
- `FBDB15K`（v7b formal）：
  - `delta_avg_mrr_mean = +0.00075`
  - `delta_avg_hits@1_mean = +0.000875`

说明：当前已实现“小幅正增益”，但需要扩展 seed 和目标域来增强结论说服力。

## 尚未完成（按任务书）

1. 目标域矩阵未补齐：
   - `fr_en`（transfer-adapt）
   - `FBYG15K`（transfer-adapt）
2. 正式统计规模需扩展到 `5-seed`。
3. 中期/终稿所需的完整误差分析与可视化图表仍需补齐。

## 下一步执行顺序

1. 补齐 `fr_en` / `FBYG15K` transfer-adapt 2-seed。  
2. 对关键目标域执行 5-seed 正式跑并汇总均值/方差。  
3. 输出中期报告主表与“方法改造 -> 指标变化”的证据链图表。  

