# Epoch10 自动决策报告

- 生成时间: `2026-03-03T08:54:42`
- 判定阈值: `delta_avg_mrr >= 0.003` 且 2-seed 同向为正
- 最终决策: `stop_after_epoch10_pilot_and_prepare_writeup`
- 执行动作: `no extra training; use pilot compare reports for writeup and error analysis`

## 数据集判定细节

| dataset | delta_avg_mrr | seed_deltas | consistent_positive | pass_threshold |
|---|---:|---|---:|---:|
| zh_en | 0.0000 | s42:0.0000, s3407:0.0000 | False | False |
| FBDB15K | 0.0000 | s42:0.0000, s3407:0.0000 | False | False |
