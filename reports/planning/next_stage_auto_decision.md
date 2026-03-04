# 自动决策报告

- 生成时间: `2026-03-03T00:32:44`
- 判定阈值: `delta_avg_mrr >= 0.003` 且 2-seed 同向为正
- 最终决策: `continue_epoch10_pilot`
- 执行动作: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_next_stage_pilot_queue.py --epoch10-only --seeds 42,3407`

## 数据集判定细节

| dataset | delta_avg_mrr | seed_deltas | consistent_positive | pass_threshold |
|---|---:|---|---:|---:|
| zh_en | 0.0003 | s42:0.0005, s3407:0.0000 | False | False |
| FBDB15K | -0.0003 | s42:-0.0005, s3407:0.0000 | False | False |
