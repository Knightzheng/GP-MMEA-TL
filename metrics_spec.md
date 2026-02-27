# Metrics Spec

## Ranking Metrics
- `Hits@K`: top-K 命中率，默认 K in {1, 10}
- `MRR`: Mean Reciprocal Rank
- `MR`: Mean Rank（可选，越低越好）

## 计算口径
- 默认 `filtered` 评估（若模型实现支持）
- 每次评估必须记录：
  - `dataset`
  - `split/fold`
  - `seed`
  - `train_ratio`
  - `candidate_space`

## 报告规范
- 单次实验：报告 Hits@1 / Hits@10 / MRR / MR
- 关键实验：至少 5 seeds，报告 `mean ± std`
- 对比实验：同预算、同数据切分、同模态可用性
