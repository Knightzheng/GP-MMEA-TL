# 迁移实验阶段报告（2026-03-07，v13 fr_en）

## 1. 本阶段目标
- 在 `v12` 回稳后，尝试低风险结构优化：
  - `source_select` 轻量启用
  - `missing_gate` 轻量启用
  - 两者联动启用
- 目标是维持稳定性的同时争取小幅提升 `fr_en` transfer-adapt。

## 2. 本阶段新增内容
- 配置：
  - `configs/transfer_adapt/tmmeada_target_fr_en_v13a_source_select_mild.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v13b_missing_gate_mild.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v13c_hybrid_mild.yaml`
- 自动脚本：
  - `scripts/run_transfer_adapt_v13_fren_auto.py`

## 3. 自动流程与决策
- 流程：`pilot(3 variants, s42) -> select best -> formal(s3407) -> 2-seed summarize`
- 结果：
  - `v13a_source_select_mild`: `delta_avg_mrr_mean = -0.00100`
  - `v13b_missing_gate_mild`: `delta_avg_mrr_mean = -0.00100`
  - `v13c_hybrid_mild`: `delta_avg_mrr_mean = -0.00100`
- 自动选优：`v13a_source_select_mild`

决策文件：
- `reports/transfer/transfer_adapt_v13_fren_decision.md`
- `reports/transfer/transfer_adapt_v13_fren_decision.json`

## 4. 最终 2-seed 结果（fr_en）
来源：
- `reports/transfer/transfer_adapt_v13_fren_2seed_compare_vs_baseline.csv`
- `reports/transfer/transfer_adapt_v13_fren_2seed_compare_vs_v12.csv`

关键结论：
- vs baseline：`delta_avg_mrr_mean = -0.00025`
- vs v12：`delta_avg_mrr_mean = 0.00000`（持平）

## 5. 结论
- `v13` 未带来可测提升，但维持了 `v12/v10` 的稳定水平。
- 在当前预算与实现下，`source_select/missing_gate` 的低权重启用未形成实质增益。

## 6. 下一步建议
1. 保留 `v10/v12/v13` 作为稳定主线，不再继续微调同类低权重模块。
2. 下一轮改为“任务层面”推进：扩展到 `FBYG15K` 或执行 `5-seed` 正式统计，提升报告说服力。
3. 并行准备误差分析图（分桶/案例），避免只给主指标表。
