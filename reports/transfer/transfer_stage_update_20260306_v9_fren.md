# 迁移优化阶段报告（2026-03-06，v9 fr_en）

## 1. 目标

- 针对 `fr_en` 在 v8 中的轻微负迁移，做定向策略优化。
- 使用自动流程完成：`pilot(2个变体, s42) -> 自动选优 -> formal(s3407) -> 2-seed汇总`。

## 2. 新增内容

- 配置：
  - `configs/transfer_adapt/tmmeada_target_fr_en_v9a_mild_da_unsup_il.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v9b_mild_da_unsup_il.yaml`
- 自动脚本：
  - `scripts/run_transfer_adapt_v9_fren_auto.py`

## 3. 自动决策结果

来源：`reports/transfer/transfer_adapt_v9_fren_decision.json`

- `pilot_seed=42`
- `formal_seed=3407`
- 候选变体：
  - `v9a_tm_src_mild_da`: `delta_avg_mrr_mean_vs_baseline = -0.0010`
  - `v9b_base_src_mild_da`: `delta_avg_mrr_mean_vs_baseline = -0.0025`
- 选择结果：`v9a_tm_src_mild_da`

## 4. 2-seed结果（fr_en）

### 4.1 vs baseline

来源：`reports/transfer/transfer_adapt_v9_fren_2seed_compare_vs_baseline.csv`

- `delta_avg_hits@1_mean = -0.000175`
- `delta_avg_hits@10_mean = -0.000175`
- `delta_avg_mrr_mean = -0.000250`

说明：仍有轻微负迁移，但相比 v8 已明显收敛。

### 4.2 vs v8 tmmeada

来源：`reports/transfer/transfer_adapt_v9_fren_2seed_compare_vs_v8.csv`

- `delta_avg_hits@1_mean = +0.001075`
- `delta_avg_mrr_mean = +0.000500`

说明：v9 相比 v8 在 `fr_en` 上实现了小幅回升。

## 5. 当前结论与下一步

- 结论：
  - v9 的保守策略有效降低了负迁移幅度；
  - 但尚未超过 baseline（仍差约 `0.00025 MRR`）。
- 下一步建议：
  1. 在 v9a 基础上继续微调 `domain_align_weight`（如 `0.025/0.035`）。  
  2. 只对 `fr_en` 做小网格，快速锁定“持平或反超 baseline”的配置。  
  3. 锁定后再扩到 5-seed 正式统计。  

