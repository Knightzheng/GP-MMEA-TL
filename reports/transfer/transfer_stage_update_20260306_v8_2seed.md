# 迁移优化阶段报告（2026-03-06，v8 2-seed）

## 1. 本阶段目标

- 在 v8 扩展设置下补齐 `seed=3407`，形成 `s42+s3407` 的正式 2-seed 对比。
- 评估 `fr_en` 与 `FBYG15K` 两个目标域在 transfer-adapt 场景下的稳定性。

## 2. 执行与产出

- 执行脚本：
  - `scripts/run_transfer_adapt_v8_expand_queue.py`
- 3407 独立阶段目录：
  - baseline: `runs/transfer/transfer_adapt_v8_expand_baseline_s3407/target_eval`
  - tmmeada: `runs/transfer/transfer_adapt_v8_expand_tmmeada_s3407/target_eval`
- 合并后的 2-seed 目录：
  - baseline: `runs/transfer/transfer_adapt_v8_expand_2seed_baseline/target_eval`
  - tmmeada: `runs/transfer/transfer_adapt_v8_expand_2seed_tmmeada/target_eval`
- 对比输出：
  - `reports/transfer/transfer_adapt_v8_expand_2seed_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v8_expand_2seed_compare_vs_baseline.md`

## 3. 2-seed 结果

来源：`reports/transfer/transfer_adapt_v8_expand_2seed_compare_vs_baseline.csv`

### 3.1 FBYG15K

- `delta_avg_hits@1_mean = +0.000925`
- `delta_avg_hits@10_mean = +0.001600`
- `delta_avg_mrr_mean = +0.000750`
- `delta_avg_mr_mean = -9.72425`

结论：跨图谱目标 `FBYG15K` 在 2-seed 下保持稳定正增益。

### 3.2 fr_en

- `delta_avg_hits@1_mean = -0.001250`
- `delta_avg_hits@10_mean = +0.000250`
- `delta_avg_mrr_mean = -0.000750`
- `delta_avg_mr_mean = +0.48450`

结论：`fr_en` 仍表现为轻微负迁移，且在 2-seed 下趋势未逆转。

## 4. 下一步优化建议

1. 针对 `fr_en` 做小范围权重搜索（降低 `domain_align/source_select/missing_gate`）。  
2. 保留 `FBYG15K` 的 mild-DA 策略作为稳定分支。  
3. 进入 `v9` 的 `fr_en` 定向调参，并至少做 2-seed 验证。  

