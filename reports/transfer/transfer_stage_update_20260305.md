# 迁移实验阶段报告（2026-03-05）

## 1. 本阶段目标

- 检查并优化 `source=DBP15K-zh_en -> target` 的可迁移表现。
- 在统一口径（Hits@1/Hits@10/MRR）下，完成 2-seed 对比并形成自动化流程。
- 重点突破 `FBDB15K` 在 transfer-adapt 设置下的指标。

## 2. 本阶段完成内容

- 完成 transfer-adapt 迭代：
  - `v3`、`v4`、`v5`、`v6_mixed`
- 新增并执行 FBDB 定向自动优化：
  - `v7a/v7b/v7c` pilot（seed=42）
  - 自动选优后运行 formal（seeds=`42,3407`）
- 新增自动脚本：
  - `scripts/run_transfer_adapt_v7_fbdb_auto.py`
- 新增配置：
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v7a_mild_da_unsup_il.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v7b_mild_da_unsup_il.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v7c_mild_da_unsup_il.yaml`

## 3. 关键结果（2-seed）

### 3.1 v6 mixed vs baseline

来源：`reports/transfer/transfer_adapt_v6_mixed_compare_vs_baseline.csv`

- `ja_en`：
  - `delta_avg_mrr_mean = +0.00075`
  - `delta_avg_hits@1_mean = +0.000825`
- `FBDB15K`：
  - `delta_avg_mrr_mean = +0.00000`（持平）

### 3.2 v7 FBDB auto vs baseline

来源：`reports/transfer/transfer_adapt_v7_fbdb_compare_vs_baseline.csv`

- 最优分支：`v7b`
- `FBDB15K`：
  - `delta_avg_hits@1_mean = +0.000875`
  - `delta_avg_hits@10_mean = +0.001425`
  - `delta_avg_mrr_mean = +0.00075`
  - `delta_avg_mr_mean = -14.805`

决策记录：
- `reports/transfer/transfer_adapt_v7_fbdb_decision.md`
- `reports/transfer/transfer_adapt_v7_fbdb_decision.json`

## 4. 当前判断

- 任务书要求中的“跨域迁移可行性验证”已形成可复现证据链：
  - 配置 -> 运行 -> run_card -> 汇总对比 -> 决策报告
- 指标提升幅度目前是“小幅正增益”，下一步重点是提升幅度与稳定性（从 2-seed 扩展到 5-seed）。

## 5. 下一步计划

1. 扩展 transfer-adapt 到 `fr_en` 与 `FBYG15K`，补齐目标域矩阵。  
2. 对 `ja_en` 与 `FBDB15K` 执行 5-seed 正式统计（均值/方差）。  
3. 生成中期报告可直接引用的主表与消融表（含误差分析分桶）。  

