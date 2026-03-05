# 迁移优化阶段报告（2026-03-06，v8 s42）

## 1. 阶段目标

- 扩展 transfer-adapt 覆盖到 `fr_en` 与 `FBYG15K`。
- 采用与前序阶段一致的评测口径（Hits@1/Hits@10/MRR）。
- 输出可直接进入中期材料的阶段对比结论。

## 2. 本阶段执行

- 新增配置：
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v8_mild_da_unsup_il.yaml`
- 新增队列脚本：
  - `scripts/run_transfer_adapt_v8_expand_queue.py`
- 实际运行策略：
  - 先完成 `seed=42` 的 baseline 与 tmmeada 双分支；
  - `seed=3407` 自动启动后中止（用于控制当前提交节奏，后续可继续补齐）。

## 3. 结果汇总（s42）

来源：
- `reports/transfer/transfer_adapt_v8_expand_s42_compare_vs_baseline.csv`

### 3.1 FBYG15K

- `delta_avg_hits@1_mean = +0.00085`
- `delta_avg_hits@10_mean = +0.00140`
- `delta_avg_mrr_mean = +0.00100`

结论：在跨图谱目标 `FBYG15K` 上，v8 的保守 mild-DA 策略表现为小幅稳定增益。

### 3.2 fr_en

- `delta_avg_hits@1_mean = -0.00080`
- `delta_avg_hits@10_mean = -0.00055`
- `delta_avg_mrr_mean = -0.00050`

结论：在 `fr_en` 上出现轻微负增益，说明该目标域仍需专门调参（尤其是 aux 权重）。

## 4. 当前判断

- v8 的“目标域分策略”方向有效，但目标域差异明显：
  - `FBYG15K` 受益；
  - `fr_en` 仍需细化策略。
- 下一步优先级：
  1. 对 `fr_en` 做轻量网格（降低 domain_align/source_select/missing_gate 权重）。
  2. 补齐 `seed=3407`，形成至少 2-seed 正式对比。
  3. 与 v6/v7 统一放入阶段总表，形成最终中期报告主表。

