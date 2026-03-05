# Transfer Adapt Pilot Update (2026-03-05, seed=42)

## 1. 本阶段完成内容
- 新增并验证了目标域自适应链路（`unsup + IL`）：
  - 目标域不再 `only_test`，而是先在目标域无标注自训练，再测试。
- 完成 `seed=42` 的 2 个目标域对照：
  - `DBP15K ja_en`
  - `FBDB15K`
- 对比对象：
  - baseline: `MEAformer`
  - method: `TMMEA-DA`（domain/source_select/missing_gate 开启）

## 2. 关键代码改造（用于报告“实现过程”）
- `scripts/run_meaformer.py`
  - 新增透传参数：`--il`, `--semi_learn_step`, `--il_start`, `--unsup`, `--unsup_k`, `--unsup_mode`
- `scripts/run_transfer_train_eval.py`
  - 新增目标域控制参数：
    - `--target-only-test`（0/1）
    - `--target-epoch`
    - `--target-save-model`
- 新增配置目录与文件：
  - `configs/transfer_adapt/meaformer_target_ja_en_unsup_il.yaml`
  - `configs/transfer_adapt/meaformer_target_fbdb15k_unsup_il.yaml`
  - `configs/transfer_adapt/tmmeada_target_ja_en_unsup_il.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_unsup_il.yaml`
- 新增队列脚本：
  - `scripts/run_transfer_adapt_pilot_queue.py`

## 3. seed=42 结果（adapt after transfer）
- baseline:
  - `ja_en`: `avg_mrr=0.5075`, `avg_hits@1=0.4289`
  - `FBDB15K`: `avg_mrr=0.0250`, `avg_hits@1=0.0107`
- TMMEA-DA:
  - `ja_en`: `avg_mrr=0.5075`, `avg_hits@1=0.4300`
  - `FBDB15K`: `avg_mrr=0.0245`, `avg_hits@1=0.0103`

对应汇总文件：
- `reports/transfer/transfer_adapt_pilot_target_eval_baseline_summary.csv`
- `reports/transfer/transfer_adapt_pilot_target_eval_tmmeada_summary.csv`
- `reports/transfer/transfer_adapt_pilot_compare_tmmeada_vs_baseline.csv`
- `reports/transfer/transfer_adapt_pilot_compare_tmmeada_vs_baseline.md`

## 4. 与“纯迁移 only_test”对比（同 seed=42）
- `ja_en`:
  - baseline: `avg_mrr 0.2055 -> 0.5075`（`+0.3020`）
  - TMMEA-DA: `avg_mrr 0.2055 -> 0.5075`（`+0.3020`）
- `FBDB15K`:
  - baseline: `avg_mrr 0.0020 -> 0.0250`（`+0.0230`）
  - TMMEA-DA: `avg_mrr 0.0020 -> 0.0245`（`+0.0225`）

结论：
- “目标域无标注自适应”对迁移性能提升显著（尤其 `ja_en`）。
- 在当前配置下，`TMMEA-DA` 相对 baseline 仍基本持平，后续需继续优化 method 特有增益。

## 5. 正在后台继续执行
- 已启动 `seed=3407` 同配置队列（`ja_en + FBDB15K`, baseline + TMMEA-DA）。
- 队列日志：
  - `runs/transfer/transfer_adapt_pilot/queue_20260305-003100.out.log`
  - `runs/transfer/transfer_adapt_pilot/queue_20260305-003100.err.log`

## 6. 下一步（待本队列结束后自动进入）
1. 汇总 `seed=42,3407` 的 adapt 对照，检查是否稳定同向提升。  
2. 若稳定，则扩展到 `fr_en` 与 `FBYG15K`。  
3. 在 `TMMEA-DA` 上做小范围权重微调（优先 `domain_align_weight/source_select_weight`），目标是在 adapt 设定下拉开与 baseline 的差距。  
