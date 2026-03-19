# Runs 阶段视图

真实 run 目录仍保留在 `runs/` 原位置；本文件负责按阶段说明“哪些 run 根目录属于哪个阶段”。

## S1 baseline 复现

- `runs/experiments/baseline/`
  - baseline 正式训练、`epoch3` 对照和 pilot 训练。

## S2 TMMEA-DA 受控开发

- `runs/experiments/tmmeada/`
  - 方法正式训练、消融和权重搜索相关 run。

## S4 transfer 主线搭建

- `runs/transfer/transfer_smoke*`
- `runs/transfer/transfer_formal*`
- `runs/transfer/transfer_adapt_v3*`
- `runs/transfer/transfer_adapt_v4*`
- `runs/transfer/transfer_adapt_v5*`
- `runs/transfer/transfer_adapt_v6*`
- `runs/transfer/transfer_adapt_v7*`
- `runs/transfer/transfer_adapt_v8*`

## S5 target-specific transfer 优化

### JA / FR

- `runs/transfer/transfer_adapt_v9*`
- `runs/transfer/transfer_adapt_v10*`
- `runs/transfer/transfer_adapt_v11*`
- `runs/transfer/transfer_adapt_v12*`
- `runs/transfer/transfer_adapt_v13*`
- `runs/transfer/transfer_adapt_v14*`
- `runs/transfer/transfer_adapt_ja_v15*`

### FBDB15K

- `runs/transfer/transfer_adapt_v16*`
- `runs/transfer/transfer_adapt_v17*`
- `runs/transfer/transfer_adapt_v18*`

### FBYG15K

- `runs/transfer/transfer_adapt_v19*`
- `runs/transfer/transfer_adapt_v20*`
- `runs/transfer/transfer_adapt_v21*`
- `runs/transfer/transfer_adapt_v22*`
- `runs/transfer/transfer_adapt_v23*`
- `runs/transfer/transfer_adapt_v24*`
- `runs/transfer/transfer_adapt_v25*`

## S6 主线收口支撑

- `runs/experiments/gpu_peak_minimal/`
  - GPU 峰值显存最小正式补测。
- `runs/system/`
  - 系统级临时配置、队列日志和辅助运行目录。
- `runs/transfer/iter_queue/`
  - 迭代队列日志。

## 当前主线正式 run

如果只关心当前正式主线，优先看：

1. `runs/transfer/transfer_adapt_ja_v15_full_baseline_matched_ref/target_eval/`
2. `runs/transfer/transfer_adapt_ja_v15_full_ref/target_eval/`
3. `runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval/`
4. `runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval/`
5. `runs/transfer/transfer_adapt_v18_fbdb_v18c_expand5_baseline_matched_ref/target_eval/`
6. `runs/transfer/transfer_adapt_v18_fbdb_v18c_expand5_ref/target_eval/`
7. `runs/transfer/transfer_adapt_v24_fbyg_v24b_expand5_baseline_matched_ref/target_eval/`
8. `runs/transfer/transfer_adapt_v24_fbyg_v24b_expand5_ref/target_eval/`
