# Configs 阶段视图

## S1 baseline 复现

- `configs/baselines/`
  - `MEAformer` baseline 的正式配置族。
  - 覆盖 DBP15K、FBDB15K、FBYG15K，以及 `epoch3 / epoch8 / epoch10` 变体。

## S2 TMMEA-DA 受控开发

- `configs/tmmeada/`
  - `domain_align_mvp`
  - `v1_best`
  - `epoch3` 正式对照
  - `wo_domain_align / wo_source_select / wo_missing_gate`

## S3 epoch10 pilot 与调参

- `configs/tmmeada/` 中包含：
  - `v2_tuned`
  - `v2a_no_hardneg`
  - `v2b_lite_hardneg`
  - `v2c_source_only`
  - `epoch10_pilot` 相关配置

## S4 transfer 主线搭建

- `configs/transfer/`
  - source checkpoint 配置
  - target eval 正式配置

## S5 target-specific transfer 优化

- `configs/transfer_adapt/`
  - `v3-v8`：bootstrap 和早期混合探索
  - `v9-v14`：`fr_en` / `ja_en` 优化链
  - `v16-v18`：`FBDB15K` 优化链
  - `v19-v25`：`FBYG15K` 优化链

## 当前使用建议

1. 如果只查当前正式主线，优先回到：
   - `configs/transfer/`
   - `configs/transfer_adapt/` 中被主表版本点名的配置
2. 如果要理解版本演化，再回到：
   - `configs/tmmeada/`
   - `configs/transfer_adapt/` 的完整版本族
