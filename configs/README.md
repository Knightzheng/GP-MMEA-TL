# Configs 布局说明

本目录的真实结构保持为四个配置族，因为脚本中有大量路径引用：

- `baselines/`
- `tmmeada/`
- `transfer/`
- `transfer_adapt/`

为了避免硬编码路径大面积失效，本轮不直接搬动真实配置目录；阶段化理解统一通过 [configs/by_stage/README.md](/d:/code/codes/cursor/BYSJ_zyf/configs/by_stage/README.md) 完成。

## 当前配置族职责

- `baselines/`
  - baseline 复现、多数据集、多 epoch 和显存约束配置。
- `tmmeada/`
  - MVP / v1 / 消融 / `epoch10` pilot 等受控方法配置。
- `transfer/`
  - `source-train -> target-eval` 的正式迁移 source / target 配置。
- `transfer_adapt/`
  - `v3-v25` 各阶段 target adaptation 策略配置。
