# 下一阶段 Pilot 实验计划（2026-03-02）

## 1. 目标

验证在更高训练预算（`epoch=8/10`）下，`TMMEA-DA v1_best` 是否能相对 baseline 产生可复现提升。

## 2. 范围与预算

- 数据集：
  - DBP15K `zh_en`
  - `FBDB15K`
- 预算：
  - `epoch=8`、`epoch=10`
  - `seed=42,3407`
- 评测：
  - `Hits@1`, `Hits@10`, `MRR`（`l2r/r2l`）

## 3. 配置文件

- baseline:
  - `configs/baselines/meaformer_zh_en_rtx3060_safe_epoch8_pilot.yaml`
  - `configs/baselines/meaformer_zh_en_rtx3060_safe_epoch10_pilot.yaml`
  - `configs/baselines/meaformer_fbdb15k_rtx3060_safe_epoch8_pilot.yaml`
  - `configs/baselines/meaformer_fbdb15k_rtx3060_safe_epoch10_pilot.yaml`
- method:
  - `configs/tmmeada/meaformer_zh_en_tmmeada_v1_best_epoch8_pilot.yaml`
  - `configs/tmmeada/meaformer_zh_en_tmmeada_v1_best_epoch10_pilot.yaml`
  - `configs/tmmeada/meaformer_fbdb15k_tmmeada_v1_best_epoch8_pilot.yaml`
  - `configs/tmmeada/meaformer_fbdb15k_tmmeada_v1_best_epoch10_pilot.yaml`

## 4. 启动命令模板

```powershell
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\run_from_base_config_multiseed.py --base-config <config_path> --seeds "42,3407"
```

## 5. 结果判定门槛

- 若任一数据集在 method 相对 baseline 上满足 `ΔMRR >= +0.003`（并在 2-seed 下方向一致），则扩展为全量 5-seed 正式实验。
- 若未达门槛，则转入“负结果分析 + 误差分桶”收口分支。
