# GPU Peak Minimal Runs

## 1. 保留原则

本目录只保留当前阶段真正完成并写入汇总表的最小正式 GPU 峰值显存补测 run。

已经清理掉的内容：

1. dry-run 留痕
2. 失败 run
3. 临时配置目录

## 2. 当前保留的正式 run

- `ja_en` baseline
  - `ja_en_baseline/20260315-033319-MEAformer-transfer-adapt-target-ja_en-transfer-tgt-gpupeak-DBP15K-ja_en-s42/`
- `ja_en` method
  - `ja_en_method/20260315-034506-TMMEA-DA-transfer-adapt-v15-target-ja_en-transfer-tgt-gpupeak-DBP15K-ja_en-s42/`
- `FBYG15K` baseline
  - `fbyg15k_baseline/20260315-035536-MEAformer-transfer-adapt-target-fbyg15k-transfer-tgt-gpupeak-FBYG15K-norm-s42/`
- `FBYG15K` method
  - `fbyg15k_method/20260315-040122-TMMEA-DA-transfer-adapt-v24b-target-fbyg15k-transfer-tgt-gpupeak-FBYG15K-norm-s42/`

## 3. 对应汇总文件

- `../../../reports/transfer/transfer_gpu_peak_minimal_per_run.csv`
- `../../../reports/transfer/transfer_gpu_peak_minimal_summary.csv`
- `../../../reports/transfer/transfer_gpu_peak_minimal_summary.md`

## 4. 使用边界

1. 这是辅助支撑项，不是主线完成前提。
2. 当前只覆盖 `seed=42`、代表性目标域 `ja_en` 与 `FBYG15K`。
3. 该补测的主要用途是给出同环境下的 PyTorch allocator 峰值显存参考，不替代正式 `5-seed` wall-clock 统计。
4. 每个 run 目录中的 `config.yaml` 是当前应优先参考的本地配置留痕；`run_card.md` 中的临时 config 路径仅代表当时的调度入口。
