# Experiments Runs Guide

## 1. 当前目录结构

- `baseline/`
  - baseline 正式训练与对照 run。
- `tmmeada/`
  - TMMEA-DA 正式训练、消融与历史方法阶段 run。
- `gpu_peak_minimal/`
  - 当前阶段保留的 GPU 峰值显存最小正式补测 run。

## 2. 如何理解这些目录

1. `baseline/`
   - 主要用于支撑 baseline 复现、epoch3 对照和多 seed 正式结果。
2. `tmmeada/`
   - 同时包含正式对照、核心消融和历史方法演进 run。
   - 使用时应优先结合结果文件判断哪些目录仍属于当前正式主线。
3. `gpu_peak_minimal/`
   - 只属于辅助支撑项，不构成主线成立前提。

## 3. 当前推荐阅读顺序

1. 先看 `../../reports/transfer/README.md`
2. 再看 `../../runs/transfer/README.md`
3. 若需要回查训练级 run，再进入本目录下对应子目录

## 4. 当前边界

1. 本目录当前不再保留 `H3` 相关实验 run。
2. GPU 峰值显存补测目录保留，是为了辅助支撑和答辩备查，不是为了抬高为主线主表。
