# Experiments Runs 目录导航

`runs/experiments/` 主要承接非 transfer 的训练级 run，可按阶段理解为：

```text
runs/experiments/
├─ baseline/          S1 baseline 复现
├─ tmmeada/           S2 TMMEA-DA 受控开发 + S3 epoch10 pilot
└─ gpu_peak_minimal/  S6 GPU 最小正式补测
```

## 各目录职责

- `baseline/`
  - 支撑 baseline 复现、`epoch3` 对照与多 seed 正式结果。
- `tmmeada/`
  - 同时保留正式对照、核心消融、权重搜索和历史方法演进 run。
- `gpu_peak_minimal/`
  - 只属于辅助支撑项，不构成主线成立前提。

## 推荐阅读顺序

1. 先看 `../../reports/transfer/README.md`
2. 再看 `../../runs/transfer/README.md`
3. 如果需要回查训练级 run，再进入本目录对应子目录

## 当前边界

1. 本目录当前不保留 `H3` 相关实验 run。
2. GPU 峰值显存补测目录保留是为了辅助支撑和答辩备查，不是为了抬高为主线主表。
