# Runs 目录导航

`runs/` 中保留了大量真实实验产物路径。本轮不直接大规模搬迁这些目录，而是按阶段重新解释它们的职责，避免“目录还在，但已经看不懂哪个阶段用了哪一批 run”。

## 逻辑阶段树

```text
runs/
├─ experiments/
│  ├─ baseline/          S1 baseline 复现
│  ├─ tmmeada/           S2-S3 TMMEA-DA 受控开发与 epoch10 pilot
│  └─ gpu_peak_minimal/  S6 GPU 最小正式补测
├─ transfer/             S4-S5 transfer 主线与目标域分支
└─ system/               队列日志、临时配置、模板与系统级辅助目录
```

## 按阶段查看

| 阶段 | 目录 | 说明 |
| --- | --- | --- |
| `S1` | `experiments/baseline/` | baseline 正式训练、多 seed 对照与早期 pilot |
| `S2-S3` | `experiments/tmmeada/` | TMMEA-DA 正式训练、消融、权重搜索、`epoch10` pilot |
| `S4-S5` | `transfer/` | transfer 主线从 smoke、formal 到四目标域正式收口的全部 run |
| `S6` | `experiments/gpu_peak_minimal/`, `system/` | GPU 辅助补测与队列日志/系统留痕 |

阶段总索引见 [runs/by_stage/README.md](/d:/code/codes/cursor/BYSJ_zyf/runs/by_stage/README.md)。

## 当前正式主线 run

如果只关心当前正式主线，请优先看 [runs/transfer/README.md](/d:/code/codes/cursor/BYSJ_zyf/runs/transfer/README.md) 中点名的 8 个正式 `target_eval/` 根目录。

## 如何区分正式 run 与历史探索

1. 正式主线 run 会在项目级总表、正式报告或 `mainline_traceability_matrix` 中被明确点名。
2. 历史探索 run 常见于 `pilot`、`queue`、`auto`、早期 `v*` 目录，主要用于保留演化过程与失败原因。
3. `system/` 中的内容主要是系统留痕和辅助运行目录，不等于正式结果目录。
