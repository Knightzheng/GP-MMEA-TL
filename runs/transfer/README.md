# Transfer Runs 目录导航

`runs/transfer/` 同时包含 `S4` 主线搭建期、`S5` 各目标域分支优化期，以及最终正式主线 run。当前最重要的不是把每个目录都看一遍，而是先分清：

1. 哪些目录是当前正式主线。
2. 哪些目录是阶段演化留痕。
3. 哪些目录只用于 pilot、queue 或失败原因分析。

## 逻辑阶段树

```text
runs/transfer/
├─ S4 主线搭建
│  ├─ transfer_smoke*
│  ├─ transfer_formal*
│  └─ transfer_adapt_v3 ~ v8*
├─ S5-A JA / FR 分支
│  └─ transfer_adapt_v9 ~ v15*
├─ S5-B FBDB15K 分支
│  └─ transfer_adapt_v16 ~ v18*
├─ S5-C FBYG15K 分支
│  └─ transfer_adapt_v19 ~ v25*
└─ 当前正式主线
   └─ 四目标域 baseline / method 共 8 个正式 target_eval 根目录
```

## 当前正式主线 run

如果只关心当前四个目标域的正式主线结果，请优先看以下 8 个目录：

| 目标域 | baseline formal runs | method formal runs |
| --- | --- | --- |
| `ja_en` | `transfer_adapt_ja_v15_full_baseline_matched_ref/target_eval/` | `transfer_adapt_ja_v15_full_ref/target_eval/` |
| `fr_en` | `transfer_adapt_v14_fren_expand5_merged_baseline/target_eval/` | `transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval/` |
| `FBDB15K` | `transfer_adapt_v18_fbdb_v18c_expand5_baseline_matched_ref/target_eval/` | `transfer_adapt_v18_fbdb_v18c_expand5_ref/target_eval/` |
| `FBYG15K` | `transfer_adapt_v24_fbyg_v24b_expand5_baseline_matched_ref/target_eval/` | `transfer_adapt_v24_fbyg_v24b_expand5_ref/target_eval/` |

这些目录对应的结果会被汇总进：

- `reports/transfer/transfer_adapt_main_results_4target.*`
- `reports/transfer/transfer_adapt_significance_summary.*`
- `reports/transfer/transfer_efficiency_summary.*`

## 正式 run 的识别规则

当前仓库中，能够直接支撑主线结论的 transfer run 通常满足：

1. 位于上表列出的 8 个目录之一。
2. 目录下保留正式 `log.txt`、`config.yaml`、`run_card.md`、`artifact_manifest.json`。
3. 对应 `5-seed` 正式汇总，而不是单 seed pilot。

## 历史探索目录如何理解

下列目录模式主要用于保留演化过程：

- `transfer_adapt_v*_pilot_*`
- `transfer_adapt_v*_queue`
- `transfer_adapt_*_baseline_ref`
- `transfer_adapt_*_auto`

它们的主要作用是：

1. 回溯方法迭代过程。
2. 支撑阶段决策和失败原因分析。
3. 保留从 pilot 到 final 的路径证据。

除非被项目级总表或正式阶段报告明确点名，否则不应直接当作当前主线正式结果引用。

## 推荐阅读顺序

1. `../../reports/notes/mainline_traceability_matrix_20260315.md`
2. `../../reports/transfer/README.md`
3. 本目录中的正式 `target_eval/` 子目录

## 当前边界

1. 本目录当前不再保留 `H3` 相关 run。
2. GPU 峰值显存最小补测位于 `../experiments/gpu_peak_minimal/`，只属于辅助支撑。
