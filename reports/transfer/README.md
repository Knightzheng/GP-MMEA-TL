# Transfer Reports 目录导航

`reports/transfer/` 是当前仓库最复杂的结果目录之一。这里同时保留了：

1. `S4` 阶段的 bootstrap / formal 早期材料。
2. `S5` 阶段各目标域分支的版本演化记录。
3. `S6` 阶段的正式主表、显著性、案例、效率与 GPU 补证。

本 README 的目标是把这些材料按阶段重新组织，而不是继续按文件名时间线去猜。

## 逻辑阶段树

```text
reports/transfer/
├─ S4 transfer 主线搭建
│  ├─ smoke / bootstrap / early formal 汇总
│  └─ v3-v8 早期自适应阶段材料
├─ S5-A JA / FR 分支优化
│  ├─ transfer_adapt_*ja*
│  └─ transfer_adapt_*fren*
├─ S5-B FBDB15K 分支优化
│  └─ transfer_adapt_*fbdb*
├─ S5-C FBYG15K 分支优化
│  └─ transfer_adapt_*fbyg*
└─ S6 主线正式收口
   ├─ transfer_adapt_main_results_4target.*
   ├─ transfer_adapt_significance_*
   ├─ transfer_case_*
   ├─ transfer_efficiency_*
   ├─ transfer_gpu_peak_minimal_*
   └─ transfer_extra_baseline_limitation_writeup.md
```

## 当前正式主线入口

如果只关心当前正式主线，请优先看：

1. `transfer_adapt_main_results_4target.md`
2. `transfer_adapt_main_results_4target.csv`
3. `transfer_adapt_significance_summary.md`
4. `transfer_case_analysis_examples.md`
5. `transfer_case_pattern_summary_20260316.md`
6. `transfer_efficiency_summary.md`
7. `transfer_gpu_peak_minimal_summary.md`
8. `transfer_extra_baseline_limitation_writeup.md`

## 四个目标域直接来源文件

| 目标域 | 当前正式 compare 文件 |
| --- | --- |
| `ja_en` | `transfer_adapt_ja_v15_expand5_compare_vs_baseline.csv` |
| `fr_en` | `transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv` |
| `FBDB15K` | `transfer_adapt_v18_fbdb_v18c_expand5_compare_vs_baseline.csv` |
| `FBYG15K` | `transfer_adapt_v24_fbyg_v24b_expand5_compare_vs_baseline.csv` |

这些文件是 `transfer_adapt_main_results_4target.*` 的直接来源之一。

## 如何阅读历史阶段文件

本目录中保留了大量 `transfer_stage_update_*.md` 与 `transfer_adapt_v*.md/csv/json`，它们主要用于回溯版本演化。

推荐入口：

1. `transfer_stage_update_20260311_ja_v15_final.md`
   - 查看 `ja_en` 如何收口到 `v15`。
2. `transfer_stage_update_20260312_v18_fbdb_bipartite_full5.md`
   - 查看 `FBDB15K` 为什么固定在 `v18c`。
3. `transfer_stage_update_20260314_fbyg_v24_strict_source_full5.md`
   - 查看 `FBYG15K` 为什么固定在 `v24b`。
4. 其他大多数 `v*` 文件
   - 默认视为探索历史，而不是当前正式证据入口。

## 当前边界

1. 本目录当前不包含 `H3` 材料。
2. GPU 峰值显存已经有补证，但仍只是辅助最小检查。
3. `transfer_case_pattern_summary_20260316.*` 是对现有案例的重新组织，不是新的统计实验。
4. 如果只需要主线闭环证据链，应回到 `../notes/mainline_traceability_matrix_20260315.md`。
