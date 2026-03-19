# Scripts 目录导航

`scripts/` 中真实文件数量较多，而且文件名保留了大量阶段痕迹。本 README 不再按“脚本清单堆叠”来组织，而是按项目阶段说明：

1. 哪批脚本属于哪个阶段。
2. 哪些脚本是当前正式主线入口。
3. 哪些脚本主要是历史探索与自动排队留痕。

## 逻辑阶段树

```text
scripts/
├─ S0 数据准备
├─ S1 baseline 复现
├─ S2 TMMEA-DA 受控开发
├─ S3 epoch10 pilot 与自动决策
├─ S4 transfer 主线搭建
├─ S5 目标域分支优化
├─ S6 主线收口与补强材料生成
└─ S7 中期模板适配
```

阶段总索引见 [scripts/by_stage/README.md](/d:/code/codes/cursor/BYSJ_zyf/scripts/by_stage/README.md)。

## 当前正式主线最重要的脚本入口

如果只关心当前正式主线与仓库维护，优先看：

1. `run_meaformer.py`
2. `run_meaformer_multiseed.py`
3. `run_meaformer_crossgraph_multiseed.py`
4. `run_tmmeada_multiseed.py`
5. `run_transfer_train_eval.py`
6. `summarize_transfer_formal.py`
7. `make_transfer_main_and_bucket_report.py`
8. `analyze_transfer_significance.py`
9. `build_transfer_case_analysis.py`
10. `summarize_transfer_efficiency.py`
11. `run_gpu_peak_minimal.py`
12. `summarize_gpu_peak_minimal.py`
13. `verify_mainline_artifacts.py`

## 脚本职责分层

### 当前正式链路

- `run_*multiseed.py`
- `run_transfer_train_eval.py`
- `summarize_transfer_formal.py`
- `make_transfer_main_and_bucket_report.py`
- `analyze_transfer_significance.py`
- `build_transfer_case_analysis.py`
- `summarize_transfer_efficiency.py`
- `run_gpu_peak_minimal.py`
- `summarize_gpu_peak_minimal.py`
- `verify_mainline_artifacts.py`

这些脚本直接对应当前主线正式结果、辅助补强材料或仓库状态校验。

### 阶段收口与整理工具

- `ensure_transfer_source_formal.py`
- `compare_transfer_summaries.py`
- `finalize_ja_fbdb_expand5_after_run.py`
- `run_and_finalize_ja_fbdb_expand5.py`
- `transfer_adapt_utils.py`

这些脚本更偏向阶段收口、结果整理和复核，不是第一入口，但仍属于当前主线有价值的工具。

### 历史探索与自动排队

- `auto_*`
- `run_transfer_adapt_v*_*.py`
- `run_transfer_adapt_*queue*.py`
- `run_next_stage_pilot_queue.py`
- `run_transfer_formal_queue.py`

这些脚本主要用于保留方法演化过程，不应直接视为“当前正式主线入口”。

## 当前边界

1. 本目录当前不包含 `H3` 相关脚本。
2. 若只想复查“当前正式主线到底依赖哪些脚本”，优先看本 README 的“正式链路”部分，而不是遍历全部历史脚本。
3. 若要判断某个脚本是否仍属于当前正式链路，应回到：
   - `reports/notes/mainline_traceability_matrix_20260315.md`
   - `reports/notes/mainline_artifact_integrity_20260315.md`
