# Scripts Guide

## 1. 当前主线最重要的脚本入口

如果只关心当前主线正式结果与项目维护，请优先看这些脚本：

1. `run_meaformer.py`
   - 单次训练 / 评测统一入口。
2. `run_meaformer_multiseed.py`
   - `DBP15K` baseline 多 seed 入口。
3. `run_meaformer_crossgraph_multiseed.py`
   - 跨图谱 baseline 多 seed 入口。
4. `run_tmmeada_multiseed.py`
   - TMMEA-DA 多 seed 入口。
5. `run_transfer_train_eval.py`
   - 统一迁移实验主入口。
6. `summarize_transfer_formal.py`
   - 汇总当前正式 transfer run。
7. `make_transfer_main_and_bucket_report.py`
   - 生成主表与误差分桶结果。
8. `analyze_transfer_significance.py`
   - 生成显著性与稳定性补强材料。
9. `build_transfer_case_analysis.py`
   - 生成当前案例分析材料。
10. `summarize_transfer_efficiency.py`
   - 汇总 wall-clock 效率材料。
11. `run_gpu_peak_minimal.py`
   - 运行 GPU 峰值显存最小正式补测。
12. `summarize_gpu_peak_minimal.py`
   - 汇总 GPU 峰值显存补测结果。
13. `verify_mainline_artifacts.py`
   - 自动校验当前主线正式材料、正式 run 与辅助补强文件的完整性。

## 2. 脚本分层建议

### 2.1 正式主线入口

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

这些脚本直接对应当前主线正式结果、辅助支撑补强或仓库状态维护。

### 2.2 阶段收口 / 辅助整理

- `ensure_transfer_source_formal.py`
- `compare_transfer_summaries.py`
- `finalize_ja_fbdb_expand5_after_run.py`
- `run_and_finalize_ja_fbdb_expand5.py`
- `transfer_adapt_utils.py`

这些脚本更偏向正式阶段收口与结果整理，不是日常第一入口，但仍属于当前主线有用工具。

### 2.3 历史探索 / 自动排队脚本

- `auto_*`
- `run_transfer_adapt_v*_*.py`
- `run_transfer_adapt_*queue*.py`
- `run_next_stage_pilot_queue.py`
- `run_transfer_formal_queue.py`

这些脚本主要保留方法演化和阶段试验留痕。当前不应把它们直接当作“当前正式主线入口”理解。

## 3. 当前边界

1. 本目录当前不再包含 `H3` 相关脚本。
2. 若只想复查“当前正式主线到底依赖哪些脚本”，应优先看第 1 节而不是遍历全部历史脚本。
3. 若要判断某个脚本是否仍属于当前正式链路，应优先回到：
   - `reports/notes/mainline_traceability_matrix_20260315.md`
   - `reports/notes/mainline_artifact_integrity_20260315.md`
