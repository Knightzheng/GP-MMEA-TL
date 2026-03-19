# Scripts 阶段视图

由于大量历史命令、run card 和脚本互相引用，`scripts/` 中的真实文件名暂不整体搬迁。本文件负责按阶段解释“当前应该把哪些脚本看成同一阶段的工具组”。

## S0 基础环境与数据准备

- `preprocess_dbp15k.py`
- `prepare_meaformer_data.py`
- `sync_official_meaformer_data.py`

## S1 baseline 复现

- `train_baseline.py`
- `run_meaformer.py`
- `run_meaformer_multiseed.py`
- `run_meaformer_crossgraph_multiseed.py`
- `collect_meaformer_results.py`
- `aggregate_meaformer_results.py`

## S2 TMMEA-DA 受控开发

- `run_tmmeada_multiseed.py`
- `run_tmmeada_v1_weight_sweep.py`
- `make_tmmeada_baseline_compare.py`
- `make_tmmeada_baseline_compare_dbp15k.py`
- `make_tmmeada_baseline_compare_all.py`
- `make_tmmeada_v1_compare_zh_en.py`
- `make_tmmeada_v1_best_compare_zh_en.py`
- `make_epoch3_pilot_compare_zh_en.py`
- `make_epoch3_multiseed_compare_zh_en.py`
- `make_epoch3_compare_dbp15k.py`
- `make_epoch3_compare_crossgraph.py`
- `make_epoch3_ablation_zh_en.py`
- `summarize_epoch3_ablation_zh_en_multiseed.py`

## S3 epoch10 pilot 与自动决策

- `run_from_base_config_multiseed.py`
- `run_next_stage_pilot_queue.py`
- `auto_decide_next_stage.py`
- `auto_decide_after_epoch10.py`
- `auto_compare_v2_tuned.py`
- `auto_next_after_v2b.py`
- `compare_epoch10_v2_tuned_vs_baseline.py`

## S4 transfer 主线搭建

- `run_transfer_train_eval.py`
- `run_transfer_formal_queue.py`
- `run_transfer_adapt_v3_queue.py`
- `run_transfer_adapt_v4_queue.py`
- `run_transfer_adapt_v5_queue.py`
- `run_transfer_adapt_v6_mixed_queue.py`
- `run_transfer_adapt_v7_fbdb_auto.py`
- `run_transfer_adapt_v8_expand_queue.py`
- `auto_after_transfer_adapt_queue.py`
- `auto_after_transfer_adapt_v3.py`
- `auto_after_transfer_adapt_v4.py`
- `auto_after_transfer_adapt_v5.py`
- `auto_after_transfer_adapt_v6_mixed.py`

## S5 target-specific transfer 优化

### JA / FR

- `run_transfer_adapt_v9_fren_auto.py`
- `run_transfer_adapt_v10_fren_auto.py`
- `run_transfer_adapt_v11_fren_auto.py`
- `run_transfer_adapt_v12_fren_auto.py`
- `run_transfer_adapt_v13_fren_auto.py`
- `run_transfer_adapt_v14_fren_auto.py`
- `run_transfer_adapt_v14_fren_expand5_resume.py`
- `run_transfer_adapt_ja_v15_pilot.py`
- `run_transfer_adapt_ja_v15_iter_queue.py`

### FBDB15K

- `run_transfer_adapt_v16_fbdb_iter_queue.py`
- `run_transfer_adapt_v17_fbdb_iter_queue.py`
- `run_transfer_adapt_v18_fbdb_iter_queue.py`

### FBYG15K

- `run_transfer_adapt_v19_fbyg_iter_queue.py`
- `run_transfer_adapt_v20_fbyg_iter_queue.py`
- `run_transfer_adapt_v21_fbyg_iter_queue.py`
- `run_transfer_adapt_v22_fbyg_iter_queue.py`
- `run_transfer_adapt_v23_fbyg_iter_queue.py`
- `run_transfer_adapt_v24_fbyg_iter_queue.py`
- `run_transfer_adapt_v25_fbyg_iter_queue.py`

### 续跑与共享工具

- `run_transfer_adapt_expand5_resume_generic.py`
- `run_transfer_adapt_fbyg_expand5_resume.py`
- `run_transfer_adapt_ja_fbdb_expand5_next.py`
- `run_and_finalize_ja_fbdb_expand5.py`
- `finalize_ja_fbdb_expand5_after_run.py`
- `ensure_transfer_source_formal.py`
- `transfer_adapt_utils.py`

## S6 主线收口与辅助支撑

- `summarize_transfer_formal.py`
- `make_transfer_main_and_bucket_report.py`
- `compare_transfer_summaries.py`
- `analyze_transfer_significance.py`
- `build_transfer_case_analysis.py`
- `summarize_transfer_efficiency.py`
- `run_gpu_peak_minimal.py`
- `summarize_gpu_peak_minimal.py`
- `verify_mainline_artifacts.py`
- `hourly_progress_reporter.py`

## S7 中期材料

- `render_midterm_report_template.py`
