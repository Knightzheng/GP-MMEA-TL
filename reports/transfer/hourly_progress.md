## 2026-03-08 19:17:58

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 1
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 1
  - 当前训练命令: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fbyg15k_unsup_il.yaml --tag baseline_transfer_adapt_fbyg_expand5_s2026 --stage-root transfer/transfer_adapt_fbyg_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0`

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: []
  - fbyg tmmeada: []

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: [2026, 7, 123]
  - tmmeada: [2026, 7, 123]

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-08 20:18:00

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 1
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 1
  - 当前训练命令: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fbyg15k_unsup_il.yaml --tag baseline_transfer_adapt_fbyg_expand5_s2026 --stage-root transfer/transfer_adapt_fbyg_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0`

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: []
  - fbyg tmmeada: []

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: [2026, 7, 123]
  - tmmeada: [2026, 7, 123]

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-08 21:18:04

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 1
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 1
  - 当前训练命令: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fbyg15k_unsup_il.yaml --tag baseline_transfer_adapt_fbyg_expand5_s2026 --stage-root transfer/transfer_adapt_fbyg_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0`

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: []
  - fbyg tmmeada: []

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: [2026, 7, 123]
  - tmmeada: [2026, 7, 123]

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-08 22:18:07

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 1
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 1
  - 当前训练命令: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fbyg15k_v8_mild_da_unsup_il.yaml --tag tmmeada_transfer_adapt_fbyg_expand5_s2026 --stage-root transfer/transfer_adapt_fbyg_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0`

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: [2026]
  - fbyg tmmeada: []

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: [2026, 7, 123]
  - tmmeada: [2026, 7, 123]

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-08 23:18:09

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 1
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 1
  - 当前训练命令: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fbyg15k_v8_mild_da_unsup_il.yaml --tag tmmeada_transfer_adapt_fbyg_expand5_s2026 --stage-root transfer/transfer_adapt_fbyg_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0`

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: [2026]
  - fbyg tmmeada: []

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: [2026, 7, 123]
  - tmmeada: [2026, 7, 123]

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-09 00:18:12

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 1
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 1
  - 当前训练命令: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fbyg15k_unsup_il.yaml --tag baseline_transfer_adapt_fbyg_expand5_s7 --stage-root transfer/transfer_adapt_fbyg_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0`

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: [2026]
  - fbyg tmmeada: [2026]

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: [2026, 7, 123]
  - tmmeada: [2026, 7, 123]

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-09 01:18:14

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 1
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 1
  - 当前训练命令: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fbyg15k_unsup_il.yaml --tag baseline_transfer_adapt_fbyg_expand5_s7 --stage-root transfer/transfer_adapt_fbyg_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0`

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: [2026]
  - fbyg tmmeada: [2026]

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: [2026, 7, 123]
  - tmmeada: [2026, 7, 123]

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-09 02:18:16

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 1
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 1
  - 当前训练命令: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fbyg15k_v8_mild_da_unsup_il.yaml --tag tmmeada_transfer_adapt_fbyg_expand5_s7 --stage-root transfer/transfer_adapt_fbyg_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0`

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: [7, 2026]
  - fbyg tmmeada: [2026]

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: [2026, 7, 123]
  - tmmeada: [2026, 7, 123]

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-09 03:18:19

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 1
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 1
  - 当前训练命令: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fbyg15k_v8_mild_da_unsup_il.yaml --tag tmmeada_transfer_adapt_fbyg_expand5_s7 --stage-root transfer/transfer_adapt_fbyg_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0`

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: [7, 2026]
  - fbyg tmmeada: [2026]

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: [2026, 7, 123]
  - tmmeada: [2026, 7, 123]

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-09 04:18:21

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 1
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 1
  - 当前训练命令: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fbyg15k_unsup_il.yaml --tag baseline_transfer_adapt_fbyg_expand5_s123 --stage-root transfer/transfer_adapt_fbyg_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0`

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: [7, 2026]
  - fbyg tmmeada: [7, 2026]

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: [2026, 7, 123]
  - tmmeada: [2026, 7, 123]

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-09 05:18:24

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 1
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 1
  - 当前训练命令: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fbyg15k_unsup_il.yaml --tag baseline_transfer_adapt_fbyg_expand5_s123 --stage-root transfer/transfer_adapt_fbyg_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0`

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: [7, 2026]
  - fbyg tmmeada: [7, 2026]

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: [2026, 7, 123]
  - tmmeada: [2026, 7, 123]

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-09 06:18:26

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 1
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 1
  - 当前训练命令: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fbyg15k_v8_mild_da_unsup_il.yaml --tag tmmeada_transfer_adapt_fbyg_expand5_s123 --stage-root transfer/transfer_adapt_fbyg_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0`

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: [7, 123, 2026]
  - fbyg tmmeada: [7, 2026]

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: [2026, 7, 123]
  - tmmeada: [2026, 7, 123]

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-09 07:18:30

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 1
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 1
  - 当前训练命令: `D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fbyg15k_v8_mild_da_unsup_il.yaml --tag tmmeada_transfer_adapt_fbyg_expand5_s123 --stage-root transfer/transfer_adapt_fbyg_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0`

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: [7, 123, 2026]
  - fbyg tmmeada: [7, 2026]

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: [2026, 7, 123]
  - tmmeada: [2026, 7, 123]

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-09 08:18:33

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 0
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 0

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: [7, 123, 2026]
  - fbyg tmmeada: [7, 123, 2026]

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: []
  - tmmeada: []

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-09 09:18:35

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 0
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 0

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: [7, 123, 2026]
  - fbyg tmmeada: [7, 123, 2026]

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: []
  - tmmeada: []

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

## 2026-03-09 10:18:37

- 进程状态:
  - v14_expand5_queue: 0
  - fbyg_expand5_queue: 0
  - auto_finalize_watcher: 0
  - run_transfer_train_eval: 0

- seed完成情况:
  - v14 baseline: [7, 123, 2026]
  - v14 tmmeada: [7, 123, 2026]
  - fbyg baseline: [7, 123, 2026]
  - fbyg tmmeada: [7, 123, 2026]

- v14 status(final_missing):
  - baseline: []
  - tmmeada: []
- fbyg status(final_missing):
  - baseline: []
  - tmmeada: []

- v14队列最新日志尾部:
```text
[INFO] initial tmmeada missing seeds: [2026, 7, 123]
[SKIP] baseline seed=42 already available
[SKIP] tmmeada seed=42 already available
[SKIP] baseline seed=3407 already available
[SKIP] tmmeada seed=3407 already available
[QUEUE] run baseline missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=2026
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s2026 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 2026 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=7
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s7 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 7 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run baseline missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/meaformer_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml --tag baseline_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_baseline --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[QUEUE] run tmmeada missing seed=123
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/run_transfer_train_eval.py --source-config configs/transfer/tmmeada_source_zh_en_epoch10.yaml --target-configs configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml --tag tmmeada_transfer_adapt_v14_fren_expand5_s123 --stage-root transfer/transfer_adapt_v14_fren_expand5_tmmeada --seed 123 --runner-python D:\Anaconda_envs\envs\bysj-meaformer\python.exe --target-only-test 0 --target-save-model 0
[RUN] D:\Anaconda_envs\envs\bysj-main\python.exe scripts/summarize_transfer_formal.py --baseline-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval --tmmeada-target-dir runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval --baseline-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv --tmmeada-out reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv --compare-out-csv reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv --compare-out-md reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md
[DONE] status json -> reports\transfer\transfer_adapt_v14_fren_expand5_status.json
[DONE] status md -> reports\transfer\transfer_adapt_v14_fren_expand5_status.md
```

