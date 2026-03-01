# TMMEA-DA v1 sweep follow-up (zh_en, 1-epoch quick setting)

## 1. Sweep setup
- Stage: `runs/tmmeada_v1_sweep`
- Dataset: `DBP15K/zh_en`
- Seeds: `42` (coarse search)
- Search grid:
  - `domain_align_weight`: `0.05, 0.1, 0.2`
  - `source_select_weight`: `0.05, 0.1`
  - `missing_align_weight`: `0.1`
  - `source_select_temp`: `1.0`
- Total runs: `6`

Artifacts:
- `reports/tmmeada_v1_sweep_summary.csv`
- `reports/tmmeada_v1_sweep_grouped.csv`
- `reports/tmmeada_v1_sweep.md`

## 2. Sweep result
- Top grouped settings (avg Hits@1) all point to `source_select_weight=0.05`.
- Representative top config selected for follow-up:
  - `domain_align_weight=0.1`
  - `source_select_weight=0.05`
  - `missing_align_weight=0.1`
  - `source_select_temp=1.0`

## 3. Multi-seed verification for selected config
- Stage: `runs/tmmeada_v1_best`
- Config: `configs/tmmeada/meaformer_zh_en_tmmeada_v1_best.yaml`
- Seeds: `42, 3407, 2026, 7, 123`
- Total runs: `5`

Artifacts:
- `reports/tmmeada_v1_best_results_summary.csv`
- `reports/tmmeada_v1_best_results_mean_std.csv`
- `reports/tmmeada_v1_best_compare_zh_en.csv`
- `reports/tmmeada_v1_best_compare_zh_en.md`

## 4. Key metrics (mean +/- std, 5 seeds)
- `v1_best` l2r: `Hits@1=0.5524 +/- 0.0056`, `Hits@10=0.8409 +/- 0.0016`, `MRR=0.6492 +/- 0.0035`
- `v1_best` r2l: `Hits@1=0.5530 +/- 0.0025`, `Hits@10=0.8404 +/- 0.0021`, `MRR=0.6490 +/- 0.0025`

Comparison:
- `v1_best` vs `v1`: near-identical under 1-epoch validation (`delta` around `0.0000~0.0001`).
- `v1_best`/`v1`/`v0` remain below current baseline in this quick setting.

## 5. Next action
- Move from 1-epoch quick validation to longer training budget for fair method judgement.
- Keep sweep-selected setting as a stable default candidate for the next training-budget step.

## 6. Epoch-3 pilot (zh_en, seed=42)
- Baseline run: `runs/baseline_epoch3/20260301-002341-MEAformer-epoch3-DBP15K-zh_en-s42/`
- Method run: `runs/tmmeada_v1_best_epoch3/20260301-005700-TMMEA-DA-v1-best-epoch3-DBP15K-zh_en-s42/`
- Compare artifacts:
  - `reports/baseline_epoch3_results_summary.csv`
  - `reports/tmmeada_v1_best_epoch3_results_summary.csv`
  - `reports/epoch3_pilot_compare_zh_en.csv`
  - `reports/epoch3_pilot_compare_zh_en.md`
- Pilot conclusion:
  - Both methods improve strongly vs 1-epoch quick setting.
  - `v1_best` and baseline are nearly tied under epoch=3 single-seed pilot.

## 7. Epoch-3 formal 5-seed comparison (zh_en)
- Baseline aggregate: `reports/baseline_epoch3_results_mean_std.csv`
- Method aggregate: `reports/tmmeada_v1_best_epoch3_results_mean_std.csv`
- Compare artifacts:
  - `reports/epoch3_multiseed_compare_zh_en.csv`
  - `reports/epoch3_multiseed_compare_zh_en.md`
- Final observation under epoch=3 multi-seed:
  - baseline and v1_best are effectively tied on Hits@1/Hits@10/MRR.
