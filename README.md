# GP-MMEA-TL

多模态实体对齐（MMEA）毕业设计实验仓库。当前阶段目标是建立统一、可复现的实验流水线，并在 `DBP15K` 与跨图谱数据上完成 baseline 复现与 TMMEA-DA 方法原型验证。

## 1. 任务定义

- 任务：多模态实体对齐（Multimodal Entity Alignment）
- 输入：两个知识图谱中的实体及其多模态信息（结构、属性、图像等）
- 输出：跨图谱实体对应关系（alignment links）
- 当前研究主线：
  - 先复现统一 baseline（MEAformer）
  - 再加入可迁移模块（当前为 TMMEA-DA 的 Domain Align MVP）
  - 最后做多 seed 统计、对比与误差分析

## 2. 数据集

### 2.1 DBP15K（跨语言）
- `zh_en`
- `ja_en`
- `fr_en`

### 2.2 跨图谱（MMKG）
- `FBDB15K`
- `FBYG15K`

说明：
- 为避免 GitHub 大文件限制，原始数据与大特征文件未上传（见 `.gitignore`）。
- 数据来源、校验和与同步记录见：
  - `data/README.md`
  - `data/official_data_manifest.json`

## 3. 指标口径

统一报告如下双向指标（`l2r` / `r2l`）：
- `Hits@1`
- `Hits@10`
- `MRR`

当前汇总文件：
- baseline 汇总：`reports/meaformer_results_mean_std.csv`
- TMMEA-DA 汇总：`reports/tmmeada_results_mean_std.csv`
- TMMEA-DA v1（zh_en）汇总：`reports/tmmeada_v1_results_mean_std.csv`
- baseline vs TMMEA-DA 对比（全数据集）：`reports/tmmeada_vs_baseline_all.md`
- baseline/v0/v1（zh_en）三方对比：`reports/tmmeada_v1_compare_zh_en.md`

## 4. 已复现 Baselines

当前已完成并记录的 baseline：
- **MEAformer（官方实现）**
  - DBP15K：`zh_en` / `ja_en` / `fr_en`，每语种 5 seeds
  - 跨图谱：`FBDB15K` / `FBYG15K`，各 5 seeds

对应脚本与配置：
- 脚本：`scripts/run_meaformer.py`
- 多 seed（DBP）：`scripts/run_meaformer_multiseed.py`
- 多 seed（跨图谱）：`scripts/run_meaformer_crossgraph_multiseed.py`
- 配置目录：`configs/baselines/`

## 5. 我做的改动（相对 baseline）

围绕 TMMEA-DA 原型，当前完成了以下最小可运行改造：
- 在 MEAformer 中新增可开关参数：
  - `--use_domain_align`
  - `--domain_align_weight`
- 在训练损失中加入 Domain Align 项（MSE on positive pairs）
  - 文件：`baselines/MEAformer/model/MEAformer.py`
- 在 v1 版本中新增：
  - `source_select`：基于模态损失的软选择辅助项
  - `missing_gate`：仅在图像可用对上计算的缺失感知图像对齐项
  - 相关文件：`baselines/MEAformer/model/MEAformer.py`, `baselines/MEAformer/src/data.py`
- 训练入口增强：
  - 支持按 stage 输出到 `runs/<stage>/...`
  - 支持方法参数透传
  - 文件：`scripts/run_meaformer.py`
- 方法实验配置：
  - `configs/tmmeada/meaformer_zh_en_domain_align_mvp.yaml`
  - `configs/tmmeada/meaformer_ja_en_domain_align_mvp.yaml`
  - `configs/tmmeada/meaformer_fr_en_domain_align_mvp.yaml`
  - `configs/tmmeada/meaformer_fbdb15k_domain_align_mvp.yaml`
  - `configs/tmmeada/meaformer_fbyg15k_domain_align_mvp.yaml`
  - `configs/tmmeada/meaformer_zh_en_tmmeada_v1_smoke.yaml`

## 6. 运行方式

### 6.1 单次运行（示例：TMMEA-DA zh_en）
```powershell
conda run -n bysj-main python scripts\run_meaformer.py --config configs\tmmeada\meaformer_zh_en_domain_align_mvp.yaml
```

### 6.2 TMMEA-DA 多 seed（示例：ja_en）
```powershell
conda run -n bysj-main python scripts\run_tmmeada_multiseed.py --base-config configs\tmmeada\meaformer_ja_en_domain_align_mvp.yaml --seeds "42,3407,2026,7,123"
```

### 6.3 TMMEA-DA 多 seed（示例：FBDB15K）
```powershell
conda run -n bysj-main python scripts\run_tmmeada_multiseed.py --base-config configs\tmmeada\meaformer_fbdb15k_domain_align_mvp.yaml --seeds "42,3407,2026,7,123"
```

### 6.4 结果收集与聚合
```powershell
conda run -n bysj-main python scripts\collect_meaformer_results.py --runs-dir runs\tmmeada --out reports\tmmeada_results_summary.csv
conda run -n bysj-main python scripts\aggregate_meaformer_results.py --in-csv reports\tmmeada_results_summary.csv --out-csv reports\tmmeada_results_mean_std.csv
```

### 6.5 与 baseline 对比（全数据集）
```powershell
conda run -n bysj-main python scripts\make_tmmeada_baseline_compare_all.py
```

## 7. 过程留痕与报告材料

- 总过程日志：`PROCESS_LOG.md`
- 中期实验草稿：`reports/midterm_results_draft.md`
- 中期实验章节：`reports/midterm_experiment_section.md`
- 方法全数据集汇总：`reports/tmmeada_dbp15k_multilang.md`

## 8. 当前阶段结论（简要）

- 流程层面：baseline 与方法分支均已形成可复现实验链路（配置-运行-汇总-对比-报告）。
- 结果层面：TMMEA-DA 当前仅含 Domain Align MVP，在 1-epoch 设置下尚未超过 baseline。
- 下一步：补充多源选择、缺失感知融合与更完整训练预算，再进行公平对比与消融。

## 9. Update (2026-03-01): v1 Weight Sweep Follow-up

- Added sweep config/runner/summary scripts:
  - `configs/tmmeada/meaformer_zh_en_tmmeada_v1_sweep.yaml`
  - `scripts/run_tmmeada_v1_weight_sweep.py`
  - `scripts/summarize_tmmeada_v1_sweep.py`
- Ran `zh_en` coarse sweep (6 runs, seed=42):
  - grid: `dw={0.05,0.1,0.2}`, `sw={0.05,0.1}`, `mw=0.1`, `temp=1.0`
- Sweep reports:
  - `reports/tmmeada_v1_sweep_summary.csv`
  - `reports/tmmeada_v1_sweep_grouped.csv`
  - `reports/tmmeada_v1_sweep.md`
- Selected follow-up config and completed 5-seed validation:
  - config: `configs/tmmeada/meaformer_zh_en_tmmeada_v1_best.yaml`
  - stage: `runs/tmmeada_v1_best`
  - reports:
    - `reports/tmmeada_v1_best_results_summary.csv`
    - `reports/tmmeada_v1_best_results_mean_std.csv`
    - `reports/tmmeada_v1_best_compare_zh_en.csv`
    - `reports/tmmeada_v1_best_compare_zh_en.md`
- Current observation under 1-epoch quick validation:
  - `v1_best` is effectively tied with `v1`.
  - `baseline` remains clearly higher than `v0/v1/v1_best` on `zh_en`.

## 10. Update (2026-03-01): Epoch-3 Budget Pilot on zh_en

- Added epoch-3 configs:
  - `configs/baselines/meaformer_zh_en_rtx3060_safe_epoch3.yaml`
  - `configs/tmmeada/meaformer_zh_en_tmmeada_v1_best_epoch3.yaml`
- Completed single-seed (`seed=42`) pilot runs:
  - baseline: `runs/baseline_epoch3/20260301-002341-MEAformer-epoch3-DBP15K-zh_en-s42/`
  - method: `runs/tmmeada_v1_best_epoch3/20260301-005700-TMMEA-DA-v1-best-epoch3-DBP15K-zh_en-s42/`
- Pilot compare report:
  - `reports/epoch3_pilot_compare_zh_en.csv`
  - `reports/epoch3_pilot_compare_zh_en.md`
- Key observation:
  - Training budget increase (`epoch: 1 -> 3`) strongly boosts both methods.
  - Under this pilot setting, `v1_best` is approximately tied with baseline.

## 11. Update (2026-03-01): Epoch-3 Formal 5-Seed Comparison on zh_en

- Completed full 5-seed runs for both settings (`42, 3407, 2026, 7, 123`):
  - baseline: `runs/baseline_epoch3/`
  - method (`v1_best`): `runs/tmmeada_v1_best_epoch3/`
- Aggregated results:
  - `reports/baseline_epoch3_results_mean_std.csv`
  - `reports/tmmeada_v1_best_epoch3_results_mean_std.csv`
- Formal compare report:
  - `reports/epoch3_multiseed_compare_zh_en.csv`
  - `reports/epoch3_multiseed_compare_zh_en.md`
- Key conclusion:
  - Under equal `epoch=3` budget and 5 seeds, `baseline` and `TMMEA-DA v1_best` are effectively tied on `zh_en`.
