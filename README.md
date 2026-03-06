# GP-MMEA-TL

多模态实体对齐（MMEA）毕业设计实验仓库。当前阶段目标是建立统一、可复现的实验流水线，并在 `DBP15K` 与跨图谱数据上完成 baseline 复现、迁移实验（source->target）与 TMMEA-DA 的目标域自适应优化。

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
- baseline 汇总：`reports/baseline/meaformer_results_mean_std.csv`
- TMMEA-DA 汇总：`reports/tmmeada/tmmeada_results_mean_std.csv`
- TMMEA-DA v1（zh_en）汇总：`reports/tmmeada/tmmeada_v1_results_mean_std.csv`
- baseline vs TMMEA-DA 对比（全数据集）：`reports/compare/tmmeada_vs_baseline_all.md`
- baseline/v0/v1（zh_en）三方对比：`reports/tmmeada/tmmeada_v1_compare_zh_en.md`
- 迁移自适应 v6 对比（2-seed）：`reports/transfer/transfer_adapt_v6_mixed_compare_vs_baseline.csv`
- 迁移自适应 v7(FBDB) 对比（2-seed）：`reports/transfer/transfer_adapt_v7_fbdb_compare_vs_baseline.csv`
- v7 自动决策记录：`reports/transfer/transfer_adapt_v7_fbdb_decision.md`

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
- 中期实验草稿：`reports/midterm/midterm_results_draft.md`
- 中期实验章节：`reports/midterm/midterm_experiment_section.md`
- 方法全数据集汇总：`reports/tmmeada/tmmeada_dbp15k_multilang.md`
- 迁移阶段报告（最新）：`reports/transfer/transfer_stage_update_20260305.md`

## 8. 当前阶段结论（简要）

- 流程层面：baseline 与方法分支均已形成可复现实验链路（配置-运行-汇总-对比-报告）。
- 结果层面（最新）：在 transfer-adapt 2-seed 设置下，`ja_en` 与 `FBDB15K` 已出现小幅正增益。
  - `ja_en`：`delta_avg_mrr_mean = +0.00075`（v6 mixed）
  - `FBDB15K`：`delta_avg_mrr_mean = +0.00075`（v7b formal）
- 下一步：扩展到 `fr_en` / `FBYG15K` 的同口径 transfer-adapt，并补齐 5-seed 正式统计与误差分析。

## 9. 阶段更新（2026-03-01）：v1 权重搜索跟进

- 新增权重搜索配置与脚本：
  - `configs/tmmeada/meaformer_zh_en_tmmeada_v1_sweep.yaml`
  - `scripts/run_tmmeada_v1_weight_sweep.py`
  - `scripts/summarize_tmmeada_v1_sweep.py`
- 在 `zh_en` 上完成单种子粗搜索（6 组，`seed=42`）：
  - 网格：`dw={0.05,0.1,0.2}`，`sw={0.05,0.1}`，`mw=0.1`，`temp=1.0`
- 输出搜索报告：
  - `reports/tmmeada/tmmeada_v1_sweep_summary.csv`
  - `reports/tmmeada/tmmeada_v1_sweep_grouped.csv`
  - `reports/tmmeada/tmmeada_v1_sweep.md`
- 选定后续配置并完成 5-seed 验证：
  - 配置：`configs/tmmeada/meaformer_zh_en_tmmeada_v1_best.yaml`
  - 阶段：`runs/experiments/tmmeada/tmmeada_v1_best`
  - 结果：
    - `reports/tmmeada/tmmeada_v1_best_results_summary.csv`
    - `reports/tmmeada/tmmeada_v1_best_results_mean_std.csv`
    - `reports/tmmeada/tmmeada_v1_best_compare_zh_en.csv`
    - `reports/tmmeada/tmmeada_v1_best_compare_zh_en.md`
- 观察：
  - 在 1-epoch 快速设置下，`v1_best` 与 `v1` 基本持平；
  - `baseline` 仍显著高于 `v0/v1/v1_best`（zh_en）。

## 10. 阶段更新（2026-03-01）：zh_en 的 epoch=3 预算试跑

- 新增 epoch=3 配置：
  - `configs/baselines/meaformer_zh_en_rtx3060_safe_epoch3.yaml`
  - `configs/tmmeada/meaformer_zh_en_tmmeada_v1_best_epoch3.yaml`
- 完成单种子试跑（`seed=42`）：
  - baseline：`runs/experiments/baseline/baseline_epoch3/20260301-002341-MEAformer-epoch3-DBP15K-zh_en-s42/`
  - method：`runs/experiments/tmmeada/tmmeada_v1_best_epoch3/20260301-005700-TMMEA-DA-v1-best-epoch3-DBP15K-zh_en-s42/`
- 输出试跑对比：
  - `reports/epoch3/epoch3_pilot_compare_zh_en.csv`
  - `reports/epoch3/epoch3_pilot_compare_zh_en.md`
- 观察：
  - 训练预算从 `epoch=1` 提升到 `epoch=3` 后，双方性能均明显提升；
  - 在该试跑设置下，`v1_best` 与 baseline 基本持平。

## 11. 阶段更新（2026-03-01）：zh_en 的 epoch=3 正式 5-seed 对比

- 在相同预算下完成 `42, 3407, 2026, 7, 123` 五个种子：
  - baseline：`runs/experiments/baseline/baseline_epoch3/`
  - method（`v1_best`）：`runs/experiments/tmmeada/tmmeada_v1_best_epoch3/`
- 聚合结果：
  - `reports/baseline/baseline_epoch3_results_mean_std.csv`
  - `reports/tmmeada/tmmeada_v1_best_epoch3_results_mean_std.csv`
- 正式对比报告：
  - `reports/epoch3/epoch3_multiseed_compare_zh_en.csv`
  - `reports/epoch3/epoch3_multiseed_compare_zh_en.md`
- 结论：
  - 在 `epoch=3 + 5-seed` 的公平设置下，`baseline` 与 `TMMEA-DA v1_best` 在 `zh_en` 基本持平。

## 12. 阶段更新（2026-03-01）：扩展到 ja_en / fr_en（试跑）

- 新增 DBP15K `ja_en` 与 `fr_en` 的 epoch=3 配置（baseline + method）。
- 在两种语言上完成 `seed=42` 试跑。
- 更新聚合文件（`zh_en` 为 5-seed，`ja_en/fr_en` 为试跑）：
  - `reports/baseline/baseline_epoch3_results_mean_std.csv`
  - `reports/tmmeada/tmmeada_v1_best_epoch3_results_mean_std.csv`
- 新增 DBP15K epoch3 对比：
  - `reports/epoch3/epoch3_compare_dbp15k.csv`
  - `reports/epoch3/epoch3_compare_dbp15k.md`
- 阶段观察：
  - `zh_en` 正式结果与 `ja_en/fr_en` 试跑结果均显示两方法接近。

## 13. 阶段更新（2026-03-02）：DBP15K epoch=3 全语种正式 5-seed 完成

- 为 `fr_en` 补齐剩余四个种子（`3407, 2026, 7, 123`），baseline 与 method 同步完成。
- 更新后的 DBP15K epoch3 结果文件：
  - `reports/baseline/baseline_epoch3_results_summary.csv`
  - `reports/baseline/baseline_epoch3_results_mean_std.csv`
  - `reports/tmmeada/tmmeada_v1_best_epoch3_results_summary.csv`
  - `reports/tmmeada/tmmeada_v1_best_epoch3_results_mean_std.csv`
  - `reports/epoch3/epoch3_compare_dbp15k.csv`
  - `reports/epoch3/epoch3_compare_dbp15k.md`
- 同步修复说明文本：
  - `scripts/make_epoch3_compare_dbp15k.py` 改为根据真实 `num_runs` 自动生成注释，避免“结果已 5-seed 但文本仍写 pilot”的不一致。
- 结论：
  - 在 `zh_en/ja_en/fr_en` 三语种上，`baseline` 与 `v1_best` 仍基本持平。

## 14. 阶段更新（2026-03-02）：跨图谱 epoch=3 试跑（FBDB15K/FBYG15K）

- 新增跨图谱 epoch=3 配置（baseline + `v1_best`）：
  - `configs/baselines/meaformer_fbdb15k_rtx3060_safe_epoch3.yaml`
  - `configs/baselines/meaformer_fbyg15k_rtx3060_safe_epoch3.yaml`
  - `configs/tmmeada/meaformer_fbdb15k_tmmeada_v1_best_epoch3.yaml`
  - `configs/tmmeada/meaformer_fbyg15k_tmmeada_v1_best_epoch3.yaml`
- 完成 `seed=42` 试跑：
  - baseline：`runs/experiments/baseline/baseline_epoch3_crossgraph/`
  - method：`runs/experiments/tmmeada/tmmeada_v1_best_epoch3_crossgraph/`
- 输出试跑对比：
  - `reports/baseline/baseline_epoch3_crossgraph_results_mean_std.csv`
  - `reports/tmmeada/tmmeada_v1_best_epoch3_crossgraph_results_mean_std.csv`
  - `reports/epoch3/epoch3_compare_crossgraph.csv`
  - `reports/epoch3/epoch3_compare_crossgraph.md`
- 观察：
  - `FBDB15K`：`v1_best` 相比 baseline 有极小正增益；
  - `FBYG15K`：两者近似持平。

## 15. 阶段更新（2026-03-02）：跨图谱 epoch=3 正式 5-seed 完成

- 在 `FBDB15K` 与 `FBYG15K` 上补齐 `3407, 2026, 7, 123`，与 `seed=42` 共同形成正式 5-seed。
- 阶段目录：
  - baseline：`runs/experiments/baseline/baseline_epoch3_crossgraph/`
  - method：`runs/experiments/tmmeada/tmmeada_v1_best_epoch3_crossgraph/`
- 结果文件：
  - `reports/baseline/baseline_epoch3_crossgraph_results_summary.csv`
  - `reports/baseline/baseline_epoch3_crossgraph_results_mean_std.csv`
  - `reports/tmmeada/tmmeada_v1_best_epoch3_crossgraph_results_summary.csv`
  - `reports/tmmeada/tmmeada_v1_best_epoch3_crossgraph_results_mean_std.csv`
  - `reports/epoch3/epoch3_compare_crossgraph.csv`
  - `reports/epoch3/epoch3_compare_crossgraph.md`
- 正式 5-seed 观察：
  - `FBDB15K`：`v1_best` 小幅优于 baseline；
  - `FBYG15K`：`v1_best` 小幅优于 baseline。
- 当前范围状态：
  - `DBP15K` epoch3 正式 5-seed 已完成（`zh_en`, `ja_en`, `fr_en`）；
  - 跨图谱 epoch3 正式 5-seed 已完成（`FBDB15K`, `FBYG15K`）。

## 16. 阶段更新（2026-03-02）：zh_en 模块消融（epoch=3，seed=42）

- 新增三组消融配置：
  - `configs/tmmeada/meaformer_zh_en_tmmeada_v1_best_epoch3_wo_domain_align.yaml`
  - `configs/tmmeada/meaformer_zh_en_tmmeada_v1_best_epoch3_wo_source_select.yaml`
  - `configs/tmmeada/meaformer_zh_en_tmmeada_v1_best_epoch3_wo_missing_gate.yaml`
- 完成三组消融运行（stage: `runs/experiments/tmmeada/tmmeada_v1_ablation_epoch3/`）。
- 新增消融汇总与对比：
  - `reports/tmmeada/tmmeada_v1_ablation_epoch3_results_summary.csv`
  - `reports/epoch3/epoch3_ablation_zh_en.csv`
  - `reports/epoch3/epoch3_ablation_zh_en.md`
- 试跑观察（zh_en + epoch3 + seed42）：
  - 三组消融与 `v1_best_full` 差异极小，`wo_source_select` 在 `l2r Hits@1` 上有 `-0.0006` 的微弱回落；
  - 整体显示当前增益量级较小，需扩展到 5-seed 才能形成稳定结论。

## 17. 阶段更新（2026-03-02）：zh_en 模块消融正式 5-seed 完成

- 在 `wo_domain_align / wo_source_select / wo_missing_gate` 三组上补齐 `3407, 2026, 7, 123`，与 `seed=42` 合并为正式 5-seed。
- 消融阶段目录：
  - `runs/experiments/tmmeada/tmmeada_v1_ablation_epoch3/`
- 更新文件：
  - `reports/tmmeada/tmmeada_v1_ablation_epoch3_results_summary.csv`（15 runs）
  - `reports/epoch3/epoch3_ablation_zh_en_multiseed.csv`
  - `reports/epoch3/epoch3_ablation_zh_en_multiseed.md`
  - `scripts/summarize_epoch3_ablation_zh_en_multiseed.py`
- 正式 5-seed 观察：
  - `wo_domain_align` 与 `v1_best_full` 基本一致；
  - `wo_missing_gate` 与 `v1_best_full` 基本一致；
  - `wo_source_select` 与 baseline 更接近，且相对 `v1_best_full` 仅有极小差值（约 `r2l H@1 -0.0001` 量级）。
- 结论：
  - 在当前 `epoch=3 + zh_en` 设置下，三模块开关带来的平均差异非常小，整体仍与 baseline 近似持平。

## 18. 阶段更新（2026-03-05）：Transfer-Adapt v3-v7（2-seed）完成

- 目标：面向任务书的“可迁移能力”主线，执行 `source=zh_en -> target` 的目标域无标注自适应实验。
- 覆盖阶段：
  - `v3` / `v4` / `v5`：持续调节 `ja_en` 与 `FBDB15K` 的自适应策略；
  - `v6_mixed`：按目标域拆分策略（`ja_en` 保持强配置，`FBDB15K` 使用 baseline source + mild DA）；
  - `v7_fbdb`：针对 `FBDB15K` 进行 `v7a/v7b/v7c` 自动试跑与自动决策。
- 关键结果（2-seed）：
  - `v6_mixed` vs baseline：
    - `ja_en`：`delta_avg_mrr_mean = +0.00075`
    - `FBDB15K`：`delta_avg_mrr_mean = +0.00000`（持平）
  - `v7_fbdb`（best variant=`v7b`）vs baseline：
    - `FBDB15K`：`delta_avg_mrr_mean = +0.00075`
    - `delta_avg_hits@1_mean = +0.000875`
    - `delta_avg_hits@10_mean = +0.001425`
- 产出文件：
  - `reports/transfer/transfer_adapt_v6_mixed_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v7_fbdb_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v7_fbdb_decision.{md,json}`
  - `runs/transfer/transfer_adapt_v7_fbdb_formal_v7b/`

## 19. 阶段更新（2026-03-05）：README 与阶段报告同步

- README 已同步到 transfer-adapt 最新进度（v7）。
- 新增阶段报告：
  - `reports/transfer/transfer_stage_update_20260305.md`

## 20. 阶段更新（2026-03-06）：Transfer-Adapt v8 扩展（s42）

- 扩展目标域：`fr_en` 与 `FBYG15K`（transfer-adapt）。
- 产出对比文件：
  - `reports/transfer/transfer_adapt_v8_expand_s42_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v8_expand_s42_compare_vs_baseline.md`
  - `reports/transfer/transfer_stage_update_20260306_v8_s42.md`
- s42 关键结果：
  - `FBYG15K`：`delta_avg_mrr_mean = +0.0010`（小幅正增益）
  - `fr_en`：`delta_avg_mrr_mean = -0.0005`（轻微回落）
- 说明：
  - `seed=3407` 在本轮为保证阶段交付时效已中止，后续可继续补齐 2-seed 正式统计。

## 21. 阶段更新（2026-03-06）：Transfer-Adapt v8 扩展（2-seed 完成）

- 已补齐 `seed=3407` 并合并 `s42+s3407` 形成正式 2-seed 对比。
- 结果文件：
  - `reports/transfer/transfer_adapt_v8_expand_2seed_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v8_expand_2seed_compare_vs_baseline.md`
  - `reports/transfer/transfer_stage_update_20260306_v8_2seed.md`
- 2-seed 关键结论：
  - `FBYG15K`：`delta_avg_mrr_mean = +0.00075`（稳定正增益）
  - `fr_en`：`delta_avg_mrr_mean = -0.00075`（仍轻微负迁移）

## 22. 阶段更新（2026-03-06）：Transfer-Adapt v9（fr_en 定向优化）

- 执行方式：`pilot(2变体, s42) -> 自动选优 -> formal(s3407)`。
- 输出文件：
  - `reports/transfer/transfer_adapt_v9_fren_decision.{md,json}`
  - `reports/transfer/transfer_adapt_v9_fren_2seed_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v9_fren_2seed_compare_vs_v8.csv`
  - `reports/transfer/transfer_stage_update_20260306_v9_fren.md`
- 结果摘要（fr_en, 2-seed）：
  - 相比 baseline：`delta_avg_mrr_mean = -0.00025`
  - 相比 v8 tmmeada：`delta_avg_mrr_mean = +0.00050`
- 结论：
  - v9 相比 v8 有改善，但尚未反超 baseline。

