# GP-MMEA-TL

多模态实体对齐（MMEA）毕业设计实验仓库。当前阶段目标是建立统一、可复现的实验流水线，并在 `DBP15K` 与跨图谱数据上完成 baseline 复现、迁移实验（source->target）与 TMMEA-DA 的目标域自适应优化。

## 外行导读

这个项目想解决的问题是：同一个现实世界对象，可能在两个不同知识图谱里被写成两个不同条目。名字可能不同，语言可能不同，图片和属性也可能不完整。模型需要根据结构、属性、图像等多种信息，自动判断“这两个条目是不是其实指向同一个对象”。

这个仓库记录的不是一次单独跑分，而是一整条研究链路：我先复现官方 baseline，再加入自己的方法模块，然后把方法做成可迁移的版本，最后对每个目标任务持续优化、比较和留痕。

- 研究任务：多模态实体对齐（Multimodal Entity Alignment）。
- 输入：两个知识图谱中的实体，以及它们的结构关系、属性、图像等多模态信息。
- 输出：两个图谱中哪些实体其实是同一个真实对象。
- 这项工作的难点不只是“对齐准不准”，还包括“能不能把在一个数据集上学到的能力迁移到另一个数据集上”。

## 怎么理解文中的常见词

| 术语 | 通俗解释 |
|---|---|
| `baseline` | 官方原始模型，用来做公平对照 |
| `seed` | 随机种子。换不同 seed 重复跑，是为了避免“只碰巧跑好一次” |
| `pilot` | 小规模试跑，通常先跑 1 个或 2 个 seed 看方向对不对 |
| `full5` / `expand5` | 扩展到 5 个 seed 的正式结果 |
| `source -> target` | 先在一个数据集上学，再迁移到另一个数据集上测试或自适应 |
| `IL` | 迭代式伪标签或伪链接生成，模型先猜一批链接，再拿它们继续训练 |
| `strict-source` | 强制每个 seed 只用严格对应的 source checkpoint，避免混用旧模型 |
| `delta_avg_mrr_mean` | 方法相对 baseline 的平均提升值。大于 0 说明方法优于 baseline |

## 项目路线概览

| 阶段 | 主要在做什么 | 这一步想回答什么 | 目前结论 |
|---|---|---|---|
| Baseline 复现 | 把 `MEAformer` 在 5 个数据集上稳定跑通 | 没有可靠 baseline，后面所有改进都站不住 | baseline 已全部复现完成 |
| TMMEA-DA MVP / v1 | 加入 `domain align`、`source_select`、`missing_gate` 等模块 | 自己的方法模块是否有潜力 | 早期模块在公平预算下大多与 baseline 接近 |
| Transfer 链路搭建 | 建立 `source_train -> target_eval` 流程 | 模型是否具备可迁移能力 | 迁移实验已经从 smoke 走到 formal |
| `ja_en` / `fr_en` 优化 | 反复调整自适应节奏、IL 刷新与轻量模块 | 跨语言迁移能否稳定增益 | `ja_en` 和 `fr_en` 最终都拿到了明显正增益 |
| `FBDB15K` 优化 | 从调权重转向改伪种子质量 | 跨图谱噪声是不是主要瓶颈 | `v18` 证明更干净的伪种子是关键 |
| `FBYG15K` 优化 | 从晚启 IL、静态过滤一路试到 staged fresh-IL、strict-source 与 adaptive top-k | 怎样在噪声较大的跨图谱场景里稳定提升 | `v24` 证明主线有效，`v25` 说明单纯 adaptive top-k 还不够 |
| 主表收口 | 汇总 4 个目标任务的正式结果 | 最后到底有没有形成稳定结论 | 当前 4 个目标均为 `5-seed` 正增益 |

## 当前正式结果（2026-03-14）

当前统一主表文件：
- `reports/transfer/transfer_adapt_main_results_4target.md`
- `reports/transfer/transfer_adapt_main_results_4target.csv`

其中 `delta_avg_mrr_mean > 0` 表示相对 baseline 有提升。当前 4 个目标都已经是 `5-seed` 正式口径：

| 目标 | 场景 | 当前主表版本 | `delta_avg_mrr_mean` | 通俗理解 |
|---|---|---|---:|---|
| `ja_en` | 跨语言 | `v15_refresh4_da0025_expand5` | `+0.01210` | 提升明显，说明迁移到日英任务是有效的 |
| `FBDB15K` | 跨图谱 | `v18c_bipartite_late_il_skiprel_expand5` | `+0.00830` | 提升稳定，关键突破来自更干净的伪种子 |
| `fr_en` | 跨语言 | `v14b_refresh4_da0025_expand5` | `+0.01210` | 从多轮失败里找到有效方向，最终提升明显 |
| `FBYG15K` | 跨图谱 | `v24b_strictsrc_staged_fresh_il_top400_expand5` | `+0.00280` | 提升幅度较小，但已经在严格口径下验证稳定成立 |

当前 4 目标平均提升：
- `delta_avg_hits@1_mean = +0.006897`
- `delta_avg_hits@10_mean = +0.012650`
- `delta_avg_mrr_mean = +0.008825`
- `delta_avg_mr_mean = -66.674325`

## 最建议先看的文件

- `README.md`
  - 首页导览，适合先建立整体认识
- `reports/notes/taskbook_gap_assessment_20260315.md`
  - 任务书 / 开题报告 / 当前项目现状的闭环检查与差距判断
- `PROJECT_OPERATION_RECORD.md`
  - 面向论文与答辩的全流程记录，说明每个阶段为什么做、做了什么、结果怎样
- `PROCESS_LOG.md`
  - 更细的原始过程日志，适合追查具体执行过程
- `reports/transfer/transfer_adapt_main_results_4target.md`
  - 当前最重要的正式主结果表
- `reports/transfer/transfer_adapt_error_bucket_summary.md`
  - 主结果之外的误差分桶分析
- `reports/transfer/transfer_stage_update_20260314_fbyg_v25_adaptive_topk_pilot.md`
  - 截至目前最新的阶段收口报告

以下内容开始进入“技术版说明”，会更偏向实验、脚本与结果口径。

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
- 迁移阶段报告（最新）：`reports/transfer/transfer_stage_update_20260314_fbyg_v25_adaptive_topk_pilot.md`
- 显著性与置信区间补强：
  - `reports/transfer/transfer_adapt_significance_summary.md`
  - `reports/transfer/transfer_adapt_significance_summary.csv`
  - `reports/transfer/transfer_adapt_significance_writeup.md`
- 案例级成功/失败分析：
  - `reports/transfer/transfer_case_analysis_examples.md`
  - `reports/transfer/transfer_case_analysis_examples.csv`
- 效率补证（当前先完成 wall-clock）：
  - `reports/transfer/transfer_efficiency_summary.md`
  - `reports/transfer/transfer_efficiency_summary.csv`
- 项目闭环评估与边界说明：
  - `reports/notes/taskbook_gap_assessment_20260315.md`
  - `reports/notes/mainline_traceability_matrix_20260315.md`
  - `reports/transfer/README.md`
  - `runs/transfer/README.md`

## 8. 当前阶段结论（简要）

- 流程层面：baseline 与方法分支均已形成可复现实验链路（配置-运行-汇总-对比-报告）。
- 结果层面（最新）：在当前 4 目标主结果表中，`ja_en`、`FBDB15K`、`fr_en`、`FBYG15K` 均为正增益。
  - `ja_en`：`delta_avg_mrr_mean = +0.01210`（v15 refresh4 da0025, 5-seed）
  - `FBDB15K`：`delta_avg_mrr_mean = +0.00830`（v18c bipartite late_il skiprel, 5-seed）
  - `fr_en`：`delta_avg_mrr_mean = +0.01210`（v14b, 5-seed）
  - `FBYG15K`：`delta_avg_mrr_mean = +0.00280`（v24b strict-source staged fresh_il top400, 5-seed）
- 置信度说明：`ja_en/FBDB15K/fr_en/FBYG15K` 当前均为 `5-seed` 正式口径。
- 证据链补强（2026-03-14 已新增）：
  - 显著性：4 个目标域在 `avg MRR` 上均为 `5/5 seed` 正增益，paired bootstrap `95% CI` 下界均大于 0，one-sided sign test / Wilcoxon 均为 `p=0.03125`。
  - 案例：已补出 `8` 个案例，覆盖 `ja_en` 失败边界案例与 `FBDB15K/FBYG15K` 成功纠错案例。
  - 效率：已可从 formal `log.txt` 统一汇总 wall-clock；GPU 峰值显存仍需一次最小代价补测。
- 项目接管判断（2026-03-15）：
  - 任务书 / 开题报告主线已基本闭环，当前优先工作是材料规范化与可追溯整理，而不是继续追加主线 rerun。
  - `H3` 相关脚本、结果与目录已从当前仓库移除，后续仅在主线完全收口后再单独重启。
  - GPU 峰值显存当前只有脚本入口与失败 / dry-run 尝试，尚无可入文正式汇总；后续需按修正后的最小补测脚本重跑。
- 方法优化最新判断：
  - `FBDB15K` 的 `P1` 伪种子质量改造已验证成功，当前主表版本切换为 `v18c`；
  - `FBYG15K` 的 `v25` 已验证 adaptive top-k 机制确实生效，但最优 pilot `v25c (+0.00250)` 仍未超过当前主表 `v24b (+0.00280)`，因此主表保持 `v24b`。
- 下一步：基于当前统一的 4 目标 `5-seed` 主表整理终稿主结果章节；若继续做方法优化，`FBYG15K` 更适合沿 `v24b` 尝试 `phase-wise consistency constraints`，而不是继续单独扫 adaptive top-k。

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
  - `reports/transfer/transfer_stage_update_20260307_v13_fren.md`

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


## 23. 阶段更新（2026-03-06）：Transfer-Adapt v10（fr_en 自动优化）

- 新增配置：
  - `configs/transfer_adapt/tmmeada_target_fr_en_v10a_unsup900.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v10b_da0025.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v10c_da0035.yaml`
- 新增自动化脚本：`scripts/run_transfer_adapt_v10_fren_auto.py`
- 执行流程：`pilot(3变体,s42) -> 自动选优 -> formal(s3407) -> 2-seed汇总`
- 自动决策：
  - best variant：`v10b_da0025`
  - 决策文件：`reports/transfer/transfer_adapt_v10_fren_decision.md`
- 最终结果（fr_en, 2-seed）：
  - vs baseline：`delta_avg_mrr_mean = -0.00025`
  - vs v9：`delta_avg_mrr_mean = 0.00000`（持平）
- 对应结果文件：
  - `reports/transfer/transfer_adapt_v10_fren_2seed_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v10_fren_2seed_compare_vs_v9.csv`
  - `reports/transfer/transfer_stage_update_20260307_v13_fren.md`

## 24. 阶段更新（2026-03-07）：Transfer-Adapt v12（fr_en 回稳优化）

- 背景：`v11` 的伪标签过滤导致 `fr_en` 明显退化（2-seed `delta_avg_mrr_mean = -0.06475`）。
- 本阶段新增：
  - `configs/transfer_adapt/tmmeada_target_fr_en_v12a_recover_v10.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v12b_mild_filter_highkeep.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v12c_mild_filter_da03.yaml`
  - `scripts/run_transfer_adapt_v12_fren_auto.py`
- 自动流程：`pilot(3变体,s42) -> 自动选优 -> formal(s3407) -> 2-seed汇总`。
- 自动决策：
  - best variant：`v12a_recover_v10`
  - pilot deltas（vs baseline, `delta_avg_mrr_mean`）：
    - v12a: `-0.00100`
    - v12b: `-0.01700`
    - v12c: `-0.02200`
- 最终 2-seed（fr_en）：
  - vs baseline：`delta_avg_mrr_mean = -0.00025`
  - vs v10：`delta_avg_mrr_mean = 0.00000`（持平）
- 结论：
  - `v12` 已恢复到 `v10` 水平，过滤版仍未带来额外增益。
- 相关文件：
  - `reports/transfer/transfer_adapt_v12_fren_decision.{md,json}`
  - `reports/transfer/transfer_adapt_v12_fren_2seed_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v12_fren_2seed_compare_vs_v10.csv`
  - `reports/transfer/transfer_stage_update_20260307_v13_fren.md`

## 25. 阶段更新（2026-03-07）：Transfer-Adapt v13（fr_en 轻量模块优化）

- 目标：在 `v12` 回稳基础上，验证低权重 `source_select` / `missing_gate` 是否能带来增益。
- 新增文件：
  - `configs/transfer_adapt/tmmeada_target_fr_en_v13a_source_select_mild.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v13b_missing_gate_mild.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v13c_hybrid_mild.yaml`
  - `scripts/run_transfer_adapt_v13_fren_auto.py`
- 自动流程：`pilot(3变体,s42) -> 自动选优 -> formal(s3407) -> 2-seed汇总`。
- pilot 结果（vs baseline, `delta_avg_mrr_mean`）：
  - `v13a_source_select_mild`: `-0.00100`
  - `v13b_missing_gate_mild`: `-0.00100`
  - `v13c_hybrid_mild`: `-0.00100`
- 选优：`v13a_source_select_mild`
- 最终 2-seed（fr_en）：
  - vs baseline：`delta_avg_mrr_mean = -0.00025`
  - vs v12：`delta_avg_mrr_mean = 0.00000`（持平）
- 结论：
  - v13 与 v12/v10 持平，未出现新增益。
- 相关文件：
  - `reports/transfer/transfer_adapt_v13_fren_decision.{md,json}`
  - `reports/transfer/transfer_adapt_v13_fren_2seed_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v13_fren_2seed_compare_vs_v12.csv`
  - `reports/transfer/transfer_stage_update_20260307_v13_fren.md`

## 26. 阶段更新（2026-03-08）：Transfer-Adapt v14（fr_en IL 刷新频率优化）

- 目标：优化伪标签更新节奏，降低中后期噪声积累，争取 `fr_en` 可迁移指标实增。
- 代码改造：
  - `baselines/MEAformer/config.py`：新增参数 `--il_refresh_interval`
  - `baselines/MEAformer/main.py`：将伪标签刷新条件从固定 `epoch*10` 改为可配置刷新间隔
  - `scripts/run_meaformer.py`：支持透传 `il_refresh_interval`
- 新增配置与自动化：
  - `configs/transfer_adapt/tmmeada_target_fr_en_v14a_refresh5_da0025.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v14c_refresh5_da0030.yaml`
  - `scripts/run_transfer_adapt_v14_fren_auto.py`
- 自动流程：`pilot(3变体,s42) -> 自动选优 -> formal(s3407) -> 2-seed汇总`
- pilot 结果（vs baseline, `delta_avg_mrr_mean`）：
  - `v14a_refresh5_da0025`: `-0.00100`
  - `v14b_refresh4_da0025`: `+0.01050`
  - `v14c_refresh5_da0030`: `-0.00100`
- 自动选优：`v14b_refresh4_da0025`
- 最终 2-seed（fr_en）：
  - vs baseline：`delta_avg_mrr_mean = +0.01075`
  - vs v13：`delta_avg_mrr_mean = +0.01100`
- 结论：
  - `v14` 在 `fr_en` 上取得当前阶段最明显的正增益，后续应优先扩展到 `5-seed` 以验证稳定性。
- 相关文件：
  - `reports/transfer/transfer_adapt_v14_fren_decision.{md,json}`
  - `reports/transfer/transfer_adapt_v14_fren_2seed_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_v14_fren_2seed_compare_vs_v13.csv`
  - `reports/transfer/transfer_stage_update_20260308_v14_fren.md`

## 27. 阶段更新（2026-03-08）：Transfer-Adapt v14 扩展到 5-seed（已启动断点续跑）

- 新增断点续跑脚本：
  - `scripts/run_transfer_adapt_v14_fren_expand5_resume.py`
- 目标：
  - 将 `fr_en` 从 2-seed 扩展到 5-seed：`42,3407,2026,7,123`
- 脚本特点：
  - 自动识别已完成 seed（含 fallback 历史目录）
  - 仅运行缺失 seed（当前缺失：`2026,7,123`）
  - 每轮完成后自动重建 merged 目录并输出 compare 报表
- 当前状态文件：
  - `reports/transfer/transfer_adapt_v14_fren_expand5_status.json`
  - `reports/transfer/transfer_adapt_v14_fren_expand5_status.md`
  - `reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv`
- 启动报告：
  - `reports/transfer/transfer_stage_update_20260308_v14_expand5_launch.md`

## 28. 阶段更新（2026-03-09）：Transfer-Adapt expand5 收官（fr_en + FBYG15K）

- 已完成 `fr_en(v14b)` 与 `FBYG15K(v8)` 的 5-seed 正式统计（`42,3407,2026,7,123`）。
- 完成状态文件：
  - `reports/transfer/transfer_adapt_v14_fren_expand5_status.{md,json}`
  - `reports/transfer/transfer_adapt_fbyg_expand5_status.{md,json}`
- 最终 compare 文件：
  - `reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv`
  - `reports/transfer/transfer_adapt_fbyg_expand5_progress_compare_vs_baseline.csv`
- 核心结果（5-seed）：
  - `fr_en`：`delta_avg_mrr_mean = +0.01210`
  - `FBYG15K`：`delta_avg_mrr_mean = +0.00110`
- 新增阶段报告：
  - `reports/transfer/transfer_stage_update_20260309_fbyg_expand5_final.md`

## 29. 阶段更新（2026-03-09）：4目标统一主结果表与误差分桶分析

- 新增自动生成脚本：
  - `scripts/make_transfer_main_and_bucket_report.py`
- 新增主结果表（当前最佳变体，4目标）：
  - `reports/transfer/transfer_adapt_main_results_4target.csv`
  - `reports/transfer/transfer_adapt_main_results_4target.md`
- 新增误差分桶分析：
  - `reports/transfer/transfer_adapt_error_bucket_summary.csv`
  - `reports/transfer/transfer_adapt_error_bucket_summary.md`
- 4目标总体平均改进（非加权）：
  - `delta_avg_hits@1_mean = +0.003210`
  - `delta_avg_hits@10_mean = +0.005381`
  - `delta_avg_mrr_mean = +0.003675`
  - `delta_avg_mr_mean = -9.080300`
- 阶段报告：
  - `reports/transfer/transfer_stage_update_20260309_main_table_bucket.md`

## 30. 阶段更新（2026-03-09）：ja_en + FBDB15K 扩展收口

- 扩展脚本：`scripts/run_transfer_adapt_ja_fbdb_expand5_next.py`
- 通用断点续跑：`scripts/run_transfer_adapt_expand5_resume_generic.py`
- 自动汇总主表与分桶：`scripts/make_transfer_main_and_bucket_report.py`
- 最终阶段报告：`reports/transfer/transfer_stage_update_20260309_ja_fbdb_expand5_final.md`

## 31. 阶段更新（2026-03-11）：ja_en v15 正式 5-seed 完成

- 完成 `ja_en v15` 的缺失 seeds 补跑：`3407, 7, 123`
- 最终 5-seed compare：
  - `reports/transfer/transfer_adapt_ja_v15_expand5_compare_vs_baseline.csv`
- 最终结果（ja_en）：
  - `delta_avg_hits@1_mean = +0.01094`
  - `delta_avg_hits@10_mean = +0.01410`
  - `delta_avg_mrr_mean = +0.01210`
- 新增恢复与收口文件：
  - `reports/transfer/transfer_stage_update_20260311_ja_v15_takeover.md`
  - `reports/transfer/transfer_stage_update_20260311_ja_v15_final.md`
- 同步修复：
  - `scripts/summarize_transfer_formal.py` 仅统计 `[DONE] return_code=0` 的完整 run
  - `scripts/run_transfer_adapt_expand5_resume_generic.py` 与相关续跑脚本改为跳过中断 run
  - `scripts/make_transfer_main_and_bucket_report.py` 将 `ja_en` 主表条目切换到 `v15`

## 32. 阶段更新（2026-03-12）：FBDB15K v17 噪声控制 pilot 完成

- 目标：验证 `FBDB15K` 的下一步优化是否应优先抑制伪标签噪声，而不是继续微调 `domain_align_weight`。
- 新增配置与自动化：
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v17a_no_il_balanced.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v17b_late_il_strict.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v17c_late_il_skiprel.yaml`
  - `scripts/run_transfer_adapt_v17_fbdb_iter_queue.py`
- pilot 结果（vs baseline，`delta_avg_mrr_mean`）：
  - `v17a`: `-0.00800`
  - `v17b`: `-0.00850`
  - `v17c`: `-0.00775`
- 诊断：
  - `v17` 初始 visual seeds 真值率提升到约 `5.67%`，但仍明显偏低；
  - `v17b/v17c` 的严格晚启 IL 在日志中 `raw=0 kept=0`，说明瓶颈已经前移到初始 seeds，而不是 IL 调度。
- 结论：
  - `FBDB15K` 继续做 `P0` 风格调参意义不大；
  - 下一步应直接进入 `P1`，改 `baselines/MEAformer/src/data.py::visual_pivot_induction` 的伪种子生成机制。
- 阶段报告：
  - `reports/transfer/transfer_stage_update_20260312_v17_fbdb_noise_control.md`

## 33. 阶段更新（2026-03-12）：FBDB15K v18 bipartite seeds 正式 5-seed 完成

- 目标：将 `FBDB15K` 从 `P0` 的配置调参切换到 `P1` 的伪种子质量改造，并验证是否能够稳定超过当前主表版本。
- 核心代码改造：
  - `baselines/MEAformer/src/data.py`：新增 `mutual nearest + margin + no fallback + unsup_k_max`
  - `baselines/MEAformer/config.py`：新增对应命令行参数
  - `scripts/run_meaformer.py`：新增参数透传
- 新增配置与自动化：
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v18a_bipartite_no_il.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v18b_bipartite_late_il.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbdb15k_v18c_bipartite_late_il_skiprel.yaml`
  - `scripts/run_transfer_adapt_v18_fbdb_iter_queue.py`
- 关键中间证据：
  - 初始 visual seeds 真值率从 `v17` 的约 `5.67%` 提升到 `v18` 的约 `15.67%`
- pilot 结果（vs baseline，`delta_avg_mrr_mean`）：
  - `v18a`: `+0.00750`
  - `v18b`: `+0.00700`
  - `v18c`: `+0.00800`
- 自动选优并扩展：
  - `best_variant_pilot = v18c`
  - `expanded_variant_to_full5 = v18c`
- 最终 5-seed（FBDB15K）：
  - `delta_avg_hits@1_mean = +0.00454`
  - `delta_avg_hits@10_mean = +0.01568`
  - `delta_avg_mrr_mean = +0.00830`
  - `delta_avg_mr_mean = -206.81670`
- 结论：
  - `v18c` 明显优于旧主表版本 `v7b (+0.0008)`，应作为新的 `FBDB15K` 主表版本。
  - `FBDB15K` 的收益主因是更干净的初始 visual seeds，而不是继续调 `DA weight`。
- 阶段报告：
  - `reports/transfer/transfer_stage_update_20260312_v18_fbdb_bipartite_full5.md`

## 34. 阶段更新（2026-03-12）：FBYG15K v19/v20 pilot 完成，主表保持 v8

- 目标：继续优化 `FBYG15K`，验证“更严格的 IL 控制”与“更保守的迁移加载”是否能超过当前主表版本 `v8`。
- 新增迁移加载能力：
  - `baselines/MEAformer/config.py` / `baselines/MEAformer/main.py`
  - 支持 `transfer_skip_prefixes`
- 新增自动化：
  - `scripts/run_transfer_adapt_v19_fbyg_iter_queue.py`
  - `scripts/run_transfer_adapt_v20_fbyg_iter_queue.py`
- 新增配置：
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v19a_late_il_strict.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v19b_late_il_skiprel.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v19c_late_il_skiprel_skipfusion.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v20a_aligned_il_skiprel_skipfusion.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v20b_aligned_il_q90_skiprel_skipfusion.yaml`
- `v19` pilot（2-seed, vs baseline）：
  - `v19a = -0.00225`
  - `v19b = -0.00250`
  - `v19c = +0.00100`
- `v20` pilot（2-seed, vs baseline）：
  - `v20a = +0.00050`
  - `v20b = +0.00050`
- 关键诊断：
  - `v19` 的 `late IL` 与当前 fresh-proposal 周期错位，实际近似“关闭 IL”；
  - `v20` 对齐周期后虽然产生了大量早期候选，但最终注入链接在 `epoch 9` 只剩 `1` 条，且真值率为 `0.0%`。
- 结论：
  - `FBYG15K` 当前最佳版本仍为 `v8_mild_da_expand5`（`5-seed delta_avg_mrr_mean = +0.00110`）
  - `v19/v20` 不扩展到 `5-seed`，主表不切换
  - 若继续优化，应改 IL 机制本身，而不是继续做轻量调度/跳过项搜索
- 阶段报告：
  - `reports/transfer/transfer_stage_update_20260312_fbyg_v19_v20_pilot.md`

## 35. 阶段更新（2026-03-12）：FBYG15K v21 fresh-IL full5 完成

- 目标：修复 `v20` 中“IL 候选在注入前塌缩”的失败模式，验证 fresh-IL 立即注入是否能稳定超过当前主表版本。
- 新增自动化：
  - `scripts/run_transfer_adapt_v21_fbyg_iter_queue.py`
- 新增配置：
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v21a_fresh_il_q80_skiprel_skipfusion.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v21b_fresh_il_q90_skiprel_skipfusion.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v21c_fresh_il_q95_skiprel_skipfusion.yaml`
- `v21` pilot（2-seed, vs baseline）：
  - `v21a = +0.00200`
  - `v21b = +0.00100`
  - `v21c = +0.00100`
- 自动决策：
  - `v21a` 超过当前 `v8` 参考值 `+0.00090`
  - 达到扩展阈值后自动扩展到 `5-seed`
- `v21a` 正式 `5-seed`（vs baseline）：
  - `delta_avg_hits@1_mean = +0.00141`
  - `delta_avg_hits@10_mean = +0.00193`
  - `delta_avg_mrr_mean = +0.00160`
  - `delta_avg_mr_mean = -35.84720`
- 关键诊断：
  - `5-seed` 日志中每个 seed 的 fresh-IL 注入规模稳定在 `397-450` 条；
  - 真值率约 `1.8% ~ 2.5%`，虽然仍低，但已明显优于 `v20` 最终塌缩为 `1` 条链接的失败模式。
- 结论：
  - `FBYG15K` 当前主表版本切换为 `v21a_fresh_il_q80_skiprel_skipfusion_expand5`
  - `5-seed delta_avg_mrr_mean` 从 `+0.00110` 提升到 `+0.00160`
  - 若继续优化，应继续提升 fresh-IL 候选质量，而不是回到晚启 IL 的轻量搜索
- 阶段报告：
  - `reports/transfer/transfer_stage_update_20260312_fbyg_v21_fresh_il_full5.md`

## 36. 阶段更新（2026-03-13）：FBYG15K v22 quality-filter pilot 完成，主表保持 v21

- 目标：在 `v21` 的 fresh-IL 立即注入基础上，继续验证“静态质量过滤 + topk cap”是否能进一步提升 `FBYG15K`。
- 新增代码能力：
  - `baselines/MEAformer/config.py`
  - `baselines/MEAformer/model/MEAformer.py`
  - `baselines/MEAformer/main.py`
  - `scripts/run_meaformer.py`
  - 支持 `il_margin_min / il_quality_quantile / il_topk_max / il_margin_weight`
- 新增自动化：
  - `scripts/run_transfer_adapt_v22_fbyg_iter_queue.py`
- 新增配置：
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v22a_fresh_il_quality_top200.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v22b_fresh_il_quality_top100.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v22c_fresh_il_quality_top300.yaml`
- `v22` pilot（2-seed, vs baseline）：
  - `v22a = +0.00050`
  - `v22b = +0.00125`
  - `v22c = +0.00125`
- 关键诊断：
  - `v22b` 在 `seed=42` 上将伪链接真值率提升到 `6.0%`，但 `seed=2026` 仍只有 `1.0%`；
  - 说明静态质量过滤能提升部分 seed 的精度，但跨 seed 稳定性不足。
- 结论：
  - `FBYG15K` 当前主表版本保持 `v21a_fresh_il_q80_skiprel_skipfusion_expand5`
  - `v22` 不扩展到 `5-seed`
  - 若继续优化，应转向分阶段/自适应注入，而不是继续做静态 filter/cap 搜索
- 阶段报告：
  - `reports/transfer/transfer_stage_update_20260313_fbyg_v22_quality_pilot.md`

## 37. 阶段更新（2026-03-13）：FBYG15K v23 staged fresh-IL full5 完成

- 目标：基于 `v22` 的负结果，验证 `FBYG15K` 上“两阶段 fresh-IL 注入”是否优于 `v21` 的单次 fresh-IL。
- 新增代码能力：
  - `baselines/MEAformer/config.py`
  - `baselines/MEAformer/model/MEAformer.py`
  - `baselines/MEAformer/main.py`
  - `scripts/run_meaformer.py`
  - 支持 `il_fresh_epochs` 与分阶段 `confidence/quantile/margin/topk` 调度
- 新增自动化：
  - `scripts/run_transfer_adapt_v23_fbyg_iter_queue.py`
- 新增配置：
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v23a_staged_fresh_il_top250.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v23b_staged_fresh_il_top400.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v23c_staged_fresh_il_epoch8_top250.yaml`
- `v23` pilot（2-seed, vs baseline）：
  - `v23a = +0.00225`
  - `v23b = +0.00300`
  - `v23c = +0.00200`
- 自动决策：
  - `best_variant_pilot = v23b`
  - `improve_over_v21_ref = +0.00140`
  - 自动扩展到 `5-seed`
- `v23b` 正式 `5-seed`（vs baseline）：
  - `delta_avg_hits@1_mean = +0.00186`
  - `delta_avg_hits@10_mean = +0.00460`
  - `delta_avg_mrr_mean = +0.00270`
  - `delta_avg_mr_mean = -43.13610`
- 关键诊断：
  - `phase 0` 在 `epoch 5` 先注入 `100` 条高精度候选；
  - `phase 1` 在 `epoch 7` 再补充 `400` 条候选；
  - staged fresh-IL 比 `v21` 的单次注入更稳定地转化为最终 `MRR` 提升。
- 结论：
  - `FBYG15K` 主表版本切换为 `v23b_staged_fresh_il_top400_expand5`
  - `5-seed delta_avg_mrr_mean` 从 `+0.00160` 提升到 `+0.00270`
  - 若继续优化，应继续沿 staged fresh-IL 做自适应 top-k 或阶段间约束，而不是回到静态 filter/cap 搜索
- 阶段报告：
  - `reports/transfer/transfer_stage_update_20260313_fbyg_v23_staged_fresh_il_full5.md`

## 38. 阶段更新（2026-03-14）：FBYG15K v24 strict-source staged fresh-IL full5 完成

- 目标：先修复 `source checkpoint` 口径不一致问题，再在 strict formal-source 条件下重跑当前最优 staged fresh-IL 路线。
- 新增基础设施：
  - `scripts/ensure_transfer_source_formal.py`
  - `scripts/transfer_adapt_utils.py`
  - 补齐 `seed=2026/7/123` 的 exact `zh_en baseline source formal` checkpoint
  - 默认 source resolver 改为只接受 exact formal-source
- 新增自动化：
  - `scripts/run_transfer_adapt_v24_fbyg_iter_queue.py`
- 新增配置：
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v24a_strictsrc_staged_fresh_il_top250.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v24b_strictsrc_staged_fresh_il_top400.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v24c_strictsrc_staged_fresh_il_epoch8_top250.yaml`
- `v24` pilot（2-seed, vs baseline）：
  - `v24a = +0.00200`
  - `v24b = +0.00300`
  - `v24c = +0.00200`
- 自动决策：
  - `best_variant_pilot = v24b`
  - `improve_over_v23_ref = +0.00030`
  - 自动扩展到 `5-seed`
- `v24b` 正式 `5-seed`（vs baseline）：
  - `delta_avg_hits@1_mean = +0.00197`
  - `delta_avg_hits@10_mean = +0.00462`
  - `delta_avg_mrr_mean = +0.00280`
  - `delta_avg_mr_mean = -42.81030`
- 关键诊断：
  - 全部 5 个 seed 都明确加载 exact `baseline_transfer_formal` source model；
  - staged fresh-IL 的两阶段注入模式仍然稳定保留；
  - 说明 `FBYG15K` 的正增益在 strict-source 口径下依然成立。
- 结论：
  - `FBYG15K` 主表版本切换为 `v24b_strictsrc_staged_fresh_il_top400_expand5`
  - `5-seed delta_avg_mrr_mean` 从 `+0.00270` 提升到 `+0.00280`
  - 当前 `FBYG15K` 结果已更适合作为论文正式主表版本
- 阶段报告：
  - `reports/transfer/transfer_stage_update_20260314_fbyg_v24_strict_source_full5.md`

## 39. 阶段更新（2026-03-14）：FBYG15K v25 adaptive top-k pilot 完成，主表保持 v24

- 目标：在 `v24b` 的 strict-source staged fresh-IL 基础上，验证 `phase-2 adaptive top-k` 是否能进一步超过当前主表版本。
- 新增代码能力：
  - `baselines/MEAformer/config.py`
  - `baselines/MEAformer/model/MEAformer.py`
  - `baselines/MEAformer/main.py`
  - `scripts/run_meaformer.py`
  - 支持 `il_adaptive_topk` 及其分阶段 `scale/min` 调度
- 新增自动化：
  - `scripts/run_transfer_adapt_v25_fbyg_iter_queue.py`
- 新增配置：
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v25a_strictsrc_staged_adaptivetopk_s100.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v25b_strictsrc_staged_adaptivetopk_s125.yaml`
  - `configs/transfer_adapt/tmmeada_target_fbyg15k_v25c_strictsrc_staged_adaptivetopk_s100_min300.yaml`
- `v25` pilot（2-seed, vs baseline）：
  - `v25a = +0.00200`
  - `v25b = +0.00200`
  - `v25c = +0.00250`
- 自动决策：
  - `best_variant_pilot = v25c`
  - `reference_v24_full5 = +0.00280`
  - `improve_over_ref = -0.00030`
  - 未达到扩展阈值，不扩展到 `5-seed`
- 关键诊断：
  - adaptive top-k 在日志中已明确生效，`phase 1` 的 `effective_topk` 会跟随 `phase 0` 的 `pre_topk` 自动变化；
  - 但 `phase 1` 新增链接真值率仍偏低，说明当前瓶颈更像是第二阶段候选一致性，而不只是注入上限数值。
- 结论：
  - `FBYG15K` 当前主表版本保持 `v24b_strictsrc_staged_fresh_il_top400_expand5`
  - `v25` 作为一次有效的机制验证保留，但不切换主表
  - 若继续优化，应优先尝试 `phase-wise consistency constraints`
- 阶段报告：
  - `reports/transfer/transfer_stage_update_20260314_fbyg_v25_adaptive_topk_pilot.md`

## 40. 阶段更新（2026-03-15）：项目主线闭环检查与辅助支撑同步

- 本轮动作：
  - 完成任务书 / 开题报告 / 当前项目现状的全局接管与差距评估；
  - 新增主线闭环评估文件：
    - `reports/notes/taskbook_gap_assessment_20260315.md`
  - 校验 `GPU peak minimal` 现状，确认当前仓库中尚无可直接入文的正式显存汇总表；
  - 修正 `scripts/run_gpu_peak_minimal.py`，避免继续生成 `epoch <= il_start` 的无效最小补测配置。
- 当前判断：
  - 主线：已闭环。
  - 仍需继续做的，是“主线材料规范化 + 辅助支撑项保守补强”。
- 辅助项边界：
  - `H3`：当前仓库已主动移除相关脚本、结果与目录，不再作为现阶段项目组成部分。
  - GPU 峰值显存：当前没有正式结果，不能写成已完成，只能写成“脚本已修正、补测待执行”。
- 推荐后来者阅读顺序：
  - `reports/notes/taskbook_gap_assessment_20260315.md`
  - `reports/transfer/transfer_adapt_main_results_4target.md`
  - `reports/transfer/transfer_adapt_significance_summary.md`
  - `reports/epoch3/epoch3_ablation_zh_en_multiseed.md`

## 41. 阶段更新（2026-03-15）：深度仓库整理与 H3 延期

- 本轮动作：
  - 删除 `H3` 相关结果、脚本、运行目录与目录说明；
  - 从 `MEAformer` 训练入口中移除人工图像缺失注入参数，避免后续误将其作为主线配置项；
  - 在项目记录与共享同步板中明确：`H3` 已延期到主线完全结束后再重新尝试。
- 当前状态：
  - 仓库聚焦于主线材料、正式 run 与可复现脚本；
  - `H3` 不再占用当前项目导航、结果目录与代码入口；
  - 论文线程不应继续从当前仓库读取或引用旧的 `H3` 留痕。

## 42. 阶段更新（2026-03-15）：主线复现与追溯总表建立

- 本轮动作：
  - 新增项目级主线追溯总表：
    - `reports/notes/mainline_traceability_matrix_20260315.md`
  - 将任务书 / 开题报告要求与正式结果、脚本入口、run 目录建立一一映射；
  - 明确当前主线外剩余缺口只剩“导航收口”和“GPU 最小正式补测”。
- 当前作用：
  - 后来者现在可以从一份文件直接定位主线正式证据；
  - 论文线程可以直接吸收“要求 -> 证据 -> 脚本 -> run”的闭环关系；
  - 后续 README / reports / runs 的继续收口有了统一锚点。

## 43. 阶段更新（2026-03-15）：主线导航再收口

- 本轮动作：
  - 为 `reports/transfer/` 新增子目录 README，明确正式主表、辅助分析与历史探索文件的使用边界；
  - 为 `runs/transfer/` 新增子目录 README，明确 4 个目标域正式 baseline / method run 目录；
  - 将根 README、reports/README、runs/README 的主线导航进一步收口到“总表 + transfer 子目录 README”这一层。
- 当前作用：
  - 后来者进入 `reports/transfer/` 与 `runs/transfer/` 后不再需要先读大量历史探索文件；
  - 主线正式结果与探索性材料的边界更清楚；
  - 为后续最小 GPU 补测保留了更干净的导航结构。
