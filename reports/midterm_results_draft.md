# 中期报告实验结果草稿（MEAformer 多 seed）

## 1. 实验设置
- 基线模型：MEAformer
- 数据集：DBP15K（`zh_en`, `ja_en`, `fr_en`）
- 训练配置：RTX3060 安全配置（1 epoch 冒烟复现）
- 随机种子：`42`, `3407`, `2026`, `7`, `123`（每语种 5 次）
- 结果来源：
  - `reports/meaformer_results_summary.csv`
  - `reports/meaformer_results_mean_std.csv`

## 2. 结果汇总（mean ± std）

| lang_pair | num_runs | l2r Hits@1 | l2r Hits@10 | l2r MRR | r2l Hits@1 | r2l Hits@10 | r2l MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| zh_en | 5 | 0.6426 ± 0.0037 | 0.8757 ± 0.0021 | 0.7224 ± 0.0031 | 0.6375 ± 0.0034 | 0.8681 ± 0.0022 | 0.7164 ± 0.0034 |
| ja_en | 5 | 0.5994 ± 0.0022 | 0.8513 ± 0.0040 | 0.6838 ± 0.0023 | 0.5984 ± 0.0026 | 0.8518 ± 0.0033 | 0.6830 ± 0.0028 |
| fr_en | 5 | 0.5408 ± 0.0049 | 0.8473 ± 0.0030 | 0.6424 ± 0.0045 | 0.5392 ± 0.0029 | 0.8509 ± 0.0039 | 0.6418 ± 0.0031 |

## 2.1 跨图谱结果（官方真实数据，5 seeds）

| dataset | num_runs | l2r Hits@1 | l2r Hits@10 | l2r MRR | r2l Hits@1 | r2l Hits@10 | r2l MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| FBDB15K | 5 | 0.0986 ± 0.0020 | 0.3243 ± 0.0048 | 0.1730 ± 0.0023 | 0.1058 ± 0.0028 | 0.3326 ± 0.0052 | 0.1810 ± 0.0027 |
| FBYG15K | 5 | 0.0857 ± 0.0017 | 0.2756 ± 0.0099 | 0.1498 ± 0.0036 | 0.0894 ± 0.0025 | 0.2759 ± 0.0102 | 0.1524 ± 0.0046 |

## 2.2 方法开发进展（TMMEA-DA MVP）

- 已在 MEAformer 中新增可开关域对齐损失（`use_domain_align/domain_align_weight`）。
- 当前已完成 `2 seeds`（`42`, `3407`）的 `DBP15K/zh_en` 1-epoch 冒烟验证。
- 阶段汇总（zh_en, 2 seeds）：
  - l2r Hits@1: `0.5480 ± 0.0009`
  - l2r Hits@10: `0.8407 ± 0.0005`
  - l2r MRR: `0.6470 ± 0.0000`
  - r2l Hits@1: `0.5524 ± 0.0002`
  - r2l Hits@10: `0.8389 ± 0.0017`
  - r2l MRR: `0.6480 ± 0.0014`
- 对应记录：
  - `reports/tmmeada_mvp_smoke.md`
  - `reports/tmmeada_results_summary.csv`
  - `reports/tmmeada_results_mean_std.csv`
  - `runs/tmmeada/20260228-044730-TMMEA-DA-MEAformer-DBP15K-zh_en-s42/`
  - `runs/tmmeada/20260228-050047-TMMEA-DA-MEAformer-DBP15K-zh_en-s3407/`

## 3. 可写入正文的结论句（草稿）
- 在当前统一设置下，`zh_en` 的对齐性能最高，`ja_en` 次之，`fr_en` 最低，初步体现了跨语言迁移难度差异。
- 三个语种在 5 个随机种子下的标准差整体较小（多数指标 std < 0.005），说明该复现实验在当前配置下具备较好的稳定性。
- 当前实验仍为 1 epoch 冒烟配置，后续将扩展到正式训练轮数并在相同统计口径下更新主结果表与消融对照表。
- 跨图谱 FBDB15K/FBYG15K 已扩展到 5 seeds，后续重点是扩展训练轮次与方法消融。
