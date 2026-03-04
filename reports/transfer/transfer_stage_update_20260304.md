# 2026-03-04 迁移实验下一步（已执行）

## 1. 目标
- 将任务书里的核心设定工程化落地：`源域训练 -> 目标域测试`。
- 支持跨数据集参数迁移加载（允许形状不一致参数自动跳过），避免直接加载失败。

## 2. 本次代码改造
- `baselines/MEAformer/config.py`
  - 新增参数：
    - `--transfer_non_strict`
    - `--transfer_skip_keys`
    - `--transfer_verbose`
- `baselines/MEAformer/main.py`
  - `_load_model` 支持迁移模式：
    - 按 `key` 过滤（例如跳过 `entity_emb`）
    - 按 `shape` 过滤（shape 不一致自动跳过）
    - 输出加载统计（loaded / skipped / missing）
- `scripts/run_meaformer.py`
  - 透传迁移相关参数：
    - `model_name_save`
    - `transfer_non_strict`
    - `transfer_skip_keys`
    - `transfer_verbose`
- 新增自动化脚本：
  - `scripts/run_transfer_train_eval.py`
  - `scripts/compare_transfer_summaries.py`
- 新增配置：
  - `configs/transfer/meaformer_*`
  - `configs/transfer/tmmeada_*`

## 3. 运行与产出
- baseline 迁移链路 run card：
  - `reports/transfer_run_card_20260304-102611_baseline_transfer_smoke_real_eval.json`
- tmmeada 迁移链路 run card：
  - `reports/transfer_run_card_20260304-102921_tmmeada_transfer_smoke_real.json`
- 汇总结果：
  - `reports/transfer_smoke_source_train_summary.csv`
  - `reports/transfer_smoke_target_eval_summary.csv`
  - `reports/transfer_smoke_tmmeada_source_train_summary.csv`
  - `reports/transfer_smoke_tmmeada_target_eval_summary.csv`
  - `reports/transfer_smoke_compare_tmmeada_vs_baseline.csv`
  - `reports/transfer_smoke_compare_tmmeada_vs_baseline.md`

## 4. 结果（seed=42，source epoch=1 smoke）

### 4.1 源域（zh_en）内测试
- baseline source：
  - l2r `H@1=0.5487, H@10=0.8403, MRR=0.647`
- tmmeada source：
  - l2r `H@1=0.5490, H@10=0.8402, MRR=0.647`

### 4.2 迁移到目标域（source=zh_en）
- baseline：
  - `ja_en` l2r `H@1=0.0697, H@10=0.1736, MRR=0.107`
  - `fr_en` l2r `H@1=0.0348, H@10=0.1136, MRR=0.063`
  - `FBDB15K` l2r `H@1=0.0002, H@10=0.0018, MRR=0.002`
- tmmeada：
  - `ja_en` l2r `H@1=0.0697, H@10=0.1739, MRR=0.107`
  - `fr_en` l2r `H@1=0.0348, H@10=0.1135, MRR=0.063`
  - `FBDB15K` l2r `H@1=0.0002, H@10=0.0018, MRR=0.002`

### 4.3 baseline vs tmmeada 差值
- `reports/transfer_smoke_compare_tmmeada_vs_baseline.md` 显示：
  - 各数据集差值基本在 `1e-4` 量级，当前 smoke 阶段几乎持平。

## 5. 关键分析
- 迁移链路已完整跑通，且具备可复现脚本、配置、run card 和结果汇总。
- 现阶段的核心问题不是“跑不通”，而是“迁移性能显著掉点”：
  - `zh_en` 内测 MRR 约 `0.647`，迁移到 `ja_en/fr_en/FBDB15K` 后明显下降。
- 这与任务书动机一致，后续重点应放在：
  - 更充分源域训练（`epoch=10` 正式版）
  - 目标域无标注自适应策略（例如伪标签/分布对齐温启动）
  - 迁移能力度量与误差分桶分析。

## 6. 下一步（建议已明确）
- 直接启动正式迁移版（`source epoch=10`）并优先跑 `seed=42,3407`：
  - baseline 与 tmmeada 各一组；
  - 目标域保留 `ja_en/fr_en/FBDB15K`；
  - 产出同口径对比表用于中期报告“动机实验”主结果。
