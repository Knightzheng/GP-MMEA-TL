# TMMEA-DA MVP Smoke 实验记录

## 1. 目标
- 在不破坏现有 MEAformer 基线的前提下，新增可开关域对齐损失（Domain Alignment Loss）。
- 先执行 1 epoch 冒烟实验，验证代码可运行、参数可控、日志可留痕。

## 2. 关键改动
- `baselines/MEAformer/config.py`
  - 新增参数：`--use_domain_align`（0/1）
  - 新增参数：`--domain_align_weight`（float）
- `baselines/MEAformer/model/MEAformer.py`
  - 新增 `_domain_align_loss()`，对 batch 内正样本对的 joint embedding 计算 `MSE`。
  - 在总损失中加入：`loss_all += domain_align_weight * domain_align_loss`（仅在开关开启时）。
  - `loss_dic` 增加 `domain_align` 字段，便于后续监控。
- `scripts/run_meaformer.py`
  - 支持 `meta.stage`，可将 run 输出到 `runs/tmmeada/`。
  - 支持 `meta.model_tag`，区分基线与方法实验。
  - 透传 `use_domain_align/domain_align_weight` 到主训练命令。

## 3. 配置与命令
- 配置文件：`configs/tmmeada/meaformer_zh_en_domain_align_mvp.yaml`
- 运行命令：
  - `conda run -n bysj-main python scripts\run_meaformer.py --config configs\tmmeada\meaformer_zh_en_domain_align_mvp.yaml`

## 4. Run 信息（当前已完成 5 seeds）
- `20260228-044730-TMMEA-DA-MEAformer-DBP15K-zh_en-s42`
- `20260228-050047-TMMEA-DA-MEAformer-DBP15K-zh_en-s3407`
- `20260228-125417-TMMEA-DA-MEAformer-DBP15K-zh_en-s2026`
- `20260228-130507-TMMEA-DA-MEAformer-DBP15K-zh_en-s7`
- `20260228-131550-TMMEA-DA-MEAformer-DBP15K-zh_en-s123`
- stage: `tmmeada`
- 数据集：`DBP15K/zh_en`
- 训练轮数：`1`

## 5. 指标（Test）

### 5.1 单次结果
- seed=42
  - l2r: Hits@1=`0.5487`, Hits@10=`0.8403`, MRR=`0.647`, MR=`11.462`
  - r2l: Hits@1=`0.5522`, Hits@10=`0.8377`, MRR=`0.647`, MR=`11.561`
- seed=3407
  - l2r: Hits@1=`0.5474`, Hits@10=`0.8410`, MRR=`0.647`, MR=`11.402`
  - r2l: Hits@1=`0.5525`, Hits@10=`0.8401`, MRR=`0.649`, MR=`12.159`
- seed=2026
  - l2r: Hits@1=`0.5489`, Hits@10=`0.8421`, MRR=`0.646`, MR=`11.590`
  - r2l: Hits@1=`0.5497`, Hits@10=`0.8387`, MRR=`0.646`, MR=`11.993`
- seed=7
  - l2r: Hits@1=`0.5594`, Hits@10=`0.8389`, MRR=`0.653`, MR=`12.816`
  - r2l: Hits@1=`0.5561`, Hits@10=`0.8416`, MRR=`0.651`, MR=`13.218`
- seed=123
  - l2r: Hits@1=`0.5571`, Hits@10=`0.8435`, MRR=`0.653`, MR=`11.786`
  - r2l: Hits@1=`0.5548`, Hits@10=`0.8430`, MRR=`0.652`, MR=`12.885`

### 5.2 阶段汇总（5 seeds）
- 汇总文件：
  - `reports/tmmeada_results_summary.csv`
  - `reports/tmmeada_results_mean_std.csv`
- mean ± std（zh_en）：
  - l2r Hits@1: `0.5523 ± 0.0055`
  - l2r Hits@10: `0.8412 ± 0.0017`
  - l2r MRR: `0.6492 ± 0.0035`
  - r2l Hits@1: `0.5531 ± 0.0025`
  - r2l Hits@10: `0.8402 ± 0.0021`
  - r2l MRR: `0.6490 ± 0.0025`
- 与基线对比文件：
  - `reports/tmmeada_vs_baseline_zh_en.csv`
  - `reports/tmmeada_vs_baseline_zh_en.md`

## 6. 结论
- 代码改造有效：新参数可被主程序识别并完成训练/测试闭环。
- 日志与工件完整：`run_card.md`、`config.yaml`、`log.txt`、`artifact_manifest.json` 均已生成。
- 当前 `5 seeds` 已补齐。由于本版仅为“单一域对齐项”的简化 MVP，且训练预算仍是 1 epoch，指标低于当前基线。
- 下一步建议：在同预算下进行模块化改造（多源选择、缺失感知融合、损失权重搜索）并扩大训练轮次，再做公平对比。
