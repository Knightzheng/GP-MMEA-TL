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

## 4. Run 信息（当前已完成 2 seeds）
- `20260228-044730-TMMEA-DA-MEAformer-DBP15K-zh_en-s42`
- `20260228-050047-TMMEA-DA-MEAformer-DBP15K-zh_en-s3407`
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

### 5.2 阶段汇总（2 seeds）
- 汇总文件：
  - `reports/tmmeada_results_summary.csv`
  - `reports/tmmeada_results_mean_std.csv`
- mean ± std（zh_en）：
  - l2r Hits@1: `0.5480 ± 0.0009`
  - l2r Hits@10: `0.8407 ± 0.0005`
  - l2r MRR: `0.6470 ± 0.0000`
  - r2l Hits@1: `0.5524 ± 0.0002`
  - r2l Hits@10: `0.8389 ± 0.0017`
  - r2l MRR: `0.6480 ± 0.0014`

## 6. 结论
- 代码改造有效：新参数可被主程序识别并完成训练/测试闭环。
- 日志与工件完整：`run_card.md`、`config.yaml`、`log.txt`、`artifact_manifest.json` 均已生成。
- 下一步建议：在同配置下补齐 `5 seeds`，与基线做公平对比（同 epoch、同数据切分、同随机种子集合）。
