# TMMEA-DA v1 Smoke 记录

## 1. 本次目标
- 在现有 Domain Align MVP 基础上加入两个增强模块并验证可运行性：
  - `source_select`（跨模态源选择辅助损失）
  - `missing_gate`（缺失感知图像对齐损失）

## 2. 代码改动
- 参数扩展：
  - `baselines/MEAformer/config.py`
  - 新增 `use_source_select/source_select_weight/source_select_temp`
  - 新增 `use_missing_gate/missing_align_weight`
- 数据加载：
  - `baselines/MEAformer/src/data.py`
  - `load_img()` 额外返回 `img_mask`，并写入 KGs
- 模型：
  - `baselines/MEAformer/model/MEAformer.py`
  - 新增 `_source_select_loss()` / `_missing_aware_img_align_loss()`
  - 在 `forward()` 中组合：joint + intra + domain + source_select + missing_align
- 运行脚本透传参数：
  - `scripts/run_meaformer.py`

## 3. 配置与命令
- 配置：`configs/tmmeada/meaformer_zh_en_tmmeada_v1_smoke.yaml`
- 命令：
  - `conda run -n bysj-main python scripts\run_meaformer.py --config configs\tmmeada\meaformer_zh_en_tmmeada_v1_smoke.yaml`

## 4. 结果（1-epoch, zh_en, 5 seeds）

### 4.1 单次 run
- `20260228-205727-TMMEA-DA-v1-DBP15K-zh_en-s42`
- `20260228-212025-TMMEA-DA-v1-DBP15K-zh_en-s3407`
- `20260228-213208-TMMEA-DA-v1-DBP15K-zh_en-s2026`
- `20260228-214257-TMMEA-DA-v1-DBP15K-zh_en-s7`
- `20260228-215418-TMMEA-DA-v1-DBP15K-zh_en-s123`

### 4.2 汇总（mean ± std）
- l2r Hits@1: `0.5524 ± 0.0056`
- l2r Hits@10: `0.8408 ± 0.0016`
- l2r MRR: `0.6492 ± 0.0035`
- r2l Hits@1: `0.5531 ± 0.0025`
- r2l Hits@10: `0.8404 ± 0.0021`
- r2l MRR: `0.6490 ± 0.0025`

对应文件：
- `reports/tmmeada_v1_results_summary.csv`
- `reports/tmmeada_v1_results_mean_std.csv`
- `reports/tmmeada_v1_compare_zh_en.md`（baseline / v0 / v1 三方对比）

## 5. 结论
- v1 模块链路已跑通，参数可配置、训练闭环正常。
- 在当前 1-epoch 预算下，v1 相比 v0 仅有极小波动（接近 0），尚无显著收益。
- 下一步：扩展训练轮次并进行权重搜索（`domain/source_select/missing_align`），再做正式消融。
