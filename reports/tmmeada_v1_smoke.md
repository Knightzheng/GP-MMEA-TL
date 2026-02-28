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

## 4. 结果（1-epoch, seed=42）
- run_id：`20260228-205727-TMMEA-DA-v1-DBP15K-zh_en-s42`
- l2r: Hits@1=`0.5488`, Hits@10=`0.8402`, MRR=`0.647`
- r2l: Hits@1=`0.5522`, Hits@10=`0.8381`, MRR=`0.647`

## 5. 结论
- v1 模块链路已跑通，参数可配置、训练闭环正常。
- 下一步：在 `zh_en` 上补齐 5-seed，再决定是否扩展到 `ja_en/fr_en` 与跨图谱。
