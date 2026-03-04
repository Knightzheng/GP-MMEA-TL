# 核心代码改造记录（2026-03-03）

## 1. 改造背景
- 在 `epoch=3`、`5-seed` 的正式口径下，`TMMEA-DA v1_best` 与 `MEAformer baseline` 基本持平，提升量级约为 `1e-4`，小于波动范围。
- 目标转为“先做小规模但明确的机制改造”，再用 `epoch=10 pilot` 验证是否能带来可重复增益。

## 2. 改造目标
- 增强域对齐损失的判别性，避免只做正样本拉近导致的“对齐但不分离”。
- 将辅助损失改为“分阶段启用”，减少训练初期对主目标（对比学习）的干扰。
- 增加可观测诊断项，便于后续在报告中做机制解释而不是只报最终分数。

## 3. 代码改造清单
### 3.1 参数层新增（可控开关）
- 文件：`baselines/MEAformer/config.py`
- 新增参数：
  - `--aux_start_epoch`
  - `--aux_ramp_epochs`
  - `--domain_align_margin`
  - `--domain_align_neg_weight`

设计意图：
- `aux_start_epoch` + `aux_ramp_epochs`：控制辅助损失从“关闭 -> 平滑放大 -> 全量生效”。
- `domain_align_margin` + `domain_align_neg_weight`：为域对齐增加 hard-negative 约束。

### 3.2 模型损失函数改造
- 文件：`baselines/MEAformer/model/MEAformer.py`
- 关键改动：
  - 新增 `_aux_scale(current_epoch)`：
    - `epoch < aux_start_epoch` 时，辅助损失系数为 0。
    - 在 `aux_ramp_epochs` 内线性升温至 1。
  - 升级 `_domain_align_loss(...)`：
    - 保留正样本 MSE（`domain_align_pos`）。
    - 新增 hard-negative hinge（`domain_align_hard`）：
      - `relu(margin + pos_dist - neg_dist)`。
    - 总损失为：`pos_loss + domain_align_neg_weight * hard_loss`。
  - 在 `forward(...)` 中引入 `aux_scale`，统一控制三类辅助项：
    - `domain_align_term`
    - `missing_align_term`
    - `source_select` 项
  - 增强 `loss_dic` 日志字段：
    - `aux_scale`
    - `domain_align_pos`
    - `domain_align_hard`
    - `domain_align_term`
    - `missing_align_term`

### 3.3 训练循环改造（让模型感知 epoch）
- 文件：`baselines/MEAformer/main.py`
- 改动：
  - 训练阶段调用改为 `self.model(batch, current_epoch=self.epoch)`（MEAformer 分支）。
  - 作用：将当前 epoch 传入模型，驱动辅助损失分阶段策略。

### 3.4 运行脚本改造（配置透传）
- 文件：`scripts/run_meaformer.py`
- 改动：
  - 将新参数透传到 MEAformer CLI：
    - `aux_start_epoch`
    - `aux_ramp_epochs`
    - `domain_align_margin`
    - `domain_align_neg_weight`

### 3.5 新增调优配置（pilot）
- 文件：`configs/tmmeada/meaformer_zh_en_tmmeada_v2_tuned_epoch10_pilot.yaml`
- 当前核心设定：
  - `domain_align_weight: 0.3`
  - `domain_align_margin: 0.4`
  - `domain_align_neg_weight: 1.0`
  - `source_select_weight: 0.08`
  - `missing_align_weight: 0.15`
  - `aux_start_epoch: 2`
  - `aux_ramp_epochs: 4`

## 4. 可复现验证步骤
1. 干跑检查：
```powershell
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\run_meaformer.py --config configs\tmmeada\meaformer_zh_en_tmmeada_v2_tuned_epoch10_pilot.yaml --dry-run
```
2. 2-seed pilot（建议先 `42,3407`）：
```powershell
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\run_from_base_config_multiseed.py --base-config configs\tmmeada\meaformer_zh_en_tmmeada_v2_tuned_epoch10_pilot.yaml --seeds "42,3407"
```
3. 结果聚合与决策：
  - 对比 `baseline epoch10 pilot` 与 `tmmeada_v2_tuned epoch10 pilot` 的 `Hits@1/Hits@10/MRR`。
  - 若 `ΔMRR` 达到预设阈值（如 `>= +0.003`）再扩展到 5-seed 正式运行。

## 5. 报告可直接引用的描述（建议）
- “针对 v1 模块在 5-seed 下增益不显著的问题，本文在 MEAformer 框架内引入了分阶段辅助损失策略与 hard-negative 域对齐项，并对训练日志增加机制级诊断字段，以区分‘优化未收敛’与‘模块本身无效’两类原因。”
- “代码改造遵循最小侵入原则：不改变主干编码器结构，仅在损失项与训练调度层进行可开关增强，保证与 baseline 的可比性。”

