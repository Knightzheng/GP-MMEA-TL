# Transfer GPU Peak Minimal Thesis Sync (2026-03-15)

## 1. 当前判断

当前不建议继续为了论文线程额外重跑更大的 GPU 峰值显存补测。原因不是这部分已经“充分完成”，而是现有最小正式补测已经足以承担辅助支撑角色，而继续增加目标域或随机种子的边际收益较低，不值得在当前阶段挤占主线材料整理时间。

当前正式入口：

- `reports/transfer/transfer_gpu_peak_minimal_summary.md`
- `reports/transfer/transfer_gpu_peak_minimal_summary.csv`
- `reports/transfer/transfer_gpu_peak_minimal_chart_ready.csv`

## 2. 为什么这轮不继续扩大 GPU 补测

1. 当前已经覆盖 `ja_en` 与 `FBYG15K` 两个代表性目标域，且都包含 `baseline / method` 对照。
2. GPU 峰值显存本身只属于辅助支撑，现有最小正式结果已经能支撑“显存变化有限、主要补证是相对参考值”的保守表述。
3. `FBYG15K method` 因原始配置中的 `il_start=5`，最小有效补测需要用到 `epoch=6`，继续扩大补测并不会显著提升结论硬度，反而容易让时间列被误解为同预算对比。

## 3. 第四章可直接吸收的分析文字

除 wall-clock 统计外，本文还在相同 `Windows + PyTorch` 环境下补充了代表性场景的 GPU 峰值显存最小正式测量。结果显示，在 `ja_en` 与 `FBYG15K` 两个目标域上，本文方法相较匹配 baseline 的 PyTorch allocator 峰值显存变化较小，说明当前迁移增强策略并未带来明显的额外显存负担。需要说明的是，该结果仅是代表性目标域下的最小补测，主要用于提供相对参考，而不能替代全目标域、全 seed 的统一显存统计。

## 4. 答辩时可直接使用的解释口径

我们没有把 GPU 峰值显存做成与主实验同规模的 `5-seed` 全覆盖统计，而是补了一版代表性目标域的最小正式结果，目的只是回答“方法是否明显更占显存”这个辅助问题。当前结果说明，在相同环境下方法相对 baseline 的 allocator-level peak memory 变化有限，因此论文把它作为 wall-clock 之外的辅助补证，而没有把它抬高成主线结论。

## 5. 当前能支持什么 / 不能支持什么

- 能支持：同环境、同目标域、同类配置下的相对 GPU 峰值显存参考。
- 不能支持：所有目标域统一的严格显存统计，或精确等价于 `nvidia-smi` 物理占用的结论。
- 不能支持：把 `1-epoch` 最小补测的时间列解释成与正式 `5-seed` 训练完全同预算的效率比较。
