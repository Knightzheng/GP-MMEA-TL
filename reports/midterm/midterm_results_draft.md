# 中期报告实验结果草稿（已更新到 epoch=3 正式 5-seed）

## 使用说明

- 本文件仅保留“结果摘要速览”。
- 中期提交正文请优先使用：`reports/midterm_report_submission.md`。

## 1. 本阶段最终实验口径

- 训练预算：`epoch=3`
- 随机种子：`42, 3407, 2026, 7, 123`
- 数据范围：
  - DBP15K：`zh_en`, `ja_en`, `fr_en`
  - 跨图谱：`FBDB15K`, `FBYG15K`
- 指标：`Hits@1`, `Hits@10`, `MRR`（`l2r/r2l`）

## 2. 关键结论（可直接引用）

1. DBP15K 与跨图谱两组正式 5-seed 对比中，`TMMEA-DA v1_best` 相对 baseline 的提升量级约为 `10^-4`，整体与 baseline 持平。  
2. `zh_en` 消融（`wo_domain_align / wo_source_select / wo_missing_gate`）与 full 几乎一致，仅 `wo_source_select` 在个别指标出现极小回落。  
3. 当前中期结论可写为：流程与复现闭环已完成，方法模块在现预算下尚未体现显著增益。

## 3. 结果来源文件

- `reports/epoch3_compare_dbp15k.csv`
- `reports/epoch3_compare_crossgraph.csv`
- `reports/epoch3_ablation_zh_en_multiseed.csv`
- `reports/midterm_report_submission.md`
