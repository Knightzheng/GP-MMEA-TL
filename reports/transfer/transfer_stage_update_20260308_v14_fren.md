# 迁移实验阶段报告（2026-03-08，v14 fr_en）

## 1. 本阶段目标
- 在 `v13` 稳定但无增益的基础上，尝试优化 IL（伪标签）更新节奏。
- 重点验证：更早刷新伪标签是否能提升 `fr_en` transfer-adapt 指标。

## 2. 本阶段新增内容
- 代码修改：
  - `baselines/MEAformer/config.py`：新增 `--il_refresh_interval`
  - `baselines/MEAformer/main.py`：刷新频率改为可配置
  - `scripts/run_meaformer.py`：新增参数透传
- 配置：
  - `configs/transfer_adapt/tmmeada_target_fr_en_v14a_refresh5_da0025.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v14b_refresh4_da0025.yaml`
  - `configs/transfer_adapt/tmmeada_target_fr_en_v14c_refresh5_da0030.yaml`
- 自动脚本：
  - `scripts/run_transfer_adapt_v14_fren_auto.py`

## 3. 断电恢复说明
- 下午运行中断电，检查后确认：
  - 三个 pilot 变体结果已完整落盘；
  - best 变体 `v14b` 的 formal 训练（`seed=3407`）也已完整结束；
  - 中断发生在“2-seed 汇总与决策文件写入”阶段。
- 恢复方式：
  - 不重复训练，直接基于已完成 run 补齐 merge/summarize/decision 文档。

## 4. 自动流程与决策
- 流程：`pilot(3 variants, s42) -> select best -> formal(s3407) -> 2-seed summarize`
- pilot 结果（`delta_avg_mrr_mean` vs baseline）：
  - `v14a_refresh5_da0025`: `-0.00100`
  - `v14b_refresh4_da0025`: `+0.01050`
  - `v14c_refresh5_da0030`: `-0.00100`
- 自动选优：`v14b_refresh4_da0025`

决策文件：
- `reports/transfer/transfer_adapt_v14_fren_decision.md`
- `reports/transfer/transfer_adapt_v14_fren_decision.json`

## 5. 最终 2-seed 结果（fr_en）
来源：
- `reports/transfer/transfer_adapt_v14_fren_2seed_compare_vs_baseline.csv`
- `reports/transfer/transfer_adapt_v14_fren_2seed_compare_vs_v13.csv`

关键结论：
- vs baseline：`delta_avg_mrr_mean = +0.01075`
- vs v13：`delta_avg_mrr_mean = +0.01100`
- 同时 `Hits@1`、`Hits@10` 均为正增益，`MR` 下降（更好）。

## 6. 结论与下一步
- `v14` 是当前 `fr_en` 迁移实验的最佳版本，已形成可报告的正向结果。
- 下一步建议：
1. 以 `v14b` 为主线扩展到 `5-seed` 正式统计（优先 `fr_en`）。
2. 按同口径扩展到 `FBYG15K`，补齐任务书中的跨图谱迁移证据。
3. 同步补充误差分析图（失败案例、分桶统计）用于中期/终稿章节。
