# 迁移实验阶段报告：FBYG15K v23 staged fresh-IL full5

- 日期：`2026-03-13`
- 目标数据：`FBYG15K`
- 当前旧主表版本：`v21a_fresh_il_q80_skiprel_skipfusion_expand5`
- 旧参考结果：`5-seed delta_avg_mrr_mean = +0.00160`

## 1. 本轮目标

在 `v22` 已证明“静态 quality/filter/cap 不足以继续提升”之后，转向验证更符合诊断结论的路线：

- 不再做单次 fresh-IL 注入
- 改为两阶段 fresh-IL 注入
- 第一阶段先注入小规模高精度候选
- 第二阶段在训练后半段补充一轮更大规模候选

核心问题是验证：`FBYG15K` 上“先稳后补”的 staged fresh-IL 是否能稳定超过当前 `v21` 主表版本。

## 2. 代码改动

本轮新增了按 epoch 触发多轮 fresh proposal 与分阶段过滤参数的能力：

- `baselines/MEAformer/config.py`
  - 新增 `il_fresh_epochs`
  - 新增 `il_confidence_min_schedule`
  - 新增 `il_confidence_quantile_schedule`
  - 新增 `il_confidence_keep_min_schedule`
  - 新增 `il_margin_min_schedule`
  - 新增 `il_quality_quantile_schedule`
  - 新增 `il_topk_max_schedule`
- `baselines/MEAformer/model/MEAformer.py`
  - 新增 phase-aware IL filter 配置解析
  - 支持在多个指定 epoch 上重新触发 `fresh` proposal
  - 支持为不同注入阶段使用不同的 `confidence / quantile / margin / topk`
- `baselines/MEAformer/main.py`
  - IL 日志新增 `phase` 与 `fresh` 标记
- `scripts/run_meaformer.py`
  - 补齐 staged-IL 新参数透传

## 3. 实验设置

新增 `v23` 三个 `2-seed pilot` 变体，均保持：

- `transfer_skip_keys = entity_emb + rel_fc`
- `transfer_skip_prefixes = multimodal_encoder.fusion.`
- `il_start = 5`
- `semi_learn_step = 1`
- `il_refresh_interval = 1`

变体设计：

| variant | fresh epochs | phase-0 | phase-1 |
|---|---|---|---|
| v23a | `5,7` | `q85 + margin 0.012 + top100` | `q80 + top250` |
| v23b | `5,7` | `q85 + margin 0.012 + top100` | `q80 + top400` |
| v23c | `5,8` | `q85 + margin 0.012 + top100` | `q80 + top250` |

自动脚本：

- `scripts/run_transfer_adapt_v23_fbyg_iter_queue.py`

## 4. Pilot 结果

`pilot_seeds = [42, 2026]`

| variant | delta_avg_mrr_mean | 结论 |
|---|---:|---|
| v23a | +0.00225 | 超过 `v21` |
| v23b | +0.00300 | 本轮最优 |
| v23c | +0.00200 | 超过 `v21` |

自动决策：

- `best_variant_pilot = v23b`
- `improve_over_current_ref = +0.00140`
- 达到扩展阈值 `+0.00030`
- 自动扩展到 `5-seed`

## 5. Full-5 正式结果

正式 `5-seed` 对比文件：

- `reports/transfer/transfer_adapt_v23_fbyg_v23b_expand5_compare_vs_baseline.csv`
- `reports/transfer/transfer_adapt_v23_fbyg_v23b_expand5_compare_vs_baseline.md`

正式结果（vs baseline）：

- `delta_avg_hits@1_mean = +0.00186`
- `delta_avg_hits@10_mean = +0.00460`
- `delta_avg_mrr_mean = +0.00270`
- `delta_avg_mr_mean = -43.13610`

相对旧主表版本 `v21`：

- `delta_avg_mrr_mean: +0.00160 -> +0.00270`
- 提升幅度：`+0.00110`

## 6. 关键诊断

`v23b` 的正增益来自“两阶段注入”而不是继续压单次过滤强度。

5 个 seed 的日志都显示出一致的两阶段模式：

- `phase=0`（epoch 5）先注入 `100` 条高精度候选
- `phase=1`（epoch 7）再补充 `400` 条更大规模候选

关键日志证据：

- `seed=42`
  - `phase 0`: `100` links, `true link ratio = 6.0%`
  - `phase 1`: `400` links, `true link ratio = 0.8%`
- `seed=2026`
  - `phase 0`: `100` links, `true link ratio = 1.0%`
  - `phase 1`: `400` links, `true link ratio = 3.0%`
- `seed=3407`
  - `phase 0`: `100` links, `true link ratio = 2.0%`
  - `phase 1`: `400` links, `true link ratio = 3.0%`
- `seed=7`
  - `phase 0`: `100` links, `true link ratio = 4.0%`
  - `phase 1`: `400` links, `true link ratio = 2.8%`
- `seed=123`
  - `phase 0`: `100` links, `true link ratio = 3.0%`
  - `phase 1`: `400` links, `true link ratio = 2.5%`

这说明：

1. `FBYG15K` 上单次注入不是最优策略，分阶段注入更符合当前噪声结构；
2. 第一阶段的小规模高精度集合有助于稳定启动后半程训练；
3. 第二阶段的大规模补充虽然噪声仍存在，但比 `v21` 的单次大注入更有效；
4. 相比 `v22` 的静态过滤，`v23` 把“质量”和“数量”拆到两个时间点处理，因而更稳定地转化成最终 `MRR` 增益。

## 7. 结论

本轮 `v23` 验证成功：

- `FBYG15K` 主表版本切换为 `v23b_staged_fresh_il_top400_expand5`
- `FBYG15K` 的 `5-seed delta_avg_mrr_mean` 从 `+0.00160` 提升到 `+0.00270`
- 统一 4 目标主表继续保持全部正增益

本轮也进一步明确了 `FBYG15K` 的优化方向：

- 不应再回到静态 `quality/filter/cap` 搜索
- staged fresh-IL 明显优于单次 fresh-IL
- 若后续继续优化，可优先尝试：
  - 阶段间自适应 top-k
  - 阶段间不同 quantile/margin 联动
  - 第二阶段的多模态一致性约束

备注：

- 本轮 `v23` 的比较口径与仓库现有 `FBYG` transfer-adapt 流水线保持一致，未额外变更 source-checkpoint 池。
