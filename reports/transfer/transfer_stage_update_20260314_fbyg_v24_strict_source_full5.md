# 迁移实验阶段报告：FBYG15K v24 strict-source staged fresh-IL full5

- 日期：`2026-03-14`
- 目标数据：`FBYG15K`
- 当前旧主表版本：`v23b_staged_fresh_il_top400_expand5`
- 旧参考结果：`5-seed delta_avg_mrr_mean = +0.00270`

## 1. 本轮目标

本轮不再继续向 `FBYG15K` 追加新的复杂机制，而是优先修复一处会影响实验口径一致性的基础问题：

- 部分 seed 缺少 exact `zh_en source formal` checkpoint
- 旧的 source resolver 会在缺失时回退到其他 `transfer_adapt` checkpoint
- 这会让 `FBYG15K` 的 transfer-adapt 结果在不同 seed 上混入不同来源的 source model

因此，本轮最优方案不是继续堆新超参，而是：

1. 补齐缺失的 exact `source formal`
2. 收紧 source resolver，只允许使用 exact formal-source
3. 在严格一致口径下重跑当前最优 staged fresh-IL 路线

## 2. 基础设施修正

### 2.1 source formal 补齐

新增工具：

- `scripts/ensure_transfer_source_formal.py`

用途：

- 为指定 seed 自动补齐 `zh_en epoch10` 的 exact baseline source formal checkpoint
- 缺什么补什么，不会重跑已经存在的 seed

本轮补齐了以下 baseline source formal：

- `seed=2026`
- `seed=7`
- `seed=123`

对应 checkpoint 现在都已存在：

- `MEAformer_DBP15K_zh_en_transfer_src_zh_en_epoch10_baseline_transfer_formal_s2026_src_s2026_`
- `MEAformer_DBP15K_zh_en_transfer_src_zh_en_epoch10_baseline_transfer_formal_s7_src_s7_`
- `MEAformer_DBP15K_zh_en_transfer_src_zh_en_epoch10_baseline_transfer_formal_s123_src_s123_`

### 2.2 source resolver 收紧

修改：

- `scripts/transfer_adapt_utils.py`

新逻辑：

- `resolve_source_model_name(..., allow_nonformal_fallback=False)` 默认只接受 exact formal-source
- 如果 exact formal-source 不存在，则返回空，而不是静默回退到旧的 `transfer_adapt` checkpoint

这一步的作用是把“隐式混用旧 source model”的风险直接封住。

## 3. v24 实验设置

本轮 `v24` 仍使用已验证有效的 staged fresh-IL 路线，但全部建立在 strict formal-source 上：

- `source_resolution = strict_formal_only`
- 仅允许加载 `transfer_src_zh_en_epoch10_baseline_transfer_formal_*`

新增 `v24` 三个变体：

| variant | 说明 |
|---|---|
| v24a | strict-source + staged fresh-IL + phase1 top250 |
| v24b | strict-source + staged fresh-IL + phase1 top400 |
| v24c | strict-source + staged fresh-IL + epoch8 second stage |

对应配置：

- `configs/transfer_adapt/tmmeada_target_fbyg15k_v24a_strictsrc_staged_fresh_il_top250.yaml`
- `configs/transfer_adapt/tmmeada_target_fbyg15k_v24b_strictsrc_staged_fresh_il_top400.yaml`
- `configs/transfer_adapt/tmmeada_target_fbyg15k_v24c_strictsrc_staged_fresh_il_epoch8_top250.yaml`

自动脚本：

- `scripts/run_transfer_adapt_v24_fbyg_iter_queue.py`

## 4. Pilot 结果

`pilot_seeds = [42, 2026]`

| variant | delta_avg_mrr_mean | 结论 |
|---|---:|---|
| v24a | +0.00200 | 正增益，但低于 `v24b` |
| v24b | +0.00300 | 本轮最优 |
| v24c | +0.00200 | 正增益，但低于 `v24b` |

自动决策：

- `best_variant_pilot = v24b`
- `reference_delta_avg_mrr_mean(v23_expand5) = +0.00270`
- `improve_over_current_ref = +0.00030`
- 达到扩展阈值 `+0.00030`
- 自动扩展到 `5-seed`

## 5. Full-5 正式结果

正式对比文件：

- `reports/transfer/transfer_adapt_v24_fbyg_v24b_expand5_compare_vs_baseline.csv`
- `reports/transfer/transfer_adapt_v24_fbyg_v24b_expand5_compare_vs_baseline.md`

正式 `5-seed` 结果（vs baseline）：

- `delta_avg_hits@1_mean = +0.00197`
- `delta_avg_hits@10_mean = +0.00462`
- `delta_avg_mrr_mean = +0.00280`
- `delta_avg_mr_mean = -42.81030`

相对旧主表版本 `v23b`：

- `delta_avg_mrr_mean: +0.00270 -> +0.00280`
- 提升幅度：`+0.00010`

## 6. 关键诊断

这次最重要的不是“再多涨了 0.0001”，而是：

- 在 strict formal-source 口径下，`staged fresh-IL` 依然成立
- 说明 `v23` 的主结论不是由 source-checkpoint 混用偶然抬出来的

`v24b` 日志还给出了一致的正向证据：

- 全部 5 个 seed 都明确加载 exact formal-source
- 日志里 `loading model [...]` 全部指向 `baseline_transfer_formal`
- staged 注入模式也稳定保留：
  - `phase 0`: `100` 条高精度候选
  - `phase 1`: `400` 条补充候选

示例证据：

- `seed=42`
  - `phase 0`: `100` links, `true link ratio = 6.0%`
  - `phase 1`: `400` links, `true link ratio = 0.8%`
- `seed=2026`
  - `phase 0`: `100` links, `true link ratio = 1.0%`
  - `phase 1`: `400` links, `true link ratio = 3.0%`
- `seed=7`
  - `phase 0`: `100` links, `true link ratio = 4.0%`
  - `phase 1`: `400` links, `true link ratio = 2.8%`
- `seed=123`
  - `phase 0`: `100` links, `true link ratio = 3.0%`
  - `phase 1`: `400` links, `true link ratio = 2.5%`

这说明：

1. staged fresh-IL 的正增益在 strict-source 口径下仍然稳定存在；
2. 这轮工作的核心价值首先是“修正实验基础口径”，其次才是“性能再抬一点点”；
3. 经过这次清洗，`FBYG15K` 当前主表结果更适合直接进入论文主表与答辩材料。

## 7. 结论

本轮 `v24` 验证成功：

- `FBYG15K` 主表版本切换为 `v24b_strictsrc_staged_fresh_il_top400_expand5`
- `5-seed delta_avg_mrr_mean` 从 `+0.00270` 提升到 `+0.00280`
- 更重要的是，`FBYG15K` 当前最优结果已经建立在 strict formal-source 的干净口径上

因此，`FBYG15K` 这条线现在的优先级判断也更明确：

- 不需要回到静态 filter/cap 网格
- 也不需要优先再做 source 链路修修补补
- 若继续方法优化，可以直接从当前 `v24b` 往下做：
  - phase-2 adaptive top-k
  - phase-2 consistency constraints
  - phase-wise multimodal agreement
