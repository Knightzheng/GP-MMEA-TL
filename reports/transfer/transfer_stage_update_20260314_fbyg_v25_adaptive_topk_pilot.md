# 迁移实验阶段报告：FBYG15K v25 adaptive-topk pilot

- 日期：`2026-03-14`
- 目标数据：`FBYG15K`
- 当前主表版本：`v24b_strictsrc_staged_fresh_il_top400_expand5`
- 当前主表结果：`5-seed delta_avg_mrr_mean = +0.00280`

## 1. 本轮目标

`v24b` 已经证明：`FBYG15K` 的 staged fresh-IL 在 strict formal-source 口径下是成立的。

因此本轮不再回到静态 `filter/cap` 搜索，而是直接验证更具体的下一步假设：

1. `phase 0` 先用高质量小规模注入稳定训练；
2. `phase 1` 再根据上一阶段实际留下来的候选规模，自适应决定最终注入上限；
3. 如果固定 `top400` 仍然偏硬，那么 `phase-2 adaptive top-k` 应该有机会进一步超过 `v24b`。

## 2. 代码改动

本轮在 `MEAformer` 的 IL 过滤链路里新增了按阶段自适应控制 `top-k` 的能力：

- `baselines/MEAformer/config.py`
  - 新增：
    - `il_adaptive_topk`
    - `il_adaptive_topk_scale`
    - `il_adaptive_topk_min`
    - `il_adaptive_topk_scale_schedule`
    - `il_adaptive_topk_min_schedule`
- `baselines/MEAformer/model/MEAformer.py`
  - 为每个 fresh-IL phase 记录 `pre_topk_count`
  - 在后续 phase 中按上一阶段 `pre_topk_count * scale` 生成新的 `effective_topk`
  - 将 `effective_topk / prev_pre_topk / adaptive_scale / adaptive_min` 写入统计信息
- `baselines/MEAformer/main.py`
  - 在日志中输出 `effective_topk`、`pre_topk` 与 adaptive 信息
- `scripts/run_meaformer.py`
  - 补齐上述参数透传

这次改动的目的不是直接“再多注一点”，而是让第二阶段的注入规模跟着每个 seed 的候选质量波动自适应变化。

## 3. v25 变体设置

新增自动脚本：

- `scripts/run_transfer_adapt_v25_fbyg_iter_queue.py`

新增配置：

- `configs/transfer_adapt/tmmeada_target_fbyg15k_v25a_strictsrc_staged_adaptivetopk_s100.yaml`
- `configs/transfer_adapt/tmmeada_target_fbyg15k_v25b_strictsrc_staged_adaptivetopk_s125.yaml`
- `configs/transfer_adapt/tmmeada_target_fbyg15k_v25c_strictsrc_staged_adaptivetopk_s100_min300.yaml`

三组变体的区别是：

| variant | 说明 |
|---|---|
| `v25a` | `phase1 adaptive scale = 1.00`, `phase1 min = 250` |
| `v25b` | `phase1 adaptive scale = 1.25`, `phase1 min = 250` |
| `v25c` | `phase1 adaptive scale = 1.00`, `phase1 min = 300` |

统一设置：

- 仍使用 strict formal-source
- 仍使用 staged fresh-IL（`epoch 5` + `epoch 7`）
- 仍保持保守迁移加载：
  - `transfer_skip_keys = multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias`
  - `transfer_skip_prefixes = multimodal_encoder.fusion.`

## 4. Pilot 结果

`pilot_seeds = [42, 2026]`

参考主表版本：

- `v24b full5 delta_avg_mrr_mean = +0.00280`

本轮 `v25` pilot（vs baseline）：

| variant | delta_avg_mrr_mean | 结论 |
|---|---:|---|
| `v25a` | `+0.00200` | 正增益，但低于当前主表 |
| `v25b` | `+0.00200` | 正增益，但低于当前主表 |
| `v25c` | `+0.00250` | 本轮最优，但仍未超过 `v24b` |

自动决策结果：

- `best_variant_pilot = v25c`
- `reference_delta_avg_mrr_mean(v24_expand5) = +0.00280`
- `improve_over_current_ref = -0.00030`
- 未达到扩展阈值 `+0.00030`
- 不扩展到 `5-seed`

对应文件：

- `reports/transfer/transfer_adapt_v25_fbyg_iter_decision.md`
- `reports/transfer/transfer_adapt_v25_fbyg_iter_decision.json`
- `reports/transfer/transfer_adapt_v25_fbyg_pilot_v25a_compare_vs_baseline.md`
- `reports/transfer/transfer_adapt_v25_fbyg_pilot_v25b_compare_vs_baseline.md`
- `reports/transfer/transfer_adapt_v25_fbyg_pilot_v25c_compare_vs_baseline.md`

## 5. 机制是否真的生效

这轮最重要的一个问题，不是“最后有没有赢”，而是“adaptive top-k 有没有真的工作起来”。

日志给出的答案是：有，而且按预期工作了。

代表性证据：

- `v25c / seed=42`
  - `phase 0`: `pre_topk=233`, `effective_topk=100`, `true link ratio = 6.0%`
  - `phase 1`: `prev_pre_topk=233`, `effective_topk=300`, `true link ratio = 1.0%`
- `v25c / seed=2026`
  - `phase 0`: `pre_topk=200`, `effective_topk=100`, `true link ratio = 1.0%`
  - `phase 1`: `prev_pre_topk=200`, `effective_topk=300`, `true link ratio = 3.3%`
- `v25b / seed=42`
  - `phase 1`: `prev_pre_topk=233`, `effective_topk=291`, `true link ratio = 1.0%`
- `v25a / seed=2026`
  - `phase 1`: `prev_pre_topk=200`, `effective_topk=250`, `true link ratio = 3.2%`

这说明：

1. 第二阶段注入规模已经不再是固定 `400`；
2. 它确实会跟随上一阶段候选规模变化；
3. 机制验证是成功的，不是“代码加了但实验里没真正用上”。

## 6. 为什么仍然没有超过 v24b

从结果看，问题不在“adaptive top-k 没启动”，而在“它带来的候选规模变化还不足以稳定转化为更高 MRR”。

当前更合理的诊断是：

1. `phase 0` 的高质量小批注入是有效的，且在不同 seed 上都能看到明显更高的真值率；
2. `phase 1` 即使改成自适应上限，新增链接的真值率仍然偏低，尤其在部分 seed 上只有约 `1.0%`；
3. 因此当前瓶颈更像是“第二阶段候选的一致性和可靠性不够”，而不是“第二阶段到底该注入 250、291、300 还是 400”。

换句话说，这次已经验证：

- 单纯把 `phase 2` 的注入数量从“固定阈值”改成“自适应阈值”，方向是合理的；
- 但只靠这一点，还不足以超过当前最优主表版本。

## 7. 结论

本轮 `v25` 的结论是：

- `adaptive top-k` 机制已经实现并验证生效；
- `v25c` 是本轮最优 pilot，但 `delta_avg_mrr_mean = +0.00250`，仍低于当前主表 `v24b (+0.00280)`；
- 因此 `FBYG15K` 主表版本保持不变，继续使用 `v24b_strictsrc_staged_fresh_il_top400_expand5`；
- 这轮不扩展到 `5-seed`。

对下一步优化的启示也更明确：

- 不需要回到静态 `quality/filter/cap` 网格；
- 也不需要继续单独搜索更多 `adaptive top-k` 数值；
- 如果继续做 `FBYG15K`，更优先的方向应是：
  - `phase-wise consistency constraints`
  - `phase-2 candidate consistency / agreement gating`
