# 迁移实验阶段报告（FBYG15K v22 quality-filter pilot）

- 日期：`2026-03-13`
- 目标域：`FBYG15K`
- 当前参考主表版本：`v21a_fresh_il_q80_skiprel_skipfusion_expand5`
- 参考结果：`5-seed delta_avg_mrr_mean = +0.00160`

## 1. 本轮目标

在 `v21` 已证明 fresh-IL 立即注入有效的基础上，继续提升 `FBYG15K` 的伪链接质量，验证“静态质量过滤 + 注入上限”是否能进一步超过当前主表版本。

本轮不再改：

- `domain_align_weight`
- `transfer_skip_prefixes`
- `late IL` 调度

本轮只改 `IL` 候选质量控制：

- `confidence min`
- `confidence-margin min`
- `quality quantile`
- `topk cap`

## 2. 代码改动

本轮在 `IL` 伪链接链路中新增了质量优先过滤能力：

- `baselines/MEAformer/config.py`
  - 新增 `il_margin_min`
  - 新增 `il_quality_quantile`
  - 新增 `il_topk_max`
  - 新增 `il_margin_weight`
- `baselines/MEAformer/model/MEAformer.py`
  - 在 `Iter_new_links` 中为互选候选计算：
    - `confidence`
    - `confidence margin`
    - `quality = confidence + margin_weight * margin`
  - 过滤顺序改为：
    - `confidence threshold`
    - `margin threshold`
    - `quality quantile`
    - `topk cap`
- `baselines/MEAformer/main.py`
  - 日志增加质量过滤统计
- `scripts/run_meaformer.py`
  - 补齐新参数透传

## 3. 实验设置

新增 `v22` 三个 `2-seed pilot` 变体，均保持 `v21` 的：

- `fresh IL`
- `skip rel_fc`
- `skip fusion`

变体如下：

| variant | confidence_min | margin_min | quality_q | topk_max |
|---|---:|---:|---:|---:|
| v22a | 0.62 | 0.008 | 0.80 | 200 |
| v22b | 0.63 | 0.012 | 0.85 | 100 |
| v22c | 0.60 | 0.005 | 0.70 | 300 |

自动脚本：

- `scripts/run_transfer_adapt_v22_fbyg_iter_queue.py`

## 4. Pilot 结果

`pilot_seeds = [42, 2026]`

| variant | delta_avg_mrr_mean | 结论 |
|---|---:|---|
| v22a | +0.00050 | 明显低于 `v21` |
| v22b | +0.00125 | 本轮最优，但仍低于 `v21` |
| v22c | +0.00125 | 与 `v22b` 持平，仍低于 `v21` |

自动决策：

- `best_variant_pilot = v22b`
- `improve_over_current_ref = -0.00035`
- 未达到扩展阈值 `+0.00030`
- 不扩展到 `5-seed`

## 5. 关键诊断

`v22` 的静态质量过滤确实提高了部分 seed 的伪链接真值率，但跨 seed 稳定性不够，最终没有超过 `v21`。

### v22a

- `seed=42`: `raw=2247 -> kept=200`, `true link ratio=3.5%`
- `seed=2026`: `raw=2139 -> kept=200`, `true link ratio=1.5%`
- `2-seed delta_avg_mrr_mean = +0.00050`

### v22b

- `seed=42`: `raw=2247 -> kept=100`, `true link ratio=6.0%`
- `seed=2026`: `raw=2139 -> kept=100`, `true link ratio=1.0%`
- `2-seed delta_avg_mrr_mean = +0.00125`

### v22c

- `seed=42`: `raw=2247 -> kept=300`, `true link ratio=2.7%`
- `seed=2026`: `raw=2139 -> kept=300`, `true link ratio=1.0%`
- `2-seed delta_avg_mrr_mean = +0.00125`

可以看出：

1. `quality + topk cap` 在 `seed=42` 上是有效的，尤其 `v22b` 把伪链接真值率提升到了 `6.0%`；
2. 但 `seed=2026` 在三种过滤强度下都只有 `1.0% ~ 1.5%`，说明问题不是“过滤还不够严”，而是静态单次过滤缺乏跨 seed 稳定性；
3. `v22` 比 `v21` 更像是在做“更干净但更少”的单次注入，精度提升没有稳定转化为更好的最终 `MRR`。

## 6. 结论

本轮 `v22` 不能替换 `v21`：

- `FBYG15K` 主表版本保持 `v21a_fresh_il_q80_skiprel_skipfusion_expand5`
- `v22` 不扩展到 `5-seed`
- 当前统一 4 目标主结果表保持不变

若继续优化 `FBYG15K`，下一步不应再做静态 `quality/filter/cap` 网格搜索，而应考虑：

1. 分阶段注入策略：
   - 先注入极小高精集合，再在后续 epoch 补充第二批候选
2. 自适应 topk：
   - 不固定 `100/200/300`，而按候选质量分布动态确定注入规模
3. 多信号一致性：
   - 不只用 joint embedding 的单次排序，而引入跨模态或前后 epoch 一致性约束
