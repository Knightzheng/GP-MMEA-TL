# 论文终稿整合包（2026-03-16）

## 1. 当前最推荐的下一步

当前最值得继续做的，不是再补新实验，而是把现有主线结果、辅助补强和边界说明收束成一份论文终稿可直接吸收的整合包。原因有三点：

1. 主线实验已经闭环，继续新增实验的边际收益明显下降。
2. 论文线程当前最需要的，不是更多原始结果，而是更短、更稳、更容易直接粘贴进第四章、第五章和附录的整合材料。
3. 当前仓库已经有主表、显著性、案例、效率、GPU、局限性和可追溯性材料，但它们分散在多个文件中，仍有整合空间。

## 2. 对项目优化线程的下一步要求

1. 不再重启 `H3`，也不再把 `H3` 当作当前阶段任务。
2. 不再为了辅助项重开大规模主线 rerun。
3. 若继续优化，优先做“论文终稿吸收效率”相关工作，而不是继续堆实验。
4. 新增内容优先整理成：
   - `md` 版章节整合说明；
   - 可直接画图或回填表格的 `csv`；
   - 第四章可直接吸收的分析段落；
   - 第五章可直接吸收的结论 / 局限性段落；
   - 附录 / 答辩可直接使用的可追溯材料。
5. 若某项不再值得继续投入，必须同步给出：
   - 为什么不做；
   - 论文里应如何保守表述；
   - 答辩时如何解释。

## 3. 第四章建议吸收路径

### 3.1 主实验结果

- 主入口：
  - `reports/transfer/transfer_adapt_main_results_4target.md`
  - `reports/transfer/transfer_adapt_main_results_4target.csv`
- 建议正文作用：
  - 作为第四章主表与总结果概述。
- 建议口径：
  - 围绕“统一迁移链路 + 4 个目标域 `5-seed` 稳定正增益”展开。

### 3.2 显著性与稳定性分析

- 主入口：
  - `reports/transfer/transfer_adapt_significance_summary.md`
  - `reports/transfer/transfer_adapt_significance_writeup.md`
- 建议正文作用：
  - 作为第四章中“统计显著性与稳定性分析”小节的直接来源。
- 建议口径：
  - 强调 `4/4` 目标域 `5/5 seed` 正增益与保守统计支持；
  - 不额外夸大到“所有模块均被严格单独验证”。

### 3.3 案例分析

- 主入口：
  - `reports/transfer/transfer_case_analysis_examples.md`
  - `reports/transfer/transfer_case_analysis_examples.csv`
  - `reports/transfer/transfer_case_analysis_thesis_sync_20260315.md`
- 建议正文作用：
  - 若篇幅有限，正文精选 `6` 个案例；
  - 剩余 `2` 个案例放附录或答辩材料。
- 建议口径：
  - 成功案例支持跨图谱 `rank recovery`；
  - 失败案例保留跨语言细粒度歧义边界。

### 3.4 效率与额外开销说明

- 主入口：
  - `reports/transfer/transfer_efficiency_summary.md`
  - `reports/transfer/transfer_gpu_peak_minimal_summary.md`
  - `reports/transfer/transfer_gpu_peak_minimal_thesis_sync_20260315.md`
  - `reports/transfer/transfer_gpu_peak_minimal_chart_ready.csv`
- 建议正文作用：
  - wall-clock 放第四章效率分析主位置；
  - GPU 放辅助补证位置。
- 建议口径：
  - GPU 仅写成代表性场景下的最小正式补测；
  - 不替代全目标域、全 seed 的统一显存统计。

### 3.5 讨论与局限性

- 主入口：
  - `reports/transfer/transfer_extra_baseline_limitation_writeup.md`
- 建议正文作用：
  - 作为第四章“讨论与局限性分析”与第五章“研究不足/展望”的直接文字来源。
- 建议口径：
  - 当前比较边界保守收在 `MEAformer-based transfer setting` 内；
  - 不把未补额外 baseline 写成“已经没有必要”。

## 4. 第五章建议吸收路径

### 4.1 结论

建议第五章继续围绕以下主线收口：

1. 本文建立了统一的 `source-train -> target-adapt -> target-eval` 迁移实验链路。
2. 在 `ja_en / fr_en / FBDB15K / FBYG15K` 四个目标域上，本文方法相对匹配 baseline 均取得了 `5-seed` 稳定正增益。
3. 目标域自适应与伪标签质量控制对迁移性能和稳定性具有积极作用。

### 4.2 局限性与展望

建议第五章保持以下边界：

1. 不把 `source_select` / `missing_gate` 写成已被充分单独严格证明。
2. 不把 GPU 最小正式补测写成完整显存统计。
3. 不把当前结论外推为对更广泛骨干模型族均已完成泛化验证。
4. 不再提 `H3` 当前仓库结果；若必须说明，只写“已延期到主线结束后再单独尝试”。

## 5. 附录与答辩建议吸收路径

- 可追溯性主入口：
  - `reports/notes/mainline_traceability_matrix_20260315.md`
  - `reports/notes/mainline_closure_onepage_20260315.md`
  - `reports/notes/mainline_artifact_integrity_20260315.md`
- 使用建议：
  - 附录中放主线追溯总表；
  - 答辩中用一页式主线闭环说明；
  - 导师追问或答辩前自查时使用完整性校验报告。

## 6. 当前不再推荐继续投入的事项

1. `H3`
   - 原因：已明确延期，不属于当前论文主线。
   - 论文写法：不纳入当前论文主体。
2. 更大规模 GPU 补测
   - 原因：边际收益低，当前最小正式补测已足够承担辅助支撑角色。
   - 论文写法：只作代表性场景下的相对参考。
3. 额外 baseline 新 rerun
   - 原因：公平接入成本过高，不适合当前阶段。
   - 论文写法：明确比较边界保守限制在 `MEAformer-based transfer setting` 内。

## 7. 可直接交给论文线程吸收的总建议

当前论文进一步优化的最佳方向，不是继续扩实验，而是集中完成“终稿整合”。也就是说，应以当前已经闭环的主线结果为核心，把显著性、案例、效率、GPU 辅助补证、局限性表述、主线闭环说明和可追溯性材料按章节重新组织起来：第四章突出主实验、统计支撑、案例证据和效率说明，第五章突出保守结论与边界控制，附录和答辩材料则使用主线追溯总表、一页式闭环说明和完整性校验报告。按照这个路径推进，能在不新增高成本实验的前提下，最大化提升论文终稿与答辩材料的完成度和说服力。
