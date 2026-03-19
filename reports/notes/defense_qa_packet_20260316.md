# 答辩追问口径包（2026-03-16）

## 1. 使用定位

这份文件服务于论文终稿附录与答辩追问，不新增实验结论，只把当前已经形成的主线口径压缩成可直接回答的问题清单。

## 2. 推荐追问与口径

### Q1. 为什么现在可以说项目主线已经基本闭环？

A: 因为任务书与开题报告要求的主线工作已经具备完整证据链：`MEAformer` baseline 已复现，统一 `source-train -> target-adapt -> target-eval` 链路已固定，`ja_en / fr_en / FBDB15K / FBYG15K` 四个目标域都已有 `5-seed` 正式结果，相对 baseline 均呈现稳定正增益，并且还有显著性、案例、效率与可追溯材料共同支撑。这里说的是“主线闭环”，不是“所有辅助项都做到最满”。

支撑文件：
- `reports/transfer/transfer_adapt_main_results_4target.md`
- `reports/transfer/transfer_adapt_significance_summary.md`
- `reports/notes/mainline_traceability_matrix_20260315.md`
- `reports/notes/four_target_evidence_map_20260316.md`

### Q2. 为什么比较范围主要收在 `MEAformer-based transfer setting` 内？

A: 因为当前最可复现、最公平、也最完整落盘的比较链路，是围绕 `MEAformer` 主干建立起来的统一迁移协议。若再补额外 baseline，不是简单多跑一个分数，而是要重新接入完整的目标域自适应、日志、run card 与结果留痕体系。当前阶段更稳妥的做法，是把结论边界明确限定在当前 `MEAformer-based transfer setting` 内，而不是把它外推成对更广泛骨干模型族的充分验证。

支撑文件：
- `reports/transfer/transfer_extra_baseline_limitation_writeup.md`
- `reports/notes/mainline_traceability_matrix_20260315.md`

### Q3. 为什么四个目标域就足以支撑当前主线？

A: 因为它们已经覆盖了两类核心迁移场景：跨语种 (`ja_en / fr_en`) 与跨图谱 (`FBDB15K / FBYG15K`)。更重要的是，这四个目标域不是单次偶然结果，而是都完成了 `5-seed` 正式留痕和一致的统计支持，所以它们足以支撑“当前方法在统一迁移链路下具备可迁移性”的主线结论。

支撑文件：
- `reports/transfer/transfer_adapt_main_results_4target.md`
- `reports/transfer/transfer_adapt_significance_summary.md`

### Q4. 为什么要强调 `5-seed`，而不是只看单次最好结果？

A: 因为论文主线要证明的是“稳定正增益”，不是“偶然挑到一个好 seed”。当前四个目标域都是 `5/5` seed 正增益，且 bootstrap 区间、sign test、Wilcoxon 都给出了保守统计支持，这比单个最好结果更能说明迁移性能和稳定性都得到提升。

支撑文件：
- `reports/transfer/transfer_adapt_significance_summary.md`
- `reports/transfer/transfer_adapt_significance_writeup.md`

### Q5. 案例分析为什么只能当定性支撑？

A: 因为当前案例包只有 `8` 个代表性样本，它的价值在于解释“增益长什么样”和“边界失败长什么样”，而不是充当大样本统计规律。现在它能稳妥支持两件事：跨图谱场景存在明显 rank recovery；跨语种场景仍保留细粒度近邻歧义边界。它不能被写成某个局部模块在所有样本上都被严格证明稳定有效。

支撑文件：
- `reports/transfer/transfer_case_analysis_examples.md`
- `reports/transfer/transfer_case_pattern_summary_20260316.md`

### Q6. 为什么 GPU 峰值显存只做最小正式补测？

A: 因为 GPU 峰值显存属于辅助支撑，不是当前主线是否成立的前提。现有最小正式补测已经能回答“方法是否明显更占显存”这个问题，并且在同环境下观测到 allocator peak 变化有限。继续扩到全目标域、全 seed 的边际收益较低，而且容易把补测时间列误读成与正式训练同预算的效率比较。

支撑文件：
- `reports/transfer/transfer_gpu_peak_minimal_summary.md`
- `reports/transfer/transfer_gpu_peak_minimal_thesis_sync_20260315.md`

### Q7. 为什么没有继续做 H3？

A: 因为 H3 已被明确延期，并且不属于任务书/开题报告当前主线闭环的必要条件。当前最优先的是把统一迁移链路、四目标域正式结果、显著性、案例、效率和可追溯材料收口，而不是为了辅助项重新拉开实验范围。论文中如需提及，只能写成“延期到主线完成后再单独尝试”。

支撑文件：
- `reports/notes/thread_sync_shared.md`
- `reports/notes/taskbook_gap_assessment_20260315.md`

### Q8. 为什么没有继续补额外 baseline？

A: 不是因为额外 baseline 不重要，而是因为真正公平的补法成本很高，需要重新建立完整迁移链路和可追溯留痕体系。当前阶段更高价值的工作，是把已经闭环的主线证据做扎实，并在论文中保守说明比较边界。

支撑文件：
- `reports/transfer/transfer_extra_baseline_limitation_writeup.md`

### Q9. 论文里哪些点不能夸大？

A: 不能把 `source_select` 或 `missing_gate` 写成已经被充分单独证明稳定有效；不能把 GPU 最小补测写成完整显存统计；不能把案例写成大样本规律；不能把当前结果外推成对更广泛外部骨干模型都已完成验证。

支撑文件：
- `reports/notes/thesis_final_integration_packet_20260316.md`
- `reports/notes/four_target_evidence_map_20260316.md`

## 3. 最简答辩收口

如果答辩时间很紧，最稳妥的收口是：

“本文先在统一的 `MEAformer-based transfer setting` 下建立了可复现的迁移实验链路，再在两个跨语种和两个跨图谱目标域上完成 `5-seed` 正式验证。结果显示，方法相对匹配 baseline 在四个目标域上都取得了稳定正增益，并且显著性、案例、效率与仓库可追溯材料共同支持这一主线结论。与此同时，我们也保守保留了比较边界、GPU 补测边界和案例定性边界，没有把辅助项夸大为新的主线结论。”
