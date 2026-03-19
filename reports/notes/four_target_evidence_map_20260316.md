# 四目标域主证据地图（2026-03-16）

## 1. 使用定位

这份文件用于把当前主线最关键的 `4` 个目标域证据压缩到一页内，方便论文终稿、附录或答辩直接引用。

- 适用场景：附录、答辩、导师追问时的主线总览
- 不宜上升为：新的正文实验结论

## 2. 四目标域主证据地图

| target | main gain | statistical support | case support | efficiency / memory support | conservative boundary |
| --- | --- | --- | --- | --- | --- |
| `ja_en` | `delta H@1=+0.010940`, `delta MRR=+0.0121` | `5/5` seeds positive; bootstrap `95% CI=[+0.0106,+0.0135]`; sign test `p=0.0312` | 保留 `3` 个跨语种失败/边界案例：`Inspiration is DEAD` (`idx=3275`), `Windows 10 Mobile` (`idx=1201`), `Fat Mike` (`idx=9563`)；用于说明细粒度近邻歧义仍存在 | wall-clock `46.69 -> 81.12 min` (`1.74x`); GPU 最小补测 allocator peak `7494.68 -> 7496.18 MB`，变化很小 | 能支持“稳定正增益 + 显存额外开销有限”；不能写成“跨语种细粒度歧义已被彻底解决” |
| `fr_en` | `delta H@1=+0.010140`, `delta MRR=+0.0121` | `5/5` seeds positive; bootstrap `95% CI=[+0.0110,+0.0134]`; sign test `p=0.0312` | 当前未单独保留 `fr_en` 代表性案例；该目标域主要由正式 `5-seed` 主表与显著性支撑 | wall-clock `52.70 -> 108.36 min` (`2.06x`); 无单独 GPU 最小补测 | 能支持“在第二个跨语种目标域上仍保持稳定正增益”；不能夸大为“所有目标域都已有同等粒度案例与显存证据” |
| `FBDB15K` | `delta H@1=+0.004540`, `delta MRR=+0.0083`, `delta MR=-206.8167` | `5/5` seeds positive; bootstrap `95% CI=[+0.0073,+0.0091]`; sign test `p=0.0312` | 保留 `3` 个跨图谱纠错案例：`Post-bop` (`idx=2283`), `The Pacific` (`idx=7880`), `JavaScript` (`idx=1959`)；集中支撑大幅 rank recovery | wall-clock `14.47 -> 4.64 min` (`0.32x`); 当前无单独 GPU 最小补测 | 能支持“跨图谱场景下存在明显 rank recovery”；不能把案例写成对单个模块机制的严格独立证明 |
| `FBYG15K` | `delta H@1=+0.001970`, `delta MRR=+0.0028`, `delta MR=-42.8103` | `5/5` seeds positive; bootstrap `95% CI=[+0.0021,+0.0034]`; sign test `p=0.0312` | 保留 `2` 个跨图谱纠错案例：`Saboteur (film)` (`idx=4851`), `Amritsar` (`idx=2903`)；支撑较难样本的 recovery | wall-clock `18.32 -> 21.02 min` (`1.15x`); GPU 最小补测 allocator peak `5832.93 -> 5833.29 MB`，变化很小 | 能支持“在较弱增益目标域上仍有稳定正增益”；`method` 最小补测实际有效 `epoch=6`，不宜把时间列解释成同预算比较 |

## 3. 当前最稳妥的一句话结论

当前四个目标域的主证据可以统一收口为：在统一的 `source-train -> target-adapt -> target-eval` 链路下，方法相对匹配 baseline 在 `4` 个目标域上都取得了 `5-seed` 稳定正增益，跨图谱场景有明确的 rank recovery 质性支撑，跨语种场景则同时保留了细粒度边界失败样本，说明结论是“稳定有效但边界受控”，而不是“所有困难样本都已被完全解决”。

## 4. 直接来源

- `reports/transfer/transfer_adapt_main_results_4target.md`
- `reports/transfer/transfer_adapt_significance_summary.md`
- `reports/transfer/transfer_case_analysis_examples.md`
- `reports/transfer/transfer_case_pattern_summary_20260316.md`
- `reports/transfer/transfer_efficiency_summary.md`
- `reports/transfer/transfer_gpu_peak_minimal_summary.md`
