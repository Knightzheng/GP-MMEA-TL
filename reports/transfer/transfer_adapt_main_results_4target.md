# Transfer Adapt Main Results (4 Targets)

| target | scenario | variant | runs(b/m) | delta H@1 | delta H@10 | delta MRR | delta MR | confidence |
|---|---|---|---:|---:|---:|---:|---:|---|
| ja_en | cross_lingual | v15_refresh4_da0025_expand5 | 5/5 | +0.010940 | +0.014100 | +0.012100 | -9.260500 | formal_5seed |
| FBDB15K | cross_graph | v18c_bipartite_late_il_skiprel_expand5 | 5/5 | +0.004540 | +0.015680 | +0.008300 | -206.816700 | formal_5seed |
| fr_en | cross_lingual | v14b_refresh4_da0025_expand5 | 5/5 | +0.010140 | +0.016200 | +0.012100 | -7.809800 | formal_5seed |
| FBYG15K | cross_graph | v21a_fresh_il_q80_skiprel_skipfusion_expand5 | 5/5 | +0.001410 | +0.001930 | +0.001600 | -35.847200 | formal_5seed |

## Notes

- All 4 targets currently use 5-seed formal snapshots.
- `ja_en` now uses the refreshed `v15_refresh4_da0025_expand5` result.
- `FBDB15K` now uses the refreshed `v18c_bipartite_late_il_skiprel_expand5` result.
- `FBYG15K` now uses the refreshed `v21a_fresh_il_q80_skiprel_skipfusion_expand5` result.
