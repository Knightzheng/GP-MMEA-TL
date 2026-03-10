# Transfer Adapt Main Results (4 Targets)

| target | scenario | variant | runs(b/m) | delta H@1 | delta H@10 | delta MRR | delta MR | confidence |
|---|---|---|---:|---:|---:|---:|---:|---|
| ja_en | cross_lingual | v6_mixed | 5/5 | -0.013560 | -0.020770 | -0.016300 | +17.290800 | formal_5seed |
| FBDB15K | cross_graph | v7b_formal | 5/5 | +0.000210 | +0.001950 | +0.000800 | -12.054500 | formal_5seed |
| fr_en | cross_lingual | v14b_refresh4_da0025_expand5 | 5/5 | +0.010140 | +0.016200 | +0.012100 | -7.809800 | formal_5seed |
| FBYG15K | cross_graph | v8_mild_da_expand5 | 5/5 | +0.001000 | +0.002100 | +0.001100 | -12.504400 | formal_5seed |

## Notes

- `ja_en` and `FBDB15K` currently use 2-seed formal snapshots.
- `fr_en` and `FBYG15K` already use 5-seed formal snapshots.
