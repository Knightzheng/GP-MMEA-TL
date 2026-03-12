# FBDB15K v18 Iteration Decision

- timestamp: `20260312-040831`
- pilot_seeds: `[42, 2026]`
- full_seeds: `[42, 3407, 2026, 7, 123]`
- reference_delta_avg_mrr_mean(v7_expand5): `0.0007999999999999986`
- best_variant_pilot: `v18c`
- best_delta_avg_mrr_mean: `0.008`
- improve_over_current_ref: `0.0072000000000000015`
- expand_threshold: `0.0008`
- expanded_variant_to_full5: `v18c`

## Pilot Summary

| variant | delta_avg_mrr_mean | selected_seeds | transfer_skip_keys |
|---|---:|---|---|
| v18a | 0.0075 | [42, 2026] | multimodal_encoder.entity_emb.weight |
| v18b | 0.006999999999999999 | [42, 2026] | multimodal_encoder.entity_emb.weight |
| v18c | 0.008 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias |

## Full-5 Expansion

- compare_csv: `D:\code\codes\cursor\BYSJ_zyf\reports\transfer\transfer_adapt_v18_fbdb_v18c_expand5_compare_vs_baseline.csv`
- delta_avg_mrr_mean: `0.008299999999999998`

