# FBYG15K v21 Iteration Decision

- timestamp: `20260312-135513`
- pilot_seeds: `[42, 2026]`
- full_seeds: `[42, 3407, 2026, 7, 123]`
- reference_delta_avg_mrr_mean(v8_expand5): `0.0011000000000000038`
- best_variant_pilot: `v21a`
- best_delta_avg_mrr_mean: `0.0020000000000000018`
- improve_over_current_ref: `0.000899999999999998`
- expand_threshold: `0.0005`
- expanded_variant_to_full5: `v21a`

## Pilot Summary

| variant | delta_avg_mrr_mean | selected_seeds | transfer_skip_keys | transfer_skip_prefixes |
|---|---:|---|---|---|
| v21a | 0.0020000000000000018 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |
| v21b | 0.0010000000000000009 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |
| v21c | 0.0010000000000000009 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |

## Full-5 Expansion

- compare_csv: `D:\code\codes\cursor\BYSJ_zyf\reports\transfer\transfer_adapt_v21_fbyg_v21a_expand5_compare_vs_baseline.csv`
- delta_avg_mrr_mean: `0.0015999999999999973`

