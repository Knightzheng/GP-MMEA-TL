# FBYG15K v23 Iteration Decision

- timestamp: `20260313-195503`
- pilot_seeds: `[42, 2026]`
- full_seeds: `[42, 3407, 2026, 7, 123]`
- reference_delta_avg_mrr_mean(v21_expand5): `0.0015999999999999973`
- best_variant_pilot: `v23b`
- best_delta_avg_mrr_mean: `0.0030000000000000027`
- improve_over_current_ref: `0.0014000000000000054`
- expand_threshold: `0.0003`
- expanded_variant_to_full5: `v23b`

## Pilot Summary

| variant | delta_avg_mrr_mean | selected_seeds | transfer_skip_keys | transfer_skip_prefixes |
|---|---:|---|---|---|
| v23a | 0.002250000000000002 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |
| v23b | 0.0030000000000000027 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |
| v23c | 0.0020000000000000018 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |

## Expanded Full-5 Summary

- variant: `v23b`
- delta_avg_mrr_mean: `0.002700000000000001`
- selected_seeds: `[7, 42, 123, 2026, 3407]`
- compare_csv: `D:\code\codes\cursor\BYSJ_zyf\reports\transfer\transfer_adapt_v23_fbyg_v23b_expand5_compare_vs_baseline.csv`
