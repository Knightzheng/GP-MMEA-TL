# FBYG15K v24 Iteration Decision

- timestamp: `20260314-054535`
- pilot_seeds: `[42, 2026]`
- full_seeds: `[42, 3407, 2026, 7, 123]`
- source_resolution: `strict_formal_only`
- reference_delta_avg_mrr_mean(v23_expand5): `0.002700000000000001`
- best_variant_pilot: `v24b`
- best_delta_avg_mrr_mean: `0.0030000000000000027`
- improve_over_current_ref: `0.00030000000000000165`
- expand_threshold: `0.0003`
- expanded_variant_to_full5: `v24b`

## Pilot Summary

| variant | delta_avg_mrr_mean | selected_seeds | transfer_skip_keys | transfer_skip_prefixes |
|---|---:|---|---|---|
| v24a | 0.0020000000000000018 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |
| v24b | 0.0030000000000000027 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |
| v24c | 0.0020000000000000018 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |

## Expanded Full-5 Summary

- variant: `v24b`
- delta_avg_mrr_mean: `0.002800000000000004`
- selected_seeds: `[7, 42, 123, 2026, 3407]`
- compare_csv: `D:\code\codes\cursor\BYSJ_zyf\reports\transfer\transfer_adapt_v24_fbyg_v24b_expand5_compare_vs_baseline.csv`
