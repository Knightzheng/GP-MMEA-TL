# FBYG15K v22 Iteration Decision

- timestamp: `20260312-233553`
- pilot_seeds: `[42, 2026]`
- full_seeds: `[42, 3407, 2026, 7, 123]`
- reference_delta_avg_mrr_mean(v21_expand5): `0.0015999999999999973`
- best_variant_pilot: `v22b`
- best_delta_avg_mrr_mean: `0.0012500000000000011`
- improve_over_current_ref: `-0.00034999999999999615`
- expand_threshold: `0.0003`
- expanded_variant_to_full5: `None`

## Pilot Summary

| variant | delta_avg_mrr_mean | selected_seeds | transfer_skip_keys | transfer_skip_prefixes |
|---|---:|---|---|---|
| v22a | 0.0005000000000000004 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |
| v22b | 0.0012500000000000011 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |
| v22c | 0.0012500000000000011 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |

