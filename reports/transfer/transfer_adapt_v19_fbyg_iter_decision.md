# FBYG15K v19 Iteration Decision

- timestamp: `20260312-111157`
- pilot_seeds: `[42, 2026]`
- full_seeds: `[42, 3407, 2026, 7, 123]`
- reference_delta_avg_mrr_mean(v8_expand5): `0.0011000000000000038`
- best_variant_pilot: `v19c`
- best_delta_avg_mrr_mean: `0.0010000000000000009`
- improve_over_current_ref: `-0.00010000000000000286`
- expand_threshold: `0.0005`
- expanded_variant_to_full5: `None`

## Pilot Summary

| variant | delta_avg_mrr_mean | selected_seeds | transfer_skip_keys | transfer_skip_prefixes |
|---|---:|---|---|---|
| v19a | -0.002249999999999995 | [42, 2026] | multimodal_encoder.entity_emb.weight |  |
| v19b | -0.0024999999999999953 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias |  |
| v19c | 0.0010000000000000009 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |

