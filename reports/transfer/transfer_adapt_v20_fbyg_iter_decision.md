# FBYG15K v20 Iteration Decision

- timestamp: `20260312-123744`
- pilot_seeds: `[42, 2026]`
- full_seeds: `[42, 3407, 2026, 7, 123]`
- reference_delta_avg_mrr_mean(v8_expand5): `0.0011000000000000038`
- best_variant_pilot: `v20a`
- best_delta_avg_mrr_mean: `0.0005000000000000004`
- improve_over_current_ref: `-0.0006000000000000033`
- expand_threshold: `0.0005`
- expanded_variant_to_full5: `None`

## Pilot Summary

| variant | delta_avg_mrr_mean | selected_seeds | transfer_skip_keys | transfer_skip_prefixes |
|---|---:|---|---|---|
| v20a | 0.0005000000000000004 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |
| v20b | 0.0005000000000000004 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |

