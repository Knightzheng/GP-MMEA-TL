# FBDB15K v17 Iteration Decision

- timestamp: `20260312-012355`
- pilot_seeds: `[42, 2026]`
- full_seeds: `[42, 3407, 2026, 7, 123]`
- reference_delta_avg_mrr_mean(v7_expand5): `0.0007999999999999986`
- best_variant_pilot: `v17c`
- best_delta_avg_mrr_mean: `-0.007750000000000003`
- improve_over_current_ref: `-0.008550000000000002`
- expand_threshold: `0.0008`
- expanded_variant_to_full5: `None`

## Pilot Summary

| variant | delta_avg_mrr_mean | selected_seeds | transfer_skip_keys |
|---|---:|---|---|
| v17a | -0.008 | [42, 2026] | multimodal_encoder.entity_emb.weight |
| v17b | -0.0085 | [42, 2026] | multimodal_encoder.entity_emb.weight |
| v17c | -0.007750000000000003 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias |

