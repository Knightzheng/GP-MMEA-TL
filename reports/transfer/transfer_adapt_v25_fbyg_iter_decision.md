# FBYG15K v25 Iteration Decision

- timestamp: `20260314-132856`
- pilot_seeds: `[42, 2026]`
- full_seeds: `[42, 3407, 2026, 7, 123]`
- optimization_theme: `phase2_adaptive_topk`
- source_resolution: `strict_formal_only`
- reference_delta_avg_mrr_mean(v24_expand5): `0.002800000000000004`
- best_variant_pilot: `v25c`
- best_delta_avg_mrr_mean: `0.0025000000000000022`
- improve_over_current_ref: `-0.00030000000000000165`
- expand_threshold: `0.0003`
- expanded_variant_to_full5: `None`

## Pilot Summary

| variant | delta_avg_mrr_mean | selected_seeds | transfer_skip_keys | transfer_skip_prefixes |
|---|---:|---|---|---|
| v25a | 0.0020000000000000018 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |
| v25b | 0.0020000000000000018 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |
| v25c | 0.0025000000000000022 | [42, 2026] | multimodal_encoder.entity_emb.weight,multimodal_encoder.rel_fc.weight,multimodal_encoder.rel_fc.bias | multimodal_encoder.fusion. |
