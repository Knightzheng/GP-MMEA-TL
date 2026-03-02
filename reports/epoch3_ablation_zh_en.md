# zh_en epoch3 ablation (seed=42)

| variant | l2r H@1 | l2r H@10 | l2r MRR | r2l H@1 | r2l H@10 | r2l MRR | d(l2r H@1) vs full | d(r2l H@1) vs full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.6272 | 0.8970 | 0.7190 | 0.6262 | 0.8952 | 0.7170 | -0.0006 | +0.0001 |
| v1_best_full | 0.6278 | 0.8969 | 0.7190 | 0.6261 | 0.8952 | 0.7170 | +0.0000 | +0.0000 |
| wo_domain_align | 0.6278 | 0.8969 | 0.7190 | 0.6261 | 0.8952 | 0.7170 | +0.0000 | +0.0000 |
| wo_source_select | 0.6272 | 0.8970 | 0.7190 | 0.6262 | 0.8952 | 0.7170 | -0.0006 | +0.0001 |
| wo_missing_gate | 0.6278 | 0.8969 | 0.7190 | 0.6261 | 0.8952 | 0.7170 | +0.0000 | +0.0000 |

Notes:
- This is a pilot ablation under zh_en + epoch3 + seed=42.
- For formal claims, extend each variant to the same 5-seed setting.