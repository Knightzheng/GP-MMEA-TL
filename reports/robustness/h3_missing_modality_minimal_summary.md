# H3 Missing-Modality Minimal Summary

- dataset: `zh_en`
- matrix: `v1_full / wo_missing_gate × drop_rate {0.0, 0.6} × seed=42`
- note: `drop_rate=0.0` rows reuse previously completed same-config `epoch3` logs; `drop_rate=0.6` rows are fresh reruns with missing-image injection and GPU peak logging
- omitted in this minimal round: `baseline`, intermediate `drop_rate=0.3`, and multi-seed repetition
- note: GPU peak numbers come from `torch.cuda.max_memory_allocated / reserved`; under Windows `WDDM`, use them mainly for relative comparison instead of direct physical-VRAM interpretation

## Paper-Ready Table

| Variant | Drop Rate | Source | avg Hits@1 | avg Hits@10 | avg MRR | GPU Peak Alloc (MB) | GPU Peak Reserv (MB) |
|---|---:|---|---:|---:|---:|---:|---:|
| v1_full | 0.00 | existing_reference_same_config | 0.6270 | 0.8961 | 0.7180 | - | - |
| v1_full | 0.60 | fresh_h3_rerun | 0.4887 | 0.8288 | 0.6020 | 7266.98 | 8750.00 |
| wo_missing_gate | 0.00 | existing_reference_same_config | 0.6270 | 0.8961 | 0.7180 | - | - |
| wo_missing_gate | 0.60 | fresh_h3_rerun | 0.4887 | 0.8288 | 0.6020 | 7267.55 | 8750.00 |

## Degradation View

| Variant | avg MRR @0.0 | avg MRR @0.6 | Delta MRR | avg Hits@1 @0.0 | avg Hits@1 @0.6 | Delta Hits@1 |
|---|---:|---:|---:|---:|---:|---:|
| v1_full | 0.7180 | 0.6020 | -0.1160 | 0.6270 | 0.4887 | -0.1383 |
| wo_missing_gate | 0.7180 | 0.6020 | -0.1160 | 0.6270 | 0.4887 | -0.1383 |

## Thesis Usage Boundary

- This minimal round can support only a **single-seed pilot** observation under severe simulated image loss.
- It can be used to describe whether `v1_full` still maintains or fails to maintain an advantage over `wo_missing_gate` at `drop_rate=0.6`.
- It cannot support any strong claim about multi-seed stability, full degradation curves, or the independent effectiveness of `missing_gate` across targets.
