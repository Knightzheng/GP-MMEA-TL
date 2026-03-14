# Transfer GPU Peak Minimal Summary

- scope: `seed=42`, representative targets `ja_en` and `FBYG15K`, 1-epoch target-adapt reruns with the same batch size and model structure as the formal configs
- note: `elapsed_minutes` here reflects the 1-epoch补测 runtime, not the formal 5-seed full-chain wall-clock already reported in the thesis
- note: GPU peak numbers come from `torch.cuda.max_memory_allocated / reserved`; under Windows `WDDM`, these allocator-level peaks may differ from `nvidia-smi` instantaneous physical usage, so they are better used for relative comparison within the same environment
- note: some variants require a higher effective epoch than the requested minimum in order to remain valid under their original `il_start` settings
- `FBYG15K` uses mixed effective epochs in this minimal supplement (`epoch=3, 6`), so runtime should not be interpreted as a same-budget comparison.

## Paper-Ready Table

| Target | Variant | Seed | Epoch | avg Hits@1 | avg Hits@10 | avg MRR | GPU Peak Alloc (MB, PyTorch) | GPU Peak Reserv (MB, PyTorch) | 1-epoch Time (min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FBYG15K | baseline | 42 | 3 | 0.0339 | 0.0856 | 0.0540 | 5832.93 | 9774.00 | 5.70 |
| FBYG15K | method | 42 | 6 | 0.0365 | 0.0990 | 0.0600 | 5833.29 | 9804.00 | 11.25 |
| ja_en | baseline | 42 | 3 | 0.3433 | 0.5135 | 0.4020 | 7494.68 | 8932.00 | 11.73 |
| ja_en | method | 42 | 3 | 0.3433 | 0.5135 | 0.4020 | 7496.18 | 8972.00 | 10.42 |

## Thesis Usage Boundary

- These runs can support a restrained statement about relative peak memory under representative target-adapt settings in the current Windows/PyTorch environment.
- They do not replace the formal 5-seed wall-clock statistics and should be cited as a supplementary memory补测.
- Because the measurement reruns only `1` epoch, the time column is only for transparency; the peak-memory column is the main result.
- If absolute values appear larger than the device's nominal physical memory, interpret them as allocator statistics rather than direct `nvidia-smi` occupancy.
