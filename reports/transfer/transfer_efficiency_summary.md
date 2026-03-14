# Transfer Efficiency Summary

## Paper-Ready Table

| Target | Baseline Time (min, mean of 5 seeds) | Ours Time (min, mean of 5 seeds) | Delta (min) | Overhead | Note |
|---|---:|---:|---:|---:|---|
| ja_en | 46.69 | 81.12 | +34.43 | 1.74x | wall-clock only; GPU peak memory not fully logged |
| FBDB15K | 14.47 | 4.64 | -9.83 | 0.32x | wall-clock only; GPU peak memory not fully logged |
| fr_en | 52.70 | 108.36 | +55.66 | 2.06x | wall-clock only; GPU peak memory not fully logged |
| FBYG15K | 18.32 | 21.02 | +2.71 | 1.15x | wall-clock only; GPU peak memory not fully logged |

## Notes

- Current repository logs are sufficient to summarize wall-clock time for the formal 5-seed chains.
- Peak GPU memory is not logged consistently in the existing runs, so it still requires one minimal-cost补测 if the thesis needs a complete time-memory comparison table.
- Recommended wording: report wall-clock time from completed logs as the primary efficiency indicator, and state GPU memory as supplementary measurement when available.
