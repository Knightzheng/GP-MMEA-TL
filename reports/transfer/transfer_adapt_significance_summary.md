# Transfer-Adapt Significance Summary

## Recommended Statistical Setting

- Primary uncertainty estimate: paired bootstrap `95% CI` on seed-wise `avg MRR` gain.
- Primary small-sample significance check: exact one-sided sign test on paired seed deltas.
- Supplementary check: exact one-sided Wilcoxon signed-rank test when available.
- Not recommended as the only evidence: paired t-test, because `n=5` is too small to rely on normality assumptions.

## Paper Table

| target | scenario | baseline avg MRR (mean+-std) | method avg MRR (mean+-std) | delta avg MRR | bootstrap 95% CI | positive seeds | sign test p (one-sided) | Wilcoxon p (one-sided) |
|---|---|---:|---:|---:|---|---:|---:|---:|
| ja_en | cross_lingual | 0.5081+-0.0011 | 0.5202+-0.0018 | +0.0121 | [+0.0106, +0.0135] | 5/5 | 0.0312 | 0.0312 |
| FBDB15K | cross_graph | 0.0261+-0.0009 | 0.0344+-0.0005 | +0.0083 | [+0.0073, +0.0091] | 5/5 | 0.0312 | 0.0312 |
| fr_en | cross_lingual | 0.5350+-0.0058 | 0.5471+-0.0050 | +0.0121 | [+0.0110, +0.0134] | 5/5 | 0.0312 | 0.0312 |
| FBYG15K | cross_graph | 0.0578+-0.0018 | 0.0606+-0.0011 | +0.0028 | [+0.0021, +0.0034] | 5/5 | 0.0312 | 0.0312 |
