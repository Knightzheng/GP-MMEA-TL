# Transfer Case Pattern Summary (2026-03-16)

## 1. Purpose

This file groups the current `8` formal case-analysis samples into a few recurring success/failure patterns for appendix or defense use.

- suitable for: appendix, defense, thesis-side qualitative grouping
- not suitable for: claiming new statistical laws

## 2. Pattern Table

| pattern_id | pattern_name | case_count | datasets | representative cases | what it supports | what it does not support |
| --- | --- | ---: | --- | --- | --- | --- |
| `P1` | cross_graph_large_rank_recovery | 4 | `FBDB15K`, `FBYG15K` | `Post-bop`, `The Pacific`, `Saboteur (film)`, `Amritsar` | In cross-graph settings, the method can recover heavily mis-ranked targets back to `top-1` on hard samples | It does not prove every cross-graph sample will show the same recovery magnitude |
| `P2` | cross_graph_attribute_guided_recovery | 1 | `FBDB15K` | `JavaScript` | The gain can also appear on attribute-heavy entities, not only on the most obvious large-rank corrections | A single case cannot prove that a specific module is independently validated |
| `P3` | cross_lingual_neighbor_ambiguity | 2 | `ja_en` | `Windows 10 Mobile`, `Fat Mike` | Cross-lingual fine-grained neighbor confusion still remains, even when the overall target-domain gain is positive | It does not overturn the aggregate improvement on `ja_en`; it only marks the boundary |
| `P4` | cross_lingual_over_transfer_regression | 1 | `ja_en` | `Inspiration is DEAD` | The method may still over-transfer toward a semantically close but wrong candidate in fine-grained music entities | It does not mean the transfer mechanism is generally harmful; it is a retained hard-case failure |

## 3. Recommended Thesis-Side Use

1. If the main text space is limited, keep the most representative `6` examples in the main body and move the remaining `2` into appendix or defense slides.
2. Use `P1 + P2` to explain why cross-graph gains are not just average-level improvements but can include visible rank recovery on hard samples.
3. Use `P3 + P4` to explain why the thesis keeps a conservative boundary on cross-lingual fine-grained ambiguity.

## 4. Conservative Boundary

1. The current grouped cases are qualitative evidence only.
2. The pattern labels summarize recurring appearances in the retained `8` cases, not full-dataset statistical buckets.
3. The grouping is useful for appendix/defense organization, but it should not be promoted to a new formal conclusion in the main text.
