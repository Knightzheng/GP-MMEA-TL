# Transfer-Adapt Significance Writeup

## Recommended Use

- Main metric for significance discussion: `avg MRR`.
- Preferred wording: use `stable under 5-seed paired setting` or `supported by paired bootstrap and seed-level tests`.
- Avoid overclaiming weak targets as universally significant if not all 5 seeds win.

## Paper-Ready Paragraph

To strengthen the stability claim of the transfer results, we further performed paired significance analysis on the final 5-seed results. Because each target uses matched baseline/method runs under the same random seeds and the sample size is small (`n=5`), we use paired bootstrap confidence intervals on the seed-wise `avg MRR` gain as the primary uncertainty estimate, and report an exact one-sided sign test as a robustness check. This choice is more appropriate than relying only on a paired t-test under such a small-sample setting.

On `ja_en`, the proposed method improves `avg MRR` by `+0.0121` over the matched baseline, with a paired bootstrap `95% CI [+0.0106, +0.0135]`. All `5/5` seeds show positive gains, and the exact one-sided sign test gives `p=0.0312`, indicating that the improvement is stable under the matched-seed setting.
On `FBDB15K`, the proposed method improves `avg MRR` by `+0.0083` over the matched baseline, with a paired bootstrap `95% CI [+0.0073, +0.0091]`. All `5/5` seeds show positive gains, and the exact one-sided sign test gives `p=0.0312`, indicating that the improvement is stable under the matched-seed setting.
On `fr_en`, the proposed method improves `avg MRR` by `+0.0121` over the matched baseline, with a paired bootstrap `95% CI [+0.0110, +0.0134]`. All `5/5` seeds show positive gains, and the exact one-sided sign test gives `p=0.0312`, indicating that the improvement is stable under the matched-seed setting.
On `FBYG15K`, the proposed method improves `avg MRR` by `+0.0028` over the matched baseline, with a paired bootstrap `95% CI [+0.0021, +0.0034]`. All `5/5` seeds show positive gains, and the exact one-sided sign test gives `p=0.0312`, indicating that the improvement is stable under the matched-seed setting.

## Defense-Ready Answers

- `ja_en` 这项我们不是只看均值，而是看了 5 个配对 seed。5/5 个 seed 都比 baseline 好，`avg MRR` 的 paired bootstrap 95% CI 也保持为正，exact one-sided sign test `p=0.0312`，所以可以说提升在当前 5-seed 配对设定下是稳定且有统计支持的。
- `FBDB15K` 这项我们不是只看均值，而是看了 5 个配对 seed。5/5 个 seed 都比 baseline 好，`avg MRR` 的 paired bootstrap 95% CI 也保持为正，exact one-sided sign test `p=0.0312`，所以可以说提升在当前 5-seed 配对设定下是稳定且有统计支持的。
- `fr_en` 这项我们不是只看均值，而是看了 5 个配对 seed。5/5 个 seed 都比 baseline 好，`avg MRR` 的 paired bootstrap 95% CI 也保持为正，exact one-sided sign test `p=0.0312`，所以可以说提升在当前 5-seed 配对设定下是稳定且有统计支持的。
- `FBYG15K` 这项我们不是只看均值，而是看了 5 个配对 seed。5/5 个 seed 都比 baseline 好，`avg MRR` 的 paired bootstrap 95% CI 也保持为正，exact one-sided sign test `p=0.0312`，所以可以说提升在当前 5-seed 配对设定下是稳定且有统计支持的。

## Suggested Thesis Footnote

Because the final transfer table is organized as matched baseline/method results under the same five random seeds, we evaluate significance on paired seed-level deltas rather than on independent samples.
