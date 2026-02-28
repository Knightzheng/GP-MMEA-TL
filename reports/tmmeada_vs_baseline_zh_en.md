# zh_en: MEAformer Baseline vs TMMEA-DA MVP (1-epoch, 5 seeds)

- Baseline runs: 5
- TMMEA-DA runs: 5

| metric | baseline (mean±std) | tmmeada (mean±std) | delta (tmmeada-baseline) |
|---|---:|---:|---:|
| l2r Hits@1 | 0.6426 ± 0.0037 | 0.5523 ± 0.0055 | -0.0903 |
| l2r Hits@10 | 0.8757 ± 0.0021 | 0.8412 ± 0.0017 | -0.0345 |
| l2r MRR | 0.7224 ± 0.0031 | 0.6492 ± 0.0035 | -0.0732 |
| r2l Hits@1 | 0.6375 ± 0.0034 | 0.5531 ± 0.0025 | -0.0844 |
| r2l Hits@10 | 0.8681 ± 0.0022 | 0.8402 ± 0.0021 | -0.0279 |
| r2l MRR | 0.7164 ± 0.0034 | 0.6490 ± 0.0025 | -0.0674 |

注：当前为 1 epoch 冒烟配置，主要用于流程与可复现验证，不用于最终 SOTA 结论。