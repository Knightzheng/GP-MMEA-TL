# Epoch10 对比报告

说明：本报告基于 epoch10 pilot（2 seeds: 42,3407）。

| dataset | metric | baseline_mean | method_mean | delta |
|---|---|---:|---:|---:|
| zh_en | l2r_hits@1 | 0.6641 | 0.6641 | +0.0000 |
| zh_en | l2r_hits@10 | 0.9163 | 0.9163 | +0.0000 |
| zh_en | l2r_mrr | 0.7500 | 0.7500 | +0.0000 |
| zh_en | r2l_hits@1 | 0.6643 | 0.6643 | +0.0000 |
| zh_en | r2l_hits@10 | 0.9147 | 0.9146 | -0.0001 |
| zh_en | r2l_mrr | 0.7505 | 0.7505 | +0.0000 |
| FBDB15K | l2r_hits@1 | 0.2182 | 0.2185 | +0.0003 |
| FBDB15K | l2r_hits@10 | 0.5278 | 0.5284 | +0.0006 |
| FBDB15K | l2r_mrr | 0.3205 | 0.3205 | +0.0000 |
| FBDB15K | r2l_hits@1 | 0.2167 | 0.2168 | +0.0001 |
| FBDB15K | r2l_hits@10 | 0.5301 | 0.5299 | -0.0002 |
| FBDB15K | r2l_mrr | 0.3205 | 0.3205 | +0.0000 |
