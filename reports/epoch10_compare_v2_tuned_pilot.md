# Epoch10 v2 tuned Pilot Compare

required_seeds=[42, 3407]; threshold=0.003

| dataset | metric | baseline_mean | method_mean | delta |
|---|---|---:|---:|---:|
| zh_en | l2r_hits@1 | 0.6641 | 0.6638 | -0.0003 |
| zh_en | l2r_hits@10 | 0.9163 | 0.9157 | -0.0006 |
| zh_en | l2r_mrr | 0.7500 | 0.7500 | +0.0000 |
| zh_en | r2l_hits@1 | 0.6643 | 0.6639 | -0.0004 |
| zh_en | r2l_hits@10 | 0.9147 | 0.9153 | +0.0006 |
| zh_en | r2l_mrr | 0.7505 | 0.7495 | -0.0010 |
| FBDB15K | l2r_hits@1 | 0.2182 | 0.2179 | -0.0003 |
| FBDB15K | l2r_hits@10 | 0.5278 | 0.5288 | +0.0010 |
| FBDB15K | l2r_mrr | 0.3205 | 0.3200 | -0.0005 |
| FBDB15K | r2l_hits@1 | 0.2167 | 0.2156 | -0.0011 |
| FBDB15K | r2l_hits@10 | 0.5301 | 0.5320 | +0.0019 |
| FBDB15K | r2l_mrr | 0.3205 | 0.3195 | -0.0010 |
