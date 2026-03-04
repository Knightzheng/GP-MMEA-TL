# Epoch10 v2 tuned Pilot Compare

required_seeds=[42, 3407]; threshold=0.003

| dataset | metric | baseline_mean | method_mean | delta |
|---|---|---:|---:|---:|
| zh_en | l2r_hits@1 | 0.6641 | 0.6636 | -0.0005 |
| zh_en | l2r_hits@10 | 0.9163 | 0.9157 | -0.0006 |
| zh_en | l2r_mrr | 0.7500 | 0.7500 | +0.0000 |
| zh_en | r2l_hits@1 | 0.6643 | 0.6638 | -0.0005 |
| zh_en | r2l_hits@10 | 0.9147 | 0.9152 | +0.0005 |
| zh_en | r2l_mrr | 0.7505 | 0.7495 | -0.0010 |
| FBDB15K | l2r_hits@1 | 0.2182 | 0.2180 | -0.0002 |
| FBDB15K | l2r_hits@10 | 0.5278 | 0.5289 | +0.0011 |
| FBDB15K | l2r_mrr | 0.3205 | 0.3200 | -0.0005 |
| FBDB15K | r2l_hits@1 | 0.2167 | 0.2155 | -0.0012 |
| FBDB15K | r2l_hits@10 | 0.5301 | 0.5322 | +0.0021 |
| FBDB15K | r2l_mrr | 0.3205 | 0.3195 | -0.0010 |
