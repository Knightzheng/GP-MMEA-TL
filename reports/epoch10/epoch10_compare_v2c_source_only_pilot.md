# Epoch10 v2 tuned Pilot Compare

required_seeds=[42, 3407]; threshold=0.003

| dataset | metric | baseline_mean | method_mean | delta |
|---|---|---:|---:|---:|
| zh_en | l2r_hits@1 | 0.6641 | 0.6640 | -0.0001 |
| zh_en | l2r_hits@10 | 0.9163 | 0.9162 | -0.0001 |
| zh_en | l2r_mrr | 0.7500 | 0.7500 | +0.0000 |
| zh_en | r2l_hits@1 | 0.6643 | 0.6643 | +0.0000 |
| zh_en | r2l_hits@10 | 0.9147 | 0.9148 | +0.0001 |
| zh_en | r2l_mrr | 0.7505 | 0.7505 | +0.0000 |
| FBDB15K | l2r_hits@1 | 0.2182 | 0.2183 | +0.0001 |
| FBDB15K | l2r_hits@10 | 0.5278 | 0.5287 | +0.0009 |
| FBDB15K | l2r_mrr | 0.3205 | 0.3205 | +0.0000 |
| FBDB15K | r2l_hits@1 | 0.2167 | 0.2167 | +0.0000 |
| FBDB15K | r2l_hits@10 | 0.5301 | 0.5303 | +0.0002 |
| FBDB15K | r2l_mrr | 0.3205 | 0.3205 | +0.0000 |
