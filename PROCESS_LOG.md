# Process Log

## 2026-02-28

### 环境初始化
- 创建并配置 `conda` 环境：`bysj-main`（Python 3.10, CUDA cu126）
- 安装主依赖：PyTorch、Transformers、MLflow、W&B、TensorBoard 等
- 导出环境证据文件：
  - `env/conda-pytorch.yaml`
  - `env/requirements.lock.txt`
  - `env/hardware_snapshot.txt`

### 代码修复
- 修复 `base.py`：
  - 避免 `conda run -c` 多行命令导致的报错
  - pip 安装增加 `--no-user`，减少装入用户目录风险

### 项目脚手架与留痕体系
- 新增需求和规范：
  - `00_requirements.md`
  - `metrics_spec.md`
  - `project_charter.yaml`
  - `EXPERIMENT_LOGGING.md`
- 新增执行脚本：
  - `scripts/preprocess_dbp15k.py`
  - `scripts/train_baseline.py`
- 新增配置与模板：
  - `configs/tmmeada/default.yaml`
  - `runs/templates/run_card.md`
  - `reports/midterm_outline.md`
  - `data/README.md`

### 留痕验证（已成功）
- 预处理占位 run：
  - `data/processed/dbp15k_zh_en/preprocess.log`
  - `data/processed/dbp15k_zh_en/split_manifest.json`
  - `data/processed/dbp15k_zh_en/data_stats.csv`
- 基线占位 run：
  - `runs/baseline/20260228-001924-MEAformer-dbp15k_zh_en-s3407/`
  - 包含 `run_card.md`, `config.yaml`, `metrics.csv`, `log.txt`, `artifact_manifest.json`

### 真实 DBP15K 数据接入（已完成）
- 新版 `scripts/preprocess_dbp15k.py` 支持：
  - 自动下载 JAPE 主仓库归档
  - 自动提取 `dbp15k.tar.gz`
  - 解析 `zh_en/ja_en/fr_en` 与 `0_1..0_5` 切分
  - 输出 `train_links.tsv`, `test_links.tsv`, `triples_1.tsv`, `triples_2.tsv`
  - 输出 `split_manifest.json`（含源归档 sha256）与 `data_stats.csv`
- 实测运行：
  - 命令：`python scripts/preprocess_dbp15k.py --lang-pair zh_en --split 0_3 --seed 3407 --train-ratio 0.3`
  - 统计：`triples_1=70414`, `triples_2=95142`, `train_links=10500`

### 新增基线留痕 run
- `runs/baseline/20260228-002353-MEAformer-dbp15k_zh_en-s3407/`
- 已生成：`run_card.md`, `config.yaml`, `metrics.csv`, `log.txt`, `artifact_manifest.json`

### 真实 MEAformer 基线接入与复现（已完成首跑）
- 新增 `baselines/MEAformer/`（从官方仓库拉取）
- 新增独立环境 `bysj-meaformer`（Python 3.9 + torch cu126 + requirement.txt）
- 新增脚本：
  - `scripts/prepare_meaformer_data.py`（将 DBP15K 转为 MEAformer 所需 `data/mmkg` 格式）
  - `scripts/run_meaformer.py`（真实调用 `main.py`，并自动写 `run_card/config/log/manifest`）
- 新增配置：
  - `configs/baselines/meaformer_zh_en.yaml`（原始配置）
  - `configs/baselines/meaformer_zh_en_rtx3060.yaml`（尝试降显存，因代码模态硬编码失败）
  - `configs/baselines/meaformer_zh_en_rtx3060_safe.yaml`（最终可运行）

### 冒烟与修复记录
- 第一次真实运行失败：缺少 `scipy` → 在 `bysj-meaformer` 安装 `scipy==1.13.1`（兼容 py3.9）
- 第二次运行失败：`ill_ent_ids` 格式不匹配（URI而非int）→ 改为 `sup_ent_ids + ref_ent_ids`
- 第三次运行失败：禁用模态导致 `IndexError`（MEAformer代码硬编码视图索引）→ 恢复模态并改用更小模型
- 第四次运行成功（1 epoch）：
  - run: `runs/baseline/20260228-003951-MEAformer-DBP15K-zh_en-s42/`
  - 关键结果（test）：
    - l2r Hits@1=0.6457, Hits@10=0.8765, MRR=0.725
    - r2l Hits@1=0.6399, Hits@10=0.8700, MRR=0.720

### 跨语言扩展复现（已完成）
- 新增配置：
  - `configs/baselines/meaformer_ja_en_rtx3060_safe.yaml`
  - `configs/baselines/meaformer_fr_en_rtx3060_safe.yaml`
- 新增结果汇总脚本：
  - `scripts/collect_meaformer_results.py`
- 完成 ja_en 首跑：
  - run: `runs/baseline/20260228-005054-MEAformer-DBP15K-ja_en-s42/`
  - test: l2r Hits@1=0.5965, Hits@10=0.8503, MRR=0.681; r2l Hits@1=0.5954, Hits@10=0.8489, MRR=0.679
- 完成 fr_en 首跑：
  - run: `runs/baseline/20260228-010242-MEAformer-DBP15K-fr_en-s42/`
  - test: l2r Hits@1=0.5343, Hits@10=0.8424, MRR=0.636; r2l Hits@1=0.5349, Hits@10=0.8446, MRR=0.637
- 生成跨语言汇总表：
  - `reports/meaformer_results_summary.csv`

### 多 seed 稳定性实验（已完成 3-seed）
- 新增脚本：
  - `scripts/run_meaformer_multiseed.py`（批量按语种+seed运行）
  - `scripts/aggregate_meaformer_results.py`（按语种输出 mean±std）
- 执行批量任务：
  - 语种：`zh_en, ja_en, fr_en`
  - seeds：`42, 3407, 2026`（其中 42 为已完成基线，新增补跑 3407/2026）
- 产出：
  - `reports/meaformer_results_summary.csv`（9 runs 明细）
  - `reports/meaformer_results_mean_std.csv`（按语种汇总）
  - `reports/midterm_results_draft.md`（中期可直接引用草稿）

### 多 seed 稳定性实验（扩展到 5-seed，已完成）
- 新增补跑 seeds：`7`, `123`（语种 `zh_en`, `ja_en`, `fr_en`）
- 新增 6 个 run：
  - `runs/baseline/20260228-022816-MEAformer-DBP15K-zh_en-s7/`
  - `runs/baseline/20260228-023730-MEAformer-DBP15K-zh_en-s123/`
  - `runs/baseline/20260228-024743-MEAformer-DBP15K-ja_en-s7/`
  - `runs/baseline/20260228-025946-MEAformer-DBP15K-ja_en-s123/`
  - `runs/baseline/20260228-031149-MEAformer-DBP15K-fr_en-s7/`
  - `runs/baseline/20260228-032430-MEAformer-DBP15K-fr_en-s123/`
- 汇总更新：
  - `reports/meaformer_results_summary.csv`（15 runs）
  - `reports/meaformer_results_mean_std.csv`（每语种5 runs的 mean±std）
  - `reports/midterm_results_draft.md`（5-seed表格草稿）
  - `reports/midterm_experiment_section.md`（可直接并入中期正文的章节草稿）

### 官方真实数据接入与跨图谱首跑（已完成）
- 下载官方数据包（Google Drive，约1.26GB）并解压至：
  - `data/raw/MEAformer_data/mmkg`
- 新增同步脚本：
  - `scripts/sync_official_meaformer_data.py`
- 同步结果：
  - 将官方 `mmkg` 全量同步到 `data/mmkg`
  - 生成 `data/official_data_manifest.json`（关键文件 SHA256）
- 新增跨图谱配置：
  - `configs/baselines/meaformer_fbdb15k_rtx3060_safe.yaml`
  - `configs/baselines/meaformer_fbyg15k_rtx3060_safe.yaml`
- 新增跨图谱 run：
  - `runs/baseline/20260228-034703-MEAformer-FBDB15K-norm-s42/`
  - `runs/baseline/20260228-035008-MEAformer-FBYG15K-norm-s42/`
- 汇总更新：
  - `reports/meaformer_results_summary.csv`（17 runs）
  - `reports/meaformer_results_mean_std.csv`（含 DBP15K 三语种 + FBDB15K/FBYG15K）

### 跨图谱多 seed 扩展（已完成 3-seed）
- 新增脚本：
  - `scripts/run_meaformer_crossgraph_multiseed.py`
- 新增补跑 seeds：`3407`, `2026`（在已完成 `seed=42` 基础上扩展）
- 新增 4 个 run：
  - `runs/baseline/20260228-035557-MEAformer-FBDB15K-norm-s3407/`
  - `runs/baseline/20260228-035841-MEAformer-FBDB15K-norm-s2026/`
  - `runs/baseline/20260228-040123-MEAformer-FBYG15K-norm-s3407/`
  - `runs/baseline/20260228-040450-MEAformer-FBYG15K-norm-s2026/`
- 汇总更新：
  - `reports/meaformer_results_summary.csv`（21 runs）
  - `reports/meaformer_results_mean_std.csv`（DBP: 5-seed；FBDB/FBYG: 3-seed）

### 跨图谱多 seed 扩展（补齐到 5-seed，已完成）
- 新增补跑 seeds：`7`, `123`
- 新增 4 个 run：
  - `runs/baseline/20260228-042721-MEAformer-FBDB15K-norm-s7/`
  - `runs/baseline/20260228-043004-MEAformer-FBDB15K-norm-s123/`
  - `runs/baseline/20260228-043247-MEAformer-FBYG15K-norm-s7/`
  - `runs/baseline/20260228-043618-MEAformer-FBYG15K-norm-s123/`
- 汇总更新：
  - `reports/meaformer_results_summary.csv`（25 runs）
  - `reports/meaformer_results_mean_std.csv`（DBP/FB 全部含 5-seed 或 5-run统计）

### 2026-02-28 TMMEA-DA MVP (Domain-Align) Smoke
- Added MEAformer args in baselines/MEAformer/config.py:
  - --use_domain_align (0/1)
  - --domain_align_weight (float)
- Added domain alignment loss in baselines/MEAformer/model/MEAformer.py:
  - domain_align_loss = MSE(joint_emb[left], joint_emb[right])
  - loss_all += domain_align_weight * domain_align_loss (when enabled)
  - output loss_dic now includes `domain_align`
- Updated scripts/run_meaformer.py:
  - support `meta.stage` and `meta.model_tag`
  - pass through `use_domain_align` and `domain_align_weight`
- Added config: configs/tmmeada/meaformer_zh_en_domain_align_mvp.yaml
- Executed smoke run:
  - run_id: 20260228-044730-TMMEA-DA-MEAformer-DBP15K-zh_en-s42
  - test l2r: H@1=0.5487, H@10=0.8403, MRR=0.647
  - test r2l: H@1=0.5522, H@10=0.8377, MRR=0.647
- Added report note: reports/tmmeada_mvp_smoke.md
- Extended TMMEA-DA MVP to multi-seed (partial):
  - Added script: scripts/run_tmmeada_multiseed.py
  - Executed seed=3407 run:
    - run_id: 20260228-050047-TMMEA-DA-MEAformer-DBP15K-zh_en-s3407
    - test l2r: H@1=0.5474, H@10=0.8410, MRR=0.647
    - test r2l: H@1=0.5525, H@10=0.8401, MRR=0.649
- Aggregated current TMMEA-DA runs:
  - reports/tmmeada_results_summary.csv (2 runs)
  - reports/tmmeada_results_mean_std.csv (zh_en, 2-seed mean±std)
- Updated report note: reports/tmmeada_mvp_smoke.md
- Corrected stale note in reports/midterm_experiment_section.md:
  - replaced "placeholder image features" with current status (official MEAformer data synced)
  - updated next-step plan to focus on full-epoch reruns and TMMEA-DA 5-seed comparison
### 2026-02-28 Afternoon TMMEA-DA continuation
- Completed remaining TMMEA-DA zh_en seeds: 2026, 7, 123
  - runs/tmmeada/20260228-125417-TMMEA-DA-MEAformer-DBP15K-zh_en-s2026/
  - runs/tmmeada/20260228-130507-TMMEA-DA-MEAformer-DBP15K-zh_en-s7/
  - runs/tmmeada/20260228-131550-TMMEA-DA-MEAformer-DBP15K-zh_en-s123/
- Refreshed summaries sequentially to avoid stale aggregation:
  - reports/tmmeada_results_summary.csv (5 runs)
  - reports/tmmeada_results_mean_std.csv (zh_en, 5-seed)
- Added baseline-vs-method comparison generator:
  - scripts/make_tmmeada_baseline_compare.py
  - outputs:
    - reports/tmmeada_vs_baseline_zh_en.csv
    - reports/tmmeada_vs_baseline_zh_en.md
- Updated writeups for latest status:
  - reports/tmmeada_mvp_smoke.md
  - reports/midterm_results_draft.md
  - reports/midterm_experiment_section.md
- Key 5-seed TMMEA-DA zh_en results:
  - l2r H@1=0.5523±0.0055, H@10=0.8412±0.0017, MRR=0.6492±0.0035
  - r2l H@1=0.5531±0.0025, H@10=0.8402±0.0021, MRR=0.6490±0.0025
- Cleanup:
  - removed dry-run folder runs/tmmeada/20260228-044720-TMMEA-DA-MEAformer-DBP15K-zh_en-s42/
  - reran tmmeada collect+aggregate sequentially to lock final 5-run stats
### 2026-02-28 Evening Extension (DBP15K full TMMEA-DA MVP)
- Added configs:
  - configs/tmmeada/meaformer_ja_en_domain_align_mvp.yaml
  - configs/tmmeada/meaformer_fr_en_domain_align_mvp.yaml
- Updated script:
  - scripts/run_tmmeada_multiseed.py (tmp config naming no longer hardcoded zh_en)
- Completed TMMEA-DA DBP15K runs (5 seeds each):
  - ja_en: s42/s3407/s2026/s7/s123
  - fr_en: s42/s3407/s2026/s7/s123
- Refreshed method summaries:
  - reports/tmmeada_results_summary.csv (15 runs)
  - reports/tmmeada_results_mean_std.csv (zh_en/ja_en/fr_en)
- Added DBP15K baseline-vs-method comparison:
  - scripts/make_tmmeada_baseline_compare_dbp15k.py
  - reports/tmmeada_vs_baseline_dbp15k.csv
  - reports/tmmeada_vs_baseline_dbp15k.md
- Added multilang method note:
  - reports/tmmeada_dbp15k_multilang.md
- Updated documentation:
  - reports/midterm_results_draft.md
  - reports/tmmeada_mvp_smoke.md
- Added project root README with task definition/dataset/run/metrics/baselines/modifications:
  - README.md
### 2026-02-28 Night Cross-Graph TMMEA-DA
- Added cross-graph TMMEA-DA configs:
  - configs/tmmeada/meaformer_fbdb15k_domain_align_mvp.yaml
  - configs/tmmeada/meaformer_fbyg15k_domain_align_mvp.yaml
- Updated scripts/run_tmmeada_multiseed.py tmp config naming to include data_choice.
- Completed TMMEA-DA cross-graph runs (5 seeds each):
  - FBDB15K: s42/s3407/s2026/s7/s123
  - FBYG15K: s42/s3407/s2026/s7/s123
- Refreshed full method summaries:
  - reports/tmmeada_results_summary.csv (25 runs)
  - reports/tmmeada_results_mean_std.csv (DBP15K + FBDB15K + FBYG15K)
- Added full-dataset baseline-vs-method comparison:
  - scripts/make_tmmeada_baseline_compare_all.py
  - reports/tmmeada_vs_baseline_all.csv
  - reports/tmmeada_vs_baseline_all.md
- Updated docs:
  - README.md
  - reports/tmmeada_dbp15k_multilang.md
  - reports/midterm_results_draft.md
  - reports/midterm_experiment_section.md
### 2026-02-28 Late Night TMMEA-DA v1 module implementation
- Added new args in baselines/MEAformer/config.py:
  - use_source_select/source_select_weight/source_select_temp
  - use_missing_gate/missing_align_weight
- Updated data loader baselines/MEAformer/src/data.py:
  - load_img now returns (img_embd, img_mask)
  - KGs now include img_mask
- Updated model baselines/MEAformer/model/MEAformer.py:
  - added _to_cuda_batch
  - added _source_select_loss
  - added _missing_aware_img_align_loss
  - integrated new losses into forward and loss_dic
- Updated runner scripts/run_meaformer.py to pass new optional args
- Added v1 smoke config:
  - configs/tmmeada/meaformer_zh_en_tmmeada_v1_smoke.yaml
- Ran v1 smoke run:
  - runs/tmmeada/20260228-205727-TMMEA-DA-v1-DBP15K-zh_en-s42/
  - l2r H@1=0.5488, H@10=0.8402, MRR=0.647
  - r2l H@1=0.5522, H@10=0.8381, MRR=0.647
- Added report note:
  - reports/tmmeada_v1_smoke.md
- Improved scripts/collect_meaformer_results.py:
  - now scans all run dirs under --runs-dir (not only *MEAformer*)
  - added optional --name-contains filter
