# 实验过程留痕规范

每次实验至少保留以下文件（放在 `runs/<stage>/<run_id>/`）：

1. `run_card.md`
2. `config.yaml`
3. `metrics.csv`
4. `log.txt`
5. `artifact_manifest.json`
6. `figs/`（可后补）

## 推荐命令

```powershell
# 1) 预处理占位（会生成 preprocess.log + split/data stats）
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\preprocess_dbp15k.py --dataset dbp15k_zh_en --split 0_3 --seed 3407 --train-ratio 0.3

# 2) 基线入口（会自动生成完整 run 证据包）
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\train_baseline.py --config configs\tmmeada\default.yaml --stage baseline --model MEAformer --dataset dbp15k_zh_en --seed 3407

# 3) 适配 MEAformer 的 DBP15K 数据目录
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\prepare_meaformer_data.py --lang-pair zh_en --split 0_3 --seed 3407

# 4) 真实 MEAformer 运行（3060 安全配置）
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\run_meaformer.py --config configs\baselines\meaformer_zh_en_rtx3060_safe.yaml

# 5) 多 seed 批量运行（示例：补 2 个种子）
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\run_meaformer_multiseed.py --langs zh_en,ja_en,fr_en --seeds 7,123

# 6) 汇总明细与 mean±std
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\collect_meaformer_results.py --runs-dir runs\baseline --out reports\meaformer_results_summary.csv
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\aggregate_meaformer_results.py --in-csv reports\meaformer_results_summary.csv --out-csv reports\meaformer_results_mean_std.csv

# 7) 同步官方 MEAformer 数据（真实 pkls + glove + FB*）
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\sync_official_meaformer_data.py --src data\raw\MEAformer_data\mmkg --dst data\mmkg

# 8) 跨图谱首轮运行
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\run_meaformer.py --config configs\baselines\meaformer_fbdb15k_rtx3060_safe.yaml
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\run_meaformer.py --config configs\baselines\meaformer_fbyg15k_rtx3060_safe.yaml

# 9) 跨图谱多 seed 扩展（示例）
D:\Anaconda_envs\envs\bysj-main\python.exe scripts\run_meaformer_crossgraph_multiseed.py --seeds 3407,2026
```
