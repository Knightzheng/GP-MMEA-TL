# 迁移实验阶段报告（2026-03-08，v14 expand5 启动）

## 1. 本阶段目标
- 将 `fr_en` 的 `v14b_refresh4_da0025` 从当前 `2-seed (42,3407)` 扩展到 `5-seed (42,3407,2026,7,123)`。
- 采用“断点续跑 + 自动汇总”方式，避免重复训练。

## 2. 本阶段新增内容
- 新增脚本：
  - `scripts/run_transfer_adapt_v14_fren_expand5_resume.py`
- 脚本能力：
  - 自动识别已完成 seed（支持 fallback 到历史阶段目录）
  - 只运行缺失 seed
  - 自动重建 merged 目录并生成对比报表
  - 输出状态文件（json/md）

## 3. 启动前状态检查（run-missing=0）
- 状态文件：
  - `reports/transfer/transfer_adapt_v14_fren_expand5_status.json`
  - `reports/transfer/transfer_adapt_v14_fren_expand5_status.md`
- 当前缺失 seed：
  - baseline: `2026, 7, 123`
  - tmmeada: `2026, 7, 123`
- 当前进展报表（仍为 2-seed）：
  - `reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv`

## 4. 后台续跑已启动
- 启动命令：
  - `python scripts/run_transfer_adapt_v14_fren_expand5_resume.py --run-missing 1`
- 父进程 PID：
  - `63212`
- 队列日志：
  - `runs/transfer/transfer_adapt_v14_fren_expand5/queue_20260308-004223.out.log`
  - `runs/transfer/transfer_adapt_v14_fren_expand5/queue_20260308-004223.err.log`
- 训练已进入：
  - `runs/transfer/transfer_adapt_v14_fren_expand5_baseline/source_train/...-s2026/`

## 5. 完成后将产出
- `reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv`
- `reports/transfer/transfer_adapt_v14_fren_expand5_status.json`
- `runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval/`
- `runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval/`

## 6. 实时查看命令
```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python.exe' -and $_.CommandLine -like '*run_transfer_adapt_v14_fren_expand5_resume.py*' } | Select-Object ProcessId,CommandLine
Get-Content runs\transfer\transfer_adapt_v14_fren_expand5\queue_20260308-004223.out.log -Tail 80
Get-Content runs\transfer\transfer_adapt_v14_fren_expand5_baseline\source_train\*\log.txt -Tail 40
```
