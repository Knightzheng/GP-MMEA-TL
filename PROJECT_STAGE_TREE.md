# 项目阶段树

本文给出当前仓库的“逻辑阶段树”。这里强调的是“如何理解仓库”，不是“所有真实目录已经物理搬迁完成”。为了保护脚本引用、run card、报告路径和论文草稿中的既有引用，真实产物路径大体保持原位，阶段化理解统一通过各目录下的 `by_stage/README.md` 完成。

```text
GP-MMEA-TL
├─ S0 基础环境与数据准备
│  ├─ env/
│  ├─ data/
│  ├─ scripts/preprocess_dbp15k.py
│  ├─ scripts/prepare_meaformer_data.py
│  └─ scripts/sync_official_meaformer_data.py
├─ S1 baseline 复现
│  ├─ configs/baselines/
│  ├─ runs/experiments/baseline/
│  └─ reports/baseline/
├─ S2 TMMEA-DA 受控开发
│  ├─ configs/tmmeada/
│  ├─ runs/experiments/tmmeada/
│  ├─ reports/tmmeada/
│  ├─ reports/epoch3/
│  └─ reports/compare/
├─ S3 epoch10 pilot 与阶段决策
│  ├─ reports/epoch10/
│  ├─ reports/planning/
│  └─ configs/tmmeada/ 中的 epoch10 / v2 系列配置
├─ S4 transfer 主线搭建
│  ├─ configs/transfer/
│  ├─ configs/transfer_adapt/
│  ├─ runs/transfer/transfer_smoke*
│  ├─ runs/transfer/transfer_formal*
│  ├─ runs/transfer/transfer_adapt_v3 ~ v8*
│  └─ reports/transfer/ 中的 bootstrap / formal 早期材料
├─ S5 目标域分支优化与主表收口
│  ├─ JA / FR 分支
│  │  ├─ runs/transfer/transfer_adapt_v9 ~ v15*
│  │  └─ reports/transfer/transfer_adapt_*fren* / *ja*
│  ├─ FBDB15K 分支
│  │  ├─ runs/transfer/transfer_adapt_v16 ~ v18*
│  │  └─ reports/transfer/transfer_adapt_*fbdb*
│  ├─ FBYG15K 分支
│  │  ├─ runs/transfer/transfer_adapt_v19 ~ v25*
│  │  └─ reports/transfer/transfer_adapt_*fbyg*
│  └─ 统一主表收口
│     ├─ reports/transfer/transfer_adapt_main_results_4target.*
│     └─ reports/transfer/transfer_adapt_error_bucket_summary.*
├─ S6 主线收口与论文/答辩支撑
│  ├─ reports/notes/
│  ├─ reports/transfer/ 中的显著性 / 案例 / 效率 / GPU 补证
│  ├─ runs/experiments/gpu_peak_minimal/
│  └─ scripts/verify_mainline_artifacts.py
└─ S7 中期与提交材料
   ├─ reports/midterm/
   └─ 本地提交材料目录
```

## 四个大分支的阶段入口

- [reports/by_stage/README.md](/d:/code/codes/cursor/BYSJ_zyf/reports/by_stage/README.md)
- [runs/by_stage/README.md](/d:/code/codes/cursor/BYSJ_zyf/runs/by_stage/README.md)
- [scripts/by_stage/README.md](/d:/code/codes/cursor/BYSJ_zyf/scripts/by_stage/README.md)
- [configs/by_stage/README.md](/d:/code/codes/cursor/BYSJ_zyf/configs/by_stage/README.md)

## 为什么本轮不直接整体搬动真实目录

1. 大量脚本仍硬编码引用 `configs/...`、`runs/...` 和历史输出路径。
2. 现有 run card、阶段报告、论文草稿与任务总结已经写入当前真实路径。
3. 在主线证据已经闭环的前提下，强行做物理迁移，会把“目录更好看”置于“可追溯与可复现”之上，风险远高于收益。

因此，本轮采用的策略是：

1. 保留真实产物路径，保护当前复现链路。
2. 用阶段视图和重写的 README/记录文件重建逻辑结构。
3. 将旧的时间线主记录转入归档，用阶段式记录替代主导航。
