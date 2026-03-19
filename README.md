# GP-MMEA-TL

多模态实体对齐迁移实验仓库。

本仓库当前采用“真实路径尽量稳定、逻辑结构按阶段重建”的整理策略：不再让使用者沿着时间线翻日志，而是优先回答“项目走到哪一步了、每一步留下了什么、当前正式主线是什么、历史探索应该去哪里看”。

## 当前结构原则

1. 真实目录尽量不搬动。
2. `reports/`、`runs/`、`scripts/`、`configs/` 统一补充 `by_stage/README.md` 作为阶段导航层。
3. 顶层记录文件改为按阶段组织，旧的时间线长记录转入归档。
4. 当前仓库中的“目录树”是逻辑阶段树，不等于所有真实文件都已经做了高风险物理迁移。

## 项目阶段总览

| 阶段 | 目标 | 主要目录 |
| --- | --- | --- |
| `S0` | 基础环境与数据准备 | `env/`, `data/`, `scripts/preprocess_dbp15k.py`, `scripts/prepare_meaformer_data.py` |
| `S1` | baseline 复现 | `configs/baselines/`, `runs/experiments/baseline/`, `reports/baseline/` |
| `S2` | TMMEA-DA 受控开发 | `configs/tmmeada/`, `runs/experiments/tmmeada/`, `reports/tmmeada/`, `reports/epoch3/`, `reports/compare/` |
| `S3` | epoch10 pilot 与阶段决策 | `reports/epoch10/`, `reports/planning/`, `configs/tmmeada/` 中的 `epoch10` 相关配置 |
| `S4` | transfer 主线搭建 | `configs/transfer/`, `configs/transfer_adapt/`, `runs/transfer/`, `reports/transfer/` 中的 bootstrap / formal 早期材料 |
| `S5` | 目标域分支优化与主表收口 | `runs/transfer/`, `reports/transfer/`, `scripts/run_transfer_adapt_*` |
| `S6` | 主线收口与论文/答辩支撑 | `reports/notes/`, `reports/transfer/`, `runs/experiments/gpu_peak_minimal/` |
| `S7` | 中期与提交材料 | `reports/midterm/` 与本地提交材料目录 |

完整逻辑树见 [PROJECT_STAGE_TREE.md](/d:/code/codes/cursor/BYSJ_zyf/PROJECT_STAGE_TREE.md)。

## 推荐阅读顺序

1. [PROJECT_STAGE_TREE.md](/d:/code/codes/cursor/BYSJ_zyf/PROJECT_STAGE_TREE.md)
2. [PROJECT_OPERATION_RECORD.md](/d:/code/codes/cursor/BYSJ_zyf/PROJECT_OPERATION_RECORD.md)
3. [PROCESS_LOG.md](/d:/code/codes/cursor/BYSJ_zyf/PROCESS_LOG.md)
4. [reports/by_stage/README.md](/d:/code/codes/cursor/BYSJ_zyf/reports/by_stage/README.md)
5. [runs/by_stage/README.md](/d:/code/codes/cursor/BYSJ_zyf/runs/by_stage/README.md)
6. [scripts/by_stage/README.md](/d:/code/codes/cursor/BYSJ_zyf/scripts/by_stage/README.md)
7. [configs/by_stage/README.md](/d:/code/codes/cursor/BYSJ_zyf/configs/by_stage/README.md)

## 四个大分支入口

- [reports/README.md](/d:/code/codes/cursor/BYSJ_zyf/reports/README.md)
- [runs/README.md](/d:/code/codes/cursor/BYSJ_zyf/runs/README.md)
- [scripts/README.md](/d:/code/codes/cursor/BYSJ_zyf/scripts/README.md)
- [configs/README.md](/d:/code/codes/cursor/BYSJ_zyf/configs/README.md)

## 当前正式主线

当前正式主线已经固定为：

1. `MEAformer` baseline 已完成复现。
2. `epoch=3` 受控对照与 `zh_en` 多 seed 消融已完成。
3. transfer 主线已经收口到四个目标域正式结果：
   - `ja_en`
   - `fr_en`
   - `FBDB15K`
   - `FBYG15K`
4. 主结果、显著性、案例与效率已有配套支撑材料，GPU 峰值显存为最小辅助补测。

主线入口文件：

- [transfer_adapt_main_results_4target.md](/d:/code/codes/cursor/BYSJ_zyf/reports/transfer/transfer_adapt_main_results_4target.md)
- [transfer_adapt_significance_summary.md](/d:/code/codes/cursor/BYSJ_zyf/reports/transfer/transfer_adapt_significance_summary.md)
- [mainline_traceability_matrix_20260315.md](/d:/code/codes/cursor/BYSJ_zyf/reports/notes/mainline_traceability_matrix_20260315.md)

## 记录文件说明

- [PROCESS_LOG.md](/d:/code/codes/cursor/BYSJ_zyf/PROCESS_LOG.md)
  - 按阶段记录“实际执行了什么”。
- [PROJECT_OPERATION_RECORD.md](/d:/code/codes/cursor/BYSJ_zyf/PROJECT_OPERATION_RECORD.md)
  - 按阶段记录“为什么这样做、产出了什么、当前边界是什么”。
- [reports/notes/archive/README.md](/d:/code/codes/cursor/BYSJ_zyf/reports/notes/archive/README.md)
  - 存放旧的按时间滚动记录。

## 当前边界

1. 本轮解决的是“结构可读性和记录组织”问题，不是高风险物理迁移工程。
2. `H3` 不属于当前正式主线。
3. GPU 峰值显存仍是辅助补测，不是完整 all-target/all-seed 显存研究。
