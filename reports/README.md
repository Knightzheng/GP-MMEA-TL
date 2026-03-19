# Reports 目录导航

`reports/` 不再按“某天新增了什么文件”来理解，而是按项目阶段来理解。真实文件仍保留在原目录中，本 README 负责回答：

1. 哪个目录属于哪个阶段。
2. 每个目录主要产出什么。
3. 当前正式主线应该先看哪些文件。
4. 历史探索和归档材料应该去哪里找。

## 逻辑阶段树

```text
reports/
├─ baseline/   S1 baseline 复现汇总
├─ tmmeada/    S2 TMMEA-DA 受控开发
├─ epoch3/     S2 epoch=3 正式对照与消融
├─ compare/    S2 baseline vs method 汇总对照
├─ epoch10/    S3 epoch10 pilot 与阶段判断
├─ planning/   S3 下一阶段规划与自动决策
├─ transfer/   S4-S6 transfer 主线、主表与补强材料
├─ notes/      S6 主线收口、论文/答辩材料、共享记录
├─ thesis/     S6 论文正文与写作同步材料
├─ midterm/    S7 中期正文、模板适配与提交件
├─ misc/       杂项历史材料
└─ tmp/        临时中间文件，不应直接引用
```

## 按阶段查看

| 阶段 | 目录 | 说明 |
| --- | --- | --- |
| `S1` | `baseline/` | baseline 多 seed 汇总、均值方差、正式对照表 |
| `S2` | `tmmeada/`, `epoch3/`, `compare/` | 方法开发、`epoch=3` 对照、消融、比较汇总 |
| `S3` | `epoch10/`, `planning/` | `epoch10` pilot、v2 系列、继续/停止投入的决策材料 |
| `S4-S5` | `transfer/` | bootstrap、各版本分支、目标域正式收口 |
| `S6` | `notes/`, `thesis/` | 主线追溯、闭环说明、完整性校验、答辩/论文浓缩材料 |
| `S7` | `midterm/` | 中期报告与学校模板适配产物 |

阶段总索引见 [reports/by_stage/README.md](/d:/code/codes/cursor/BYSJ_zyf/reports/by_stage/README.md)。

## 当前正式主线入口

如果只关心当前正式主线，请优先看：

1. [transfer_adapt_main_results_4target.md](/d:/code/codes/cursor/BYSJ_zyf/reports/transfer/transfer_adapt_main_results_4target.md)
2. [transfer_adapt_significance_summary.md](/d:/code/codes/cursor/BYSJ_zyf/reports/transfer/transfer_adapt_significance_summary.md)
3. [transfer_case_analysis_examples.md](/d:/code/codes/cursor/BYSJ_zyf/reports/transfer/transfer_case_analysis_examples.md)
4. [transfer_efficiency_summary.md](/d:/code/codes/cursor/BYSJ_zyf/reports/transfer/transfer_efficiency_summary.md)
5. [mainline_traceability_matrix_20260315.md](/d:/code/codes/cursor/BYSJ_zyf/reports/notes/mainline_traceability_matrix_20260315.md)
6. [mainline_closure_onepage_20260315.md](/d:/code/codes/cursor/BYSJ_zyf/reports/notes/mainline_closure_onepage_20260315.md)
7. [mainline_artifact_integrity_20260315.md](/d:/code/codes/cursor/BYSJ_zyf/reports/notes/mainline_artifact_integrity_20260315.md)

## 历史与共享记录

- [reports/notes/README.md](/d:/code/codes/cursor/BYSJ_zyf/reports/notes/README.md)
  - 主线收口、论文/答辩材料与共享记录入口。
- [reports/notes/thread_sync_shared.md](/d:/code/codes/cursor/BYSJ_zyf/reports/notes/thread_sync_shared.md)
  - 线程共享交接板。
- [reports/notes/archive/README.md](/d:/code/codes/cursor/BYSJ_zyf/reports/notes/archive/README.md)
  - 已归档的旧时间线记录和历史版本。

## 当前边界

1. `transfer/` 下大量 `v*` 文件保留为历史过程留痕，不能默认视为当前正式证据。
2. `tmp/` 和 `misc/` 更偏中间产物或杂项，不应直接作为论文主证据引用。
3. 本目录已经改成“按阶段导航”，但真实文件路径仍保持原位。
