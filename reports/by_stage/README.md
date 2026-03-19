# Reports 阶段视图

真实报告文件仍保留在 `reports/` 原有目录中；本文件负责把这些目录按阶段重新组织。

## 阶段树

### S1 baseline 复现

- `reports/baseline/`
  - baseline 多 seed 汇总、均值方差和正式对照表。

### S2 TMMEA-DA 受控开发

- `reports/tmmeada/`
  - 方法汇总、权重搜索、MVP/v1 对照。
- `reports/epoch3/`
  - `epoch=3` 正式对照与消融。
- `reports/compare/`
  - baseline 与方法的汇总对照表。

### S3 epoch10 pilot 与阶段调参

- `reports/epoch10/`
  - `epoch10` pilot、v2 系列和自动决策。
- `reports/planning/`
  - 下一阶段方案与阶段收口判断。

### S4-S5 transfer 主线与分支优化

- `reports/transfer/`
  - bootstrap、各版本分支、主表、显著性、案例、效率、GPU 补证。
  - 进一步阅读：`reports/transfer/README.md`

### S6 主线收口与论文/答辩支撑

- `reports/notes/`
  - 任务书对齐、主线追溯、一页式闭环、完整性校验、答辩材料包。
- `reports/thesis/`
  - 论文正文和论文写作中的同步材料。

### S7 中期与提交材料

- `reports/midterm/`
  - 中期正文、模板说明和最终提交件。

### 归档/临时

- `reports/misc/`
  - 杂项材料。
- `reports/tmp/`
  - 临时汇总与不应直接引用的中间文件。

## 阅读建议

1. 看 baseline 与受控实验：从 `baseline/ -> tmmeada/ -> epoch3/ -> compare/`。
2. 看 transfer 主线：直接看 `transfer/README.md`。
3. 看论文/答辩整理：直接看 `notes/README.md`。
