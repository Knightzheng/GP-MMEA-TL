# 任务书对齐更新（2026-03-05）

## 已新增完成
- 动机实验链路从“纯迁移 only_test”扩展到“目标域无标注自适应（unsup+IL）”并实跑成功。
- 已完成 `seed=42` 的 `zh_en -> ja_en` 与 `zh_en -> FBDB15K` 两组 baseline/TMMEA-DA 对照。
- 结果显示迁移性能显著提升（相较 only_test 设定）：
  - `ja_en`: `avg_mrr` 提升约 `+0.302`
  - `FBDB15K`: `avg_mrr` 提升约 `+0.023`

## 仍在进行
- `seed=3407` 的同配置队列已启动后台执行（用于稳定性验证）。

## 下一步对齐项
1. 扩展到 `fr_en` 与 `FBYG15K`，补齐任务书中的目标域矩阵。  
2. 在 adapt 设定下继续调优 `TMMEA-DA`，目标是形成“优于 baseline”的证据。  
3. 将 `2-seed` 稳定后再扩展 `5-seed` 正式结果用于终稿。  
