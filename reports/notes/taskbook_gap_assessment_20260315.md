# 任务书 / 开题报告 / 当前项目现状差距评估（2026-03-15）

## 1. 本轮接管范围

本轮已系统通读并交叉核对以下材料：

- 根目录与说明：
  - `README.md`
  - `00_requirements.md`
  - `PROCESS_LOG.md`
  - `PROJECT_OPERATION_RECORD.md`
  - `reports/README.md`
- 任务与研究目标：
  - `多模态实体对齐模型可迁移能力研究_任务书整理版.md`
  - `reports/midterm/midterm_report_submission.md`
  - `reports/thesis/本科毕业论文初稿_v1.md`
  - `reports/notes/thread_sync_shared.md`
- 主线实验与分析：
  - `reports/transfer/transfer_adapt_main_results_4target.md`
  - `reports/transfer/transfer_adapt_significance_summary.md`
  - `reports/transfer/transfer_case_analysis_examples.md`
  - `reports/transfer/transfer_efficiency_summary.md`
  - `reports/transfer/transfer_stage_update_20260311_ja_v15_final.md`
  - `reports/transfer/transfer_stage_update_20260312_v18_fbdb_bipartite_full5.md`
  - `reports/transfer/transfer_stage_update_20260314_fbyg_v24_strict_source_full5.md`
  - `reports/epoch3/epoch3_compare_dbp15k.md`
  - `reports/epoch3/epoch3_compare_crossgraph.md`
  - `reports/epoch3/epoch3_ablation_zh_en_multiseed.md`
- 辅助支撑与运行产物：
  - `reports/robustness/robustness_stage_update_20260314_h3_gpu_setup.md`
  - `reports/robustness/h3_missing_modality_minimal_summary.md`
  - `runs/experiments/h3_missing_modality_minimal/`
  - `runs/experiments/gpu_peak_minimal/`
- 关键脚本与入口：
  - `scripts/analyze_transfer_significance.py`
  - `scripts/build_transfer_case_analysis.py`
  - `scripts/summarize_transfer_efficiency.py`
  - `scripts/run_h3_missing_modality_minimal.py`
  - `scripts/build_h3_missing_modality_paper_summary.py`
  - `scripts/run_gpu_peak_minimal.py`
  - `scripts/summarize_gpu_peak_minimal.py`
  - `scripts/run_meaformer.py`
  - `baselines/MEAformer/main.py`
  - `baselines/MEAformer/src/data.py`
  - `baselines/MEAformer/config.py`

## 2. 当前项目哪些核心要求已经完成

### 2.1 任务书与开题报告主线

以下内容已经形成较完整闭环：

1. 已完成基于 `MEAformer` 的统一 baseline 复现，并覆盖 `DBP15K zh_en/ja_en/fr_en` 与 `FBDB15K/FBYG15K`。
2. 已建立统一的 `source-train -> target-adapt` 迁移实验链路，且关键配置、日志、run card、汇总表均已落盘。
3. 已完成 4 个目标域的正式主实验结果，当前固定主表为：
   - `ja_en`
   - `fr_en`
   - `FBDB15K`
   - `FBYG15K`
4. 上述 4 个目标域均已形成 `5-seed` 正增益正式口径，主结果文件已固定在：
   - `reports/transfer/transfer_adapt_main_results_4target.md`
   - `reports/transfer/transfer_adapt_main_results_4target.csv`
5. 已完成主线所需的核心结果分析补强：
   - 统计显著性
   - zh_en 多 seed 消融
   - 误差分桶与阶段机理
   - 案例分析
   - wall-clock 效率统计
6. 已保留较完整的过程记录与阶段报告，能支撑导师审阅和答辩追溯。

### 2.2 与任务书“动机实验 / 迁移问题提出”相关的材料

这部分已经具备论文可用支撑，但材料分散在多个位置：

1. `reports/midterm/midterm_report_submission.md` 中保留了中期阶段的“迁移退化/目标域差异”问题提出。
2. 迁移链路从 `smoke -> formal -> 4 target main table -> stage update` 的演进证据完整存在于 `reports/transfer/`。
3. 论文初稿已经把“为什么需要研究可迁移能力”与“为什么主线收束到目标域自适应 + 伪标签质量控制”写顺。

结论：动机实验材料不是“没做”，而是“已经做过且结论已收口，只需要更清楚地导航与回填”。

## 3. 哪些内容已经基本完成，但组织还不够规范

以下内容已经有结果或已有明确结论，但此前没有完全同步进项目级导航与记录：

1. 主线闭环判断本身。
   - 之前仓库里有很多阶段性收口材料，但缺少一份明确说明“任务书/开题主线已经基本闭环”的项目评估文件。
2. 显著性、案例、效率三类补强材料。
   - 文件已存在，但此前更偏“结果补强”，缺少与任务书主线关系的明确定位说明。
3. `H3` 缺失模态最小版结果。
   - 现在已经不是“只有能力入口”，而是存在 `zh_en, seed=42, drop_rate={0.0,0.6}` 的单 seed pilot 与汇总文件。
   - 但它仍未被统一写清楚“只能作为辅助观察，不能当作 H3 正式验证”。
4. GPU 峰值显存补测状态。
   - 代码能力与脚本入口已接入；
   - 但当前仓库中的最小补测 run 仍以 dry-run/失败尝试为主，尚无可入文正式表格；
   - 这一点此前没有被项目级文档清楚写明。
5. `reports/README.md` 与 `runs/README.md` 对 `robustness/`、`h3_missing_modality_minimal/`、`gpu_peak_minimal/` 的目录说明不足。

## 4. 哪些内容仍存在明显差距

### 4.1 主线差距

当前判断：主线实验本身没有明显“必须继续补跑”的硬缺口，真正的差距主要是整理层面的。

仍需补齐的主线侧工作主要有：

1. 明确写清楚“主线已闭环，后续以材料规范化和辅助支撑为主”。
2. 把主线结果、脚本入口、阶段报告、过程记录之间的对应关系再同步清楚。
3. 保持对 `source_select` / `missing_gate` 的保守表述，避免 README 或共享文件中出现隐性夸大。

### 4.2 辅助支撑差距

1. `H3` 目前只有 `zh_en` 单 seed 最小 pilot。
   - 不能支持多 seed 稳定性；
   - 不能支持跨目标域结论；
   - 也不能支持“missing_gate 已被严格验证有效”。
2. GPU 峰值显存正式补测尚未完成。
   - 当前 `run_gpu_peak_minimal.py` 生成的 `epoch=1` 配置会与部分 transfer 配置中的 `il_start` 冲突；
   - 已有尝试中出现 `AssertionError: self.args.il_start < self.args.epoch`；
   - 因此当前没有可直接入论文的 GPU 峰值汇总表。
3. 额外 baseline 仍未补齐。
   - 但这不构成任务书主线未完成，只能算扩展对照不足。

## 5. 哪些内容属于主线必须补齐

当前优先级判断如下：

### 5.1 主线必须补齐

1. 项目级差距评估与闭环判断文档。
2. README / 过程日志 / 项目操作记录 / 共享文件中的主线状态同步。
3. 对辅助支撑项边界的明确说明，防止论文线程误把辅助项写成主线完成标志。

### 5.2 主线之后再做

1. GPU 峰值显存最小版正式补测。
2. `H3` 扩展到 baseline、多 seed 或更多 drop rate。
3. 扩展案例到 8 个以上。
4. 额外 baseline。
5. 图表友好型整理数据。

## 6. 总体结论

截至 `2026-03-15`，当前项目最重要的判断是：

1. 任务书与开题报告要求的主线实验已经基本闭环。
2. 当前项目最高价值工作不再是继续追加主线 rerun，而是把已有主线材料组织得更清楚、更可追溯、更便于论文与答辩直接引用。
3. `H3` 与 GPU 峰值显存应统一定位为辅助支撑项：
   - `H3` 目前只可保守写成单 seed pilot 观察；
   - GPU 峰值显存目前仍不能写成已完成结果，只能写成“脚本入口已修正，正式补测待执行”。
4. 后续若继续补实验，优先级应为：
   - 先保证项目材料规范化完成；
   - 再以最小成本补 GPU；
   - 最后再评估是否继续扩额外 baseline。

## 7. 2026-03-15 仓库整理覆盖说明

1. 本评估文件形成后，项目已进一步完成一次“主线优先”的仓库整理。
2. `H3` 相关脚本、结果文件、运行目录与代码入口已从当前仓库移除。
3. 因此，本文档前文中关于 `H3` pilot、`reports/robustness/`、`runs/experiments/h3_missing_modality_minimal/` 的描述，仅代表整理前的历史状态，不再代表当前仓库现状。
4. 当前应采用的新口径是：
   - 主线闭环与材料规范化继续推进；
   - GPU 峰值显存仍可在后续做最小补测；
   - `H3` 延期到主线完整结束后再单独重启，不纳入当前仓库管理范围。
