# Reports Layout

- `baseline/`
  - Baseline summary and aggregate outputs.
- `tmmeada/`
  - TMMEA-DA summaries, sweeps, and ablations.
- `epoch3/`
  - Epoch-3 pilot/formal compare artifacts.
- `epoch10/`
  - Epoch-10 pilot/tuning compare and decisions.
- `transfer/`
  - Source-train -> target-adapt -> target-eval artifacts and run-card-facing summaries.
  - See `transfer/README.md` for the current official transfer entry points.
- `midterm/`
  - Midterm draft, submission materials, guidance-record drafts, and teacher-comment reference.
  - See `midterm/README.md` for the current recommended midterm entry points.
- `planning/`
  - Next-stage plans and auto decisions.
- `notes/`
  - Current-state notes, mainline closure documents, thread sync, and thesis/defense integration materials.

## Key Current Note Files

- `notes/thread_sync_shared.md`
  - Shared handoff board between the thesis-writing thread and the project-optimization thread.
- `notes/taskbook_gap_assessment_20260315.md`
  - Taskbook/proposal/mainline closure assessment and gap audit.
- `notes/mainline_traceability_matrix_20260315.md`
  - Requirement-to-result-to-script-to-run mapping.
- `notes/mainline_closure_onepage_20260315.md`
  - One-page defense/acceptance summary of the mainline.
- `notes/mainline_artifact_integrity_20260315.md`
  - Auto-generated integrity check for formal runs, case/GPU supplements, and H3-removal state.
- `notes/thesis_final_integration_packet_20260316.md`
  - Chapter-oriented thesis finalization packet.
- `notes/four_target_evidence_map_20260316.md`
  - One-page per-target evidence map for appendix/defense use.
- `notes/defense_qa_packet_20260316.md`
  - Defense-ready wording pack for scope and boundary questions.

## Key Current Transfer Files

- `transfer/transfer_adapt_main_results_4target.md`
  - Formal 4-target main result summary.
- `transfer/transfer_adapt_significance_summary.md`
  - Significance and stability support for the 4-target `5-seed` package.
- `transfer/transfer_case_analysis_examples.md`
  - Current formal case-analysis examples.
- `transfer/transfer_case_pattern_summary_20260316.md`
  - Grouped success/failure patterns over the current 8 formal cases.
- `transfer/transfer_efficiency_summary.md`
  - Wall-clock efficiency summary.
- `transfer/transfer_gpu_peak_minimal_summary.md`
  - Minimal formal GPU-peak-memory supplement.

## Key Current Midterm Files

- `midterm/midterm_report_filled_20260319.md`
  - Filled midterm-report draft aligned to the school template requirements.
- `midterm/guidance_records_10_20260319.md`
  - Ten guidance-record entries ready for system submission.
- `midterm/guidance_records_10_20260319.csv`
  - Structured version of the guidance records.
- `midterm/teacher_comment_draft_20260319.md`
  - Reference wording for the supervisor's midterm comment.

## Current Boundary

1. The current repository mainline does not include `H3`.
2. GPU peak memory remains a supplementary minimal check, not a full all-target/all-seed memory study.
3. Thesis/defense concentration files improve organization and reuse, but they are not new experimental conclusions.
