import argparse
import csv
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "transfer"


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def read_target_delta(compare_csv: Path, target: str):
    if not compare_csv.exists():
        return None
    with compare_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("target") == target:
            try:
                return {
                    "baseline_num_runs": int(float(row.get("baseline_num_runs", "0"))),
                    "tmmeada_num_runs": int(float(row.get("tmmeada_num_runs", "0"))),
                    "delta_avg_hits@1_mean": float(row.get("delta_avg_hits@1_mean", "0")),
                    "delta_avg_hits@10_mean": float(row.get("delta_avg_hits@10_mean", "0")),
                    "delta_avg_mrr_mean": float(row.get("delta_avg_mrr_mean", "0")),
                    "delta_avg_mr_mean": float(row.get("delta_avg_mr_mean", "0")),
                }
            except Exception:
                return None
    return None


def run_generic_expand(
    runner_python: str,
    meaformer_python: str,
    run_missing: int,
    target: str,
    status_title: str,
    baseline_source_config: str,
    baseline_target_config: str,
    tmmeada_source_config: str,
    tmmeada_target_config: str,
    baseline_stage_root: str,
    tmmeada_stage_root: str,
    baseline_fallback_target_eval: str,
    tmmeada_fallback_target_eval: str,
    merged_baseline_target_eval: str,
    merged_tmmeada_target_eval: str,
    report_prefix: str,
    status_json: str,
    status_md: str,
):
    cmd = [
        runner_python,
        "scripts/run_transfer_adapt_expand5_resume_generic.py",
        "--runner-python",
        runner_python,
        "--meaformer-python",
        meaformer_python,
        "--seeds",
        "42,3407,2026,7,123",
        "--target",
        target,
        "--status-title",
        status_title,
        "--baseline-source-config",
        baseline_source_config,
        "--baseline-target-config",
        baseline_target_config,
        "--tmmeada-source-config",
        tmmeada_source_config,
        "--tmmeada-target-config",
        tmmeada_target_config,
        "--baseline-stage-root",
        baseline_stage_root,
        "--tmmeada-stage-root",
        tmmeada_stage_root,
        "--baseline-fallback-target-eval",
        baseline_fallback_target_eval,
        "--tmmeada-fallback-target-eval",
        tmmeada_fallback_target_eval,
        "--merged-baseline-target-eval",
        merged_baseline_target_eval,
        "--merged-tmmeada-target-eval",
        merged_tmmeada_target_eval,
        "--report-prefix",
        report_prefix,
        "--status-json",
        status_json,
        "--status-md",
        status_md,
        "--run-missing",
        str(run_missing),
    ]
    run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Next stage: expand ja_en + FBDB15K to 5-seed and refresh transfer main/bucket reports."
    )
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--meaformer-python", default=r"D:\Anaconda_envs\envs\bysj-meaformer\python.exe")
    parser.add_argument(
        "--run-missing",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: run missing seeds; 0: summarize only",
    )
    args = parser.parse_args()

    # 1) ja_en (method = v6 mixed ja branch)
    run_generic_expand(
        runner_python=args.runner_python,
        meaformer_python=args.meaformer_python,
        run_missing=args.run_missing,
        target="ja_en",
        status_title="Transfer Adapt ja_en expand5 Status",
        baseline_source_config="configs/transfer/meaformer_source_zh_en_epoch10.yaml",
        baseline_target_config="configs/transfer_adapt/meaformer_target_ja_en_unsup_il.yaml",
        tmmeada_source_config="configs/transfer/tmmeada_source_zh_en_epoch10.yaml",
        tmmeada_target_config="configs/transfer_adapt/tmmeada_target_ja_en_v5_unsup_il.yaml",
        baseline_stage_root="transfer/transfer_adapt_ja_expand5_baseline",
        tmmeada_stage_root="transfer/transfer_adapt_ja_expand5_tmmeada",
        baseline_fallback_target_eval="runs/transfer/transfer_adapt_pilot/target_eval",
        tmmeada_fallback_target_eval="runs/transfer/transfer_adapt_v6_mixed/target_eval",
        merged_baseline_target_eval="runs/transfer/transfer_adapt_ja_expand5_merged_baseline/target_eval",
        merged_tmmeada_target_eval="runs/transfer/transfer_adapt_ja_expand5_merged_tmmeada/target_eval",
        report_prefix="reports/transfer/transfer_adapt_v6_mixed_ja_expand5",
        status_json="reports/transfer/transfer_adapt_ja_expand5_status.json",
        status_md="reports/transfer/transfer_adapt_ja_expand5_status.md",
    )

    # 2) FBDB15K (method = v7b formal)
    run_generic_expand(
        runner_python=args.runner_python,
        meaformer_python=args.meaformer_python,
        run_missing=args.run_missing,
        target="FBDB15K",
        status_title="Transfer Adapt FBDB15K expand5 Status",
        baseline_source_config="configs/transfer/meaformer_source_zh_en_epoch10.yaml",
        baseline_target_config="configs/transfer_adapt/meaformer_target_fbdb15k_unsup_il.yaml",
        tmmeada_source_config="configs/transfer/meaformer_source_zh_en_epoch10.yaml",
        tmmeada_target_config="configs/transfer_adapt/tmmeada_target_fbdb15k_v7b_mild_da_unsup_il.yaml",
        baseline_stage_root="transfer/transfer_adapt_fbdb_expand5_baseline",
        tmmeada_stage_root="transfer/transfer_adapt_fbdb_expand5_tmmeada",
        baseline_fallback_target_eval="runs/transfer/transfer_adapt_pilot/target_eval",
        tmmeada_fallback_target_eval="runs/transfer/transfer_adapt_v7_fbdb_formal_v7b/target_eval",
        merged_baseline_target_eval="runs/transfer/transfer_adapt_fbdb_expand5_merged_baseline/target_eval",
        merged_tmmeada_target_eval="runs/transfer/transfer_adapt_fbdb_expand5_merged_tmmeada/target_eval",
        report_prefix="reports/transfer/transfer_adapt_v7_fbdb_expand5",
        status_json="reports/transfer/transfer_adapt_fbdb_expand5_status.json",
        status_md="reports/transfer/transfer_adapt_fbdb_expand5_status.md",
    )

    # 3) refresh 4-target main table + bucket analysis
    run_cmd([args.runner_python, "scripts/make_transfer_main_and_bucket_report.py"])

    # 4) stage update report
    ja_csv = REPORT_DIR / "transfer_adapt_v6_mixed_ja_expand5_compare_vs_baseline.csv"
    fbdb_csv = REPORT_DIR / "transfer_adapt_v7_fbdb_expand5_compare_vs_baseline.csv"
    ja = read_target_delta(ja_csv, "ja_en")
    fbdb = read_target_delta(fbdb_csv, "FBDB15K")

    lines = [
        "# 迁移实验阶段报告（ja_en + FBDB15K 扩展到 5-seed）",
        "",
        f"- 时间戳: `{now_ts()}`",
        "- 覆盖目标: `ja_en`, `FBDB15K`",
        "- 目标: 将两目标从 2-seed 扩展到 5-seed，并刷新 4目标主结果表。",
        "",
        "## 输出文件",
        "",
        "- `reports/transfer/transfer_adapt_ja_expand5_status.{md,json}`",
        "- `reports/transfer/transfer_adapt_fbdb_expand5_status.{md,json}`",
        "- `reports/transfer/transfer_adapt_v6_mixed_ja_expand5_compare_vs_baseline.{csv,md}`",
        "- `reports/transfer/transfer_adapt_v7_fbdb_expand5_compare_vs_baseline.{csv,md}`",
        "- `reports/transfer/transfer_adapt_main_results_4target.{csv,md}`",
        "- `reports/transfer/transfer_adapt_error_bucket_summary.{csv,md}`",
        "",
    ]

    if ja is not None:
        lines.extend(
            [
                "## ja_en（5-seed）",
                "",
                f"- runs(b/m): `{ja['baseline_num_runs']}/{ja['tmmeada_num_runs']}`",
                f"- `delta_avg_hits@1_mean = {ja['delta_avg_hits@1_mean']:+.6f}`",
                f"- `delta_avg_hits@10_mean = {ja['delta_avg_hits@10_mean']:+.6f}`",
                f"- `delta_avg_mrr_mean = {ja['delta_avg_mrr_mean']:+.6f}`",
                f"- `delta_avg_mr_mean = {ja['delta_avg_mr_mean']:+.6f}`",
                "",
            ]
        )
    if fbdb is not None:
        lines.extend(
            [
                "## FBDB15K（5-seed）",
                "",
                f"- runs(b/m): `{fbdb['baseline_num_runs']}/{fbdb['tmmeada_num_runs']}`",
                f"- `delta_avg_hits@1_mean = {fbdb['delta_avg_hits@1_mean']:+.6f}`",
                f"- `delta_avg_hits@10_mean = {fbdb['delta_avg_hits@10_mean']:+.6f}`",
                f"- `delta_avg_mrr_mean = {fbdb['delta_avg_mrr_mean']:+.6f}`",
                f"- `delta_avg_mr_mean = {fbdb['delta_avg_mr_mean']:+.6f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## 结论",
            "",
            "1. 两目标已具备 5-seed 正式口径后，4目标主表可用于论文主结果。",
            "2. 若某目标仍无显著提升，可继续做目标域伪标签策略微调（仅在该目标域单独推进）。",
        ]
    )
    out_md = REPORT_DIR / "transfer_stage_update_20260309_ja_fbdb_expand5.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] stage update -> {out_md}")


if __name__ == "__main__":
    main()
