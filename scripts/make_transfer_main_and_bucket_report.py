import csv
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "transfer"


SOURCE_SPECS = [
    {
        "target": "ja_en",
        "scenario": "cross_lingual",
        "method_variant": "v6_mixed",
        "source_csv": REPORT_DIR / "transfer_adapt_v6_mixed_compare_vs_baseline.csv",
    },
    {
        "target": "FBDB15K",
        "scenario": "cross_graph",
        "method_variant": "v7b_formal",
        "source_csv": REPORT_DIR / "transfer_adapt_v7_fbdb_compare_vs_baseline.csv",
    },
    {
        "target": "fr_en",
        "scenario": "cross_lingual",
        "method_variant": "v14b_refresh4_da0025_expand5",
        "source_csv": REPORT_DIR / "transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv",
    },
    {
        "target": "FBYG15K",
        "scenario": "cross_graph",
        "method_variant": "v8_mild_da_expand5",
        "source_csv": REPORT_DIR / "transfer_adapt_fbyg_expand5_progress_compare_vs_baseline.csv",
    },
]


MAIN_OUT_CSV = REPORT_DIR / "transfer_adapt_main_results_4target.csv"
MAIN_OUT_MD = REPORT_DIR / "transfer_adapt_main_results_4target.md"
BUCKET_OUT_CSV = REPORT_DIR / "transfer_adapt_error_bucket_summary.csv"
BUCKET_OUT_MD = REPORT_DIR / "transfer_adapt_error_bucket_summary.md"


METRIC_KEYS = [
    "delta_avg_hits@1_mean",
    "delta_avg_hits@10_mean",
    "delta_avg_mrr_mean",
    "delta_avg_mr_mean",
]


def to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing csv: {path}")
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_row(rows: List[Dict[str, str]], target: str) -> Dict[str, str]:
    for row in rows:
        if row.get("target") == target:
            return row
    raise ValueError(f"Target {target} not found in source csv")


def confidence_level(baseline_runs: int, method_runs: int) -> str:
    n = min(baseline_runs, method_runs)
    if n >= 5:
        return "formal_5seed"
    if n >= 3:
        return "mid_3seed"
    return "pilot_2seed"


def difficulty_bucket(baseline_mrr: float) -> str:
    if baseline_mrr < 0.05:
        return "very_hard"
    if baseline_mrr < 0.20:
        return "hard"
    if baseline_mrr < 0.40:
        return "moderate"
    return "easy"


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_main_rows() -> List[Dict[str, object]]:
    out = []
    for spec in SOURCE_SPECS:
        rows = read_csv_rows(spec["source_csv"])
        row = find_row(rows, spec["target"])
        b_runs = int(float(row["baseline_num_runs"]))
        m_runs = int(float(row["tmmeada_num_runs"]))
        b_mrr = to_float(row["baseline_avg_mrr_mean"])
        main_row = {
            "target": spec["target"],
            "scenario": spec["scenario"],
            "method_variant": spec["method_variant"],
            "baseline_num_runs": b_runs,
            "tmmeada_num_runs": m_runs,
            "confidence_level": confidence_level(b_runs, m_runs),
            "difficulty_bucket": difficulty_bucket(b_mrr),
            "baseline_avg_hits@1_mean": to_float(row["baseline_avg_hits@1_mean"]),
            "tmmeada_avg_hits@1_mean": to_float(row["tmmeada_avg_hits@1_mean"]),
            "delta_avg_hits@1_mean": to_float(row["delta_avg_hits@1_mean"]),
            "baseline_avg_hits@10_mean": to_float(row["baseline_avg_hits@10_mean"]),
            "tmmeada_avg_hits@10_mean": to_float(row["tmmeada_avg_hits@10_mean"]),
            "delta_avg_hits@10_mean": to_float(row["delta_avg_hits@10_mean"]),
            "baseline_avg_mrr_mean": b_mrr,
            "tmmeada_avg_mrr_mean": to_float(row["tmmeada_avg_mrr_mean"]),
            "delta_avg_mrr_mean": to_float(row["delta_avg_mrr_mean"]),
            "baseline_avg_mr_mean": to_float(row["baseline_avg_mr_mean"]),
            "tmmeada_avg_mr_mean": to_float(row["tmmeada_avg_mr_mean"]),
            "delta_avg_mr_mean": to_float(row["delta_avg_mr_mean"]),
            "source_compare_csv": str(spec["source_csv"].relative_to(ROOT)).replace("\\", "/"),
        }
        out.append(main_row)
    return out


def write_main_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "scenario",
        "method_variant",
        "baseline_num_runs",
        "tmmeada_num_runs",
        "confidence_level",
        "difficulty_bucket",
        "baseline_avg_hits@1_mean",
        "tmmeada_avg_hits@1_mean",
        "delta_avg_hits@1_mean",
        "baseline_avg_hits@10_mean",
        "tmmeada_avg_hits@10_mean",
        "delta_avg_hits@10_mean",
        "baseline_avg_mrr_mean",
        "tmmeada_avg_mrr_mean",
        "delta_avg_mrr_mean",
        "baseline_avg_mr_mean",
        "tmmeada_avg_mr_mean",
        "delta_avg_mr_mean",
        "source_compare_csv",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_main_md(rows: List[Dict[str, object]], out_md: Path) -> None:
    lines = []
    lines.append("# Transfer Adapt Main Results (4 Targets)")
    lines.append("")
    lines.append("| target | scenario | variant | runs(b/m) | delta H@1 | delta H@10 | delta MRR | delta MR | confidence |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['target']} | {r['scenario']} | {r['method_variant']} | "
            f"{r['baseline_num_runs']}/{r['tmmeada_num_runs']} | "
            f"{r['delta_avg_hits@1_mean']:+.6f} | {r['delta_avg_hits@10_mean']:+.6f} | "
            f"{r['delta_avg_mrr_mean']:+.6f} | {r['delta_avg_mr_mean']:+.6f} | {r['confidence_level']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `ja_en` and `FBDB15K` currently use 2-seed formal snapshots.")
    lines.append("- `fr_en` and `FBYG15K` already use 5-seed formal snapshots.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def group_by(rows: List[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        groups.setdefault(str(r[key]), []).append(r)
    out = []
    for gk in sorted(groups.keys()):
        items = groups[gk]
        out.append(
            {
                "bucket_type": key,
                "bucket_name": gk,
                "num_targets": len(items),
                "mean_delta_avg_hits@1_mean": mean([float(x["delta_avg_hits@1_mean"]) for x in items]),
                "mean_delta_avg_hits@10_mean": mean([float(x["delta_avg_hits@10_mean"]) for x in items]),
                "mean_delta_avg_mrr_mean": mean([float(x["delta_avg_mrr_mean"]) for x in items]),
                "mean_delta_avg_mr_mean": mean([float(x["delta_avg_mr_mean"]) for x in items]),
            }
        )
    return out


def write_bucket_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket_type",
        "bucket_name",
        "num_targets",
        "mean_delta_avg_hits@1_mean",
        "mean_delta_avg_hits@10_mean",
        "mean_delta_avg_mrr_mean",
        "mean_delta_avg_mr_mean",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_bucket_md(rows: List[Dict[str, object]], out_md: Path) -> None:
    lines = []
    lines.append("# Transfer Adapt Error-Bucket Summary")
    lines.append("")
    lines.append("| bucket_type | bucket_name | n_targets | mean delta H@1 | mean delta H@10 | mean delta MRR | mean delta MR |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['bucket_type']} | {r['bucket_name']} | {r['num_targets']} | "
            f"{r['mean_delta_avg_hits@1_mean']:+.6f} | {r['mean_delta_avg_hits@10_mean']:+.6f} | "
            f"{r['mean_delta_avg_mrr_mean']:+.6f} | {r['mean_delta_avg_mr_mean']:+.6f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Positive `delta_avg_mrr_mean` indicates transfer gain over baseline.")
    lines.append("- Negative `delta_avg_mr_mean` indicates lower mean rank (better).")
    lines.append("- Bucket views are intended for report-side error analysis summarization.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    main_rows = build_main_rows()
    write_main_csv(main_rows, MAIN_OUT_CSV)
    write_main_md(main_rows, MAIN_OUT_MD)

    bucket_rows = []
    bucket_rows.extend(group_by(main_rows, "scenario"))
    bucket_rows.extend(group_by(main_rows, "confidence_level"))
    bucket_rows.extend(group_by(main_rows, "difficulty_bucket"))
    write_bucket_csv(bucket_rows, BUCKET_OUT_CSV)
    write_bucket_md(bucket_rows, BUCKET_OUT_MD)

    overall = {
        mk: mean([float(r[mk]) for r in main_rows]) for mk in METRIC_KEYS
    }
    print("[DONE] main csv:", MAIN_OUT_CSV)
    print("[DONE] main md :", MAIN_OUT_MD)
    print("[DONE] bucket csv:", BUCKET_OUT_CSV)
    print("[DONE] bucket md :", BUCKET_OUT_MD)
    print(
        "[DONE] overall delta: "
        f"H@1={overall['delta_avg_hits@1_mean']:+.6f}, "
        f"H@10={overall['delta_avg_hits@10_mean']:+.6f}, "
        f"MRR={overall['delta_avg_mrr_mean']:+.6f}, "
        f"MR={overall['delta_avg_mr_mean']:+.6f}"
    )


if __name__ == "__main__":
    main()
