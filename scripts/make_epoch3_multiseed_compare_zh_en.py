import csv
from pathlib import Path


METRICS = [
    ("l2r_hits@1_mean", "l2r_hits@1_std", "l2r Hits@1"),
    ("l2r_hits@10_mean", "l2r_hits@10_std", "l2r Hits@10"),
    ("l2r_mrr_mean", "l2r_mrr_std", "l2r MRR"),
    ("r2l_hits@1_mean", "r2l_hits@1_std", "r2l Hits@1"),
    ("r2l_hits@10_mean", "r2l_hits@10_std", "r2l Hits@10"),
    ("r2l_mrr_mean", "r2l_mrr_std", "r2l MRR"),
]


def read_row(path: Path, key: str = "zh_en"):
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["lang_pair"] == key:
                return row
    raise ValueError(f"missing {key} in {path}")


def main():
    baseline = read_row(Path("reports/baseline_epoch3_results_mean_std.csv"))
    method = read_row(Path("reports/tmmeada_v1_best_epoch3_results_mean_std.csv"))

    out_csv = Path("reports/epoch3_multiseed_compare_zh_en.csv")
    out_md = Path("reports/epoch3_multiseed_compare_zh_en.md")

    rows = []
    for mean_key, std_key, metric_name in METRICS:
        b_mean = float(baseline[mean_key])
        b_std = float(baseline[std_key])
        m_mean = float(method[mean_key])
        m_std = float(method[std_key])
        rows.append(
            {
                "metric": metric_name,
                "baseline_epoch3_mean": round(b_mean, 4),
                "baseline_epoch3_std": round(b_std, 4),
                "tmmeada_v1_best_epoch3_mean": round(m_mean, 4),
                "tmmeada_v1_best_epoch3_std": round(m_std, 4),
                "delta_method_minus_baseline": round(m_mean - b_mean, 4),
            }
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# zh_en epoch3 multi-seed: baseline vs TMMEA-DA v1_best",
        "",
        f"- baseline num_runs: `{baseline['num_runs']}`",
        f"- method num_runs: `{method['num_runs']}`",
        "",
        "| metric | baseline_epoch3 | tmmeada_v1_best_epoch3 | delta(method-baseline) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | "
            f"{row['baseline_epoch3_mean']:.4f} +/- {row['baseline_epoch3_std']:.4f} | "
            f"{row['tmmeada_v1_best_epoch3_mean']:.4f} +/- {row['tmmeada_v1_best_epoch3_std']:.4f} | "
            f"{row['delta_method_minus_baseline']:+.4f} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- Both methods share the same epoch=3 training budget and seed set.")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
