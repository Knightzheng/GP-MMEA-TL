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
    baseline = read_row(Path("reports/meaformer_results_mean_std.csv"))
    v0 = read_row(Path("reports/tmmeada_results_mean_std.csv"))
    v1 = read_row(Path("reports/tmmeada_v1_results_mean_std.csv"))
    v1_best = read_row(Path("reports/tmmeada_v1_best_results_mean_std.csv"))

    out_csv = Path("reports/tmmeada_v1_best_compare_zh_en.csv")
    out_md = Path("reports/tmmeada_v1_best_compare_zh_en.md")

    rows = []
    for mean_key, std_key, metric_name in METRICS:
        b_mean = float(baseline[mean_key])
        b_std = float(baseline[std_key])
        v0_mean = float(v0[mean_key])
        v0_std = float(v0[std_key])
        v1_mean = float(v1[mean_key])
        v1_std = float(v1[std_key])
        vb_mean = float(v1_best[mean_key])
        vb_std = float(v1_best[std_key])
        rows.append(
            {
                "metric": metric_name,
                "baseline_mean": round(b_mean, 4),
                "baseline_std": round(b_std, 4),
                "v0_mean": round(v0_mean, 4),
                "v0_std": round(v0_std, 4),
                "v1_mean": round(v1_mean, 4),
                "v1_std": round(v1_std, 4),
                "v1_best_mean": round(vb_mean, 4),
                "v1_best_std": round(vb_std, 4),
                "delta_v0_minus_baseline": round(v0_mean - b_mean, 4),
                "delta_v1_minus_baseline": round(v1_mean - b_mean, 4),
                "delta_v1_best_minus_baseline": round(vb_mean - b_mean, 4),
                "delta_v1_best_minus_v1": round(vb_mean - v1_mean, 4),
            }
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# zh_en: Baseline vs TMMEA-DA v0 vs v1 vs v1_best (1-epoch, 5 seeds)",
        "",
        "| metric | baseline | v0 | v1 | v1_best | delta(v1_best-v1) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | "
            f"{row['baseline_mean']:.4f} +/- {row['baseline_std']:.4f} | "
            f"{row['v0_mean']:.4f} +/- {row['v0_std']:.4f} | "
            f"{row['v1_mean']:.4f} +/- {row['v1_std']:.4f} | "
            f"{row['v1_best_mean']:.4f} +/- {row['v1_best_std']:.4f} | "
            f"{row['delta_v1_best_minus_v1']:+.4f} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- v1_best uses sweep-selected setting: dw=0.1, sw=0.05, mw=0.1, temp=1.0.")
    lines.append("- Current results are under the 1-epoch quick validation setup.")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
