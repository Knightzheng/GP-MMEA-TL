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

    out_csv = Path("reports/tmmeada_v1_compare_zh_en.csv")
    out_md = Path("reports/tmmeada_v1_compare_zh_en.md")

    rows = []
    for mean_key, std_key, metric_name in METRICS:
        b_mean = float(baseline[mean_key])
        b_std = float(baseline[std_key])
        v0_mean = float(v0[mean_key])
        v0_std = float(v0[std_key])
        v1_mean = float(v1[mean_key])
        v1_std = float(v1[std_key])
        rows.append(
            {
                "metric": metric_name,
                "baseline_mean": round(b_mean, 4),
                "baseline_std": round(b_std, 4),
                "v0_mean": round(v0_mean, 4),
                "v0_std": round(v0_std, 4),
                "v1_mean": round(v1_mean, 4),
                "v1_std": round(v1_std, 4),
                "delta_v0_minus_baseline": round(v0_mean - b_mean, 4),
                "delta_v1_minus_baseline": round(v1_mean - b_mean, 4),
                "delta_v1_minus_v0": round(v1_mean - v0_mean, 4),
            }
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# zh_en: Baseline vs TMMEA-DA v0 vs TMMEA-DA v1 (1-epoch, 5 seeds)",
        "",
        "| metric | baseline | v0 | v1 | v1-v0 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['metric']} | "
            f"{row['baseline_mean']:.4f} ± {row['baseline_std']:.4f} | "
            f"{row['v0_mean']:.4f} ± {row['v0_std']:.4f} | "
            f"{row['v1_mean']:.4f} ± {row['v1_std']:.4f} | "
            f"{row['delta_v1_minus_v0']:+.4f} |"
        )
    md.append("")
    md.append("注：v1 在 v0 基础上增加 source_select + missing_gate。")
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
