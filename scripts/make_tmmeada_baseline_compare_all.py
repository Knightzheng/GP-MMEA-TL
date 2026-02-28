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


def read_rows(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["lang_pair"]] = row
    return rows


def main():
    baseline_path = Path("reports/meaformer_results_mean_std.csv")
    tmmeada_path = Path("reports/tmmeada_results_mean_std.csv")
    out_csv = Path("reports/tmmeada_vs_baseline_all.csv")
    out_md = Path("reports/tmmeada_vs_baseline_all.md")

    baseline = read_rows(baseline_path)
    tmmeada = read_rows(tmmeada_path)

    keys = sorted(set(baseline.keys()) & set(tmmeada.keys()))
    rows = []
    for key in keys:
        for mean_key, std_key, metric_name in METRICS:
            b_mean = float(baseline[key][mean_key])
            b_std = float(baseline[key][std_key])
            t_mean = float(tmmeada[key][mean_key])
            t_std = float(tmmeada[key][std_key])
            rows.append(
                {
                    "dataset": key,
                    "metric": metric_name,
                    "baseline_mean": round(b_mean, 4),
                    "baseline_std": round(b_std, 4),
                    "tmmeada_mean": round(t_mean, 4),
                    "tmmeada_std": round(t_std, 4),
                    "delta_tmmeada_minus_baseline": round(t_mean - b_mean, 4),
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# All Datasets: MEAformer Baseline vs TMMEA-DA MVP (1-epoch, 5 seeds)",
        "",
        "注：该对比用于阶段性流程/模块诊断，非最终训练预算下的SOTA比较。",
        "",
    ]
    for key in keys:
        md.append(f"## {key}")
        md.append("| metric | baseline (mean±std) | tmmeada (mean±std) | delta |")
        md.append("|---|---:|---:|---:|")
        for row in rows:
            if row["dataset"] != key:
                continue
            md.append(
                f"| {row['metric']} | {row['baseline_mean']:.4f} ± {row['baseline_std']:.4f} | "
                f"{row['tmmeada_mean']:.4f} ± {row['tmmeada_std']:.4f} | "
                f"{row['delta_tmmeada_minus_baseline']:+.4f} |"
            )
        md.append("")

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
