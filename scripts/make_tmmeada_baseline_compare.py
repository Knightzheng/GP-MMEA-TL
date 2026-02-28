import csv
from pathlib import Path


def read_row(path: Path, lang_pair: str):
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["lang_pair"] == lang_pair:
                return row
    raise ValueError(f"missing lang_pair={lang_pair} in {path}")


def main():
    baseline_path = Path("reports/meaformer_results_mean_std.csv")
    tmmeada_path = Path("reports/tmmeada_results_mean_std.csv")
    out_csv = Path("reports/tmmeada_vs_baseline_zh_en.csv")
    out_md = Path("reports/tmmeada_vs_baseline_zh_en.md")

    baseline = read_row(baseline_path, "zh_en")
    tmmeada = read_row(tmmeada_path, "zh_en")

    metrics = [
        ("l2r_hits@1_mean", "l2r_hits@1_std", "l2r Hits@1"),
        ("l2r_hits@10_mean", "l2r_hits@10_std", "l2r Hits@10"),
        ("l2r_mrr_mean", "l2r_mrr_std", "l2r MRR"),
        ("r2l_hits@1_mean", "r2l_hits@1_std", "r2l Hits@1"),
        ("r2l_hits@10_mean", "r2l_hits@10_std", "r2l Hits@10"),
        ("r2l_mrr_mean", "r2l_mrr_std", "r2l MRR"),
    ]

    rows = []
    for mean_key, std_key, metric_name in metrics:
        b_mean = float(baseline[mean_key])
        b_std = float(baseline[std_key])
        t_mean = float(tmmeada[mean_key])
        t_std = float(tmmeada[std_key])
        rows.append(
            {
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

    md_lines = [
        "# zh_en: MEAformer Baseline vs TMMEA-DA MVP (1-epoch, 5 seeds)",
        "",
        f"- Baseline runs: {baseline['num_runs']}",
        f"- TMMEA-DA runs: {tmmeada['num_runs']}",
        "",
        "| metric | baseline (mean±std) | tmmeada (mean±std) | delta (tmmeada-baseline) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['metric']} | {row['baseline_mean']:.4f} ± {row['baseline_std']:.4f} | "
            f"{row['tmmeada_mean']:.4f} ± {row['tmmeada_std']:.4f} | "
            f"{row['delta_tmmeada_minus_baseline']:+.4f} |"
        )
    md_lines.append("")
    md_lines.append("注：当前为 1 epoch 冒烟配置，主要用于流程与可复现验证，不用于最终 SOTA 结论。")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
