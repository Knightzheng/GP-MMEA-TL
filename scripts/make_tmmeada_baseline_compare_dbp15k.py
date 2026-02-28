import csv
from pathlib import Path


DBP_LANGS = ["zh_en", "ja_en", "fr_en"]
METRICS = [
    ("l2r_hits@1_mean", "l2r_hits@1_std", "l2r Hits@1"),
    ("l2r_hits@10_mean", "l2r_hits@10_std", "l2r Hits@10"),
    ("l2r_mrr_mean", "l2r_mrr_std", "l2r MRR"),
    ("r2l_hits@1_mean", "r2l_hits@1_std", "r2l Hits@1"),
    ("r2l_hits@10_mean", "r2l_hits@10_std", "r2l Hits@10"),
    ("r2l_mrr_mean", "r2l_mrr_std", "r2l MRR"),
]


def read_agg(path: Path):
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["lang_pair"]] = row
    return data


def main():
    base_path = Path("reports/meaformer_results_mean_std.csv")
    tmmeada_path = Path("reports/tmmeada_results_mean_std.csv")
    out_csv = Path("reports/tmmeada_vs_baseline_dbp15k.csv")
    out_md = Path("reports/tmmeada_vs_baseline_dbp15k.md")

    baseline = read_agg(base_path)
    tmmeada = read_agg(tmmeada_path)

    rows = []
    for lang in DBP_LANGS:
        if lang not in baseline or lang not in tmmeada:
            continue
        for mean_key, std_key, metric_name in METRICS:
            b_mean = float(baseline[lang][mean_key])
            b_std = float(baseline[lang][std_key])
            t_mean = float(tmmeada[lang][mean_key])
            t_std = float(tmmeada[lang][std_key])
            rows.append(
                {
                    "lang_pair": lang,
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
        "# DBP15K: MEAformer Baseline vs TMMEA-DA MVP (1-epoch, 5 seeds)",
        "",
        "注：当前对比为 1-epoch 冒烟设置，主要用于流程验证与模块诊断。",
        "",
    ]
    for lang in DBP_LANGS:
        lang_rows = [r for r in rows if r["lang_pair"] == lang]
        if not lang_rows:
            continue
        md.append(f"## {lang}")
        md.append("| metric | baseline (mean±std) | tmmeada (mean±std) | delta |")
        md.append("|---|---:|---:|---:|")
        for r in lang_rows:
            md.append(
                f"| {r['metric']} | {r['baseline_mean']:.4f} ± {r['baseline_std']:.4f} | "
                f"{r['tmmeada_mean']:.4f} ± {r['tmmeada_std']:.4f} | "
                f"{r['delta_tmmeada_minus_baseline']:+.4f} |"
            )
        md.append("")

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
