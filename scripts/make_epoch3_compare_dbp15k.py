import csv
from pathlib import Path


METRICS = [
    ("l2r_hits@1_mean", "l2r Hits@1"),
    ("l2r_hits@10_mean", "l2r Hits@10"),
    ("l2r_mrr_mean", "l2r MRR"),
    ("r2l_hits@1_mean", "r2l Hits@1"),
    ("r2l_hits@10_mean", "r2l Hits@10"),
    ("r2l_mrr_mean", "r2l MRR"),
]


def read_csv_as_map(path: Path):
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["lang_pair"]] = row
    return out


def main():
    baseline_map = read_csv_as_map(Path("reports/baseline_epoch3_results_mean_std.csv"))
    method_map = read_csv_as_map(Path("reports/tmmeada_v1_best_epoch3_results_mean_std.csv"))
    langs = [x for x in ("zh_en", "ja_en", "fr_en") if x in baseline_map and x in method_map]

    out_csv = Path("reports/epoch3_compare_dbp15k.csv")
    out_md = Path("reports/epoch3_compare_dbp15k.md")

    rows = []
    for lang in langs:
        b = baseline_map[lang]
        m = method_map[lang]
        for key, metric_name in METRICS:
            rows.append(
                {
                    "lang_pair": lang,
                    "metric": metric_name,
                    "baseline_epoch3_mean": round(float(b[key]), 4),
                    "tmmeada_v1_best_epoch3_mean": round(float(m[key]), 4),
                    "delta_method_minus_baseline": round(float(m[key]) - float(b[key]), 4),
                    "baseline_num_runs": int(b["num_runs"]),
                    "method_num_runs": int(m["num_runs"]),
                }
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# DBP15K epoch3: baseline vs TMMEA-DA v1_best",
        "",
        "| lang_pair | baseline_runs | method_runs | l2r H@1 delta | l2r H@10 delta | l2r MRR delta | r2l H@1 delta | r2l H@10 delta | r2l MRR delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lang in langs:
        d = {r["metric"]: r["delta_method_minus_baseline"] for r in rows if r["lang_pair"] == lang}
        base_runs = next(r["baseline_num_runs"] for r in rows if r["lang_pair"] == lang)
        meth_runs = next(r["method_num_runs"] for r in rows if r["lang_pair"] == lang)
        lines.append(
            f"| {lang} | {base_runs} | {meth_runs} | "
            f"{d['l2r Hits@1']:+.4f} | {d['l2r Hits@10']:+.4f} | {d['l2r MRR']:+.4f} | "
            f"{d['r2l Hits@1']:+.4f} | {d['r2l Hits@10']:+.4f} | {d['r2l MRR']:+.4f} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- zh_en is a 5-seed formal comparison; ja_en/fr_en are current pilot single-seed results.")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
