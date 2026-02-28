import csv
from pathlib import Path


METRICS = [
    ("l2r_hits@1", "l2r Hits@1"),
    ("l2r_hits@10", "l2r Hits@10"),
    ("l2r_mrr", "l2r MRR"),
    ("r2l_hits@1", "r2l Hits@1"),
    ("r2l_hits@10", "r2l Hits@10"),
    ("r2l_mrr", "r2l MRR"),
]


def read_row(path: Path, key: str = "zh_en"):
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["lang_pair"] == key:
                return row
    raise ValueError(f"missing {key} in {path}")


def main():
    baseline = read_row(Path("reports/baseline_epoch3_results_summary.csv"))
    method = read_row(Path("reports/tmmeada_v1_best_epoch3_results_summary.csv"))

    out_csv = Path("reports/epoch3_pilot_compare_zh_en.csv")
    out_md = Path("reports/epoch3_pilot_compare_zh_en.md")

    rows = []
    for key, name in METRICS:
        b = float(baseline[key])
        m = float(method[key])
        rows.append(
            {
                "metric": name,
                "baseline_epoch3": round(b, 4),
                "tmmeada_v1_best_epoch3": round(m, 4),
                "delta_method_minus_baseline": round(m - b, 4),
            }
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# zh_en epoch3 pilot: baseline vs TMMEA-DA v1_best (seed=42)",
        "",
        f"- baseline_run: `{baseline['run_id']}`",
        f"- method_run: `{method['run_id']}`",
        "",
        "| metric | baseline_epoch3 | tmmeada_v1_best_epoch3 | delta(method-baseline) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['baseline_epoch3']:.4f} | "
            f"{row['tmmeada_v1_best_epoch3']:.4f} | {row['delta_method_minus_baseline']:+.4f} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- This is a single-seed pilot under epoch=3, used for training-budget trend check only.")
    lines.append("- Next formal step should be multi-seed runs with the same epoch budget.")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
