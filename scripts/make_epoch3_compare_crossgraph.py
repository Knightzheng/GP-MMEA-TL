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
    baseline_map = read_csv_as_map(Path("reports/baseline_epoch3_crossgraph_results_mean_std.csv"))
    method_map = read_csv_as_map(Path("reports/tmmeada_v1_best_epoch3_crossgraph_results_mean_std.csv"))
    datasets = [x for x in ("FBDB15K", "FBYG15K") if x in baseline_map and x in method_map]

    out_csv = Path("reports/epoch3_compare_crossgraph.csv")
    out_md = Path("reports/epoch3_compare_crossgraph.md")

    rows = []
    for ds in datasets:
        b = baseline_map[ds]
        m = method_map[ds]
        for key, metric_name in METRICS:
            rows.append(
                {
                    "dataset": ds,
                    "metric": metric_name,
                    "baseline_epoch3_mean": round(float(b[key]), 4),
                    "tmmeada_v1_best_epoch3_mean": round(float(m[key]), 4),
                    "delta_method_minus_baseline": round(float(m[key]) - float(b[key]), 4),
                    "baseline_num_runs": int(b["num_runs"]),
                    "method_num_runs": int(m["num_runs"]),
                }
            )

    if not rows:
        raise RuntimeError("No overlapping datasets found in mean/std files.")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Cross-graph epoch3: baseline vs TMMEA-DA v1_best",
        "",
        "| dataset | baseline_runs | method_runs | l2r H@1 delta | l2r H@10 delta | l2r MRR delta | r2l H@1 delta | r2l H@10 delta | r2l MRR delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in datasets:
        d = {r["metric"]: r["delta_method_minus_baseline"] for r in rows if r["dataset"] == ds}
        base_runs = next(r["baseline_num_runs"] for r in rows if r["dataset"] == ds)
        meth_runs = next(r["method_num_runs"] for r in rows if r["dataset"] == ds)
        lines.append(
            f"| {ds} | {base_runs} | {meth_runs} | "
            f"{d['l2r Hits@1']:+.4f} | {d['l2r Hits@10']:+.4f} | {d['l2r MRR']:+.4f} | "
            f"{d['r2l Hits@1']:+.4f} | {d['r2l Hits@10']:+.4f} | {d['r2l MRR']:+.4f} |"
        )

    lines.append("")
    lines.append("Notes:")
    run_desc = []
    all_formal = True
    for ds in datasets:
        base_runs = next(r["baseline_num_runs"] for r in rows if r["dataset"] == ds)
        meth_runs = next(r["method_num_runs"] for r in rows if r["dataset"] == ds)
        run_desc.append(f"{ds}={base_runs}/{meth_runs}")
        if min(base_runs, meth_runs) < 5:
            all_formal = False
    lines.append("- run counts (baseline/method): " + ", ".join(run_desc))
    if all_formal:
        lines.append("- all listed datasets are formal 5-seed comparisons.")
    else:
        lines.append("- current stage includes pilot comparisons (fewer than 5 seeds).")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
