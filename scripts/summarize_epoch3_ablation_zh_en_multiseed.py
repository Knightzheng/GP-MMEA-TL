import csv
import math
from collections import defaultdict
from pathlib import Path


METRICS = [
    "l2r_hits@1",
    "l2r_hits@10",
    "l2r_mrr",
    "r2l_hits@1",
    "r2l_hits@10",
    "r2l_mrr",
]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_variant_values():
    baseline_rows = read_rows(Path("reports/baseline_epoch3_results_summary.csv"))
    full_rows = read_rows(Path("reports/tmmeada_v1_best_epoch3_results_summary.csv"))
    ablation_rows = read_rows(Path("reports/tmmeada_v1_ablation_epoch3_results_summary.csv"))

    vals = defaultdict(lambda: defaultdict(list))

    for r in baseline_rows:
        if r["lang_pair"] == "zh_en" and "-zh_en-s" in r["run_id"]:
            for k in METRICS:
                vals["baseline"][k].append(float(r[k]))

    for r in full_rows:
        if r["lang_pair"] == "zh_en" and "-zh_en-s" in r["run_id"]:
            for k in METRICS:
                vals["v1_best_full"][k].append(float(r[k]))

    for r in ablation_rows:
        if r["lang_pair"] != "zh_en" or "-zh_en-s" not in r["run_id"]:
            continue
        run_id = r["run_id"]
        if "wo-domain" in run_id:
            key = "wo_domain_align"
        elif "wo-source" in run_id:
            key = "wo_source_select"
        elif "wo-missing" in run_id:
            key = "wo_missing_gate"
        else:
            continue
        for k in METRICS:
            vals[key][k].append(float(r[k]))

    return vals


def main():
    vals = collect_variant_values()
    if "v1_best_full" not in vals:
        raise RuntimeError("missing v1_best_full rows for zh_en in epoch3 summaries")

    order = [
        "baseline",
        "v1_best_full",
        "wo_domain_align",
        "wo_source_select",
        "wo_missing_gate",
    ]
    present = [k for k in order if k in vals]

    full_means = {k: mean(vals["v1_best_full"][k]) for k in METRICS}
    out_rows = []
    for variant in present:
        row = {"variant": variant, "num_runs": len(vals[variant][METRICS[0]])}
        for k in METRICS:
            row[f"{k}_mean"] = round(mean(vals[variant][k]), 4)
            row[f"{k}_std"] = round(std(vals[variant][k]), 4)
        row["delta_l2r_h1_vs_full"] = round(row["l2r_hits@1_mean"] - full_means["l2r_hits@1"], 4)
        row["delta_l2r_mrr_vs_full"] = round(row["l2r_mrr_mean"] - full_means["l2r_mrr"], 4)
        row["delta_r2l_h1_vs_full"] = round(row["r2l_hits@1_mean"] - full_means["r2l_hits@1"], 4)
        row["delta_r2l_mrr_vs_full"] = round(row["r2l_mrr_mean"] - full_means["r2l_mrr"], 4)
        out_rows.append(row)

    out_csv = Path("reports/epoch3_ablation_zh_en_multiseed.csv")
    out_md = Path("reports/epoch3_ablation_zh_en_multiseed.md")

    fields = ["variant", "num_runs"]
    for k in METRICS:
        fields.extend([f"{k}_mean", f"{k}_std"])
    fields.extend(
        [
            "delta_l2r_h1_vs_full",
            "delta_l2r_mrr_vs_full",
            "delta_r2l_h1_vs_full",
            "delta_r2l_mrr_vs_full",
        ]
    )
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    lines = [
        "# zh_en epoch3 ablation (multi-seed)",
        "",
        "| variant | runs | l2r H@1 | l2r H@10 | l2r MRR | r2l H@1 | r2l H@10 | r2l MRR | d(l2r H@1) vs full | d(r2l H@1) vs full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in out_rows:
        lines.append(
            f"| {r['variant']} | {r['num_runs']} | "
            f"{r['l2r_hits@1_mean']:.4f} +/- {r['l2r_hits@1_std']:.4f} | "
            f"{r['l2r_hits@10_mean']:.4f} +/- {r['l2r_hits@10_std']:.4f} | "
            f"{r['l2r_mrr_mean']:.4f} +/- {r['l2r_mrr_std']:.4f} | "
            f"{r['r2l_hits@1_mean']:.4f} +/- {r['r2l_hits@1_std']:.4f} | "
            f"{r['r2l_hits@10_mean']:.4f} +/- {r['r2l_hits@10_std']:.4f} | "
            f"{r['r2l_mrr_mean']:.4f} +/- {r['r2l_mrr_std']:.4f} | "
            f"{r['delta_l2r_h1_vs_full']:+.4f} | {r['delta_r2l_h1_vs_full']:+.4f} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- Full method is `v1_best_full` with domain_align + source_select + missing_gate enabled.")
    lines.append("- Deltas are computed against full method means.")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
