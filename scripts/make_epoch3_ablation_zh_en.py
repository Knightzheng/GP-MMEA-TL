import csv
from pathlib import Path


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_unique(rows, predicate, name):
    matched = [r for r in rows if predicate(r)]
    if len(matched) != 1:
        raise RuntimeError(f"{name}: expected 1 row, got {len(matched)}")
    return matched[0]


def to_float_map(row):
    return {
        "l2r_hits@1": float(row["l2r_hits@1"]),
        "l2r_hits@10": float(row["l2r_hits@10"]),
        "l2r_mrr": float(row["l2r_mrr"]),
        "r2l_hits@1": float(row["r2l_hits@1"]),
        "r2l_hits@10": float(row["r2l_hits@10"]),
        "r2l_mrr": float(row["r2l_mrr"]),
    }


def main():
    baseline_rows = read_rows(Path("reports/baseline_epoch3_results_summary.csv"))
    full_rows = read_rows(Path("reports/tmmeada_v1_best_epoch3_results_summary.csv"))
    ablation_rows = read_rows(Path("reports/tmmeada_v1_ablation_epoch3_results_summary.csv"))

    baseline = pick_unique(
        baseline_rows,
        lambda r: r["lang_pair"] == "zh_en" and "-zh_en-s42" in r["run_id"],
        "baseline zh_en seed42",
    )
    full = pick_unique(
        full_rows,
        lambda r: r["lang_pair"] == "zh_en" and "-zh_en-s42" in r["run_id"],
        "v1_best full zh_en seed42",
    )
    wo_domain = pick_unique(
        ablation_rows,
        lambda r: r["lang_pair"] == "zh_en" and "wo-domain" in r["run_id"] and "-zh_en-s42" in r["run_id"],
        "wo_domain zh_en seed42",
    )
    wo_source = pick_unique(
        ablation_rows,
        lambda r: r["lang_pair"] == "zh_en" and "wo-source" in r["run_id"] and "-zh_en-s42" in r["run_id"],
        "wo_source zh_en seed42",
    )
    wo_missing = pick_unique(
        ablation_rows,
        lambda r: r["lang_pair"] == "zh_en" and "wo-missing" in r["run_id"] and "-zh_en-s42" in r["run_id"],
        "wo_missing zh_en seed42",
    )

    full_vals = to_float_map(full)
    variants = [
        ("baseline", to_float_map(baseline)),
        ("v1_best_full", full_vals),
        ("wo_domain_align", to_float_map(wo_domain)),
        ("wo_source_select", to_float_map(wo_source)),
        ("wo_missing_gate", to_float_map(wo_missing)),
    ]

    out_rows = []
    for name, vals in variants:
        out_rows.append(
            {
                "variant": name,
                "l2r_hits@1": round(vals["l2r_hits@1"], 4),
                "l2r_hits@10": round(vals["l2r_hits@10"], 4),
                "l2r_mrr": round(vals["l2r_mrr"], 4),
                "r2l_hits@1": round(vals["r2l_hits@1"], 4),
                "r2l_hits@10": round(vals["r2l_hits@10"], 4),
                "r2l_mrr": round(vals["r2l_mrr"], 4),
                "delta_l2r_h1_vs_full": round(vals["l2r_hits@1"] - full_vals["l2r_hits@1"], 4),
                "delta_l2r_mrr_vs_full": round(vals["l2r_mrr"] - full_vals["l2r_mrr"], 4),
                "delta_r2l_h1_vs_full": round(vals["r2l_hits@1"] - full_vals["r2l_hits@1"], 4),
                "delta_r2l_mrr_vs_full": round(vals["r2l_mrr"] - full_vals["r2l_mrr"], 4),
            }
        )

    out_csv = Path("reports/epoch3_ablation_zh_en.csv")
    out_md = Path("reports/epoch3_ablation_zh_en.md")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    md = [
        "# zh_en epoch3 ablation (seed=42)",
        "",
        "| variant | l2r H@1 | l2r H@10 | l2r MRR | r2l H@1 | r2l H@10 | r2l MRR | d(l2r H@1) vs full | d(r2l H@1) vs full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in out_rows:
        md.append(
            f"| {r['variant']} | {r['l2r_hits@1']:.4f} | {r['l2r_hits@10']:.4f} | {r['l2r_mrr']:.4f} | "
            f"{r['r2l_hits@1']:.4f} | {r['r2l_hits@10']:.4f} | {r['r2l_mrr']:.4f} | "
            f"{r['delta_l2r_h1_vs_full']:+.4f} | {r['delta_r2l_h1_vs_full']:+.4f} |"
        )
    md.append("")
    md.append("Notes:")
    md.append("- This is a pilot ablation under zh_en + epoch3 + seed=42.")
    md.append("- For formal claims, extend each variant to the same 5-seed setting.")
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
