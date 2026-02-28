import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import yaml


L2R_RE = re.compile(r"l2r: acc of top \[1, 10, 50\] = \[([0-9.\s]+)\], mr = ([0-9.]+), mrr = ([0-9.]+)")
R2L_RE = re.compile(r"r2l: acc of top \[1, 10, 50\] = \[([0-9.\s]+)\], mr = ([0-9.]+), mrr = ([0-9.]+)")


def parse_top_vals(text: str):
    parts = [p for p in text.strip().split() if p]
    vals = [float(x) for x in parts[:3]]
    while len(vals) < 3:
        vals.append(0.0)
    return vals


def extract_metrics(log_text: str):
    l2r = None
    r2l = None
    for line in log_text.splitlines():
        if "Ep" in line and "l2r: acc of top [1, 10, 50]" in line:
            m = L2R_RE.search(line)
            if m:
                tops = parse_top_vals(m.group(1))
                l2r = {
                    "hits@1": tops[0],
                    "hits@10": tops[1],
                    "mrr": float(m.group(3)),
                }
        if "Ep" in line and "r2l: acc of top [1, 10, 50]" in line:
            m = R2L_RE.search(line)
            if m:
                tops = parse_top_vals(m.group(1))
                r2l = {
                    "hits@1": tops[0],
                    "hits@10": tops[1],
                    "mrr": float(m.group(3)),
                }
    return l2r, r2l


def mean_std(values):
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, math.sqrt(v)


def round4(x):
    return round(float(x), 4)


def main():
    parser = argparse.ArgumentParser(description="Summarize TMMEA-DA v1 sweep runs.")
    parser.add_argument("--runs-dir", default="runs/tmmeada_v1_sweep")
    parser.add_argument("--out-summary", default="reports/tmmeada_v1_sweep_summary.csv")
    parser.add_argument("--out-grouped", default="reports/tmmeada_v1_sweep_grouped.csv")
    parser.add_argument("--out-md", default="reports/tmmeada_v1_sweep.md")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.yaml"
        log_path = run_dir / "log.txt"
        if not cfg_path.exists() or not log_path.exists():
            continue

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8", errors="replace"))
        meta = cfg.get("meta", {})
        if str(meta.get("stage", "")) != "tmmeada_v1_sweep":
            continue
        mea = cfg["meaformer"]

        l2r, r2l = extract_metrics(log_path.read_text(encoding="utf-8", errors="replace"))
        if not l2r or not r2l:
            continue

        rows.append(
            {
                "run_id": run_dir.name,
                "data_choice": str(mea.get("data_choice", "")),
                "data_split": str(mea.get("data_split", "")),
                "seed": int(mea.get("random_seed", -1)),
                "domain_align_weight": float(mea.get("domain_align_weight", 0.0)),
                "source_select_weight": float(mea.get("source_select_weight", 0.0)),
                "missing_align_weight": float(mea.get("missing_align_weight", 0.0)),
                "source_select_temp": float(mea.get("source_select_temp", 1.0)),
                "l2r_hits@1": float(l2r["hits@1"]),
                "l2r_hits@10": float(l2r["hits@10"]),
                "l2r_mrr": float(l2r["mrr"]),
                "r2l_hits@1": float(r2l["hits@1"]),
                "r2l_hits@10": float(r2l["hits@10"]),
                "r2l_mrr": float(r2l["mrr"]),
            }
        )

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "run_id",
        "data_choice",
        "data_split",
        "seed",
        "domain_align_weight",
        "source_select_weight",
        "missing_align_weight",
        "source_select_temp",
        "l2r_hits@1",
        "l2r_hits@10",
        "l2r_mrr",
        "r2l_hits@1",
        "r2l_hits@10",
        "r2l_mrr",
    ]
    with out_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        key = (
            row["data_choice"],
            row["data_split"],
            row["domain_align_weight"],
            row["source_select_weight"],
            row["missing_align_weight"],
            row["source_select_temp"],
        )
        groups[key].append(row)

    grouped_rows = []
    for key, members in groups.items():
        metric_names = [
            "l2r_hits@1",
            "l2r_hits@10",
            "l2r_mrr",
            "r2l_hits@1",
            "r2l_hits@10",
            "r2l_mrr",
        ]
        agg = {
            "data_choice": key[0],
            "data_split": key[1],
            "domain_align_weight": key[2],
            "source_select_weight": key[3],
            "missing_align_weight": key[4],
            "source_select_temp": key[5],
            "num_runs": len(members),
        }
        for metric in metric_names:
            vals = [float(x[metric]) for x in members]
            m, s = mean_std(vals)
            agg[f"{metric}_mean"] = round4(m)
            agg[f"{metric}_std"] = round4(s)
        agg["avg_hits@1_mean"] = round4((agg["l2r_hits@1_mean"] + agg["r2l_hits@1_mean"]) / 2.0)
        grouped_rows.append(agg)

    grouped_rows.sort(
        key=lambda x: (x["avg_hits@1_mean"], x["l2r_mrr_mean"], x["r2l_mrr_mean"]),
        reverse=True,
    )

    grouped_fields = [
        "data_choice",
        "data_split",
        "domain_align_weight",
        "source_select_weight",
        "missing_align_weight",
        "source_select_temp",
        "num_runs",
        "avg_hits@1_mean",
        "l2r_hits@1_mean",
        "l2r_hits@1_std",
        "l2r_hits@10_mean",
        "l2r_hits@10_std",
        "l2r_mrr_mean",
        "l2r_mrr_std",
        "r2l_hits@1_mean",
        "r2l_hits@1_std",
        "r2l_hits@10_mean",
        "r2l_hits@10_std",
        "r2l_mrr_mean",
        "r2l_mrr_std",
    ]
    out_grouped = Path(args.out_grouped)
    with out_grouped.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=grouped_fields)
        writer.writeheader()
        writer.writerows(grouped_rows)

    out_md = Path(args.out_md)
    lines = [
        "# TMMEA-DA v1 weight sweep (zh_en)",
        "",
        f"- runs_dir: `{runs_dir}`",
        f"- total_runs: `{len(rows)}`",
        f"- total_groups: `{len(grouped_rows)}`",
        "",
        "| rank | dw | sw | mw | temp | n | avg H@1 | l2r H@1 | r2l H@1 | l2r MRR | r2l MRR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(grouped_rows, start=1):
        lines.append(
            f"| {i} | {row['domain_align_weight']:.4f} | {row['source_select_weight']:.4f} | "
            f"{row['missing_align_weight']:.4f} | {row['source_select_temp']:.4f} | {row['num_runs']} | "
            f"{row['avg_hits@1_mean']:.4f} | {row['l2r_hits@1_mean']:.4f} | {row['r2l_hits@1_mean']:.4f} | "
            f"{row['l2r_mrr_mean']:.4f} | {row['r2l_mrr_mean']:.4f} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- Ranking key: avg Hits@1 mean (tie-breaker: l2r/r2l MRR mean).")
    lines.append("- Use this table to pick top-k settings for next multi-seed verification.")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote summary: {out_summary} ({len(rows)} runs)")
    print(f"Wrote grouped: {out_grouped} ({len(grouped_rows)} groups)")
    print(f"Wrote markdown: {out_md}")


if __name__ == "__main__":
    main()
