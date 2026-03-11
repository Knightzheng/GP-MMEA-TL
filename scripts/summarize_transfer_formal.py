import argparse
import csv
import re
from pathlib import Path
from statistics import mean, pstdev

import yaml


L2R_RE = re.compile(
    r"Ep (?:Test|[0-9]+) \| l2r: acc of top \[1, 10, 50\] = \[(?P<h1>[0-9.]+)\s+(?P<h10>[0-9.]+)\s+(?P<h50>[0-9.]+)\s*\], mr = (?P<mr>[0-9.]+), mrr = (?P<mrr>[0-9.]+)"
)
R2L_RE = re.compile(
    r"Ep (?:Test|[0-9]+) \| r2l: acc of top \[1, 10, 50\] = \[(?P<h1>[0-9.]+)\s+(?P<h10>[0-9.]+)\s+(?P<h50>[0-9.]+)\s*\], mr = (?P<mr>[0-9.]+), mrr = (?P<mrr>[0-9.]+)"
)
RETURN_CODE_OK_MARKER = "[DONE] return_code=0"


def parse_eval_metrics(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if RETURN_CODE_OK_MARKER not in text:
        return None
    l2r_match = None
    r2l_match = None
    for line in text.splitlines():
        m1 = L2R_RE.search(line)
        if m1:
            l2r_match = m1
        m2 = R2L_RE.search(line)
        if m2:
            r2l_match = m2
    if l2r_match is None or r2l_match is None:
        return None
    l2r = {k: float(v) for k, v in l2r_match.groupdict().items()}
    r2l = {k: float(v) for k, v in r2l_match.groupdict().items()}
    return l2r, r2l


def infer_target(run_dir: Path):
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return "unknown", "unknown", "unknown"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    m = cfg.get("meaformer", {})
    data_choice = str(m.get("data_choice", "unknown"))
    data_split = str(m.get("data_split", "unknown"))
    if data_choice == "DBP15K":
        target = data_split
    else:
        target = data_choice
    return data_choice, data_split, target


def collect(target_eval_dir: Path):
    rows = []
    for run_dir in sorted(target_eval_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        log_path = run_dir / "log.txt"
        if not log_path.exists():
            continue
        parsed = parse_eval_metrics(log_path)
        if parsed is None:
            continue
        l2r, r2l = parsed
        data_choice, data_split, target = infer_target(run_dir)
        row = {
            "run_id": run_dir.name,
            "data_choice": data_choice,
            "data_split": data_split,
            "target": target,
            "l2r_hits@1": l2r["h1"],
            "l2r_hits@10": l2r["h10"],
            "l2r_mrr": l2r["mrr"],
            "l2r_mr": l2r["mr"],
            "r2l_hits@1": r2l["h1"],
            "r2l_hits@10": r2l["h10"],
            "r2l_mrr": r2l["mrr"],
            "r2l_mr": r2l["mr"],
        }
        row["avg_hits@1"] = (row["l2r_hits@1"] + row["r2l_hits@1"]) / 2.0
        row["avg_hits@10"] = (row["l2r_hits@10"] + row["r2l_hits@10"]) / 2.0
        row["avg_mrr"] = (row["l2r_mrr"] + row["r2l_mrr"]) / 2.0
        row["avg_mr"] = (row["l2r_mr"] + row["r2l_mr"]) / 2.0
        rows.append(row)
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "data_choice",
        "data_split",
        "target",
        "l2r_hits@1",
        "l2r_hits@10",
        "l2r_mrr",
        "l2r_mr",
        "r2l_hits@1",
        "r2l_hits@10",
        "r2l_mrr",
        "r2l_mr",
        "avg_hits@1",
        "avg_hits@10",
        "avg_mrr",
        "avg_mr",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_by_target(rows):
    grouped = {}
    metrics = ["avg_hits@1", "avg_hits@10", "avg_mrr", "avg_mr"]
    for row in rows:
        grouped.setdefault(row["target"], []).append(row)

    out = []
    for target, items in grouped.items():
        record = {"target": target, "num_runs": len(items)}
        for m in metrics:
            vals = [float(x[m]) for x in items]
            record[f"{m}_mean"] = mean(vals)
            record[f"{m}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(record)
    out.sort(key=lambda x: x["target"])
    return out


def build_compare_rows(baseline_rows, method_rows):
    baseline = {x["target"]: x for x in aggregate_by_target(baseline_rows)}
    method = {x["target"]: x for x in aggregate_by_target(method_rows)}
    targets = sorted(set(baseline.keys()) | set(method.keys()))
    out = []
    for t in targets:
        b = baseline.get(t)
        m = method.get(t)
        if b is None or m is None:
            continue
        out.append(
            {
                "target": t,
                "baseline_num_runs": b["num_runs"],
                "tmmeada_num_runs": m["num_runs"],
                "baseline_avg_hits@1_mean": b["avg_hits@1_mean"],
                "baseline_avg_hits@1_std": b["avg_hits@1_std"],
                "tmmeada_avg_hits@1_mean": m["avg_hits@1_mean"],
                "tmmeada_avg_hits@1_std": m["avg_hits@1_std"],
                "delta_avg_hits@1_mean": m["avg_hits@1_mean"] - b["avg_hits@1_mean"],
                "baseline_avg_hits@10_mean": b["avg_hits@10_mean"],
                "baseline_avg_hits@10_std": b["avg_hits@10_std"],
                "tmmeada_avg_hits@10_mean": m["avg_hits@10_mean"],
                "tmmeada_avg_hits@10_std": m["avg_hits@10_std"],
                "delta_avg_hits@10_mean": m["avg_hits@10_mean"] - b["avg_hits@10_mean"],
                "baseline_avg_mrr_mean": b["avg_mrr_mean"],
                "baseline_avg_mrr_std": b["avg_mrr_std"],
                "tmmeada_avg_mrr_mean": m["avg_mrr_mean"],
                "tmmeada_avg_mrr_std": m["avg_mrr_std"],
                "delta_avg_mrr_mean": m["avg_mrr_mean"] - b["avg_mrr_mean"],
                "baseline_avg_mr_mean": b["avg_mr_mean"],
                "baseline_avg_mr_std": b["avg_mr_std"],
                "tmmeada_avg_mr_mean": m["avg_mr_mean"],
                "tmmeada_avg_mr_std": m["avg_mr_std"],
                "delta_avg_mr_mean": m["avg_mr_mean"] - b["avg_mr_mean"],
            }
        )
    return out


def write_compare_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "target",
        "baseline_num_runs",
        "tmmeada_num_runs",
        "baseline_avg_hits@1_mean",
        "baseline_avg_hits@1_std",
        "tmmeada_avg_hits@1_mean",
        "tmmeada_avg_hits@1_std",
        "delta_avg_hits@1_mean",
        "baseline_avg_hits@10_mean",
        "baseline_avg_hits@10_std",
        "tmmeada_avg_hits@10_mean",
        "tmmeada_avg_hits@10_std",
        "delta_avg_hits@10_mean",
        "baseline_avg_mrr_mean",
        "baseline_avg_mrr_std",
        "tmmeada_avg_mrr_mean",
        "tmmeada_avg_mrr_std",
        "delta_avg_mrr_mean",
        "baseline_avg_mr_mean",
        "baseline_avg_mr_std",
        "tmmeada_avg_mr_mean",
        "tmmeada_avg_mr_std",
        "delta_avg_mr_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_compare_md(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Formal Transfer Compare: TMMEA-DA vs Baseline")
    lines.append("")
    lines.append("| target | n_baseline | n_tmmeada | baseline_avg_mrr(mean±std) | tmmeada_avg_mrr(mean±std) | delta_avg_mrr_mean | baseline_avg_hits@1(mean±std) | tmmeada_avg_hits@1(mean±std) | delta_avg_hits@1_mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['target']} | "
            f"{int(r['baseline_num_runs'])} | {int(r['tmmeada_num_runs'])} | "
            f"{r['baseline_avg_mrr_mean']:.4f}±{r['baseline_avg_mrr_std']:.4f} | "
            f"{r['tmmeada_avg_mrr_mean']:.4f}±{r['tmmeada_avg_mrr_std']:.4f} | "
            f"{r['delta_avg_mrr_mean']:+.4f} | "
            f"{r['baseline_avg_hits@1_mean']:.4f}±{r['baseline_avg_hits@1_std']:.4f} | "
            f"{r['tmmeada_avg_hits@1_mean']:.4f}±{r['tmmeada_avg_hits@1_std']:.4f} | "
            f"{r['delta_avg_hits@1_mean']:+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize and compare formal transfer target eval logs.")
    parser.add_argument(
        "--baseline-target-dir",
        default="runs/transfer/transfer_formal/target_eval",
    )
    parser.add_argument(
        "--tmmeada-target-dir",
        default="runs/transfer/transfer_formal_tmmeada/target_eval",
    )
    parser.add_argument(
        "--baseline-out",
        default="reports/transfer/transfer_formal_target_eval_baseline_summary.csv",
    )
    parser.add_argument(
        "--tmmeada-out",
        default="reports/transfer/transfer_formal_target_eval_tmmeada_summary.csv",
    )
    parser.add_argument(
        "--compare-out-csv",
        default="reports/transfer/transfer_formal_compare_tmmeada_vs_baseline.csv",
    )
    parser.add_argument(
        "--compare-out-md",
        default="reports/transfer/transfer_formal_compare_tmmeada_vs_baseline.md",
    )
    args = parser.parse_args()

    baseline_rows = collect(Path(args.baseline_target_dir))
    tmmeada_rows = collect(Path(args.tmmeada_target_dir))
    write_csv(Path(args.baseline_out), baseline_rows)
    write_csv(Path(args.tmmeada_out), tmmeada_rows)
    compare_rows = build_compare_rows(baseline_rows, tmmeada_rows)
    write_compare_csv(Path(args.compare_out_csv), compare_rows)
    write_compare_md(Path(args.compare_out_md), compare_rows)

    print(f"[DONE] baseline summary -> {args.baseline_out} ({len(baseline_rows)} rows)")
    print(f"[DONE] tmmeada summary -> {args.tmmeada_out} ({len(tmmeada_rows)} rows)")
    print(f"[DONE] compare csv -> {args.compare_out_csv} ({len(compare_rows)} rows)")
    print(f"[DONE] compare md -> {args.compare_out_md} ({len(compare_rows)} rows)")


if __name__ == "__main__":
    main()
