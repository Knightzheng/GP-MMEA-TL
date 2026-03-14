import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

L2R_RE = re.compile(r"l2r: acc of top \[1, 10, 50\] = \[([0-9.\s]+)\], mr = ([0-9.]+), mrr = ([0-9.]+)")
R2L_RE = re.compile(r"r2l: acc of top \[1, 10, 50\] = \[([0-9.\s]+)\], mr = ([0-9.]+), mrr = ([0-9.]+)")
GPU_RE = re.compile(r"\[gpu_peak\]\s+allocated_mb=([0-9.]+)\s+reserved_mb=([0-9.]+)")


def parse_top_vals(s):
    vals = [float(item) for item in s.strip().split() if item][:3]
    while len(vals) < 3:
        vals.append(0.0)
    return vals


def extract_metrics(log_text):
    l2r = None
    r2l = None
    gpu_alloc = None
    gpu_reserved = None
    for line in log_text.splitlines():
        l2r_match = L2R_RE.search(line)
        if l2r_match:
            tops = parse_top_vals(l2r_match.group(1))
            l2r = {
                "hits@1": tops[0],
                "hits@10": tops[1],
                "mrr": float(l2r_match.group(3)),
            }
        r2l_match = R2L_RE.search(line)
        if r2l_match:
            tops = parse_top_vals(r2l_match.group(1))
            r2l = {
                "hits@1": tops[0],
                "hits@10": tops[1],
                "mrr": float(r2l_match.group(3)),
            }
        gpu_match = GPU_RE.search(line)
        if gpu_match:
            gpu_alloc = float(gpu_match.group(1))
            gpu_reserved = float(gpu_match.group(2))
    return l2r, r2l, gpu_alloc, gpu_reserved


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def iter_run_dirs(root: Path):
    for variant_dir in sorted(root.iterdir()):
        if not variant_dir.is_dir():
            continue
        for run_dir in sorted(variant_dir.iterdir()):
            if run_dir.is_dir():
                yield variant_dir.name, run_dir


def main():
    parser = argparse.ArgumentParser(description="Summarize the minimal H3 missing-modality pressure-test runs.")
    parser.add_argument("--runs-root", default="runs/experiments/h3_missing_modality_minimal")
    parser.add_argument("--out-dir", default="reports/robustness")
    args = parser.parse_args()

    runs_root = ROOT / args.runs_root
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows = []
    grouped = defaultdict(list)

    if runs_root.exists():
        for variant, run_dir in iter_run_dirs(runs_root):
            log_path = run_dir / "log.txt"
            cfg_path = run_dir / "config.yaml"
            if not log_path.exists() or not cfg_path.exists():
                continue

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            drop_rate = float(cfg["meaformer"].get("img_mask_drop_rate", 0.0))
            seed = int(cfg["meaformer"].get("random_seed", 0))
            text = log_path.read_text(encoding="utf-8", errors="replace")
            l2r, r2l, gpu_alloc, gpu_reserved = extract_metrics(text)
            if not l2r or not r2l:
                continue

            avg_hits1 = (l2r["hits@1"] + r2l["hits@1"]) / 2.0
            avg_hits10 = (l2r["hits@10"] + r2l["hits@10"]) / 2.0
            avg_mrr = (l2r["mrr"] + r2l["mrr"]) / 2.0
            row = {
                "variant": variant,
                "drop_rate": f"{drop_rate:.2f}",
                "seed": str(seed),
                "run_id": run_dir.name,
                "avg_hits@1": f"{avg_hits1:.4f}",
                "avg_hits@10": f"{avg_hits10:.4f}",
                "avg_mrr": f"{avg_mrr:.4f}",
                "gpu_peak_allocated_mb": f"{gpu_alloc:.2f}" if gpu_alloc is not None else "",
                "gpu_peak_reserved_mb": f"{gpu_reserved:.2f}" if gpu_reserved is not None else "",
            }
            per_run_rows.append(row)
            grouped[(variant, drop_rate)].append(
                {
                    "avg_hits@1": avg_hits1,
                    "avg_hits@10": avg_hits10,
                    "avg_mrr": avg_mrr,
                    "gpu_peak_allocated_mb": gpu_alloc,
                    "gpu_peak_reserved_mb": gpu_reserved,
                }
            )

    per_run_path = out_dir / "h3_missing_modality_minimal_per_run.csv"
    with per_run_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "drop_rate",
                "seed",
                "run_id",
                "avg_hits@1",
                "avg_hits@10",
                "avg_mrr",
                "gpu_peak_allocated_mb",
                "gpu_peak_reserved_mb",
            ],
        )
        writer.writeheader()
        writer.writerows(per_run_rows)

    summary_rows = []
    for (variant, drop_rate), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "variant": variant,
                "drop_rate": f"{drop_rate:.2f}",
                "num_runs": str(len(rows)),
                "avg_hits@1_mean": f"{mean([row['avg_hits@1'] for row in rows]):.4f}",
                "avg_hits@1_std": f"{std([row['avg_hits@1'] for row in rows]):.4f}",
                "avg_hits@10_mean": f"{mean([row['avg_hits@10'] for row in rows]):.4f}",
                "avg_hits@10_std": f"{std([row['avg_hits@10'] for row in rows]):.4f}",
                "avg_mrr_mean": f"{mean([row['avg_mrr'] for row in rows]):.4f}",
                "avg_mrr_std": f"{std([row['avg_mrr'] for row in rows]):.4f}",
                "gpu_peak_allocated_mb_mean": (
                    f"{mean([row['gpu_peak_allocated_mb'] for row in rows if row['gpu_peak_allocated_mb'] is not None]):.2f}"
                    if any(row["gpu_peak_allocated_mb"] is not None for row in rows)
                    else ""
                ),
                "gpu_peak_reserved_mb_mean": (
                    f"{mean([row['gpu_peak_reserved_mb'] for row in rows if row['gpu_peak_reserved_mb'] is not None]):.2f}"
                    if any(row["gpu_peak_reserved_mb"] is not None for row in rows)
                    else ""
                ),
            }
        )

    summary_path = out_dir / "h3_missing_modality_minimal_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "drop_rate",
                "num_runs",
                "avg_hits@1_mean",
                "avg_hits@1_std",
                "avg_hits@10_mean",
                "avg_hits@10_std",
                "avg_mrr_mean",
                "avg_mrr_std",
                "gpu_peak_allocated_mb_mean",
                "gpu_peak_reserved_mb_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    md_lines = [
        "# H3 Missing-Modality Minimal Summary",
        "",
        f"- runs_root: `{runs_root}`",
        "",
        "## Aggregated Table",
        "",
        "| Variant | Drop Rate | Runs | avg Hits@1 | avg Hits@10 | avg MRR | GPU Peak Alloc (MB) | GPU Peak Reserv (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not summary_rows:
        md_lines.append("| - | - | 0 | - | - | - | - | - |")
    for row in summary_rows:
        md_lines.append(
            f"| {row['variant']} | {row['drop_rate']} | {row['num_runs']} | "
            f"{row['avg_hits@1_mean']} +- {row['avg_hits@1_std']} | "
            f"{row['avg_hits@10_mean']} +- {row['avg_hits@10_std']} | "
            f"{row['avg_mrr_mean']} +- {row['avg_mrr_std']} | "
            f"{row['gpu_peak_allocated_mb_mean'] or '-'} | {row['gpu_peak_reserved_mb_mean'] or '-'} |"
        )
    md_lines.extend(
        [
            "",
            "## Suggested Thesis Usage",
            "",
            "- Compare degradation trends across `baseline`, `v1_full`, and `wo_missing_gate` under the same drop rates.",
            "- If `v1_full` degrades more slowly than `wo_missing_gate`, this can be written as evidence that missing-aware design helps under simulated modality loss.",
            "- If gains are small, use a restrained wording such as `provides partial support for H3 under the current budget`.",
            "",
        ]
    )
    (out_dir / "h3_missing_modality_minimal_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] wrote {per_run_path} and {summary_path}")


if __name__ == "__main__":
    main()
