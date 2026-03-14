import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

L2R_RE = re.compile(
    r"l2r: acc of top \[1, 10, 50\] = \[([0-9.\s]+)\], mr = ([0-9.]+), mrr = ([0-9.]+)"
)
R2L_RE = re.compile(
    r"r2l: acc of top \[1, 10, 50\] = \[([0-9.\s]+)\], mr = ([0-9.]+), mrr = ([0-9.]+)"
)
GPU_RE = re.compile(r"\[gpu_peak\]\s+allocated_mb=([0-9.]+)\s+reserved_mb=([0-9.]+)")


REFERENCE_RUNS = {
    ("v1_full", "0.00"): ROOT
    / "runs/experiments/tmmeada/tmmeada_v1_best_epoch3/20260301-005700-TMMEA-DA-v1-best-epoch3-DBP15K-zh_en-s42/log.txt",
    ("wo_missing_gate", "0.00"): ROOT
    / "runs/experiments/tmmeada/tmmeada_v1_ablation_epoch3/20260302-101712-TMMEA-DA-v1-best-epoch3-wo-missing-DBP15K-zh_en-s42/log.txt",
}


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
    if not l2r or not r2l:
        return None
    return {
        "avg_hits@1": (l2r["hits@1"] + r2l["hits@1"]) / 2.0,
        "avg_hits@10": (l2r["hits@10"] + r2l["hits@10"]) / 2.0,
        "avg_mrr": (l2r["mrr"] + r2l["mrr"]) / 2.0,
        "gpu_peak_allocated_mb": gpu_alloc,
        "gpu_peak_reserved_mb": gpu_reserved,
    }


def latest_log(variant: str, drop_tag: str) -> Path | None:
    root = ROOT / "runs/experiments/h3_missing_modality_minimal" / variant
    if not root.exists():
        return None
    matches = sorted(root.glob(f"*{drop_tag}*/log.txt"))
    return matches[-1] if matches else None


def format_float(value, digits):
    return f"{value:.{digits}f}" if value is not None else ""


def main():
    parser = argparse.ArgumentParser(
        description="Build a paper-ready H3 missing-modality summary by combining reference no-drop logs and fresh high-drop reruns."
    )
    parser.add_argument("--out-dir", default="reports/robustness")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [
        {
            "variant": "v1_full",
            "drop_rate": "0.00",
            "seed": "42",
            "data_source": "existing_reference_same_config",
            "log_path": REFERENCE_RUNS[("v1_full", "0.00")],
        },
        {
            "variant": "wo_missing_gate",
            "drop_rate": "0.00",
            "seed": "42",
            "data_source": "existing_reference_same_config",
            "log_path": REFERENCE_RUNS[("wo_missing_gate", "0.00")],
        },
        {
            "variant": "v1_full",
            "drop_rate": "0.60",
            "seed": "42",
            "data_source": "fresh_h3_rerun",
            "log_path": latest_log("v1_full", "miss60"),
        },
        {
            "variant": "wo_missing_gate",
            "drop_rate": "0.60",
            "seed": "42",
            "data_source": "fresh_h3_rerun",
            "log_path": latest_log("wo_missing_gate", "miss60"),
        },
    ]

    rows = []
    for spec in run_specs:
        log_path = spec["log_path"]
        if log_path is None or not log_path.exists():
            continue
        metrics = extract_metrics(log_path.read_text(encoding="utf-8", errors="replace"))
        if metrics is None:
            continue
        rows.append(
            {
                "variant": spec["variant"],
                "drop_rate": spec["drop_rate"],
                "seed": spec["seed"],
                "data_source": spec["data_source"],
                "log_path": str(log_path.relative_to(ROOT)),
                "avg_hits@1": format_float(metrics["avg_hits@1"], 4),
                "avg_hits@10": format_float(metrics["avg_hits@10"], 4),
                "avg_mrr": format_float(metrics["avg_mrr"], 4),
                "gpu_peak_allocated_mb": format_float(metrics["gpu_peak_allocated_mb"], 2),
                "gpu_peak_reserved_mb": format_float(metrics["gpu_peak_reserved_mb"], 2),
            }
        )

    rows.sort(key=lambda row: (row["variant"], row["drop_rate"]))

    summary_path = out_dir / "h3_missing_modality_minimal_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "drop_rate",
                "seed",
                "data_source",
                "log_path",
                "avg_hits@1",
                "avg_hits@10",
                "avg_mrr",
                "gpu_peak_allocated_mb",
                "gpu_peak_reserved_mb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_path = out_dir / "h3_missing_modality_minimal_plot.csv"
    with plot_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "drop_rate", "avg_hits@1", "avg_hits@10", "avg_mrr"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row["variant"],
                    "drop_rate": row["drop_rate"],
                    "avg_hits@1": row["avg_hits@1"],
                    "avg_hits@10": row["avg_hits@10"],
                    "avg_mrr": row["avg_mrr"],
                }
            )

    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], {})[row["drop_rate"]] = row

    md_lines = [
        "# H3 Missing-Modality Minimal Summary",
        "",
        "- dataset: `zh_en`",
        "- matrix: `v1_full / wo_missing_gate × drop_rate {0.0, 0.6} × seed=42`",
        "- note: `drop_rate=0.0` rows reuse previously completed same-config `epoch3` logs; `drop_rate=0.6` rows are fresh reruns with missing-image injection and GPU peak logging",
        "- omitted in this minimal round: `baseline`, intermediate `drop_rate=0.3`, and multi-seed repetition",
        "- note: GPU peak numbers come from `torch.cuda.max_memory_allocated / reserved`; under Windows `WDDM`, use them mainly for relative comparison instead of direct physical-VRAM interpretation",
        "",
        "## Paper-Ready Table",
        "",
        "| Variant | Drop Rate | Source | avg Hits@1 | avg Hits@10 | avg MRR | GPU Peak Alloc (MB) | GPU Peak Reserv (MB) |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    if not rows:
        md_lines.append("| - | - | - | - | - | - | - | - |")
    for row in rows:
        md_lines.append(
            f"| {row['variant']} | {row['drop_rate']} | {row['data_source']} | "
            f"{row['avg_hits@1']} | {row['avg_hits@10']} | {row['avg_mrr']} | "
            f"{row['gpu_peak_allocated_mb'] or '-'} | {row['gpu_peak_reserved_mb'] or '-'} |"
        )

    md_lines.extend(
        [
            "",
            "## Degradation View",
            "",
            "| Variant | avg MRR @0.0 | avg MRR @0.6 | Delta MRR | avg Hits@1 @0.0 | avg Hits@1 @0.6 | Delta Hits@1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in ["v1_full", "wo_missing_gate"]:
        base = by_variant.get(variant, {}).get("0.00")
        stress = by_variant.get(variant, {}).get("0.60")
        if not base or not stress:
            md_lines.append(f"| {variant} | - | - | - | - | - | - |")
            continue
        base_mrr = float(base["avg_mrr"])
        stress_mrr = float(stress["avg_mrr"])
        base_h1 = float(base["avg_hits@1"])
        stress_h1 = float(stress["avg_hits@1"])
        md_lines.append(
            f"| {variant} | {base_mrr:.4f} | {stress_mrr:.4f} | {stress_mrr - base_mrr:+.4f} | "
            f"{base_h1:.4f} | {stress_h1:.4f} | {stress_h1 - base_h1:+.4f} |"
        )

    md_lines.extend(
        [
            "",
            "## Thesis Usage Boundary",
            "",
            "- This minimal round can support only a **single-seed pilot** observation under severe simulated image loss.",
            "- It can be used to describe whether `v1_full` still maintains or fails to maintain an advantage over `wo_missing_gate` at `drop_rate=0.6`.",
            "- It cannot support any strong claim about multi-seed stability, full degradation curves, or the independent effectiveness of `missing_gate` across targets.",
            "",
        ]
    )
    (out_dir / "h3_missing_modality_minimal_summary.md").write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )
    print(f"[OK] wrote {summary_path} and {plot_path}")


if __name__ == "__main__":
    main()
