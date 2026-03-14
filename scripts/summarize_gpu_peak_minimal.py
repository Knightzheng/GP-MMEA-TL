import argparse
import csv
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

L2R_RE = re.compile(
    r"l2r: acc of top \[1, 10, 50\] = \[([0-9.\s]+)\], mr = ([0-9.]+), mrr = ([0-9.]+)"
)
R2L_RE = re.compile(
    r"r2l: acc of top \[1, 10, 50\] = \[([0-9.\s]+)\], mr = ([0-9.]+), mrr = ([0-9.]+)"
)
GPU_RE = re.compile(r"\[gpu_peak\]\s+allocated_mb=([0-9.]+)\s+reserved_mb=([0-9.]+)")
ELAPSED_RE = re.compile(r"INFO - .* - ([0-9]+:[0-9]+:[0-9]+) - done!")


def parse_top_vals(s):
    vals = [float(item) for item in s.strip().split() if item][:3]
    while len(vals) < 3:
        vals.append(0.0)
    return vals


def hhmmss_to_minutes(value: str) -> float:
    hours, minutes, seconds = [int(item) for item in value.split(":")]
    return hours * 60.0 + minutes + seconds / 60.0


def extract_metrics(log_text):
    l2r = None
    r2l = None
    gpu_alloc = None
    gpu_reserved = None
    elapsed_minutes = None
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
        elapsed_match = ELAPSED_RE.search(line)
        if elapsed_match:
            elapsed_minutes = hhmmss_to_minutes(elapsed_match.group(1))
    return l2r, r2l, gpu_alloc, gpu_reserved, elapsed_minutes


def iter_run_dirs(root: Path):
    for variant_dir in sorted(root.iterdir()):
        if not variant_dir.is_dir():
            continue
        for run_dir in sorted(variant_dir.iterdir()):
            if run_dir.is_dir():
                yield variant_dir.name, run_dir


def target_label(cfg):
    data_choice = str(cfg["meaformer"].get("data_choice", "")).upper()
    data_split = str(cfg["meaformer"].get("data_split", ""))
    if data_choice == "DBP15K":
        return data_split
    return data_choice


def variant_label(variant: str):
    return "method" if variant.endswith("_method") else "baseline"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize minimal peak-GPU-memory measurements for representative transfer-adapt targets."
    )
    parser.add_argument("--runs-root", default="runs/experiments/gpu_peak_minimal")
    parser.add_argument("--out-dir", default="reports/transfer")
    args = parser.parse_args()

    runs_root = ROOT / args.runs_root
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if runs_root.exists():
        for variant, run_dir in iter_run_dirs(runs_root):
            log_path = run_dir / "log.txt"
            cfg_path = run_dir / "config.yaml"
            if not log_path.exists() or not cfg_path.exists():
                continue

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            l2r, r2l, gpu_alloc, gpu_reserved, elapsed_minutes = extract_metrics(log_text)
            if not l2r or not r2l:
                continue

            avg_hits1 = (l2r["hits@1"] + r2l["hits@1"]) / 2.0
            avg_hits10 = (l2r["hits@10"] + r2l["hits@10"]) / 2.0
            avg_mrr = (l2r["mrr"] + r2l["mrr"]) / 2.0
            rows.append(
                {
                    "target": target_label(cfg),
                    "variant": variant_label(variant),
                    "config_variant": variant,
                    "seed": str(cfg["meaformer"].get("random_seed", "")),
                    "epoch": str(cfg["meaformer"].get("epoch", "")),
                    "run_id": run_dir.name,
                    "avg_hits@1": f"{avg_hits1:.4f}",
                    "avg_hits@10": f"{avg_hits10:.4f}",
                    "avg_mrr": f"{avg_mrr:.4f}",
                    "gpu_peak_allocated_mb": f"{gpu_alloc:.2f}" if gpu_alloc is not None else "",
                    "gpu_peak_reserved_mb": f"{gpu_reserved:.2f}" if gpu_reserved is not None else "",
                    "elapsed_minutes": f"{elapsed_minutes:.2f}" if elapsed_minutes is not None else "",
                }
            )

    rows.sort(key=lambda row: (row["target"], row["variant"]))

    per_run_path = out_dir / "transfer_gpu_peak_minimal_per_run.csv"
    with per_run_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "variant",
                "config_variant",
                "seed",
                "epoch",
                "run_id",
                "avg_hits@1",
                "avg_hits@10",
                "avg_mrr",
                "gpu_peak_allocated_mb",
                "gpu_peak_reserved_mb",
                "elapsed_minutes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "transfer_gpu_peak_minimal_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "variant",
                "seed",
                "epoch",
                "avg_hits@1",
                "avg_hits@10",
                "avg_mrr",
                "gpu_peak_allocated_mb",
                "gpu_peak_reserved_mb",
                "elapsed_minutes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "target": row["target"],
                    "variant": row["variant"],
                    "seed": row["seed"],
                    "epoch": row["epoch"],
                    "avg_hits@1": row["avg_hits@1"],
                    "avg_hits@10": row["avg_hits@10"],
                    "avg_mrr": row["avg_mrr"],
                    "gpu_peak_allocated_mb": row["gpu_peak_allocated_mb"],
                    "gpu_peak_reserved_mb": row["gpu_peak_reserved_mb"],
                    "elapsed_minutes": row["elapsed_minutes"],
                }
            )

    md_lines = [
        "# Transfer GPU Peak Minimal Summary",
        "",
        "- scope: `seed=42`, representative targets `ja_en` and `FBYG15K`, 1-epoch target-adapt reruns with the same batch size and model structure as the formal configs",
        "- note: `elapsed_minutes` here reflects the 1-epoch补测 runtime, not the formal 5-seed full-chain wall-clock already reported in the thesis",
        "- note: GPU peak numbers come from `torch.cuda.max_memory_allocated / reserved`; under Windows `WDDM`, these allocator-level peaks may differ from `nvidia-smi` instantaneous physical usage, so they are better used for relative comparison within the same environment",
        "",
        "## Paper-Ready Table",
        "",
        "| Target | Variant | Seed | Epoch | avg Hits@1 | avg Hits@10 | avg MRR | GPU Peak Alloc (MB, PyTorch) | GPU Peak Reserv (MB, PyTorch) | 1-epoch Time (min) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not rows:
        md_lines.append("| - | - | - | - | - | - | - | - | - | - |")
    for row in rows:
        md_lines.append(
            f"| {row['target']} | {row['variant']} | {row['seed']} | {row['epoch']} | "
            f"{row['avg_hits@1']} | {row['avg_hits@10']} | {row['avg_mrr']} | "
            f"{row['gpu_peak_allocated_mb'] or '-'} | {row['gpu_peak_reserved_mb'] or '-'} | "
            f"{row['elapsed_minutes'] or '-'} |"
        )

    md_lines.extend(
        [
            "",
            "## Thesis Usage Boundary",
            "",
            "- These runs can support a restrained statement about relative peak memory under representative target-adapt settings in the current Windows/PyTorch environment.",
            "- They do not replace the formal 5-seed wall-clock statistics and should be cited as a supplementary memory补测.",
            "- Because the measurement reruns only `1` epoch, the time column is only for transparency; the peak-memory column is the main result.",
            "- If absolute values appear larger than the device's nominal physical memory, interpret them as allocator statistics rather than direct `nvidia-smi` occupancy.",
            "",
        ]
    )
    (out_dir / "transfer_gpu_peak_minimal_summary.md").write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )
    print(f"[OK] wrote {per_run_path} and {summary_path}")


if __name__ == "__main__":
    main()
