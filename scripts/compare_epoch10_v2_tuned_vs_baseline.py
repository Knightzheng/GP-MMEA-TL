import argparse
import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


METRICS = [
    "l2r_hits@1",
    "l2r_hits@10",
    "l2r_mrr",
    "r2l_hits@1",
    "r2l_hits@10",
    "r2l_mrr",
]
SEED_RE = re.compile(r"-s(\d+)$")


def run_cmd(cmd: List[str]):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_seeds(seed_str: str) -> List[int]:
    return [int(x.strip()) for x in seed_str.split(",") if x.strip()]


def extract_seed(run_id: str):
    m = SEED_RE.search(run_id)
    return int(m.group(1)) if m else None


def summarize_runs(runner_python: str, runs_dir: Path, out_prefix: Path) -> Tuple[Path, Path]:
    summary_csv = out_prefix.with_suffix(".summary.csv")
    mean_csv = out_prefix.with_suffix(".mean_std.csv")
    run_cmd(
        [
            runner_python,
            "scripts/collect_meaformer_results.py",
            "--runs-dir",
            str(runs_dir),
            "--out",
            str(summary_csv),
        ]
    )
    run_cmd(
        [
            runner_python,
            "scripts/aggregate_meaformer_results.py",
            "--in-csv",
            str(summary_csv),
            "--out-csv",
            str(mean_csv),
        ]
    )
    return summary_csv, mean_csv


def read_mean_row(mean_csv: Path, lang_tag: str) -> Dict[str, float]:
    with mean_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["lang_pair"] == lang_tag:
                return {k: float(v) if k not in ("lang_pair",) else v for k, v in row.items()}
    raise ValueError(f"lang_tag={lang_tag} not found in {mean_csv}")


def read_seed_avg_mrr(summary_csv: Path, required_seeds: List[int]) -> Dict[int, float]:
    selected: Dict[int, Tuple[str, float]] = {}
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row["run_id"]
            seed = extract_seed(run_id)
            if seed is None or seed not in required_seeds:
                continue
            avg_mrr = (float(row["l2r_mrr"]) + float(row["r2l_mrr"])) / 2.0
            prev = selected.get(seed)
            if prev is None or run_id > prev[0]:
                selected[seed] = (run_id, avg_mrr)
    return {k: v[1] for k, v in selected.items()}


def write_compare(compare_csv: Path, compare_md: Path, rows: List[Dict[str, object]], note: str):
    fields = [
        "dataset",
        "metric",
        "baseline_mean",
        "method_mean",
        "delta_method_minus_baseline",
        "baseline_num_runs",
        "method_num_runs",
    ]
    compare_csv.parent.mkdir(parents=True, exist_ok=True)
    with compare_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Epoch10 v2 tuned Pilot Compare",
        "",
        note,
        "",
        "| dataset | metric | baseline_mean | method_mean | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['dataset']} | {row['metric']} | {float(row['baseline_mean']):.4f} | "
            f"{float(row['method_mean']):.4f} | {float(row['delta_method_minus_baseline']):+.4f} |"
        )
    compare_md.write_text("\n".join(md) + "\n", encoding="utf-8")


def write_decision(md_path: Path, json_path: Path, payload: Dict):
    md = [
        "# Epoch10 v2 tuned Decision",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- threshold: `delta_avg_mrr >= {payload['threshold']}` + all required seeds positive",
        f"- decision: `{payload['decision']}`",
        "",
        "| dataset | delta_avg_mrr | seed_deltas | consistent_positive | pass_threshold |",
        "|---|---:|---|---:|---:|",
    ]
    for d in payload["datasets"]:
        seed_txt = ", ".join([f"s{item['seed']}:{item['delta']:.4f}" for item in d["seed_deltas"]]) or "N/A"
        md.append(
            f"| {d['dataset']} | {d['delta_avg_mrr']:.4f} | {seed_txt} | "
            f"{str(d['consistent_positive'])} | {str(d['pass_threshold'])} |"
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare epoch10 baseline pilot vs TMMEA-DA v2 tuned pilot.")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--required-seeds", default="42,3407")
    parser.add_argument("--threshold", type=float, default=0.003)
    parser.add_argument("--baseline-zh-dir", default="runs/baseline_pilot_epoch10")
    parser.add_argument("--baseline-fbdb-dir", default="runs/baseline_pilot_epoch10_crossgraph")
    parser.add_argument("--method-zh-dir", default="runs/tmmeada_v2_tuned_pilot_epoch10")
    parser.add_argument("--method-fbdb-dir", default="runs/tmmeada_v2_tuned_pilot_epoch10_crossgraph")
    parser.add_argument("--compare-csv", default="reports/epoch10_compare_v2_tuned_pilot.csv")
    parser.add_argument("--compare-md", default="reports/epoch10_compare_v2_tuned_pilot.md")
    parser.add_argument("--decision-json", default="reports/epoch10_v2_tuned_decision.json")
    parser.add_argument("--decision-md", default="reports/epoch10_v2_tuned_decision.md")
    args = parser.parse_args()

    required_seeds = parse_seeds(args.required_seeds)
    report_tmp = Path("reports/auto_decision_tmp/v2_tuned")
    report_tmp.mkdir(parents=True, exist_ok=True)

    stage_specs = [
        ("zh_en", Path(args.baseline_zh_dir), Path(args.method_zh_dir), "zh_en"),
        ("FBDB15K", Path(args.baseline_fbdb_dir), Path(args.method_fbdb_dir), "FBDB15K"),
    ]

    rows: List[Dict[str, object]] = []
    dataset_reports = []
    all_pass = True

    for name, b_dir, m_dir, tag in stage_specs:
        b_summary, b_mean = summarize_runs(args.runner_python, b_dir, report_tmp / f"{name}_baseline")
        m_summary, m_mean = summarize_runs(args.runner_python, m_dir, report_tmp / f"{name}_method")

        b_row = read_mean_row(b_mean, tag)
        m_row = read_mean_row(m_mean, tag)
        for metric in METRICS:
            key = f"{metric}_mean"
            rows.append(
                {
                    "dataset": name,
                    "metric": metric,
                    "baseline_mean": round(float(b_row[key]), 6),
                    "method_mean": round(float(m_row[key]), 6),
                    "delta_method_minus_baseline": round(float(m_row[key]) - float(b_row[key]), 6),
                    "baseline_num_runs": int(b_row["num_runs"]),
                    "method_num_runs": int(m_row["num_runs"]),
                }
            )

        b_avg_mrr = (float(b_row["l2r_mrr_mean"]) + float(b_row["r2l_mrr_mean"])) / 2.0
        m_avg_mrr = (float(m_row["l2r_mrr_mean"]) + float(m_row["r2l_mrr_mean"])) / 2.0
        delta_avg_mrr = m_avg_mrr - b_avg_mrr

        b_seed_mrr = read_seed_avg_mrr(b_summary, required_seeds)
        m_seed_mrr = read_seed_avg_mrr(m_summary, required_seeds)
        seed_deltas = []
        for seed in required_seeds:
            if seed in b_seed_mrr and seed in m_seed_mrr:
                seed_deltas.append({"seed": seed, "delta": m_seed_mrr[seed] - b_seed_mrr[seed]})

        consistent_positive = len(seed_deltas) == len(required_seeds) and all(d["delta"] > 0 for d in seed_deltas)
        pass_threshold = delta_avg_mrr >= args.threshold and consistent_positive
        all_pass = all_pass and pass_threshold
        dataset_reports.append(
            {
                "dataset": name,
                "delta_avg_mrr": round(delta_avg_mrr, 6),
                "seed_deltas": seed_deltas,
                "consistent_positive": consistent_positive,
                "pass_threshold": pass_threshold,
            }
        )

    write_compare(
        compare_csv=Path(args.compare_csv),
        compare_md=Path(args.compare_md),
        rows=rows,
        note=f"required_seeds={required_seeds}; threshold={args.threshold}",
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "required_seeds": required_seeds,
        "threshold": args.threshold,
        "decision": "promote_to_formal_5seed_epoch10" if all_pass else "continue_tuning_or_error_analysis",
        "datasets": dataset_reports,
    }
    write_decision(Path(args.decision_md), Path(args.decision_json), payload)
    print(f"[DONE] compare -> {args.compare_csv}")
    print(f"[DONE] decision -> {args.decision_json}")


if __name__ == "__main__":
    main()
