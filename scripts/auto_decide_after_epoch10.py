import argparse
import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


SEED_RE = re.compile(r"-s(\d+)$")
METRICS = [
    "l2r_hits@1",
    "l2r_hits@10",
    "l2r_mrr",
    "r2l_hits@1",
    "r2l_hits@10",
    "r2l_mrr",
]


@dataclass
class StageSpec:
    name: str
    baseline_runs_dir: Path
    method_runs_dir: Path
    lang_tag: str


def run_cmd(cmd: List[str]):
    print(f"[AUTO-E10] RUN {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_seeds(seed_str: str) -> List[int]:
    return [int(x.strip()) for x in seed_str.split(",") if x.strip()]


def extract_seed(run_id: str):
    m = SEED_RE.search(run_id)
    return int(m.group(1)) if m else None


def is_run_success(run_dir: Path) -> bool:
    log_path = run_dir / "log.txt"
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return "[DONE] return_code=0" in text


def completed_seed_map(runs_dir: Path) -> Dict[int, str]:
    done: Dict[int, str] = {}
    if not runs_dir.exists():
        return done
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        seed = extract_seed(run_dir.name)
        if seed is None or not is_run_success(run_dir):
            continue
        prev = done.get(seed)
        if prev is None or run_dir.name > prev:
            done[seed] = run_dir.name
    return done


def wait_for_required_seeds(stages: List[StageSpec], required_seeds: List[int], poll_seconds: int, timeout_hours: float):
    deadline = time.time() + timeout_hours * 3600
    while True:
        all_ready = True
        print(f"[AUTO-E10] checking completion at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for spec in stages:
            b_done = completed_seed_map(spec.baseline_runs_dir)
            m_done = completed_seed_map(spec.method_runs_dir)
            b_ready = [s for s in required_seeds if s in b_done]
            m_ready = [s for s in required_seeds if s in m_done]
            print(
                f"[AUTO-E10] {spec.name}: baseline {len(b_ready)}/{len(required_seeds)}, "
                f"method {len(m_ready)}/{len(required_seeds)}"
            )
            if len(b_ready) < len(required_seeds) or len(m_ready) < len(required_seeds):
                all_ready = False
        if all_ready:
            print("[AUTO-E10] required epoch10 pilot runs are complete.")
            return
        if time.time() > deadline:
            raise TimeoutError("Timeout while waiting for epoch10 pilot completion.")
        time.sleep(poll_seconds)


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


def read_mean_rows(mean_csv: Path) -> Dict[str, Dict[str, float]]:
    rows = {}
    with mean_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["lang_pair"]
            rows[key] = {k: float(v) if k != "lang_pair" else v for k, v in row.items()}
    return rows


def read_mean_row(mean_csv: Path, lang_tag: str) -> Dict[str, float]:
    rows = read_mean_rows(mean_csv)
    if lang_tag not in rows:
        raise ValueError(f"lang_tag={lang_tag} not found in {mean_csv}")
    return rows[lang_tag]


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


def write_compare_report(compare_csv: Path, compare_md: Path, rows: List[Dict[str, object]], note: str):
    compare_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "metric",
        "baseline_mean",
        "method_mean",
        "delta_method_minus_baseline",
        "baseline_num_runs",
        "method_num_runs",
    ]
    with compare_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Epoch10 对比报告",
        "",
        note,
        "",
        "| dataset | metric | baseline_mean | method_mean | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['dataset']} | {row['metric']} | {float(row['baseline_mean']):.4f} | "
            f"{float(row['method_mean']):.4f} | {float(row['delta_method_minus_baseline']):+.4f} |"
        )
    compare_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def build_compare_rows(
    stage_specs: List[StageSpec],
    baseline_means: Dict[str, Path],
    method_means: Dict[str, Path],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for spec in stage_specs:
        b_row = read_mean_row(baseline_means[spec.name], spec.lang_tag)
        m_row = read_mean_row(method_means[spec.name], spec.lang_tag)
        for metric in METRICS:
            b_key = f"{metric}_mean"
            m_key = f"{metric}_mean"
            rows.append(
                {
                    "dataset": spec.name,
                    "metric": metric,
                    "baseline_mean": round(float(b_row[b_key]), 6),
                    "method_mean": round(float(m_row[m_key]), 6),
                    "delta_method_minus_baseline": round(float(m_row[m_key]) - float(b_row[b_key]), 6),
                    "baseline_num_runs": int(b_row["num_runs"]),
                    "method_num_runs": int(m_row["num_runs"]),
                }
            )
    return rows


def write_decision_reports(md_path: Path, json_path: Path, payload: Dict):
    md_lines = [
        "# Epoch10 自动决策报告",
        "",
        f"- 生成时间: `{payload['generated_at']}`",
        f"- 判定阈值: `delta_avg_mrr >= {payload['threshold']}` 且 2-seed 同向为正",
        f"- 最终决策: `{payload['decision']}`",
        f"- 执行动作: `{payload['action_cmd']}`",
        "",
        "## 数据集判定细节",
        "",
        "| dataset | delta_avg_mrr | seed_deltas | consistent_positive | pass_threshold |",
        "|---|---:|---|---:|---:|",
    ]
    for d in payload["datasets"]:
        seed_delta_text = ", ".join([f"s{item['seed']}:{item['delta']:.4f}" for item in d["seed_deltas"]]) or "N/A"
        md_lines.append(
            f"| {d['dataset']} | {d['delta_avg_mrr']:.4f} | {seed_delta_text} | "
            f"{str(d['consistent_positive'])} | {str(d['pass_threshold'])} |"
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Auto decide and dispatch after epoch10 pilot completion.")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--wait-seeds", default="42,3407")
    parser.add_argument("--formal-expand-seeds", default="2026,7,123")
    parser.add_argument("--threshold", type=float, default=0.003)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    parser.add_argument("--decision-md", default="reports/epoch10/epoch10_auto_decision.md")
    parser.add_argument("--decision-json", default="reports/epoch10/epoch10_auto_decision.json")
    parser.add_argument("--pilot-compare-csv", default="reports/epoch10/epoch10_compare_pilot.csv")
    parser.add_argument("--pilot-compare-md", default="reports/epoch10/epoch10_compare_pilot.md")
    parser.add_argument("--formal-compare-csv", default="reports/epoch10/epoch10_compare_formal.csv")
    parser.add_argument("--formal-compare-md", default="reports/epoch10/epoch10_compare_formal.md")
    args = parser.parse_args()

    wait_seeds = parse_seeds(args.wait_seeds)
    formal_expand_seeds = parse_seeds(args.formal_expand_seeds)

    stages = [
        StageSpec(
            name="zh_en",
            baseline_runs_dir=Path("runs/experiments/baseline/baseline_pilot_epoch10"),
            method_runs_dir=Path("runs/experiments/tmmeada/tmmeada_v1_best_pilot_epoch10"),
            lang_tag="zh_en",
        ),
        StageSpec(
            name="FBDB15K",
            baseline_runs_dir=Path("runs/experiments/baseline/baseline_pilot_epoch10_crossgraph"),
            method_runs_dir=Path("runs/experiments/tmmeada/tmmeada_v1_best_pilot_epoch10_crossgraph"),
            lang_tag="FBDB15K",
        ),
    ]

    wait_for_required_seeds(
        stages=stages,
        required_seeds=wait_seeds,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )

    report_dir = Path("reports/tmp/auto_decision_tmp")
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset_reports = []
    passed_datasets = []
    baseline_mean_paths: Dict[str, Path] = {}
    method_mean_paths: Dict[str, Path] = {}

    for spec in stages:
        b_summary, b_mean = summarize_runs(
            runner_python=args.runner_python,
            runs_dir=spec.baseline_runs_dir,
            out_prefix=report_dir / f"{spec.name}_baseline_epoch10",
        )
        m_summary, m_mean = summarize_runs(
            runner_python=args.runner_python,
            runs_dir=spec.method_runs_dir,
            out_prefix=report_dir / f"{spec.name}_method_epoch10",
        )
        baseline_mean_paths[spec.name] = b_mean
        method_mean_paths[spec.name] = m_mean

        b_row = read_mean_row(b_mean, spec.lang_tag)
        m_row = read_mean_row(m_mean, spec.lang_tag)
        b_avg_mrr = (b_row["l2r_mrr_mean"] + b_row["r2l_mrr_mean"]) / 2.0
        m_avg_mrr = (m_row["l2r_mrr_mean"] + m_row["r2l_mrr_mean"]) / 2.0
        delta_avg_mrr = m_avg_mrr - b_avg_mrr

        b_seed_mrr = read_seed_avg_mrr(b_summary, wait_seeds)
        m_seed_mrr = read_seed_avg_mrr(m_summary, wait_seeds)
        seed_deltas = []
        for seed in wait_seeds:
            if seed in b_seed_mrr and seed in m_seed_mrr:
                seed_deltas.append({"seed": seed, "delta": m_seed_mrr[seed] - b_seed_mrr[seed]})

        consistent_positive = len(seed_deltas) == len(wait_seeds) and all(d["delta"] > 0 for d in seed_deltas)
        pass_threshold = delta_avg_mrr >= args.threshold and consistent_positive
        if pass_threshold:
            passed_datasets.append(spec.name)

        dataset_reports.append(
            {
                "dataset": spec.name,
                "baseline_avg_mrr": round(b_avg_mrr, 6),
                "method_avg_mrr": round(m_avg_mrr, 6),
                "delta_avg_mrr": round(delta_avg_mrr, 6),
                "seed_deltas": seed_deltas,
                "consistent_positive": consistent_positive,
                "pass_threshold": pass_threshold,
                "baseline_summary_csv": str(b_summary),
                "method_summary_csv": str(m_summary),
            }
        )

    pilot_rows = build_compare_rows(stages, baseline_mean_paths, method_mean_paths)
    write_compare_report(
        compare_csv=Path(args.pilot_compare_csv),
        compare_md=Path(args.pilot_compare_md),
        rows=pilot_rows,
        note="说明：本报告基于 epoch10 pilot（2 seeds: 42,3407）。",
    )

    if passed_datasets:
        decision = "expand_to_5seed_formal_epoch10"
        action_cmd = (
            f"{args.runner_python} scripts/run_next_stage_pilot_queue.py --epoch10-only --seeds "
            + ",".join(str(s) for s in formal_expand_seeds)
        )
        dispatch_cmd = [
            args.runner_python,
            "scripts/run_next_stage_pilot_queue.py",
            "--epoch10-only",
            "--seeds",
            ",".join(str(s) for s in formal_expand_seeds),
        ]
        run_cmd(dispatch_cmd)

        # Refresh and produce formal compare report (now includes all seeds present in stage dirs).
        baseline_mean_paths = {}
        method_mean_paths = {}
        for spec in stages:
            _, b_mean = summarize_runs(
                runner_python=args.runner_python,
                runs_dir=spec.baseline_runs_dir,
                out_prefix=report_dir / f"{spec.name}_baseline_epoch10_formal",
            )
            _, m_mean = summarize_runs(
                runner_python=args.runner_python,
                runs_dir=spec.method_runs_dir,
                out_prefix=report_dir / f"{spec.name}_method_epoch10_formal",
            )
            baseline_mean_paths[spec.name] = b_mean
            method_mean_paths[spec.name] = m_mean

        formal_rows = build_compare_rows(stages, baseline_mean_paths, method_mean_paths)
        write_compare_report(
            compare_csv=Path(args.formal_compare_csv),
            compare_md=Path(args.formal_compare_md),
            rows=formal_rows,
            note="说明：本报告基于 epoch10 formal（已包含 stage 目录中的全部可用 seeds）。",
        )
    else:
        decision = "stop_after_epoch10_pilot_and_prepare_writeup"
        action_cmd = "no extra training; use pilot compare reports for writeup and error analysis"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "threshold": args.threshold,
        "wait_seeds": wait_seeds,
        "formal_expand_seeds": formal_expand_seeds,
        "passed_datasets": passed_datasets,
        "decision": decision,
        "action_cmd": action_cmd,
        "datasets": dataset_reports,
        "pilot_compare_csv": args.pilot_compare_csv,
        "pilot_compare_md": args.pilot_compare_md,
        "formal_compare_csv": args.formal_compare_csv if passed_datasets else "",
        "formal_compare_md": args.formal_compare_md if passed_datasets else "",
    }
    write_decision_reports(
        md_path=Path(args.decision_md),
        json_path=Path(args.decision_json),
        payload=payload,
    )
    print(f"[AUTO-E10] decision: {decision}")
    print(f"[AUTO-E10] action: {action_cmd}")


if __name__ == "__main__":
    main()

