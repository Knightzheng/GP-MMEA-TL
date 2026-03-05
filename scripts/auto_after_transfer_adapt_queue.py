import argparse
import csv
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


SEED_RE = re.compile(r"-s(?P<seed>\d+)$")


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_cmd(cmd: List[str]):
    print(f"[AUTO-ADAPT] RUN {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_seeds(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_targets(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def tail_contains(path: Path, marker: str, read_bytes: int = 8192) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        offset = max(0, size - read_bytes)
        f.seek(offset)
        chunk = f.read().decode("utf-8", errors="replace")
    return marker in chunk


def infer_seed_target(run_dir: Path) -> Tuple[int, str] | None:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        m = payload.get("meaformer", {})
        seed = int(m.get("random_seed", -1))
        data_choice = str(m.get("data_choice", ""))
        data_split = str(m.get("data_split", ""))
        target = data_split if data_choice == "DBP15K" else data_choice
        return seed, target
    except Exception:
        return None


def collect_latest_status(target_eval_dir: Path) -> Dict[Tuple[int, str], Dict]:
    result: Dict[Tuple[int, str], Dict] = {}
    if not target_eval_dir.exists():
        return result
    for run_dir in target_eval_dir.iterdir():
        if not run_dir.is_dir():
            continue
        parsed = infer_seed_target(run_dir)
        if parsed is None:
            continue
        seed, target = parsed
        key = (seed, target)
        log_path = run_dir / "log.txt"
        done = tail_contains(log_path, "[DONE] return_code=0")
        mtime = run_dir.stat().st_mtime
        prev = result.get(key)
        if prev is None or mtime > prev["mtime"]:
            result[key] = {
                "run_dir": str(run_dir),
                "done": done,
                "mtime": mtime,
            }
    return result


def get_missing(
    status_map: Dict[Tuple[int, str], Dict], required_seeds: List[int], required_targets: List[str]
) -> List[str]:
    missing = []
    for seed in required_seeds:
        for target in required_targets:
            key = (seed, target)
            info = status_map.get(key)
            if info is None:
                missing.append(f"{target}:s{seed}:missing_run")
            elif not info.get("done", False):
                missing.append(f"{target}:s{seed}:not_done")
    return missing


def wait_for_queue_done(
    baseline_dir: Path,
    tmmeada_dir: Path,
    wait_seeds: List[int],
    required_targets: List[str],
    poll_seconds: int,
    timeout_hours: float,
):
    deadline = time.time() + timeout_hours * 3600
    while True:
        b_status = collect_latest_status(baseline_dir)
        t_status = collect_latest_status(tmmeada_dir)
        b_missing = get_missing(b_status, wait_seeds, required_targets)
        t_missing = get_missing(t_status, wait_seeds, required_targets)
        if not b_missing and not t_missing:
            print(f"[AUTO-ADAPT] queue done at {now_str()}")
            return
        if time.time() > deadline:
            raise TimeoutError(
                "Timeout waiting queue done. "
                f"baseline_missing={b_missing}, tmmeada_missing={t_missing}"
            )
        print(
            f"[AUTO-ADAPT] waiting {now_str()} | "
            f"baseline_missing={b_missing} | tmmeada_missing={t_missing}"
        )
        time.sleep(poll_seconds)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_seed_from_run_id(run_id: str) -> int | None:
    m = SEED_RE.search(run_id)
    if not m:
        return None
    return int(m.group("seed"))


def seed_target_map(rows: List[Dict[str, str]]) -> Dict[Tuple[str, int], Dict[str, str]]:
    out: Dict[Tuple[str, int], Dict[str, str]] = {}
    for r in rows:
        target = r.get("target", "")
        seed = parse_seed_from_run_id(r.get("run_id", ""))
        if target and seed is not None:
            out[(target, seed)] = r
    return out


def to_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0"))
    except Exception:
        return 0.0


def build_decision_details(
    compare_rows: List[Dict[str, str]],
    baseline_rows: List[Dict[str, str]],
    tmmeada_rows: List[Dict[str, str]],
    required_targets: List[str],
    reference_seeds: List[int],
    threshold: float,
) -> Tuple[List[Dict], bool]:
    compare_map = {r.get("target", ""): r for r in compare_rows}
    b_map = seed_target_map(baseline_rows)
    t_map = seed_target_map(tmmeada_rows)

    details = []
    all_pass = True
    for target in required_targets:
        c = compare_map.get(target)
        if c is None:
            details.append(
                {
                    "target": target,
                    "error": "missing_target_in_compare",
                    "pass_threshold": False,
                }
            )
            all_pass = False
            continue

        baseline_runs = int(float(c.get("baseline_num_runs", "0")))
        tmmeada_runs = int(float(c.get("tmmeada_num_runs", "0")))
        delta_avg_mrr = to_float(c, "delta_avg_mrr_mean")

        seed_deltas = []
        consistent_positive = True
        for seed in reference_seeds:
            b_row = b_map.get((target, seed))
            t_row = t_map.get((target, seed))
            if b_row is None or t_row is None:
                consistent_positive = False
                seed_deltas.append({"seed": seed, "delta_avg_mrr": None, "status": "missing"})
                continue
            d = to_float(t_row, "avg_mrr") - to_float(b_row, "avg_mrr")
            seed_deltas.append({"seed": seed, "delta_avg_mrr": round(d, 6), "status": "ok"})
            if d < 0:
                consistent_positive = False

        enough_runs = baseline_runs >= len(reference_seeds) and tmmeada_runs >= len(reference_seeds)
        pass_threshold = delta_avg_mrr >= threshold and consistent_positive and enough_runs
        all_pass = all_pass and pass_threshold
        details.append(
            {
                "target": target,
                "baseline_num_runs": baseline_runs,
                "tmmeada_num_runs": tmmeada_runs,
                "delta_avg_mrr_mean": round(delta_avg_mrr, 6),
                "seed_deltas": seed_deltas,
                "consistent_positive": consistent_positive,
                "enough_runs": enough_runs,
                "pass_threshold": pass_threshold,
            }
        )
    return details, all_pass


def write_decision(md_path: Path, json_path: Path, payload: Dict):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Transfer Adapt Auto Decision",
        "",
        f"- timestamp: `{payload['timestamp']}`",
        f"- threshold: `delta_avg_mrr_mean >= {payload['threshold']}` and seed-wise non-negative",
        f"- required_targets: `{payload['required_targets']}`",
        f"- reference_seeds: `{payload['reference_seeds']}`",
        f"- decision: `{payload['decision']}`",
        f"- next_action: `{payload['next_action']}`",
        "",
        "| target | delta_avg_mrr_mean | consistent_positive | enough_runs | pass_threshold |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in payload["details"]:
        if "error" in d:
            lines.append(f"| {d['target']} | N/A | False | False | False |")
            continue
        lines.append(
            f"| {d['target']} | {d['delta_avg_mrr_mean']:+.6f} | "
            f"{str(d['consistent_positive'])} | {str(d['enough_runs'])} | {str(d['pass_threshold'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Wait transfer-adapt queue, auto decide expand/not-expand, then auto run next step."
    )
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--meaformer-python", default=r"D:\Anaconda_envs\envs\bysj-meaformer\python.exe")
    parser.add_argument("--wait-stage-root", default="transfer/transfer_adapt_pilot")
    parser.add_argument("--wait-stage-root-tmmeada", default="transfer/transfer_adapt_pilot_tmmeada")
    parser.add_argument("--wait-seeds", default="3407")
    parser.add_argument("--reference-seeds", default="42,3407")
    parser.add_argument("--required-targets", default="ja_en,FBDB15K")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    parser.add_argument(
        "--decision-json",
        default="reports/transfer/transfer_adapt_auto_decision.json",
    )
    parser.add_argument(
        "--decision-md",
        default="reports/transfer/transfer_adapt_auto_decision.md",
    )
    args = parser.parse_args()

    wait_seeds = parse_seeds(args.wait_seeds)
    reference_seeds = parse_seeds(args.reference_seeds)
    required_targets = parse_targets(args.required_targets)

    baseline_target_dir = Path("runs") / args.wait_stage_root / "target_eval"
    tmmeada_target_dir = Path("runs") / args.wait_stage_root_tmmeada / "target_eval"

    wait_for_queue_done(
        baseline_dir=baseline_target_dir,
        tmmeada_dir=tmmeada_target_dir,
        wait_seeds=wait_seeds,
        required_targets=required_targets,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )

    baseline_summary = Path("reports/transfer/transfer_adapt_pilot_target_eval_baseline_summary.csv")
    tmmeada_summary = Path("reports/transfer/transfer_adapt_pilot_target_eval_tmmeada_summary.csv")
    compare_csv = Path("reports/transfer/transfer_adapt_pilot_compare_tmmeada_vs_baseline.csv")
    compare_md = Path("reports/transfer/transfer_adapt_pilot_compare_tmmeada_vs_baseline.md")

    run_cmd(
        [
            args.runner_python,
            "scripts/summarize_transfer_formal.py",
            "--baseline-target-dir",
            str(baseline_target_dir),
            "--tmmeada-target-dir",
            str(tmmeada_target_dir),
            "--baseline-out",
            str(baseline_summary),
            "--tmmeada-out",
            str(tmmeada_summary),
            "--compare-out-csv",
            str(compare_csv),
            "--compare-out-md",
            str(compare_md),
        ]
    )

    compare_rows = read_csv_rows(compare_csv)
    baseline_rows = read_csv_rows(baseline_summary)
    tmmeada_rows = read_csv_rows(tmmeada_summary)
    details, all_pass = build_decision_details(
        compare_rows=compare_rows,
        baseline_rows=baseline_rows,
        tmmeada_rows=tmmeada_rows,
        required_targets=required_targets,
        reference_seeds=reference_seeds,
        threshold=args.threshold,
    )

    if all_pass:
        decision = "expand_to_fr_en_fbyg15k_adapt"
        next_action = "run_transfer_adapt_expand_queue"
    else:
        decision = "run_tmmeada_tuned_lite_on_ja_fbdb"
        next_action = "run_transfer_adapt_tuned_queue"

    payload = {
        "timestamp": now_str(),
        "threshold": args.threshold,
        "required_targets": required_targets,
        "wait_seeds": wait_seeds,
        "reference_seeds": reference_seeds,
        "decision": decision,
        "next_action": next_action,
        "details": details,
    }
    write_decision(Path(args.decision_md), Path(args.decision_json), payload)
    print(f"[AUTO-ADAPT] decision={decision}")

    if decision == "expand_to_fr_en_fbyg15k_adapt":
        run_cmd(
            [
                args.runner_python,
                "scripts/run_transfer_adapt_expand_queue.py",
                "--seeds",
                ",".join(str(s) for s in reference_seeds),
                "--runner-python",
                args.runner_python,
                "--meaformer-python",
                args.meaformer_python,
            ]
        )
        run_cmd(
            [
                args.runner_python,
                "scripts/summarize_transfer_formal.py",
                "--baseline-target-dir",
                "runs/transfer/transfer_adapt_expand/target_eval",
                "--tmmeada-target-dir",
                "runs/transfer/transfer_adapt_expand_tmmeada/target_eval",
                "--baseline-out",
                "reports/transfer/transfer_adapt_expand_target_eval_baseline_summary.csv",
                "--tmmeada-out",
                "reports/transfer/transfer_adapt_expand_target_eval_tmmeada_summary.csv",
                "--compare-out-csv",
                "reports/transfer/transfer_adapt_expand_compare_tmmeada_vs_baseline.csv",
                "--compare-out-md",
                "reports/transfer/transfer_adapt_expand_compare_tmmeada_vs_baseline.md",
            ]
        )
    else:
        run_cmd(
            [
                args.runner_python,
                "scripts/run_transfer_adapt_tuned_queue.py",
                "--seeds",
                ",".join(str(s) for s in reference_seeds),
                "--runner-python",
                args.runner_python,
                "--meaformer-python",
                args.meaformer_python,
            ]
        )
        run_cmd(
            [
                args.runner_python,
                "scripts/summarize_transfer_formal.py",
                "--baseline-target-dir",
                str(baseline_target_dir),
                "--tmmeada-target-dir",
                "runs/transfer/transfer_adapt_tuned/target_eval",
                "--baseline-out",
                "reports/transfer/transfer_adapt_tuned_lite_baseline_ref_summary.csv",
                "--tmmeada-out",
                "reports/transfer/transfer_adapt_tuned_lite_tmmeada_summary.csv",
                "--compare-out-csv",
                "reports/transfer/transfer_adapt_tuned_lite_compare_vs_baseline.csv",
                "--compare-out-md",
                "reports/transfer/transfer_adapt_tuned_lite_compare_vs_baseline.md",
            ]
        )

    print("[AUTO-ADAPT] done.")


if __name__ == "__main__":
    main()
