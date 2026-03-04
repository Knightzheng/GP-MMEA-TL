import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

import subprocess


def run_cmd(cmd: List[str]):
    print(f"[AUTO-NEXT] RUN {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def wait_for_decision(path: Path, poll_seconds: int, timeout_hours: float):
    deadline = time.time() + timeout_hours * 3600
    while True:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and "decision" in payload:
                    return payload
            except Exception:
                pass
        if time.time() > deadline:
            raise TimeoutError(f"Timeout waiting for {path}")
        print(f"[AUTO-NEXT] waiting for v2b decision... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(poll_seconds)


def run_formal_v2b(args):
    print("[AUTO-NEXT] v2b passed threshold, promoting to formal 5-seed.")
    extra_seeds = "2026,7,123"

    # Extend baseline to 5-seed for fair comparison
    run_cmd(
        [
            args.runner_python,
            "scripts/run_from_base_config_multiseed.py",
            "--base-config",
            "configs/baselines/meaformer_zh_en_rtx3060_safe_epoch10_pilot.yaml",
            "--seeds",
            extra_seeds,
        ]
    )
    run_cmd(
        [
            args.runner_python,
            "scripts/run_from_base_config_multiseed.py",
            "--base-config",
            "configs/baselines/meaformer_fbdb15k_rtx3060_safe_epoch10_pilot.yaml",
            "--seeds",
            extra_seeds,
        ]
    )

    # Extend method to 5-seed
    run_cmd(
        [
            args.runner_python,
            "scripts/run_from_base_config_multiseed.py",
            "--base-config",
            "configs/tmmeada/meaformer_zh_en_tmmeada_v2b_lite_hardneg_epoch10_pilot.yaml",
            "--seeds",
            extra_seeds,
        ]
    )
    run_cmd(
        [
            args.runner_python,
            "scripts/run_from_base_config_multiseed.py",
            "--base-config",
            "configs/tmmeada/meaformer_fbdb15k_tmmeada_v2b_lite_hardneg_epoch10_pilot.yaml",
            "--seeds",
            extra_seeds,
        ]
    )

    # Formal compare
    run_cmd(
        [
            args.runner_python,
            "scripts/compare_epoch10_v2_tuned_vs_baseline.py",
            "--required-seeds",
            "42,3407,2026,7,123",
            "--threshold",
            str(args.threshold),
            "--method-zh-dir",
            "runs/experiments/tmmeada/tmmeada_v2b_lite_hardneg_pilot_epoch10",
            "--method-fbdb-dir",
            "runs/experiments/tmmeada/tmmeada_v2b_lite_hardneg_pilot_epoch10_crossgraph",
            "--compare-csv",
            "reports/epoch10/epoch10_compare_v2b_lite_hardneg_formal.csv",
            "--compare-md",
            "reports/epoch10/epoch10_compare_v2b_lite_hardneg_formal.md",
            "--decision-json",
            "reports/epoch10/epoch10_v2b_lite_hardneg_formal_decision.json",
            "--decision-md",
            "reports/epoch10/epoch10_v2b_lite_hardneg_formal_decision.md",
        ]
    )


def run_fallback_v2c(args):
    print("[AUTO-NEXT] v2b not passed, starting v2c source-select-only pilot.")
    seeds = "42,3407"
    run_cmd(
        [
            args.runner_python,
            "scripts/run_from_base_config_multiseed.py",
            "--base-config",
            "configs/tmmeada/meaformer_zh_en_tmmeada_v2c_source_only_epoch10_pilot.yaml",
            "--seeds",
            seeds,
        ]
    )
    run_cmd(
        [
            args.runner_python,
            "scripts/run_from_base_config_multiseed.py",
            "--base-config",
            "configs/tmmeada/meaformer_fbdb15k_tmmeada_v2c_source_only_epoch10_pilot.yaml",
            "--seeds",
            seeds,
        ]
    )
    run_cmd(
        [
            args.runner_python,
            "scripts/compare_epoch10_v2_tuned_vs_baseline.py",
            "--required-seeds",
            seeds,
            "--threshold",
            str(args.threshold),
            "--method-zh-dir",
            "runs/experiments/tmmeada/tmmeada_v2c_source_only_pilot_epoch10",
            "--method-fbdb-dir",
            "runs/experiments/tmmeada/tmmeada_v2c_source_only_pilot_epoch10_crossgraph",
            "--compare-csv",
            "reports/epoch10/epoch10_compare_v2c_source_only_pilot.csv",
            "--compare-md",
            "reports/epoch10/epoch10_compare_v2c_source_only_pilot.md",
            "--decision-json",
            "reports/epoch10/epoch10_v2c_source_only_decision.json",
            "--decision-md",
            "reports/epoch10/epoch10_v2c_source_only_decision.md",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Auto-run next stage once v2b decision is ready.")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--decision-json", default="reports/epoch10/epoch10_v2b_lite_hardneg_decision.json")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    parser.add_argument("--threshold", type=float, default=0.003)
    args = parser.parse_args()

    decision_path = Path(args.decision_json)
    payload = wait_for_decision(decision_path, args.poll_seconds, args.timeout_hours)
    decision = str(payload.get("decision", "")).strip()
    print(f"[AUTO-NEXT] v2b decision = {decision}")

    if decision == "promote_to_formal_5seed_epoch10":
        run_formal_v2b(args)
    else:
        run_fallback_v2c(args)

    print("[AUTO-NEXT] done.")


if __name__ == "__main__":
    main()

