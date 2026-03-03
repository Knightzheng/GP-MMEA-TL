import argparse
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


SEED_RE = re.compile(r"-s(\d+)$")


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


def run_cmd(cmd: List[str]):
    print(f"[AUTO-V2] RUN {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def wait_until_ready(method_dirs: List[Path], required_seeds: List[int], poll_seconds: int, timeout_hours: float):
    deadline = time.time() + timeout_hours * 3600
    while True:
        all_ready = True
        print(f"[AUTO-V2] checking completion at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for runs_dir in method_dirs:
            done = completed_seed_map(runs_dir)
            ready = [s for s in required_seeds if s in done]
            print(f"[AUTO-V2] {runs_dir}: {len(ready)}/{len(required_seeds)}")
            if len(ready) < len(required_seeds):
                all_ready = False
        if all_ready:
            print("[AUTO-V2] required v2 runs are complete.")
            return
        if time.time() > deadline:
            raise TimeoutError("Timeout while waiting for v2 tuned runs.")
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description="Auto-compare v2 tuned pilot once required runs complete.")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--required-seeds", default="42,3407")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    parser.add_argument("--threshold", type=float, default=0.003)
    parser.add_argument("--method-zh-dir", default="runs/tmmeada_v2_tuned_pilot_epoch10")
    parser.add_argument("--method-fbdb-dir", default="runs/tmmeada_v2_tuned_pilot_epoch10_crossgraph")
    args = parser.parse_args()

    required_seeds = parse_seeds(args.required_seeds)
    method_dirs = [Path(args.method_zh_dir), Path(args.method_fbdb_dir)]

    wait_until_ready(
        method_dirs=method_dirs,
        required_seeds=required_seeds,
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )

    run_cmd(
        [
            args.runner_python,
            "scripts/compare_epoch10_v2_tuned_vs_baseline.py",
            "--required-seeds",
            args.required_seeds,
            "--threshold",
            str(args.threshold),
            "--method-zh-dir",
            args.method_zh_dir,
            "--method-fbdb-dir",
            args.method_fbdb_dir,
        ]
    )


if __name__ == "__main__":
    main()
