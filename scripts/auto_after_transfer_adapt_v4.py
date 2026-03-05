import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_cmd(cmd: List[str]):
    print(f"[AUTO-V4] RUN {' '.join(cmd)}")
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


def wait_for_done(
    target_dir: Path, required_seeds: List[int], required_targets: List[str], poll_seconds: int, timeout_hours: float
):
    deadline = time.time() + timeout_hours * 3600
    while True:
        status = collect_latest_status(target_dir)
        missing = get_missing(status, required_seeds, required_targets)
        if not missing:
            print(f"[AUTO-V4] v4 queue done at {now_str()}")
            return
        if time.time() > deadline:
            raise TimeoutError(f"Timeout waiting v4 done: {missing}")
        print(f"[AUTO-V4] waiting {now_str()} | missing={missing}")
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description="Wait transfer_adapt_v4 queue then auto summarize.")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--target-dir", default="runs/transfer/transfer_adapt_v4/target_eval")
    parser.add_argument("--seeds", default="42,3407")
    parser.add_argument("--targets", default="ja_en,FBDB15K")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-hours", type=float, default=48.0)
    args = parser.parse_args()

    wait_for_done(
        target_dir=Path(args.target_dir),
        required_seeds=parse_seeds(args.seeds),
        required_targets=parse_targets(args.targets),
        poll_seconds=args.poll_seconds,
        timeout_hours=args.timeout_hours,
    )

    run_cmd(
        [
            args.runner_python,
            "scripts/summarize_transfer_formal.py",
            "--baseline-target-dir",
            "runs/transfer/transfer_adapt_pilot/target_eval",
            "--tmmeada-target-dir",
            "runs/transfer/transfer_adapt_v4/target_eval",
            "--baseline-out",
            "reports/transfer/transfer_adapt_v4_baseline_ref_summary.csv",
            "--tmmeada-out",
            "reports/transfer/transfer_adapt_v4_tmmeada_summary.csv",
            "--compare-out-csv",
            "reports/transfer/transfer_adapt_v4_compare_vs_baseline.csv",
            "--compare-out-md",
            "reports/transfer/transfer_adapt_v4_compare_vs_baseline.md",
        ]
    )
    run_cmd(
        [
            args.runner_python,
            "scripts/summarize_transfer_formal.py",
            "--baseline-target-dir",
            "runs/transfer/transfer_adapt_v3/target_eval",
            "--tmmeada-target-dir",
            "runs/transfer/transfer_adapt_v4/target_eval",
            "--baseline-out",
            "reports/transfer/transfer_adapt_v4_v3_ref_summary.csv",
            "--tmmeada-out",
            "reports/transfer/transfer_adapt_v4_tmmeada_summary.csv",
            "--compare-out-csv",
            "reports/transfer/transfer_adapt_v4_compare_vs_v3.csv",
            "--compare-out-md",
            "reports/transfer/transfer_adapt_v4_compare_vs_v3.md",
        ]
    )
    print("[AUTO-V4] done.")


if __name__ == "__main__":
    main()
