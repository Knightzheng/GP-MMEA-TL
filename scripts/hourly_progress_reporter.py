import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


def run_powershell(cmd: str) -> str:
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def process_matches(substr: str):
    ps = (
        "Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.Name -like 'python.exe' -and $_.CommandLine -like '*{substr}*' }} | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    out = run_powershell(ps)
    if not out:
        return []
    try:
        data = json.loads(out)
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    return data


def latest_line(path: Path, n=40):
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return lines[-n:]
    except Exception:
        return []


def list_run_cards(report_dir: Path, pattern: str):
    cards = sorted(report_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return [c.name for c in cards]


def parse_seeds_from_card_names(card_names, key: str):
    pat = re.compile(rf"transfer_run_card_\d{{8}}-\d{{6}}_{key}_s(\d+)\.json")
    seeds = []
    for name in card_names:
        m = pat.search(name)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(set(seeds))


def safe_read_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_snapshot(out_file: Path, repo_root: Path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_dir = repo_root / "reports" / "transfer"

    v14_procs = process_matches("run_transfer_adapt_v14_fren_expand5_resume.py")
    fbyg_procs = process_matches("run_transfer_adapt_fbyg_expand5_resume.py")
    auto_procs = process_matches("auto_after_v14_expand5_then_next.py")
    train_procs = process_matches("run_transfer_train_eval.py")

    v14_cards = list_run_cards(report_dir, "transfer_run_card_*v14_fren_expand5_s*.json")
    fbyg_cards = list_run_cards(report_dir, "transfer_run_card_*fbyg_expand5_s*.json")

    v14_baseline_seeds = parse_seeds_from_card_names(v14_cards, "baseline_transfer_adapt_v14_fren_expand5")
    v14_tmmeada_seeds = parse_seeds_from_card_names(v14_cards, "tmmeada_transfer_adapt_v14_fren_expand5")
    fbyg_baseline_seeds = parse_seeds_from_card_names(fbyg_cards, "baseline_transfer_adapt_fbyg_expand5")
    fbyg_tmmeada_seeds = parse_seeds_from_card_names(fbyg_cards, "tmmeada_transfer_adapt_fbyg_expand5")

    v14_status = safe_read_json(report_dir / "transfer_adapt_v14_fren_expand5_status.json")
    fbyg_status = safe_read_json(report_dir / "transfer_adapt_fbyg_expand5_status.json")

    queue_logs = sorted(
        (repo_root / "runs" / "transfer" / "transfer_adapt_v14_fren_expand5").glob("*.out.log"),
        key=lambda p: p.stat().st_mtime,
    )
    queue_tail = latest_line(queue_logs[-1], n=20) if queue_logs else []

    lines = []
    lines.append(f"## {now}")
    lines.append("")
    lines.append("- 进程状态:")
    lines.append(f"  - v14_expand5_queue: {len(v14_procs)}")
    lines.append(f"  - fbyg_expand5_queue: {len(fbyg_procs)}")
    lines.append(f"  - auto_finalize_watcher: {len(auto_procs)}")
    lines.append(f"  - run_transfer_train_eval: {len(train_procs)}")
    if train_procs:
        lines.append(f"  - 当前训练命令: `{train_procs[0].get('CommandLine', '')}`")
    lines.append("")
    lines.append("- seed完成情况:")
    lines.append(f"  - v14 baseline: {v14_baseline_seeds}")
    lines.append(f"  - v14 tmmeada: {v14_tmmeada_seeds}")
    lines.append(f"  - fbyg baseline: {fbyg_baseline_seeds}")
    lines.append(f"  - fbyg tmmeada: {fbyg_tmmeada_seeds}")
    lines.append("")
    if v14_status:
        lines.append("- v14 status(final_missing):")
        lines.append(
            f"  - baseline: {v14_status.get('baseline', {}).get('final_missing_seeds', [])}"
        )
        lines.append(
            f"  - tmmeada: {v14_status.get('tmmeada', {}).get('final_missing_seeds', [])}"
        )
    if fbyg_status:
        lines.append("- fbyg status(final_missing):")
        lines.append(
            f"  - baseline: {fbyg_status.get('baseline', {}).get('final_missing_seeds', [])}"
        )
        lines.append(
            f"  - tmmeada: {fbyg_status.get('tmmeada', {}).get('final_missing_seeds', [])}"
        )
    lines.append("")
    if queue_tail:
        lines.append("- v14队列最新日志尾部:")
        lines.append("```text")
        lines.extend(queue_tail)
        lines.append("```")
        lines.append("")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    prev = out_file.read_text(encoding="utf-8", errors="ignore") if out_file.exists() else ""
    out_file.write_text(prev + "\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Write hourly progress snapshots to markdown.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="reports/transfer/hourly_progress.md")
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_file = repo_root / args.output

    loops = max(1, args.hours)
    for i in range(loops):
        write_snapshot(out_file=out_file, repo_root=repo_root)
        if i < loops - 1:
            time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    main()
