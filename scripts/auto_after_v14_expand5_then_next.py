import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_cmd(cmd, cwd=None, check=True):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, check=False, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip())
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def process_exists(substr: str) -> bool:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_Process | "
            f"Where-Object {{ $_.Name -like 'python.exe' -and $_.CommandLine -like '*{substr}*' }} | "
            "Select-Object -ExpandProperty ProcessId"
        ),
    ]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        return False
    return bool(proc.stdout.strip())


def read_delta_avg_mrr(compare_csv: Path, target: str) -> float | None:
    if not compare_csv.exists():
        return None
    with compare_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("target") == target:
            try:
                return float(row.get("delta_avg_mrr_mean", "nan"))
            except Exception:
                return None
    return None


def append_process_log(repo_root: Path, lines):
    p = repo_root / "PROCESS_LOG.md"
    text = p.read_text(encoding="utf-8")
    text = text.rstrip() + "\n\n" + "\n".join(lines) + "\n"
    p.write_text(text, encoding="utf-8")


def has_staged_changes(repo_root: Path) -> bool:
    proc = run_cmd(["git", "diff", "--cached", "--name-only"], cwd=repo_root, check=True)
    return bool(proc.stdout.strip())


def commit_and_push(repo_root: Path, commit_msg: str, add_paths):
    run_cmd(["git", "add", *add_paths], cwd=repo_root, check=True)
    if not has_staged_changes(repo_root):
        print("[INFO] no changes to commit.")
        return False
    run_cmd(["git", "commit", "-m", commit_msg], cwd=repo_root, check=True)
    run_cmd(["git", "push", "origin", "sort"], cwd=repo_root, check=True)
    return True


def write_v14_final_report(repo_root: Path, status_json: Path, compare_csv: Path, out_md: Path):
    status = {}
    if status_json.exists():
        status = json.loads(status_json.read_text(encoding="utf-8"))
    delta = read_delta_avg_mrr(compare_csv=compare_csv, target="fr_en")
    b_missing = status.get("baseline", {}).get("final_missing_seeds", [])
    t_missing = status.get("tmmeada", {}).get("final_missing_seeds", [])
    done_5seed = (not b_missing) and (not t_missing)

    lines = [
        "# 迁移实验阶段报告（v14 fr_en expand5 完成）",
        "",
        f"- 时间戳: `{now_ts()}`",
        f"- baseline 缺失 seeds: `{b_missing}`",
        f"- tmmeada 缺失 seeds: `{t_missing}`",
        f"- 是否完成 5-seed: `{done_5seed}`",
        f"- delta_avg_mrr_mean (fr_en, vs baseline): `{delta}`",
        "",
        "## 结论",
    ]
    if delta is None:
        lines.append("- 未读取到最终对比指标，请检查 compare csv。")
    elif delta > 0:
        lines.append("- v14 在 fr_en 上保持正增益，5-seed 结果支持继续扩展到跨图谱目标。")
    elif abs(delta) < 1e-12:
        lines.append("- v14 在 fr_en 上与 baseline 持平，建议进入跨图谱验证。")
    else:
        lines.append("- v14 在 fr_en 上未达预期，建议回退到更稳配置并做误差分析。")

    lines.extend(
        [
            "",
            "## 下一步",
            "1. 启动 FBYG15K expand5（断点续跑）以补齐跨图谱迁移证据。",
            "2. 产出同口径 compare_vs_baseline 报表并并入中期/终稿图表。",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fbyg_launch_report(repo_root: Path, status_json: Path, out_md: Path, queue_out: str, queue_err: str):
    status = {}
    if status_json.exists():
        status = json.loads(status_json.read_text(encoding="utf-8"))
    b_missing = status.get("baseline", {}).get("final_missing_seeds", [])
    t_missing = status.get("tmmeada", {}).get("final_missing_seeds", [])

    lines = [
        "# 迁移实验阶段报告（FBYG expand5 启动）",
        "",
        f"- 时间戳: `{now_ts()}`",
        f"- baseline 缺失 seeds: `{b_missing}`",
        f"- tmmeada 缺失 seeds: `{t_missing}`",
        "",
        "## 已启动后台续跑",
        f"- out log: `{queue_out}`",
        f"- err log: `{queue_err}`",
        "",
        "## 目标",
        "- 将 FBYG15K 从 2-seed 扩展到 5-seed，并自动输出 compare_vs_baseline。",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor v14 expand5 queue. After finish: finalize + push, then launch next step and push."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--watch-substr", default="run_transfer_adapt_v14_fren_expand5_resume.py")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    auto_log_dir = repo_root / "runs" / "system" / "auto_watch"
    auto_log_dir.mkdir(parents=True, exist_ok=True)
    auto_log = auto_log_dir / f"auto_after_v14_expand5_{now_ts()}.log"

    def log(msg: str):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        auto_log.write_text(auto_log.read_text(encoding="utf-8") + line + "\n", encoding="utf-8") if auto_log.exists() else auto_log.write_text(line + "\n", encoding="utf-8")

    log("start monitoring v14 expand5 queue")
    while process_exists(args.watch_substr):
        log("queue still running...")
        time.sleep(max(15, args.poll_seconds))
    log("queue finished, start finalization")

    # 1) finalize fr_en expand5 summary
    run_cmd(
        [
            args.runner_python,
            "scripts/run_transfer_adapt_v14_fren_expand5_resume.py",
            "--run-missing",
            "0",
        ],
        cwd=repo_root,
        check=True,
    )

    v14_status_json = repo_root / "reports" / "transfer" / "transfer_adapt_v14_fren_expand5_status.json"
    v14_compare_csv = repo_root / "reports" / "transfer" / "transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv"
    v14_final_report = repo_root / "reports" / "transfer" / "transfer_stage_update_20260308_v14_expand5_final.md"
    write_v14_final_report(
        repo_root=repo_root,
        status_json=v14_status_json,
        compare_csv=v14_compare_csv,
        out_md=v14_final_report,
    )

    append_process_log(
        repo_root=repo_root,
        lines=[
            "## 28. 2026-03-08 v14 fr_en expand5 auto-finalized (ASCII summary)",
            "- monitored queue completed, then rebuilt final merged summaries.",
            "- file: reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv",
            "- file: reports/transfer/transfer_stage_update_20260308_v14_expand5_final.md",
            "- next step decided: launch FBYG expand5 resume queue.",
        ],
    )

    # first sync
    commit_and_push(
        repo_root=repo_root,
        commit_msg="Finalize v14 fr_en expand5 and publish final report",
        add_paths=[
            "scripts/run_transfer_adapt_v14_fren_expand5_resume.py",
            "runs/transfer/transfer_adapt_v14_fren_expand5_baseline",
            "runs/transfer/transfer_adapt_v14_fren_expand5_tmmeada",
            "runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline",
            "runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada",
            "reports/transfer/transfer_adapt_v14_fren_expand5_progress_baseline_ref_summary.csv",
            "reports/transfer/transfer_adapt_v14_fren_expand5_progress_tmmeada_summary.csv",
            "reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.csv",
            "reports/transfer/transfer_adapt_v14_fren_expand5_progress_compare_vs_baseline.md",
            "reports/transfer/transfer_adapt_v14_fren_expand5_status.json",
            "reports/transfer/transfer_adapt_v14_fren_expand5_status.md",
            "reports/transfer/transfer_stage_update_20260308_v14_expand5_final.md",
            "PROCESS_LOG.md",
        ],
    )
    log("first sync completed")

    # 2) next step: FBYG expand5
    run_cmd(
        [
            args.runner_python,
            "scripts/run_transfer_adapt_fbyg_expand5_resume.py",
            "--run-missing",
            "0",
        ],
        cwd=repo_root,
        check=True,
    )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    qdir = repo_root / "runs" / "transfer" / "transfer_adapt_fbyg_expand5"
    qdir.mkdir(parents=True, exist_ok=True)
    qout = qdir / f"queue_{ts}.out.log"
    qerr = qdir / f"queue_{ts}.err.log"

    with qout.open("w", encoding="utf-8") as out_fp, qerr.open("w", encoding="utf-8") as err_fp:
        subprocess.Popen(
            [
                args.runner_python,
                "scripts/run_transfer_adapt_fbyg_expand5_resume.py",
                "--run-missing",
                "1",
            ],
            cwd=repo_root,
            stdout=out_fp,
            stderr=err_fp,
        )

    fbyg_status_json = repo_root / "reports" / "transfer" / "transfer_adapt_fbyg_expand5_status.json"
    fbyg_launch_report = repo_root / "reports" / "transfer" / "transfer_stage_update_20260308_fbyg_expand5_launch.md"
    write_fbyg_launch_report(
        repo_root=repo_root,
        status_json=fbyg_status_json,
        out_md=fbyg_launch_report,
        queue_out=str(qout.relative_to(repo_root)).replace("\\", "/"),
        queue_err=str(qerr.relative_to(repo_root)).replace("\\", "/"),
    )

    append_process_log(
        repo_root=repo_root,
        lines=[
            "## 29. 2026-03-08 FBYG expand5 launched after v14 finalize (ASCII summary)",
            "- added resume script: scripts/run_transfer_adapt_fbyg_expand5_resume.py",
            "- generated pre-check status:",
            "  - reports/transfer/transfer_adapt_fbyg_expand5_status.json",
            "  - reports/transfer/transfer_adapt_fbyg_expand5_status.md",
            "- launched queue logs:",
            f"  - {str(qout.relative_to(repo_root)).replace(chr(92), '/')}",
            f"  - {str(qerr.relative_to(repo_root)).replace(chr(92), '/')}",
            "- stage note: reports/transfer/transfer_stage_update_20260308_fbyg_expand5_launch.md",
        ],
    )

    # final sync
    commit_and_push(
        repo_root=repo_root,
        commit_msg="Launch FBYG expand5 next stage after v14 finalize",
        add_paths=[
            "scripts/run_transfer_adapt_fbyg_expand5_resume.py",
            "reports/transfer/transfer_adapt_fbyg_expand5_progress_baseline_ref_summary.csv",
            "reports/transfer/transfer_adapt_fbyg_expand5_progress_tmmeada_summary.csv",
            "reports/transfer/transfer_adapt_fbyg_expand5_progress_compare_vs_baseline.csv",
            "reports/transfer/transfer_adapt_fbyg_expand5_progress_compare_vs_baseline.md",
            "reports/transfer/transfer_adapt_fbyg_expand5_status.json",
            "reports/transfer/transfer_adapt_fbyg_expand5_status.md",
            "reports/transfer/transfer_stage_update_20260308_fbyg_expand5_launch.md",
            "PROCESS_LOG.md",
        ],
    )
    log("final sync completed")


if __name__ == "__main__":
    main()
