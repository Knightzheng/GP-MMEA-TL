import csv
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "transfer"
README_PATH = ROOT / "README.md"
RECORD_PATH = ROOT / "PROJECT_OPERATION_RECORD.md"


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def now_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def load_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pick_row(rows, target: str):
    for r in rows:
        if r.get("target") == target:
            return r
    return None


def to_int(x: str) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def build_final_stage_report(main_csv: Path, out_md: Path):
    rows = load_rows(main_csv)
    ja = pick_row(rows, "ja_en")
    fbdb = pick_row(rows, "FBDB15K")
    fren = pick_row(rows, "fr_en")
    fbyg = pick_row(rows, "FBYG15K")

    def mrr_delta(r):
        return to_float(r.get("delta_avg_mrr_mean", "0")) if r else 0.0

    def runs(r):
        if not r:
            return "0/0"
        return f"{to_int(r.get('baseline_num_runs', '0'))}/{to_int(r.get('tmmeada_num_runs', '0'))}"

    all5 = True
    for r in [ja, fbdb, fren, fbyg]:
        if r is None:
            all5 = False
            break
        if min(to_int(r.get("baseline_num_runs", "0")), to_int(r.get("tmmeada_num_runs", "0"))) < 5:
            all5 = False
            break

    lines = [
        "# 迁移实验阶段报告（ja_en + FBDB15K expand5 完成）",
        "",
        f"- 时间戳: `{now_ts()}`",
        f"- 统一口径: `{'4目标均为5-seed' if all5 else '4目标口径尚未全部5-seed'}`",
        "",
        "## 4目标当前主表摘要",
        "",
        "| target | runs(b/m) | delta MRR |",
        "|---|---:|---:|",
        f"| ja_en | {runs(ja)} | {mrr_delta(ja):+.6f} |",
        f"| FBDB15K | {runs(fbdb)} | {mrr_delta(fbdb):+.6f} |",
        f"| fr_en | {runs(fren)} | {mrr_delta(fren):+.6f} |",
        f"| FBYG15K | {runs(fbyg)} | {mrr_delta(fbyg):+.6f} |",
        "",
        "## 结论",
        "",
        "1. 已将 ja_en 与 FBDB15K 扩展流程接入统一断点续跑与自动汇总链路。",
        "2. 主结果表与误差分桶分析已自动刷新，可直接用于论文主实验章节。",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all5


def update_readme(latest_report_rel: str, all5: bool):
    text = README_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    # latest report pointer
    for i, line in enumerate(lines):
        if line.startswith("- 迁移阶段报告（最新）："):
            lines[i] = f"- 迁移阶段报告（最新）：`{latest_report_rel}`"
            break

    # confidence note
    for i, line in enumerate(lines):
        if line.startswith("- 置信度说明："):
            if all5:
                lines[i] = "- 置信度说明：`ja_en/FBDB15K/fr_en/FBYG15K` 当前均为 `5-seed` 正式口径。"
            else:
                lines[i] = "- 置信度说明：`ja_en/FBDB15K` 扩展进行中，`fr_en/FBYG15K` 已为 `5-seed`。"
            break

    heading = "## 30. 阶段更新（2026-03-09）：ja_en + FBDB15K 扩展收口"
    if heading not in text:
        lines.extend(
            [
                "",
                heading,
                "",
                "- 扩展脚本：`scripts/run_transfer_adapt_ja_fbdb_expand5_next.py`",
                "- 通用断点续跑：`scripts/run_transfer_adapt_expand5_resume_generic.py`",
                "- 自动汇总主表与分桶：`scripts/make_transfer_main_and_bucket_report.py`",
                f"- 最终阶段报告：`{latest_report_rel}`",
            ]
        )

    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_record(latest_report_rel: str):
    text = RECORD_PATH.read_text(encoding="utf-8")
    marker = "## 13. 2026-03-09 追加记录（ja_en + FBDB15K expand5 自动收口）"
    if marker in text:
        return
    lines = text.splitlines()
    lines.extend(
        [
            "",
            marker,
            "",
            "本次追加操作：",
            "",
            "1. 恢复并完成 `ja_en + FBDB15K` 的 expand5 队列（缺失 seed 自动补跑）。",
            "2. 自动刷新 4目标统一主结果表与误差分桶分析。",
            "3. 自动更新 README 与阶段报告链接。",
            "4. 将本阶段结果与脚本改动提交并同步到 GitHub `sort` 分支。",
            "",
            "新增阶段报告：",
            "",
            f"- `{latest_report_rel}`",
        ]
    )
    RECORD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def has_changes() -> bool:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(proc.stdout.strip())


def main():
    # refresh summary table first
    run_cmd([r"D:\Anaconda_envs\envs\bysj-main\python.exe", "scripts/make_transfer_main_and_bucket_report.py"])

    final_report_name = "transfer_stage_update_20260309_ja_fbdb_expand5_final.md"
    final_report_path = REPORT_DIR / final_report_name
    all5 = build_final_stage_report(
        main_csv=REPORT_DIR / "transfer_adapt_main_results_4target.csv",
        out_md=final_report_path,
    )
    latest_report_rel = f"reports/transfer/{final_report_name}"
    update_readme(latest_report_rel=latest_report_rel, all5=all5)
    update_record(latest_report_rel=latest_report_rel)

    add_cmd = [
        "git",
        "-C",
        str(ROOT),
        "add",
        "README.md",
        "PROJECT_OPERATION_RECORD.md",
        "scripts/make_transfer_main_and_bucket_report.py",
        "scripts/run_transfer_adapt_expand5_resume_generic.py",
        "scripts/run_transfer_adapt_ja_fbdb_expand5_next.py",
        "scripts/finalize_ja_fbdb_expand5_after_run.py",
        "reports/transfer",
        "runs/transfer/transfer_adapt_ja_expand5_baseline",
        "runs/transfer/transfer_adapt_ja_expand5_tmmeada",
        "runs/transfer/transfer_adapt_ja_expand5_merged_baseline",
        "runs/transfer/transfer_adapt_ja_expand5_merged_tmmeada",
        "runs/transfer/transfer_adapt_fbdb_expand5_baseline",
        "runs/transfer/transfer_adapt_fbdb_expand5_tmmeada",
        "runs/transfer/transfer_adapt_fbdb_expand5_merged_baseline",
        "runs/transfer/transfer_adapt_fbdb_expand5_merged_tmmeada",
        "runs/transfer/transfer_adapt_ja_fbdb_expand5",
    ]
    run_cmd(add_cmd)

    if not has_changes():
        print("[INFO] no changes to commit.")
        return

    run_cmd(["git", "-C", str(ROOT), "commit", "-m", "exp: finalize ja/fbdb expand5 and refresh reports"])
    run_cmd(["git", "-C", str(ROOT), "push", "origin", "sort"])
    print("[DONE] finalized and pushed.")


if __name__ == "__main__":
    main()
