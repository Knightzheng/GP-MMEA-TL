import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "transfer"

ELAPSED_RE = re.compile(r" - (\d+:\d{2}:\d{2}) - ")
DONE_RE = re.compile(r"\[DONE\]\s+return_code=(\d+)")


@dataclass(frozen=True)
class RunGroup:
    dataset: str
    role: str
    run_dir: Path


RUN_GROUPS = [
    RunGroup("ja_en", "baseline", ROOT / "runs/transfer/transfer_adapt_ja_v15_full_baseline_matched_ref/target_eval"),
    RunGroup("ja_en", "method", ROOT / "runs/transfer/transfer_adapt_ja_v15_full_ref/target_eval"),
    RunGroup("FBDB15K", "baseline", ROOT / "runs/transfer/transfer_adapt_v18_fbdb_v18c_expand5_baseline_matched_ref/target_eval"),
    RunGroup("FBDB15K", "method", ROOT / "runs/transfer/transfer_adapt_v18_fbdb_v18c_expand5_ref/target_eval"),
    RunGroup("fr_en", "baseline", ROOT / "runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval"),
    RunGroup("fr_en", "method", ROOT / "runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval"),
    RunGroup("FBYG15K", "baseline", ROOT / "runs/transfer/transfer_adapt_v24_fbyg_v24b_expand5_baseline_matched_ref/target_eval"),
    RunGroup("FBYG15K", "method", ROOT / "runs/transfer/transfer_adapt_v24_fbyg_v24b_expand5_ref/target_eval"),
]


def elapsed_to_minutes(text: str) -> float:
    hours, minutes, seconds = [int(part) for part in text.split(":")]
    return hours * 60 + minutes + seconds / 60.0


def parse_log(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    done_match = DONE_RE.search(text)
    if not done_match or done_match.group(1) != "0":
        raise ValueError(f"run not successful: {log_path}")
    matches = ELAPSED_RE.findall(text)
    if not matches:
        raise ValueError(f"no elapsed time found: {log_path}")
    return elapsed_to_minutes(matches[-1])


def summarize_group(group: RunGroup) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in sorted(group.run_dir.iterdir()):
        log_path = item / "log.txt"
        if not log_path.exists():
            continue
        elapsed_min = parse_log(log_path)
        rows.append(
            {
                "dataset": group.dataset,
                "role": group.role,
                "run_name": item.name,
                "elapsed_min": f"{elapsed_min:.2f}",
            }
        )
    return rows


def write_csv(per_run: List[Dict[str, str]], summary: List[Dict[str, str]]) -> None:
    with (REPORT_DIR / "transfer_efficiency_per_run.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "role", "run_name", "elapsed_min"])
        writer.writeheader()
        writer.writerows(per_run)
    with (REPORT_DIR / "transfer_efficiency_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "baseline_mean_min",
                "method_mean_min",
                "delta_min",
                "overhead_ratio",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)


def write_markdown(summary: List[Dict[str, str]]) -> None:
    lines = [
        "# Transfer Efficiency Summary",
        "",
        "## Paper-Ready Table",
        "",
        "| Target | Baseline Time (min, mean of 5 seeds) | Ours Time (min, mean of 5 seeds) | Delta (min) | Overhead | Note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['baseline_mean_min']} | {row['method_mean_min']} | "
            f"{row['delta_min']} | {row['overhead_ratio']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Current repository logs are sufficient to summarize wall-clock time for the formal 5-seed chains.",
            "- Peak GPU memory is not logged consistently in the existing runs, so it still requires one minimal-cost补测 if the thesis needs a complete time-memory comparison table.",
            "- Recommended wording: report wall-clock time from completed logs as the primary efficiency indicator, and state GPU memory as supplementary measurement when available.",
            "",
        ]
    )
    (REPORT_DIR / "transfer_efficiency_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per_run: List[Dict[str, str]] = []
    grouped: Dict[str, Dict[str, List[float]]] = {}

    for group in RUN_GROUPS:
        rows = summarize_group(group)
        per_run.extend(rows)
        grouped.setdefault(group.dataset, {}).setdefault(group.role, [])
        grouped[group.dataset][group.role] = [float(row["elapsed_min"]) for row in rows]

    summary: List[Dict[str, str]] = []
    for dataset in ["ja_en", "FBDB15K", "fr_en", "FBYG15K"]:
        baseline = grouped[dataset]["baseline"]
        method = grouped[dataset]["method"]
        baseline_mean = sum(baseline) / len(baseline)
        method_mean = sum(method) / len(method)
        delta = method_mean - baseline_mean
        ratio = method_mean / baseline_mean if baseline_mean else 0.0
        summary.append(
            {
                "dataset": dataset,
                "baseline_mean_min": f"{baseline_mean:.2f}",
                "method_mean_min": f"{method_mean:.2f}",
                "delta_min": f"{delta:+.2f}",
                "overhead_ratio": f"{ratio:.2f}x",
                "note": "wall-clock only; GPU peak memory not fully logged",
            }
        )

    write_csv(per_run, summary)
    write_markdown(summary)
    print(f"[OK] wrote {len(per_run)} per-run rows and {len(summary)} summary rows")


if __name__ == "__main__":
    main()
