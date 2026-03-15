from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class CheckRow:
    category: str
    item: str
    status: str
    detail: str


def ok(category: str, item: str, detail: str) -> CheckRow:
    return CheckRow(category=category, item=item, status="OK", detail=detail)


def fail(category: str, item: str, detail: str) -> CheckRow:
    return CheckRow(category=category, item=item, status="FAIL", detail=detail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that current mainline artifacts and official entry points are present."
    )
    parser.add_argument(
        "--out-md",
        default="reports/notes/mainline_artifact_integrity_20260315.md",
        help="Markdown report path, relative to repo root.",
    )
    parser.add_argument(
        "--out-csv",
        default="reports/notes/mainline_artifact_integrity_20260315.csv",
        help="CSV report path, relative to repo root.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check_exists(rows: list[CheckRow], category: str, root: Path, rel_path: str) -> None:
    path = root / rel_path
    if path.exists():
        rows.append(ok(category, rel_path, "present"))
    else:
        rows.append(fail(category, rel_path, "missing"))


def iter_run_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()])


def check_run_bundle(rows: list[CheckRow], category: str, item: str, path: Path, expected_count: int) -> None:
    if not path.exists():
        rows.append(fail(category, item, f"run root missing: {path.as_posix()}"))
        return

    run_dirs = iter_run_dirs(path)
    if len(run_dirs) != expected_count:
        rows.append(
            fail(
                category,
                item,
                f"expected {expected_count} run dirs, found {len(run_dirs)} at {path.as_posix()}",
            )
        )
        return

    required = {"artifact_manifest.json", "config.yaml", "log.txt", "run_card.md"}
    missing_details: list[str] = []
    for run_dir in run_dirs:
        names = {p.name for p in run_dir.iterdir()}
        missing = sorted(required - names)
        if missing:
            missing_details.append(f"{run_dir.name}: missing {', '.join(missing)}")

    if missing_details:
        rows.append(fail(category, item, "; ".join(missing_details)))
    else:
        rows.append(ok(category, item, f"{expected_count} run dirs with required files"))


def check_gpu_variant_runs(rows: list[CheckRow], root: Path) -> None:
    gpu_root = root / "runs/experiments/gpu_peak_minimal"
    variants = ["ja_en_baseline", "ja_en_method", "fbyg15k_baseline", "fbyg15k_method"]
    for variant in variants:
        variant_path = gpu_root / variant
        if not variant_path.exists():
            rows.append(fail("gpu_runs", variant, "variant directory missing"))
            continue
        run_dirs = iter_run_dirs(variant_path)
        if len(run_dirs) != 1:
            rows.append(
                fail("gpu_runs", variant, f"expected 1 run dir, found {len(run_dirs)}")
            )
            continue
        run_dir = run_dirs[0]
        names = {p.name for p in run_dir.iterdir()}
        required = {"artifact_manifest.json", "config.yaml", "log.txt", "run_card.md"}
        missing = sorted(required - names)
        if missing:
            rows.append(fail("gpu_runs", variant, f"missing {', '.join(missing)}"))
        else:
            rows.append(ok("gpu_runs", variant, f"1 run dir ready: {run_dir.name}"))


def check_case_package(rows: list[CheckRow], root: Path) -> None:
    csv_path = root / "reports/transfer/transfer_case_analysis_examples.csv"
    rows_csv = read_csv_rows(csv_path)
    if len(rows_csv) != 8:
        rows.append(fail("case_package", "case_row_count", f"expected 8, found {len(rows_csv)}"))
    else:
        rows.append(ok("case_package", "case_row_count", "8 rows"))

    dataset_case_counter = Counter((r["dataset"], r["case_type"]) for r in rows_csv)
    ja_failures = dataset_case_counter.get(("ja_en", "failure"), 0)
    crossgraph_success = sum(
        1 for r in rows_csv if r["dataset"] in {"FBDB15K", "FBYG15K"} and r["case_type"] == "success"
    )
    rows.append(
        ok(
            "case_package",
            "dataset_breakdown",
            f"ja_en failures={ja_failures}, crossgraph successes={crossgraph_success}",
        )
    )

    expected_entities = {"Fat Mike", "JavaScript"}
    observed_entities = {
        r["source_entity"] for r in rows_csv
    } | {
        r["ground_truth"] for r in rows_csv
    }
    missing_entities = sorted(expected_entities - observed_entities)
    if missing_entities:
        rows.append(
            fail(
                "case_package",
                "thesis_requested_examples",
                f"missing expected examples: {', '.join(missing_entities)}",
            )
        )
    else:
        rows.append(ok("case_package", "thesis_requested_examples", "Fat Mike and JavaScript present"))


def check_gpu_package(rows: list[CheckRow], root: Path) -> None:
    csv_path = root / "reports/transfer/transfer_gpu_peak_minimal_chart_ready.csv"
    rows_csv = read_csv_rows(csv_path)
    if len(rows_csv) != 8:
        rows.append(fail("gpu_package", "gpu_chart_row_count", f"expected 8, found {len(rows_csv)}"))
        return

    rows.append(ok("gpu_package", "gpu_chart_row_count", "8 rows"))
    targets = sorted({r["target"] for r in rows_csv})
    variants = sorted({r["variant"] for r in rows_csv})
    metrics = sorted({r["metric"] for r in rows_csv})
    rows.append(
        ok(
            "gpu_package",
            "gpu_chart_breakdown",
            f"targets={targets}, variants={variants}, metrics={metrics}",
        )
    )


def check_h3_absence(rows: list[CheckRow], root: Path) -> None:
    removed_paths = [
        "scripts/run_h3_missing_modality_minimal.py",
        "scripts/summarize_h3_missing_modality.py",
        "scripts/build_h3_missing_modality_paper_summary.py",
        "reports/robustness",
        "runs/experiments/h3_missing_modality_minimal",
    ]
    unexpected_present: list[str] = []
    for rel_path in removed_paths:
        if (root / rel_path).exists():
            unexpected_present.append(rel_path)

    if unexpected_present:
        rows.append(
            fail(
                "h3_cleanup",
                "removed_artifacts",
                f"unexpectedly present: {', '.join(unexpected_present)}",
            )
        )
    else:
        rows.append(ok("h3_cleanup", "removed_artifacts", "expected H3 paths remain absent"))


def build_report(rows: Iterable[CheckRow]) -> str:
    rows = list(rows)
    total = len(rows)
    ok_count = sum(1 for row in rows if row.status == "OK")
    fail_count = total - ok_count
    lines = [
        "# Mainline Artifact Integrity Report (2026-03-15)",
        "",
        "## Summary",
        "",
        f"- total checks: `{total}`",
        f"- passed: `{ok_count}`",
        f"- failed: `{fail_count}`",
        "",
        "## Check Table",
        "",
        "| Category | Item | Status | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row.category} | {row.item} | {row.status} | {row.detail} |")

    lines.extend(
        [
            "",
            "## Current Reading",
            "",
            "- If all checks pass, the repository currently retains the intended mainline entry points, formal transfer runs, case-analysis supplement, GPU supplement, and H3-removal state.",
            "- This report does not claim that every historical note in the repository is current; it only verifies the concrete current-state artifacts that support the mainline.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    out_md = repo_root / args.out_md
    out_csv = repo_root / args.out_csv

    rows: list[CheckRow] = []

    for rel_path in [
        "README.md",
        "reports/README.md",
        "reports/notes/taskbook_gap_assessment_20260315.md",
        "reports/notes/mainline_traceability_matrix_20260315.md",
        "reports/notes/mainline_closure_onepage_20260315.md",
        "reports/transfer/README.md",
        "runs/README.md",
        "runs/transfer/README.md",
        "reports/transfer/transfer_adapt_main_results_4target.md",
        "reports/transfer/transfer_adapt_main_results_4target.csv",
        "reports/transfer/transfer_adapt_significance_summary.md",
        "reports/transfer/transfer_case_analysis_examples.md",
        "reports/transfer/transfer_case_analysis_examples.csv",
        "reports/transfer/transfer_case_analysis_thesis_sync_20260315.md",
        "reports/transfer/transfer_efficiency_summary.md",
        "reports/transfer/transfer_gpu_peak_minimal_summary.md",
        "reports/transfer/transfer_gpu_peak_minimal_summary.csv",
        "reports/transfer/transfer_gpu_peak_minimal_thesis_sync_20260315.md",
        "reports/transfer/transfer_gpu_peak_minimal_chart_ready.csv",
        "reports/transfer/transfer_extra_baseline_limitation_writeup.md",
    ]:
        check_exists(rows, "files", repo_root, rel_path)

    check_run_bundle(
        rows,
        "formal_runs",
        "ja_en_baseline",
        repo_root / "runs/transfer/transfer_adapt_ja_v15_full_baseline_matched_ref/target_eval",
        expected_count=5,
    )
    check_run_bundle(
        rows,
        "formal_runs",
        "ja_en_method",
        repo_root / "runs/transfer/transfer_adapt_ja_v15_full_ref/target_eval",
        expected_count=5,
    )
    check_run_bundle(
        rows,
        "formal_runs",
        "fr_en_baseline",
        repo_root / "runs/transfer/transfer_adapt_v14_fren_expand5_merged_baseline/target_eval",
        expected_count=5,
    )
    check_run_bundle(
        rows,
        "formal_runs",
        "fr_en_method",
        repo_root / "runs/transfer/transfer_adapt_v14_fren_expand5_merged_tmmeada/target_eval",
        expected_count=5,
    )
    check_run_bundle(
        rows,
        "formal_runs",
        "fbdb_baseline",
        repo_root / "runs/transfer/transfer_adapt_v18_fbdb_v18c_expand5_baseline_matched_ref/target_eval",
        expected_count=5,
    )
    check_run_bundle(
        rows,
        "formal_runs",
        "fbdb_method",
        repo_root / "runs/transfer/transfer_adapt_v18_fbdb_v18c_expand5_ref/target_eval",
        expected_count=5,
    )
    check_run_bundle(
        rows,
        "formal_runs",
        "fbyg_baseline",
        repo_root / "runs/transfer/transfer_adapt_v24_fbyg_v24b_expand5_baseline_matched_ref/target_eval",
        expected_count=5,
    )
    check_run_bundle(
        rows,
        "formal_runs",
        "fbyg_method",
        repo_root / "runs/transfer/transfer_adapt_v24_fbyg_v24b_expand5_ref/target_eval",
        expected_count=5,
    )

    check_case_package(rows, repo_root)
    check_gpu_package(rows, repo_root)
    check_gpu_variant_runs(rows, repo_root)
    check_h3_absence(rows, repo_root)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_report(rows), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "item", "status", "detail"])
        for row in rows:
            writer.writerow([row.category, row.item, row.status, row.detail])

    total = len(rows)
    failed = sum(1 for row in rows if row.status == "FAIL")
    print(f"wrote {out_md.as_posix()} and {out_csv.as_posix()} ({total} checks, {failed} failed)")


if __name__ == "__main__":
    main()
