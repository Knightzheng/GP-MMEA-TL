import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

from transfer_adapt_utils import (
    latest_complete_run_for_seed_target_from_roots,
    rebuild_merged_target_eval,
    resolve_source_model_name,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PY = r"D:\Anaconda_envs\envs\bysj-main\python.exe"
MEAFORMER_PY = r"D:\Anaconda_envs\envs\bysj-meaformer\python.exe"

TARGET = "FBYG15K"
SOURCE_CONFIG = "configs/transfer/meaformer_source_zh_en_epoch10.yaml"
BASELINE_MERGED_TARGET_EVAL = ROOT / "runs/transfer/transfer_adapt_fbyg_expand5_merged_baseline/target_eval"
CURRENT_BEST_COMPARE_CSV = ROOT / "reports/transfer/transfer_adapt_fbyg_expand5_progress_compare_vs_baseline.csv"

DEFAULT_PILOT_SEEDS = [42, 2026]
DEFAULT_FULL_SEEDS = [42, 3407, 2026, 7, 123]

VARIANTS = {
    "v19a": {
        "config": "configs/transfer_adapt/tmmeada_target_fbyg15k_v19a_late_il_strict.yaml",
        "stage_root": "transfer/transfer_adapt_v19_fbyg_pilot_v19a",
        "tag_prefix": "tmmeada_transfer_adapt_v19_fbyg_v19a",
        "pilot_ref_dir": ROOT / "runs/transfer/transfer_adapt_v19_fbyg_v19a_pilot_ref/target_eval",
        "full_ref_dir": ROOT / "runs/transfer/transfer_adapt_v19_fbyg_v19a_expand5_ref/target_eval",
        "transfer_skip_keys": "multimodal_encoder.entity_emb.weight",
        "transfer_skip_prefixes": "",
    },
    "v19b": {
        "config": "configs/transfer_adapt/tmmeada_target_fbyg15k_v19b_late_il_skiprel.yaml",
        "stage_root": "transfer/transfer_adapt_v19_fbyg_pilot_v19b",
        "tag_prefix": "tmmeada_transfer_adapt_v19_fbyg_v19b",
        "pilot_ref_dir": ROOT / "runs/transfer/transfer_adapt_v19_fbyg_v19b_pilot_ref/target_eval",
        "full_ref_dir": ROOT / "runs/transfer/transfer_adapt_v19_fbyg_v19b_expand5_ref/target_eval",
        "transfer_skip_keys": (
            "multimodal_encoder.entity_emb.weight,"
            "multimodal_encoder.rel_fc.weight,"
            "multimodal_encoder.rel_fc.bias"
        ),
        "transfer_skip_prefixes": "",
    },
    "v19c": {
        "config": "configs/transfer_adapt/tmmeada_target_fbyg15k_v19c_late_il_skiprel_skipfusion.yaml",
        "stage_root": "transfer/transfer_adapt_v19_fbyg_pilot_v19c",
        "tag_prefix": "tmmeada_transfer_adapt_v19_fbyg_v19c",
        "pilot_ref_dir": ROOT / "runs/transfer/transfer_adapt_v19_fbyg_v19c_pilot_ref/target_eval",
        "full_ref_dir": ROOT / "runs/transfer/transfer_adapt_v19_fbyg_v19c_expand5_ref/target_eval",
        "transfer_skip_keys": (
            "multimodal_encoder.entity_emb.weight,"
            "multimodal_encoder.rel_fc.weight,"
            "multimodal_encoder.rel_fc.bias"
        ),
        "transfer_skip_prefixes": "multimodal_encoder.fusion.",
    },
}


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_cmd(cmd: list[str]):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_seeds(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def ensure_variant_runs(variant_key: str, seeds: list[int], run_missing: int):
    info = VARIANTS[variant_key]
    cfg = info["config"]
    stage_root = info["stage_root"]
    tag_prefix = info["tag_prefix"]
    target_eval = ROOT / "runs" / stage_root / "target_eval"

    for seed in seeds:
        existing = latest_complete_run_for_seed_target_from_roots(
            target_eval_roots=[target_eval],
            seed=seed,
            target=TARGET,
        )
        if existing is not None:
            print(f"[SKIP] {variant_key} seed={seed} already has a complete run.")
            continue
        if run_missing == 0:
            print(f"[INFO] {variant_key} seed={seed} missing complete run, but run-missing=0 so skip training.")
            continue

        src_model = resolve_source_model_name(seed=seed, tmmeada=False)
        if not src_model:
            raise FileNotFoundError(f"source model for seed={seed} not found under data/mmkg/MEAformer/save")
        tag = f"{tag_prefix}_s{seed}"
        cmd = [
            RUNNER_PY,
            "scripts/run_transfer_train_eval.py",
            "--source-config",
            SOURCE_CONFIG,
            "--target-configs",
            cfg,
            "--runner-python",
            MEAFORMER_PY,
            "--stage-root",
            stage_root,
            "--tag",
            tag,
            "--seed",
            str(seed),
            "--target-only-test",
            "0",
            "--target-save-model",
            "0",
            "--source-model-name",
            src_model,
            "--transfer-skip-keys",
            info["transfer_skip_keys"],
        ]
        if info["transfer_skip_prefixes"]:
            cmd.extend(["--transfer-skip-prefixes", info["transfer_skip_prefixes"]])
        run_cmd(cmd)


def summarize_compare(prefix: str, baseline_target_dir: Path, tmmeada_target_dir: Path) -> Path:
    run_cmd(
        [
            RUNNER_PY,
            "scripts/summarize_transfer_formal.py",
            "--baseline-target-dir",
            str(baseline_target_dir),
            "--tmmeada-target-dir",
            str(tmmeada_target_dir),
            "--baseline-out",
            f"reports/transfer/{prefix}_baseline_ref_summary.csv",
            "--tmmeada-out",
            f"reports/transfer/{prefix}_tmmeada_summary.csv",
            "--compare-out-csv",
            f"reports/transfer/{prefix}_compare_vs_baseline.csv",
            "--compare-out-md",
            f"reports/transfer/{prefix}_compare_vs_baseline.md",
        ]
    )
    return ROOT / f"reports/transfer/{prefix}_compare_vs_baseline.csv"


def read_delta_mrr(compare_csv: Path, target: str = TARGET) -> float:
    with compare_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("target") == target:
            return float(row.get("delta_avg_mrr_mean", "0"))
    raise ValueError(f"target={target} not found in {compare_csv}")


def read_ref_delta(compare_csv: Path, target: str = TARGET):
    if not compare_csv.exists():
        return None
    try:
        return read_delta_mrr(compare_csv, target=target)
    except Exception:
        return None


def rebuild_variant_subset(variant_key: str, seeds: list[int], dst_target_eval: Path):
    info = VARIANTS[variant_key]
    return rebuild_merged_target_eval(
        seeds=seeds,
        candidate_roots=[ROOT / "runs" / info["stage_root"] / "target_eval"],
        merged_target_eval=dst_target_eval,
        target=TARGET,
    )


def seed_list_from_selected(selected: dict[str, str]) -> list[int]:
    return sorted(int(x) for x in selected.keys())


def main():
    parser = argparse.ArgumentParser(description="Iterate FBYG15K v19 variants with resumable pilot/full5 summaries.")
    parser.add_argument(
        "--run-missing",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: run missing seeds; 0: only rebuild refs and reports",
    )
    parser.add_argument("--pilot-seeds", default="42,2026")
    parser.add_argument("--full-seeds", default="42,3407,2026,7,123")
    parser.add_argument(
        "--expand-threshold",
        type=float,
        default=0.0005,
        help="expand to full-5 only when pilot best exceeds current ref by this delta on MRR",
    )
    args = parser.parse_args()

    pilot_seeds = parse_seeds(args.pilot_seeds) if args.pilot_seeds else list(DEFAULT_PILOT_SEEDS)
    full_seeds = parse_seeds(args.full_seeds) if args.full_seeds else list(DEFAULT_FULL_SEEDS)
    ref_delta = read_ref_delta(CURRENT_BEST_COMPARE_CSV)

    records = {
        "timestamp": now_ts(),
        "target": TARGET,
        "pilot_seeds": pilot_seeds,
        "full_seeds": full_seeds,
        "run_missing": args.run_missing,
        "reference_compare_csv": str(CURRENT_BEST_COMPARE_CSV),
        "reference_delta_avg_mrr_mean": ref_delta,
        "expand_threshold": args.expand_threshold,
        "variant_results": {},
        "decision": {},
    }

    baseline_pilot_all_ref = ROOT / "runs/transfer/transfer_adapt_v19_fbyg_pilot_baseline_ref/target_eval"
    _, baseline_pilot_missing = rebuild_merged_target_eval(
        seeds=pilot_seeds,
        candidate_roots=[BASELINE_MERGED_TARGET_EVAL],
        merged_target_eval=baseline_pilot_all_ref,
        target=TARGET,
    )

    pilot_scores = {}
    for key in ["v19a", "v19b", "v19c"]:
        ensure_variant_runs(key, pilot_seeds, run_missing=args.run_missing)
        selected, missing = rebuild_variant_subset(key, pilot_seeds, VARIANTS[key]["pilot_ref_dir"])
        selected_seeds = seed_list_from_selected(selected)

        compare_csv = None
        delta = None
        baseline_variant_ref = ROOT / "runs/transfer" / f"transfer_adapt_v19_fbyg_{key}_pilot_baseline_matched_ref" / "target_eval"
        if selected_seeds:
            baseline_pilot_selected, baseline_pilot_compare_missing = rebuild_merged_target_eval(
                seeds=selected_seeds,
                candidate_roots=[BASELINE_MERGED_TARGET_EVAL],
                merged_target_eval=baseline_variant_ref,
                target=TARGET,
            )
        else:
            baseline_pilot_selected = {}
            baseline_pilot_compare_missing = []

        if baseline_pilot_selected and selected:
            compare_csv_path = summarize_compare(
                prefix=f"transfer_adapt_v19_fbyg_pilot_{key}",
                baseline_target_dir=baseline_variant_ref,
                tmmeada_target_dir=VARIANTS[key]["pilot_ref_dir"],
            )
            compare_csv = str(compare_csv_path)
            delta = read_delta_mrr(compare_csv_path)

        is_complete_pilot = (not baseline_pilot_missing) and (not missing)
        if is_complete_pilot and delta is not None:
            pilot_scores[key] = delta

        records["variant_results"][f"{key}_pilot"] = {
            "delta_avg_mrr_mean": delta,
            "compare_csv": compare_csv,
            "selected_seeds": selected_seeds,
            "missing_seeds": missing,
            "baseline_missing_seeds": baseline_pilot_missing,
            "baseline_compare_missing_seeds": baseline_pilot_compare_missing,
            "is_complete_pilot": is_complete_pilot,
            "transfer_skip_keys": VARIANTS[key]["transfer_skip_keys"],
            "transfer_skip_prefixes": VARIANTS[key]["transfer_skip_prefixes"],
        }

    best_variant = None
    best_delta = None
    improve_over_ref = None
    if pilot_scores:
        best_variant = max(pilot_scores, key=lambda k: pilot_scores[k])
        best_delta = pilot_scores[best_variant]
        if ref_delta is not None:
            improve_over_ref = best_delta - ref_delta

    records["decision"] = {
        "best_variant_pilot": best_variant,
        "best_delta_avg_mrr_mean": best_delta,
        "improve_over_current_ref": improve_over_ref,
        "reference_delta_avg_mrr_mean": ref_delta,
    }

    should_expand = (
        best_variant is not None
        and best_delta is not None
        and ref_delta is not None
        and improve_over_ref is not None
        and improve_over_ref >= args.expand_threshold
    )

    if should_expand:
        ensure_variant_runs(best_variant, full_seeds, run_missing=args.run_missing)
        best_selected, best_missing = rebuild_variant_subset(
            best_variant,
            full_seeds,
            VARIANTS[best_variant]["full_ref_dir"],
        )
        best_selected_seeds = seed_list_from_selected(best_selected)
        if best_selected_seeds:
            best_baseline_ref = ROOT / "runs/transfer" / f"transfer_adapt_v19_fbyg_{best_variant}_expand5_baseline_matched_ref" / "target_eval"
            best_baseline_selected, best_baseline_compare_missing = rebuild_merged_target_eval(
                seeds=best_selected_seeds,
                candidate_roots=[BASELINE_MERGED_TARGET_EVAL],
                merged_target_eval=best_baseline_ref,
                target=TARGET,
            )
        else:
            best_baseline_ref = ROOT / "runs/transfer" / f"transfer_adapt_v19_fbyg_{best_variant}_expand5_baseline_matched_ref" / "target_eval"
            best_baseline_selected = {}
            best_baseline_compare_missing = []

        if best_baseline_selected and best_selected:
            compare_csv = summarize_compare(
                prefix=f"transfer_adapt_v19_fbyg_{best_variant}_expand5",
                baseline_target_dir=best_baseline_ref,
                tmmeada_target_dir=VARIANTS[best_variant]["full_ref_dir"],
            )
            expanded_delta = read_delta_mrr(compare_csv)
            expanded_csv = str(compare_csv)
        else:
            expanded_delta = None
            expanded_csv = None

        records["decision"]["expanded_variant_to_full5"] = best_variant
        records["decision"]["expanded_variant_full5_delta_avg_mrr_mean"] = expanded_delta
        records["decision"]["expanded_variant_full5_compare_csv"] = expanded_csv
        records["decision"]["expanded_variant_full5_missing_seeds"] = best_missing
        records["decision"]["expanded_variant_full5_baseline_compare_missing_seeds"] = best_baseline_compare_missing
    else:
        records["decision"]["expanded_variant_to_full5"] = None

    out_json = ROOT / "reports/transfer/transfer_adapt_v19_fbyg_iter_decision.json"
    out_md = ROOT / "reports/transfer/transfer_adapt_v19_fbyg_iter_decision.md"
    out_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# FBYG15K v19 Iteration Decision",
        "",
        f"- timestamp: `{records['timestamp']}`",
        f"- pilot_seeds: `{pilot_seeds}`",
        f"- full_seeds: `{full_seeds}`",
        f"- reference_delta_avg_mrr_mean(v8_expand5): `{ref_delta}`",
        f"- best_variant_pilot: `{best_variant}`",
        f"- best_delta_avg_mrr_mean: `{best_delta}`",
        f"- improve_over_current_ref: `{improve_over_ref}`",
        f"- expand_threshold: `{args.expand_threshold}`",
        f"- expanded_variant_to_full5: `{records['decision']['expanded_variant_to_full5']}`",
        "",
        "## Pilot Summary",
        "",
        "| variant | delta_avg_mrr_mean | selected_seeds | transfer_skip_keys | transfer_skip_prefixes |",
        "|---|---:|---|---|---|",
    ]
    for key in ["v19a", "v19b", "v19c"]:
        item = records["variant_results"].get(f"{key}_pilot", {})
        lines.append(
            f"| {key} | {item.get('delta_avg_mrr_mean')} | {item.get('selected_seeds')} | {item.get('transfer_skip_keys')} | {item.get('transfer_skip_prefixes')} |"
        )
    lines.append("")
    if records["decision"]["expanded_variant_to_full5"] is not None:
        lines.append("## Full-5 Expansion")
        lines.append("")
        lines.append(
            f"- compare_csv: `{records['decision'].get('expanded_variant_full5_compare_csv')}`"
        )
        lines.append(
            f"- delta_avg_mrr_mean: `{records['decision'].get('expanded_variant_full5_delta_avg_mrr_mean')}`"
        )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] decision json -> {out_json}")
    print(f"[DONE] decision md -> {out_md}")


if __name__ == "__main__":
    main()
