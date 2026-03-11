import csv
import json
import argparse
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

SOURCE_CONFIG = "configs/transfer/meaformer_source_zh_en_epoch10.yaml"
BASELINE_MERGED_TARGET_EVAL = ROOT / "runs/transfer/transfer_adapt_ja_expand5_merged_baseline/target_eval"
TARGET = "ja_en"

FULL_SEEDS = [42, 3407, 2026, 7, 123]
PILOT_SEEDS = [42, 2026]
SOURCE_MODELS = {
    42: "MEAformer_DBP15K_zh_en_transfer_src_zh_en_epoch10_baseline_transfer_formal_s42_src_s42_",
    3407: "MEAformer_DBP15K_zh_en_transfer_src_zh_en_epoch10_baseline_transfer_formal_s3407_src_s3407_",
    2026: "MEAformer_DBP15K_zh_en_transfer_src_zh_en_epoch10_baseline_transfer_adapt_ja_en_expand5_s2026_src_s2026_",
    7: "MEAformer_DBP15K_zh_en_transfer_src_zh_en_epoch10_baseline_transfer_adapt_ja_en_expand5_s7_src_s7_",
    123: "MEAformer_DBP15K_zh_en_transfer_src_zh_en_epoch10_baseline_transfer_adapt_ja_en_expand5_s123_src_s123_",
}

VARIANTS = {
    "v15": {
        "config": "configs/transfer_adapt/tmmeada_target_ja_en_v15_refresh4_da0025.yaml",
        "stage_root": "transfer/transfer_adapt_ja_v15_pilot",
        "tag_prefix": "tmmeada_transfer_adapt_ja_v15",
        "pilot_ref_dir": ROOT / "runs/transfer/transfer_adapt_ja_v15_pilot_ref/target_eval",
        "full_ref_dir": ROOT / "runs/transfer/transfer_adapt_ja_v15_full_ref/target_eval",
    },
    "v15a": {
        "config": "configs/transfer_adapt/tmmeada_target_ja_en_v15a_refresh4_da0020.yaml",
        "stage_root": "transfer/transfer_adapt_ja_v15a_pilot2seed",
        "tag_prefix": "tmmeada_transfer_adapt_ja_v15a",
        "pilot_ref_dir": ROOT / "runs/transfer/transfer_adapt_ja_v15a_pilot2seed_ref/target_eval",
        "full_ref_dir": ROOT / "runs/transfer/transfer_adapt_ja_v15a_expand5_ref/target_eval",
    },
    "v15b": {
        "config": "configs/transfer_adapt/tmmeada_target_ja_en_v15b_refresh4_da0030.yaml",
        "stage_root": "transfer/transfer_adapt_ja_v15b_pilot2seed",
        "tag_prefix": "tmmeada_transfer_adapt_ja_v15b",
        "pilot_ref_dir": ROOT / "runs/transfer/transfer_adapt_ja_v15b_pilot2seed_ref/target_eval",
        "full_ref_dir": ROOT / "runs/transfer/transfer_adapt_ja_v15b_expand5_ref/target_eval",
    },
    "v15c": {
        "config": "configs/transfer_adapt/tmmeada_target_ja_en_v15c_refresh3_da0025.yaml",
        "stage_root": "transfer/transfer_adapt_ja_v15c_pilot2seed",
        "tag_prefix": "tmmeada_transfer_adapt_ja_v15c",
        "pilot_ref_dir": ROOT / "runs/transfer/transfer_adapt_ja_v15c_pilot2seed_ref/target_eval",
        "full_ref_dir": ROOT / "runs/transfer/transfer_adapt_ja_v15c_expand5_ref/target_eval",
    },
}


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_cmd(cmd: list[str]):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


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

        src_model = SOURCE_MODELS.get(seed) or resolve_source_model_name(seed=seed, tmmeada=False)
        if not src_model:
            raise FileNotFoundError(f"source model for seed={seed} not found under data/mmkg/MEAformer/save")
        tag = f"{tag_prefix}_s{seed}"
        run_cmd(
            [
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
            ]
        )


def summarize_compare(prefix: str, baseline_target_dir: Path, tmmeada_target_dir: Path):
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


def read_delta_mrr(compare_csv: Path, target: str = "ja_en") -> float:
    with compare_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("target") == target:
            return float(row.get("delta_avg_mrr_mean", "0"))
    raise ValueError(f"target={target} not found in {compare_csv}")


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
    parser = argparse.ArgumentParser(description="Iterate ja_en v15 variants with resumable summaries.")
    parser.add_argument(
        "--run-missing",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: run missing seeds; 0: only rebuild refs and reports",
    )
    parser.add_argument(
        "--v15-full-only",
        action="store_true",
        help="only complete/rebuild v15 full-seed status, skip v15a/v15b/v15c pilot variants",
    )
    args = parser.parse_args()

    records = {
        "timestamp": now_ts(),
        "variant_results": {},
        "decision": {},
    }

    # A) v15 expand to full 5-seed
    ensure_variant_runs("v15", FULL_SEEDS, run_missing=args.run_missing)
    baseline_full_all_ref = ROOT / "runs/transfer/transfer_adapt_ja_v15_full_baseline_ref/target_eval"
    baseline_full_all_selected, baseline_full_missing = rebuild_merged_target_eval(
        seeds=FULL_SEEDS,
        candidate_roots=[BASELINE_MERGED_TARGET_EVAL],
        merged_target_eval=baseline_full_all_ref,
        target=TARGET,
    )
    v15_full_selected, v15_full_missing = rebuild_variant_subset("v15", FULL_SEEDS, VARIANTS["v15"]["full_ref_dir"])
    v15_full_compare_seeds = seed_list_from_selected(v15_full_selected)
    v15_full_delta = None
    v15_full_compare_csv = None
    if v15_full_compare_seeds:
        baseline_full_ref = ROOT / "runs/transfer/transfer_adapt_ja_v15_full_baseline_matched_ref/target_eval"
        baseline_full_selected, baseline_full_compare_missing = rebuild_merged_target_eval(
            seeds=v15_full_compare_seeds,
            candidate_roots=[BASELINE_MERGED_TARGET_EVAL],
            merged_target_eval=baseline_full_ref,
            target=TARGET,
        )
    else:
        baseline_full_ref = ROOT / "runs/transfer/transfer_adapt_ja_v15_full_baseline_matched_ref/target_eval"
        baseline_full_selected = {}
        baseline_full_compare_missing = []
    if baseline_full_selected and v15_full_selected:
        v15_full_csv = summarize_compare(
            prefix="transfer_adapt_ja_v15_expand5",
            baseline_target_dir=baseline_full_ref,
            tmmeada_target_dir=VARIANTS["v15"]["full_ref_dir"],
        )
        v15_full_delta = read_delta_mrr(v15_full_csv)
        v15_full_compare_csv = str(v15_full_csv)
    records["variant_results"]["v15_full5"] = {
        "delta_avg_mrr_mean": v15_full_delta,
        "compare_csv": v15_full_compare_csv,
        "selected_seeds": v15_full_compare_seeds,
        "matched_compare_seeds": v15_full_compare_seeds,
        "missing_seeds": v15_full_missing,
        "baseline_missing_seeds": baseline_full_missing,
        "baseline_compare_missing_seeds": baseline_full_compare_missing,
    }

    # B) 2-seed variant pilots
    baseline_pilot_all_ref = ROOT / "runs/transfer/transfer_adapt_ja_v15_pilot_baseline_ref_2seed_iter/target_eval"
    baseline_pilot_all_selected, baseline_pilot_missing = rebuild_merged_target_eval(
        seeds=PILOT_SEEDS,
        candidate_roots=[BASELINE_MERGED_TARGET_EVAL],
        merged_target_eval=baseline_pilot_all_ref,
        target=TARGET,
    )

    pilot_scores = {}
    pilot_variant_keys = ["v15"] if args.v15_full_only else ["v15", "v15a", "v15b", "v15c"]
    for key in pilot_variant_keys:
        ensure_variant_runs(key, PILOT_SEEDS, run_missing=args.run_missing)
        selected, missing = rebuild_variant_subset(key, PILOT_SEEDS, VARIANTS[key]["pilot_ref_dir"])
        selected_seeds = seed_list_from_selected(selected)
        prefix = f"transfer_adapt_ja_{key}_pilot2seed"
        compare_csv = None
        delta = None
        baseline_variant_ref = ROOT / "runs/transfer" / f"transfer_adapt_ja_{key}_pilot2seed_baseline_matched_ref" / "target_eval"
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
                prefix=prefix,
                baseline_target_dir=baseline_variant_ref,
                tmmeada_target_dir=VARIANTS[key]["pilot_ref_dir"],
            )
            compare_csv = str(compare_csv_path)
            delta = read_delta_mrr(compare_csv_path)
        is_complete_pilot = (not baseline_pilot_missing) and (not missing)
        if is_complete_pilot and delta is not None:
            pilot_scores[key] = delta
        records["variant_results"][f"{key}_pilot2"] = {
            "delta_avg_mrr_mean": delta,
            "compare_csv": compare_csv,
            "selected_seeds": selected_seeds,
            "matched_compare_seeds": selected_seeds,
            "missing_seeds": missing,
            "baseline_missing_seeds": baseline_pilot_missing,
            "baseline_compare_missing_seeds": baseline_pilot_compare_missing,
            "is_complete_pilot": is_complete_pilot,
        }

    if args.v15_full_only:
        for key in ["v15a", "v15b", "v15c"]:
            records["variant_results"][f"{key}_pilot2"] = {
                "delta_avg_mrr_mean": None,
                "compare_csv": None,
                "selected_seeds": [],
                "matched_compare_seeds": [],
                "missing_seeds": PILOT_SEEDS,
                "baseline_missing_seeds": baseline_pilot_missing,
                "baseline_compare_missing_seeds": [],
                "is_complete_pilot": False,
                "skipped_by_flag": True,
            }

    base_variant = "v15"
    best_variant = None
    improve_over_v15 = None
    if pilot_scores:
        best_variant = max(pilot_scores, key=lambda k: pilot_scores[k])
        if base_variant in pilot_scores:
            improve_over_v15 = pilot_scores[best_variant] - pilot_scores[base_variant]
    records["decision"] = {
        "best_variant_pilot2": best_variant,
        "best_delta_avg_mrr_mean": pilot_scores.get(best_variant) if best_variant else None,
        "v15_delta_avg_mrr_mean": pilot_scores.get(base_variant),
        "improve_over_v15": improve_over_v15,
    }

    # C) Optional full-5 expansion for best variant if clearly better
    if best_variant is not None and best_variant != "v15" and improve_over_v15 is not None and improve_over_v15 > 0.001:
        ensure_variant_runs(best_variant, FULL_SEEDS, run_missing=args.run_missing)
        best_selected, best_missing = rebuild_variant_subset(
            best_variant,
            FULL_SEEDS,
            VARIANTS[best_variant]["full_ref_dir"],
        )
        best_selected_seeds = seed_list_from_selected(best_selected)
        if best_selected_seeds:
            best_baseline_ref = ROOT / "runs/transfer" / f"transfer_adapt_ja_{best_variant}_expand5_baseline_matched_ref" / "target_eval"
            best_baseline_selected, best_baseline_compare_missing = rebuild_merged_target_eval(
                seeds=best_selected_seeds,
                candidate_roots=[BASELINE_MERGED_TARGET_EVAL],
                merged_target_eval=best_baseline_ref,
                target=TARGET,
            )
        else:
            best_baseline_ref = ROOT / "runs/transfer" / f"transfer_adapt_ja_{best_variant}_expand5_baseline_matched_ref" / "target_eval"
            best_baseline_selected = {}
            best_baseline_compare_missing = []
        if best_baseline_selected and best_selected:
            compare_csv = summarize_compare(
                prefix=f"transfer_adapt_ja_{best_variant}_expand5",
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

    # D) Write decision report
    out_json = ROOT / "reports/transfer/transfer_adapt_ja_v15_iter_decision.json"
    out_md = ROOT / "reports/transfer/transfer_adapt_ja_v15_iter_decision.md"
    out_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# ja_en v15 Iteration Decision",
        "",
        f"- timestamp: `{records['timestamp']}`",
        f"- v15 current matched-seed delta MRR: `{v15_full_delta:+.6f}`" if v15_full_delta is not None else "- v15 current matched-seed delta MRR: `N/A`",
        f"- v15 current compare seeds: `{v15_full_compare_seeds}`",
        f"- v15 full5 missing seeds: `{v15_full_missing}`",
        f"- pilot best variant: `{best_variant}`",
        f"- pilot best delta MRR: `{pilot_scores[best_variant]:+.6f}`" if best_variant in pilot_scores else "- pilot best delta MRR: `N/A`",
        f"- improve over v15 pilot: `{improve_over_v15:+.6f}`" if improve_over_v15 is not None else "- improve over v15 pilot: `N/A`",
        f"- expanded to full5: `{records['decision']['expanded_variant_to_full5']}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] decision json -> {out_json}")
    print(f"[DONE] decision md -> {out_md}")


if __name__ == "__main__":
    main()
