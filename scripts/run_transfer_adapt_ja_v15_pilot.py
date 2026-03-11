import argparse
import subprocess
from pathlib import Path

from transfer_adapt_utils import (
    latest_complete_run_for_seed_target_from_roots,
    rebuild_merged_target_eval,
    resolve_source_model_name,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PY = r"D:\Anaconda_envs\envs\bysj-main\python.exe"
MEAFORMER_PY = r"D:\Anaconda_envs\envs\bysj-meaformer\python.exe"

SEEDS = [42, 2026]
TARGET = "ja_en"
SOURCE_MODELS = {
    42: "MEAformer_DBP15K_zh_en_transfer_src_zh_en_epoch10_baseline_transfer_formal_s42_src_s42_",
    2026: "MEAformer_DBP15K_zh_en_transfer_src_zh_en_epoch10_baseline_transfer_adapt_ja_en_expand5_s2026_src_s2026_",
}

SOURCE_CONFIG = "configs/transfer/meaformer_source_zh_en_epoch10.yaml"
TARGET_CONFIG = "configs/transfer_adapt/tmmeada_target_ja_en_v15_refresh4_da0025.yaml"
STAGE_ROOT = "transfer/transfer_adapt_ja_v15_pilot"

BASELINE_MERGED = ROOT / "runs/transfer/transfer_adapt_ja_expand5_merged_baseline/target_eval"
BASELINE_PILOT_REF = ROOT / "runs/transfer/transfer_adapt_ja_v15_pilot_baseline_ref/target_eval"
TMMEADA_PILOT_REF = ROOT / "runs/transfer/transfer_adapt_ja_v15_pilot_tmmeada_ref/target_eval"
TMMEADA_TARGET = ROOT / "runs/transfer/transfer_adapt_ja_v15_pilot/target_eval"

REPORT_PREFIX = ROOT / "reports/transfer/transfer_adapt_ja_v15_pilot_2seed"


def run_cmd(cmd: list[str]):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def summarize():
    baseline_selected, baseline_missing = rebuild_merged_target_eval(
        seeds=SEEDS,
        candidate_roots=[BASELINE_MERGED],
        merged_target_eval=BASELINE_PILOT_REF,
        target=TARGET,
    )
    tmmeada_selected, tmmeada_missing = rebuild_merged_target_eval(
        seeds=SEEDS,
        candidate_roots=[TMMEADA_TARGET],
        merged_target_eval=TMMEADA_PILOT_REF,
        target=TARGET,
    )
    print(f"[INFO] baseline selected seeds: {sorted(int(x) for x in baseline_selected.keys())}")
    print(f"[INFO] baseline missing seeds: {baseline_missing}")
    print(f"[INFO] tmmeada selected seeds: {sorted(int(x) for x in tmmeada_selected.keys())}")
    print(f"[INFO] tmmeada missing seeds: {tmmeada_missing}")
    if not baseline_selected or not tmmeada_selected:
        print("[WARN] skip summarize because one branch has no complete run.")
        return
    run_cmd(
        [
            RUNNER_PY,
            "scripts/summarize_transfer_formal.py",
            "--baseline-target-dir",
            str(BASELINE_PILOT_REF),
            "--tmmeada-target-dir",
            str(TMMEADA_PILOT_REF),
            "--baseline-out",
            str(REPORT_PREFIX) + "_baseline_ref_summary.csv",
            "--tmmeada-out",
            str(REPORT_PREFIX) + "_tmmeada_summary.csv",
            "--compare-out-csv",
            str(REPORT_PREFIX) + "_compare_vs_baseline.csv",
            "--compare-out-md",
            str(REPORT_PREFIX) + "_compare_vs_baseline.md",
        ]
    )


def ensure_seed_runs(run_missing: int):
    for seed in SEEDS:
        existing = latest_complete_run_for_seed_target_from_roots(
            target_eval_roots=[TMMEADA_TARGET],
            seed=seed,
            target=TARGET,
        )
        if existing is not None:
            print(f"[SKIP] seed={seed} already has a complete target run.")
            continue
        if run_missing == 0:
            print(f"[INFO] seed={seed} missing complete run, but run-missing=0 so skip training.")
            continue
        src_model = SOURCE_MODELS.get(seed) or resolve_source_model_name(seed=seed, tmmeada=False)
        if not src_model:
            raise FileNotFoundError(f"source model for seed={seed} not found under data/mmkg/MEAformer/save")
        tag = f"tmmeada_transfer_adapt_ja_v15_pilot_s{seed}"
        run_cmd(
            [
                RUNNER_PY,
                "scripts/run_transfer_train_eval.py",
                "--source-config",
                SOURCE_CONFIG,
                "--target-configs",
                TARGET_CONFIG,
                "--runner-python",
                MEAFORMER_PY,
                "--stage-root",
                STAGE_ROOT,
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


def main():
    parser = argparse.ArgumentParser(description="Run or rebuild the ja_en v15 pilot summary.")
    parser.add_argument(
        "--run-missing",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: run missing pilot seeds; 0: only rebuild merged refs and reports",
    )
    args = parser.parse_args()

    ensure_seed_runs(run_missing=args.run_missing)
    summarize()
    print("[DONE] ja v15 pilot completed.")


if __name__ == "__main__":
    main()
