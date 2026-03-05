import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def resolve_source_model_name(seed: int, tmmeada: bool) -> str:
    save_dir = Path("data/mmkg/MEAformer/save")
    if not save_dir.exists():
        return ""

    if tmmeada:
        exact = save_dir / (
            "MEAformer_DBP15K_zh_en_"
            f"tmmeada_transfer_src_zh_en_epoch10_tmmeada_transfer_formal_s{seed}_src_s{seed}_.pkl"
        )
        if exact.exists():
            return exact.stem
        pattern = f"*tmmeada_transfer_formal_s{seed}*src_s{seed}*.pkl"
    else:
        exact = save_dir / (
            "MEAformer_DBP15K_zh_en_"
            f"transfer_src_zh_en_epoch10_baseline_transfer_formal_s{seed}_src_s{seed}_.pkl"
        )
        if exact.exists():
            return exact.stem
        pattern = f"*baseline_transfer_formal_s{seed}*src_s{seed}*.pkl"

    matches = sorted(save_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].stem if matches else ""


def run_baseline(seed: int, runner_python: str, meaformer_python: str, stage_root: str):
    target_cfgs = ",".join(
        [
            "configs/transfer_adapt/meaformer_target_fr_en_unsup_il.yaml",
            "configs/transfer_adapt/meaformer_target_fbyg15k_unsup_il.yaml",
        ]
    )
    ckpt = resolve_source_model_name(seed=seed, tmmeada=False)
    cmd = [
        runner_python,
        "scripts/run_transfer_train_eval.py",
        "--source-config",
        "configs/transfer/meaformer_source_zh_en_epoch10.yaml",
        "--target-configs",
        target_cfgs,
        "--tag",
        f"baseline_transfer_adapt_v8_expand_s{seed}",
        "--stage-root",
        stage_root,
        "--seed",
        str(seed),
        "--runner-python",
        meaformer_python,
        "--target-only-test",
        "0",
        "--target-save-model",
        "0",
    ]
    if ckpt:
        cmd.extend(["--source-model-name", ckpt])
    run_cmd(cmd)


def run_tmmeada(seed: int, runner_python: str, meaformer_python: str, stage_root: str):
    tmmeada_ckpt = resolve_source_model_name(seed=seed, tmmeada=True)
    baseline_ckpt = resolve_source_model_name(seed=seed, tmmeada=False)

    fr_cmd = [
        runner_python,
        "scripts/run_transfer_train_eval.py",
        "--source-config",
        "configs/transfer/tmmeada_source_zh_en_epoch10.yaml",
        "--target-configs",
        "configs/transfer_adapt/tmmeada_target_fr_en_unsup_il.yaml",
        "--tag",
        f"tmmeada_transfer_adapt_v8_expand_fr_s{seed}",
        "--stage-root",
        stage_root,
        "--seed",
        str(seed),
        "--runner-python",
        meaformer_python,
        "--target-only-test",
        "0",
        "--target-save-model",
        "0",
    ]
    if tmmeada_ckpt:
        fr_cmd.extend(["--source-model-name", tmmeada_ckpt])
    run_cmd(fr_cmd)

    fbyg_cmd = [
        runner_python,
        "scripts/run_transfer_train_eval.py",
        "--source-config",
        "configs/transfer/meaformer_source_zh_en_epoch10.yaml",
        "--target-configs",
        "configs/transfer_adapt/tmmeada_target_fbyg15k_v8_mild_da_unsup_il.yaml",
        "--tag",
        f"tmmeada_transfer_adapt_v8_expand_fbyg_s{seed}",
        "--stage-root",
        stage_root,
        "--seed",
        str(seed),
        "--runner-python",
        meaformer_python,
        "--target-only-test",
        "0",
        "--target-save-model",
        "0",
    ]
    if baseline_ckpt:
        fbyg_cmd.extend(["--source-model-name", baseline_ckpt])
    run_cmd(fbyg_cmd)


def summarize(runner_python: str, baseline_target_dir: str, tmmeada_target_dir: str):
    run_cmd(
        [
            runner_python,
            "scripts/summarize_transfer_formal.py",
            "--baseline-target-dir",
            baseline_target_dir,
            "--tmmeada-target-dir",
            tmmeada_target_dir,
            "--baseline-out",
            "reports/transfer/transfer_adapt_v8_expand_baseline_ref_summary.csv",
            "--tmmeada-out",
            "reports/transfer/transfer_adapt_v8_expand_tmmeada_summary.csv",
            "--compare-out-csv",
            "reports/transfer/transfer_adapt_v8_expand_compare_vs_baseline.csv",
            "--compare-out-md",
            "reports/transfer/transfer_adapt_v8_expand_compare_vs_baseline.md",
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run transfer adapt v8 expand queue (fr_en + FBYG15K, baseline and TMMEA-DA)."
    )
    parser.add_argument("--seeds", default="42,3407")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--meaformer-python", default=r"D:\Anaconda_envs\envs\bysj-meaformer\python.exe")
    parser.add_argument("--baseline-stage-root", default="transfer/transfer_adapt_v8_expand_baseline")
    parser.add_argument("--tmmeada-stage-root", default="transfer/transfer_adapt_v8_expand_tmmeada")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    for seed in seeds:
        print(f"[QUEUE] baseline seed={seed}")
        run_baseline(
            seed=seed,
            runner_python=args.runner_python,
            meaformer_python=args.meaformer_python,
            stage_root=args.baseline_stage_root,
        )
        print(f"[QUEUE] tmmeada seed={seed}")
        run_tmmeada(
            seed=seed,
            runner_python=args.runner_python,
            meaformer_python=args.meaformer_python,
            stage_root=args.tmmeada_stage_root,
        )

    summarize(
        runner_python=args.runner_python,
        baseline_target_dir=f"runs/{args.baseline_stage_root}/target_eval",
        tmmeada_target_dir=f"runs/{args.tmmeada_stage_root}/target_eval",
    )
    print("[DONE] transfer adapt v8 expand queue completed.")


if __name__ == "__main__":
    main()
