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


def run_one(seed: int, runner_python: str, meaformer_python: str, stage_root: str):
    # Branch 1: ja_en uses tmmeada source + current best ja_en target config (v5).
    tmmeada_ckpt = resolve_source_model_name(seed=seed, tmmeada=True)
    ja_cmd = [
        runner_python,
        "scripts/run_transfer_train_eval.py",
        "--source-config",
        "configs/transfer/tmmeada_source_zh_en_epoch10.yaml",
        "--target-configs",
        "configs/transfer_adapt/tmmeada_target_ja_en_v5_unsup_il.yaml",
        "--tag",
        f"tmmeada_transfer_adapt_v6_mixed_ja_s{seed}",
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
        ja_cmd.extend(["--source-model-name", tmmeada_ckpt])

    # Branch 2: FBDB uses baseline source + mild domain-align target config.
    baseline_ckpt = resolve_source_model_name(seed=seed, tmmeada=False)
    fbdb_cmd = [
        runner_python,
        "scripts/run_transfer_train_eval.py",
        "--source-config",
        "configs/transfer/meaformer_source_zh_en_epoch10.yaml",
        "--target-configs",
        "configs/transfer_adapt/tmmeada_target_fbdb15k_v6_mild_da_unsup_il.yaml",
        "--tag",
        f"tmmeada_transfer_adapt_v6_mixed_fbdb_s{seed}",
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
        fbdb_cmd.extend(["--source-model-name", baseline_ckpt])

    run_cmd(ja_cmd)
    run_cmd(fbdb_cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Run transfer adapt v6 mixed-source queue (ja from tmmeada source, fbdb from baseline source)."
    )
    parser.add_argument("--seeds", default="42,3407")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--meaformer-python", default=r"D:\Anaconda_envs\envs\bysj-meaformer\python.exe")
    parser.add_argument("--stage-root", default="transfer/transfer_adapt_v6_mixed")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    for seed in seeds:
        print(f"[QUEUE] seed={seed}")
        run_one(
            seed=seed,
            runner_python=args.runner_python,
            meaformer_python=args.meaformer_python,
            stage_root=args.stage_root,
        )

    print("[DONE] transfer adapt v6 mixed queue completed.")


if __name__ == "__main__":
    main()
