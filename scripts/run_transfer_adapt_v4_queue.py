import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def resolve_source_model_name(seed: int) -> str:
    save_dir = Path("data/mmkg/MEAformer/save")
    if not save_dir.exists():
        return ""
    exact = save_dir / (
        "MEAformer_DBP15K_zh_en_"
        f"tmmeada_transfer_src_zh_en_epoch10_tmmeada_transfer_formal_s{seed}_src_s{seed}_.pkl"
    )
    if exact.exists():
        return exact.stem
    pattern = f"*tmmeada_transfer_formal_s{seed}*src_s{seed}*.pkl"
    matches = sorted(save_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].stem if matches else ""


def run_one(seed: int, runner_python: str, meaformer_python: str, stage_root: str):
    target_cfgs = ",".join(
        [
            "configs/transfer_adapt/tmmeada_target_ja_en_v4_unsup_il.yaml",
            "configs/transfer_adapt/tmmeada_target_fbdb15k_v4_unsup_il.yaml",
        ]
    )
    ckpt = resolve_source_model_name(seed=seed)
    cmd = [
        runner_python,
        "scripts/run_transfer_train_eval.py",
        "--source-config",
        "configs/transfer/tmmeada_source_zh_en_epoch10.yaml",
        "--target-configs",
        target_cfgs,
        "--tag",
        f"tmmeada_transfer_adapt_v4_s{seed}",
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


def main():
    parser = argparse.ArgumentParser(
        description="Run transfer adapt v4 queue on ja_en and FBDB15K (TMMEA-DA only)."
    )
    parser.add_argument("--seeds", default="42,3407")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--meaformer-python", default=r"D:\Anaconda_envs\envs\bysj-meaformer\python.exe")
    parser.add_argument("--stage-root", default="transfer/transfer_adapt_v4")
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

    print("[DONE] transfer adapt v4 queue completed.")


if __name__ == "__main__":
    main()
