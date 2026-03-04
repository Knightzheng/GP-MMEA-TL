import argparse
import subprocess


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run formal transfer queue: baseline then TMMEA-DA.")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--meaformer-python", default=r"D:\Anaconda_envs\envs\bysj-meaformer\python.exe")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--stage-root", default="transfer_formal")
    parser.add_argument("--source-epoch", type=int, default=10)
    args = parser.parse_args()

    common_targets_baseline = ",".join(
        [
            "configs/transfer/meaformer_target_ja_en_eval.yaml",
            "configs/transfer/meaformer_target_fr_en_eval.yaml",
            "configs/transfer/meaformer_target_fbdb15k_eval.yaml",
        ]
    )
    common_targets_tmmeada = ",".join(
        [
            "configs/transfer/tmmeada_target_ja_en_eval.yaml",
            "configs/transfer/tmmeada_target_fr_en_eval.yaml",
            "configs/transfer/tmmeada_target_fbdb15k_eval.yaml",
        ]
    )

    baseline_cmd = [
        args.runner_python,
        "scripts/run_transfer_train_eval.py",
        "--source-config",
        "configs/transfer/meaformer_source_zh_en_epoch10.yaml",
        "--target-configs",
        common_targets_baseline,
        "--tag",
        f"baseline_transfer_formal_s{args.seed}",
        "--stage-root",
        args.stage_root,
        "--seed",
        str(args.seed),
        "--source-epoch",
        str(args.source_epoch),
        "--runner-python",
        args.meaformer_python,
    ]
    tmmeada_cmd = [
        args.runner_python,
        "scripts/run_transfer_train_eval.py",
        "--source-config",
        "configs/transfer/tmmeada_source_zh_en_epoch10.yaml",
        "--target-configs",
        common_targets_tmmeada,
        "--tag",
        f"tmmeada_transfer_formal_s{args.seed}",
        "--stage-root",
        f"{args.stage_root}_tmmeada",
        "--seed",
        str(args.seed),
        "--source-epoch",
        str(args.source_epoch),
        "--runner-python",
        args.meaformer_python,
    ]

    run_cmd(baseline_cmd)
    run_cmd(tmmeada_cmd)
    print("[DONE] transfer formal queue completed.")


if __name__ == "__main__":
    main()
