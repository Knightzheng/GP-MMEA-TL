import argparse
import subprocess
from pathlib import Path


PYTHON = r"D:\Anaconda_envs\envs\bysj-main\python.exe"
RUNNER = "scripts/run_from_base_config_multiseed.py"


def run_cmd(cmd):
    print(f"[QUEUE] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run next-stage pilot queue sequentially.")
    parser.add_argument("--seeds", default="42,3407")
    parser.add_argument("--include-epoch10", action="store_true")
    parser.add_argument("--epoch10-only", action="store_true")
    args = parser.parse_args()

    epoch8_cfgs = [
        Path("configs/baselines/meaformer_zh_en_rtx3060_safe_epoch8_pilot.yaml"),
        Path("configs/tmmeada/meaformer_zh_en_tmmeada_v1_best_epoch8_pilot.yaml"),
        Path("configs/baselines/meaformer_fbdb15k_rtx3060_safe_epoch8_pilot.yaml"),
        Path("configs/tmmeada/meaformer_fbdb15k_tmmeada_v1_best_epoch8_pilot.yaml"),
    ]
    epoch10_cfgs = [
        Path("configs/baselines/meaformer_zh_en_rtx3060_safe_epoch10_pilot.yaml"),
        Path("configs/tmmeada/meaformer_zh_en_tmmeada_v1_best_epoch10_pilot.yaml"),
        Path("configs/baselines/meaformer_fbdb15k_rtx3060_safe_epoch10_pilot.yaml"),
        Path("configs/tmmeada/meaformer_fbdb15k_tmmeada_v1_best_epoch10_pilot.yaml"),
    ]

    if args.epoch10_only:
        cfgs = epoch10_cfgs
    else:
        cfgs = list(epoch8_cfgs)
        if args.include_epoch10:
            cfgs.extend(epoch10_cfgs)

    for cfg in cfgs:
        if not cfg.exists():
            raise FileNotFoundError(f"Missing config: {cfg}")
        run_cmd(
            [
                PYTHON,
                RUNNER,
                "--base-config",
                str(cfg),
                "--seeds",
                args.seeds,
            ]
        )


if __name__ == "__main__":
    main()
