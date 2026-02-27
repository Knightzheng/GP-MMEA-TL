import argparse
import subprocess
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run FBDB15K/FBYG15K MEAformer across multiple seeds.")
    parser.add_argument("--seeds", default="3407,2026")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--runner-script", default="scripts/run_meaformer.py")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    base_cfgs = [
        Path("configs/baselines/meaformer_fbdb15k_rtx3060_safe.yaml"),
        Path("configs/baselines/meaformer_fbyg15k_rtx3060_safe.yaml"),
    ]
    tmp_dir = Path("runs/multiseed_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for base_cfg in base_cfgs:
        if not base_cfg.exists():
            raise FileNotFoundError(f"Missing base config: {base_cfg}")
        for seed in seeds:
            cfg = load_yaml(base_cfg)
            m = cfg["meaformer"]
            m["random_seed"] = seed
            m["exp_name"] = f"{m['exp_name']}_multiseed"
            m["exp_id"] = f"{m['exp_id']}_s{seed}"
            tmp_cfg = tmp_dir / f"{base_cfg.stem}_s{seed}.yaml"
            dump_yaml(tmp_cfg, cfg)
            run_cmd([args.runner_python, args.runner_script, "--config", str(tmp_cfg)])


if __name__ == "__main__":
    main()
