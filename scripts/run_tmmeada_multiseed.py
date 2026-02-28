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


def run_cmd(cmd, cwd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run TMMEA-DA MVP across multiple seeds.")
    parser.add_argument("--base-config", default="configs/tmmeada/meaformer_zh_en_domain_align_mvp.yaml")
    parser.add_argument("--seeds", default="3407,2026,7,123")
    parser.add_argument("--runner-script", default="scripts/run_meaformer.py")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    args = parser.parse_args()

    base_cfg = Path(args.base_config)
    if not base_cfg.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg}")

    cfg = load_yaml(base_cfg)
    split = str(cfg["meaformer"]["data_split"])
    data_choice = str(cfg["meaformer"]["data_choice"])
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tmp_dir = Path("runs/multiseed_tmp/tmmeada")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        run_cfg = load_yaml(base_cfg)
        run_cfg["meaformer"]["random_seed"] = seed
        run_cfg["meaformer"]["exp_id"] = f"{cfg['meaformer']['exp_id']}_s{seed}"
        run_cfg["meaformer"]["exp_name"] = f"{cfg['meaformer']['exp_name']}_multiseed"

        tmp_cfg = tmp_dir / f"tmmeada_{data_choice}_{split}_s{seed}.yaml"
        dump_yaml(tmp_cfg, run_cfg)
        cmd = [args.runner_python, args.runner_script, "--config", str(tmp_cfg)]
        run_cmd(cmd, cwd=Path.cwd())


if __name__ == "__main__":
    main()
