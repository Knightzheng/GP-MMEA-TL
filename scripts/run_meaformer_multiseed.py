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
    parser = argparse.ArgumentParser(description="Run MEAformer across multiple lang pairs and seeds.")
    parser.add_argument("--langs", default="zh_en,ja_en,fr_en")
    parser.add_argument("--seeds", default="3407,2026")
    parser.add_argument("--base-config-dir", default="configs/baselines")
    parser.add_argument("--runner-script", default="scripts/run_meaformer.py")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    args = parser.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tmp_dir = Path("runs/multiseed_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        base_cfg = Path(args.base_config_dir) / f"meaformer_{lang}_rtx3060_safe.yaml"
        if not base_cfg.exists():
            raise FileNotFoundError(f"Base config not found: {base_cfg}")

        for seed in seeds:
            cfg = load_yaml(base_cfg)
            cfg["meaformer"]["random_seed"] = seed
            cfg["meaformer"]["exp_id"] = f"{cfg['meaformer']['exp_id']}_s{seed}"
            cfg["meaformer"]["exp_name"] = f"{cfg['meaformer']['exp_name']}_multiseed"

            tmp_cfg = tmp_dir / f"meaformer_{lang}_s{seed}.yaml"
            dump_yaml(tmp_cfg, cfg)
            cmd = [args.runner_python, args.runner_script, "--config", str(tmp_cfg)]
            run_cmd(cmd, cwd=Path.cwd())


if __name__ == "__main__":
    main()
