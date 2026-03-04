import argparse
import subprocess
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run a config across multiple seeds.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--seeds", default="42,3407")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--runner-script", default="scripts/run_meaformer.py")
    parser.add_argument("--tag-suffix", default="")
    args = parser.parse_args()

    base_cfg = Path(args.base_config)
    if not base_cfg.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg}")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tmp_dir = Path("runs/system/multiseed_tmp/custom")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    original = load_yaml(base_cfg)
    data_choice = str(original["meaformer"].get("data_choice", "data"))
    data_split = str(original["meaformer"].get("data_split", "split"))

    for seed in seeds:
        run_cfg = load_yaml(base_cfg)
        m = run_cfg["meaformer"]
        m["random_seed"] = seed
        m["exp_id"] = f"{m['exp_id']}_s{seed}{args.tag_suffix}"
        m["exp_name"] = f"{m['exp_name']}_multiseed"

        tmp_cfg = tmp_dir / f"{base_cfg.stem}_{data_choice}_{data_split}_s{seed}.yaml"
        dump_yaml(tmp_cfg, run_cfg)
        run_cmd([args.runner_python, args.runner_script, "--config", str(tmp_cfg)])


if __name__ == "__main__":
    main()

