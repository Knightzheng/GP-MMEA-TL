import argparse
import itertools
import subprocess
from pathlib import Path

import yaml


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def parse_float_list(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_seed_list(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def fmt_tag(x: float):
    text = f"{x:.4g}"
    return text.replace("-", "m").replace(".", "p")


def run_cmd(cmd, cwd: Path):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run TMMEA-DA v1 weight sweep on zh_en.")
    parser.add_argument("--base-config", default="configs/tmmeada/meaformer_zh_en_tmmeada_v1_sweep.yaml")
    parser.add_argument("--domain-align-weights", default="0.05,0.1,0.2")
    parser.add_argument("--source-select-weights", default="0.05,0.1")
    parser.add_argument("--missing-align-weights", default="0.05,0.1")
    parser.add_argument("--source-select-temps", default="1.0")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--runner-script", default="scripts/run_meaformer.py")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    args = parser.parse_args()

    base_cfg_path = Path(args.base_config)
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")
    base_cfg = load_yaml(base_cfg_path)

    dws = parse_float_list(args.domain_align_weights)
    sws = parse_float_list(args.source_select_weights)
    mws = parse_float_list(args.missing_align_weights)
    temps = parse_float_list(args.source_select_temps)
    seeds = parse_seed_list(args.seeds)

    combos = list(itertools.product(dws, sws, mws, temps, seeds))
    tmp_dir = Path("runs/multiseed_tmp/tmmeada_v1_sweep")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    total = len(combos)
    for i, (dw, sw, mw, temp, seed) in enumerate(combos, start=1):
        run_cfg = load_yaml(base_cfg_path)
        meta = run_cfg.setdefault("meta", {})
        mea = run_cfg["meaformer"]
        split = str(mea["data_split"])
        data_choice = str(mea["data_choice"])

        mea["use_domain_align"] = 1
        mea["use_source_select"] = 1
        mea["use_missing_gate"] = 1
        mea["domain_align_weight"] = dw
        mea["source_select_weight"] = sw
        mea["missing_align_weight"] = mw
        mea["source_select_temp"] = temp
        mea["random_seed"] = seed

        tag = (
            f"dw{fmt_tag(dw)}_sw{fmt_tag(sw)}_mw{fmt_tag(mw)}_tp{fmt_tag(temp)}_s{seed}"
        )
        meta["stage"] = "tmmeada_v1_sweep"
        meta["model_tag"] = f"TMMEA-DA-v1-{tag}"
        mea["exp_name"] = "BYSJ_TMMEA_DA_v1_sweep"
        mea["exp_id"] = f"v1_sweep_{split}_{tag}"

        tmp_cfg = tmp_dir / f"v1_sweep_{data_choice}_{split}_{tag}.yaml"
        dump_yaml(tmp_cfg, run_cfg)
        print(f"[{i}/{total}] {tag}")
        cmd = [args.runner_python, args.runner_script, "--config", str(tmp_cfg)]
        run_cmd(cmd, cwd=Path.cwd())


if __name__ == "__main__":
    main()
