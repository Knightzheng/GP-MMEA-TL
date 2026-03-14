import argparse
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VARIANTS = {
    "baseline": ROOT / "configs/baselines/meaformer_zh_en_rtx3060_safe_epoch3.yaml",
    "v1_full": ROOT / "configs/tmmeada/meaformer_zh_en_tmmeada_v1_best_epoch3.yaml",
    "wo_missing_gate": ROOT / "configs/tmmeada/meaformer_zh_en_tmmeada_v1_best_epoch3_wo_missing_gate.yaml",
}


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the minimal H3 missing-modality pressure-test matrix on zh_en.")
    parser.add_argument("--variants", default="baseline,v1_full,wo_missing_gate")
    parser.add_argument("--drop-rates", default="0.0,0.3,0.6")
    parser.add_argument("--seeds", default="42,2026")
    parser.add_argument("--runner-script", default="scripts/run_meaformer.py")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    drop_rates = [float(item.strip()) for item in args.drop_rates.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    tmp_root = ROOT / "runs/system/h3_missing_modality_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        if variant not in DEFAULT_VARIANTS:
            raise KeyError(f"unknown variant: {variant}")
        base_cfg_path = DEFAULT_VARIANTS[variant]
        base_cfg = load_yaml(base_cfg_path)
        base_stage = f"experiments/h3_missing_modality_minimal/{variant}"

        for drop_rate in drop_rates:
            drop_tag = f"miss{int(round(drop_rate * 100)):02d}"
            for seed in seeds:
                cfg = load_yaml(base_cfg_path)
                cfg.setdefault("meta", {})
                cfg["meta"]["stage"] = base_stage
                cfg["meta"]["model_tag"] = f"{cfg['meta'].get('model_tag', 'MEAformer')}-{drop_tag}"
                cfg["meaformer"]["random_seed"] = seed
                cfg["meaformer"]["img_mask_drop_rate"] = drop_rate
                cfg["meaformer"]["img_mask_drop_seed"] = seed
                cfg["meaformer"]["exp_name"] = f"{cfg['meaformer']['exp_name']}_h3_missing"
                cfg["meaformer"]["exp_id"] = f"{cfg['meaformer']['exp_id']}_{drop_tag}_s{seed}"

                tmp_cfg = tmp_root / f"{variant}_{drop_tag}_s{seed}.yaml"
                dump_yaml(tmp_cfg, cfg)

                cmd = [args.runner_python, args.runner_script, "--config", str(tmp_cfg)]
                if args.dry_run:
                    cmd.append("--dry-run")
                run_cmd(cmd)


if __name__ == "__main__":
    main()
