import argparse
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VARIANTS = {
    "ja_en_baseline": ROOT
    / "runs/transfer/transfer_adapt_ja_expand5_merged_baseline/target_eval/20260304-221002-MEAformer-transfer-adapt-target-ja_en-transfer-tgt-DBP15K-ja_en-s42/config.yaml",
    "ja_en_method": ROOT
    / "runs/transfer/transfer_adapt_ja_v15_full_ref/target_eval/20260311-010054-TMMEA-DA-transfer-adapt-v15-target-ja_en-transfer-tgt-DBP15K-ja_en-s42/config.yaml",
    "fbyg15k_baseline": ROOT
    / "runs/transfer/transfer_adapt_fbyg_expand5_merged_baseline/target_eval/20260306-002756-MEAformer-transfer-adapt-target-fbyg15k-transfer-tgt-FBYG15K-norm-s42/config.yaml",
    "fbyg15k_method": ROOT
    / "runs/transfer/transfer_adapt_v24_fbyg_v24b_expand5_ref/target_eval/20260314-062815-TMMEA-DA-transfer-adapt-v24b-target-fbyg15k-transfer-tgt-FBYG15K-norm-s42/config.yaml",
}


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run minimal peak-GPU-memory measurements for representative transfer-adapt targets."
    )
    parser.add_argument(
        "--variants",
        default="ja_en_baseline,ja_en_method,fbyg15k_baseline,fbyg15k_method",
    )
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runner-script", default="scripts/run_meaformer.py")
    parser.add_argument(
        "--runner-python",
        default=r"D:\Anaconda_envs\envs\bysj-main\python.exe",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    tmp_root = ROOT / "runs/system/gpu_peak_minimal_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        if variant not in DEFAULT_VARIANTS:
            raise KeyError(f"unknown variant: {variant}")

        cfg = load_yaml(DEFAULT_VARIANTS[variant])
        meta = cfg.setdefault("meta", {})
        meaformer = cfg["meaformer"]

        dataset = str(meaformer.get("data_choice", "dataset")).lower()
        split = str(meaformer.get("data_split", "split")).lower()
        stage = f"experiments/gpu_peak_minimal/{variant}"
        meta["stage"] = stage
        meta["model_tag"] = f"{meta.get('model_tag', 'MEAformer')}-gpupeak"

        requested_epoch = int(args.epoch)
        effective_epoch = requested_epoch
        if int(meaformer.get("il", 0)):
            il_start = int(meaformer.get("il_start", 0))
            if effective_epoch <= il_start:
                # Keep the rerun minimal, but still valid for transfer configs that enable IL.
                effective_epoch = il_start + 1

        meaformer["epoch"] = effective_epoch
        meaformer["random_seed"] = int(args.seed)
        meaformer["save_model"] = 0
        meaformer["eval_epoch"] = 1
        meaformer["exp_name"] = f"{meaformer.get('exp_name', 'BYSJ')}_gpu_peak_minimal"
        meaformer["exp_id"] = (
            f"{meaformer.get('exp_id', variant)}_gpupeak_e{effective_epoch}_s{args.seed}"
        )

        tmp_cfg = tmp_root / f"{variant}_{dataset}_{split}_e{effective_epoch}_s{args.seed}.yaml"
        dump_yaml(tmp_cfg, cfg)

        cmd = [args.runner_python, args.runner_script, "--config", str(tmp_cfg)]
        if args.dry_run:
            cmd.append("--dry-run")
        run_cmd(cmd)


if __name__ == "__main__":
    main()
