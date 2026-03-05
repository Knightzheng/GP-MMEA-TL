import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def model_save_name(exp_id: str) -> str:
    # Keep consistent with MEAformer Runner._save_name_define (when no dist/il prefix).
    return f"{exp_id}_"


def find_saved_checkpoint(src_exp_id: str) -> Path | None:
    save_dir = Path("data/mmkg/MEAformer/save")
    if not save_dir.exists():
        return None
    matches = sorted(
        save_dir.glob(f"*{src_exp_id}_*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    # fallback for names ending exactly with "<src_exp_id>_.pkl"
    matches = sorted(
        save_dir.glob(f"*{src_exp_id}_.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(
        description="Train on source config then evaluate transfer on target config(s)."
    )
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-configs", required=True, help="comma-separated config paths")
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--runner-script", default="scripts/run_meaformer.py")
    parser.add_argument("--stage-root", default="transfer/transfer_pilot")
    parser.add_argument("--tag", default="xfer")
    parser.add_argument("--seed", type=int, default=None, help="override source/target seed if set")
    parser.add_argument("--source-epoch", type=int, default=None, help="override source training epochs")
    parser.add_argument(
        "--target-only-test",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: evaluate-only target; 0: train/adapt on target then evaluate",
    )
    parser.add_argument(
        "--target-epoch",
        type=int,
        default=None,
        help="override target epoch when target-only-test=0",
    )
    parser.add_argument(
        "--target-save-model",
        type=int,
        default=0,
        choices=[0, 1],
        help="whether to save model in target stage",
    )
    parser.add_argument(
        "--source-model-name",
        default="",
        help="existing checkpoint stem for loading on target; if set, source training is skipped",
    )
    parser.add_argument(
        "--transfer-skip-keys",
        default="multimodal_encoder.entity_emb.weight",
        help="comma-separated state_dict keys skipped in target load",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = now_tag()
    tmp_root = Path("runs/system/transfer_tmp") / f"{ts}_{args.tag}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    source_cfg_path = Path(args.source_config)
    if not source_cfg_path.exists():
        raise FileNotFoundError(f"source config not found: {source_cfg_path}")

    target_paths = [Path(x.strip()) for x in args.target_configs.split(",") if x.strip()]
    if not target_paths:
        raise ValueError("target-configs is empty")
    for p in target_paths:
        if not p.exists():
            raise FileNotFoundError(f"target config not found: {p}")

    source_cfg = load_yaml(source_cfg_path)
    source_m = source_cfg["meaformer"]
    source_meta = source_cfg.setdefault("meta", {})
    src_choice = str(source_m.get("data_choice", "SRC"))
    src_split = str(source_m.get("data_split", "split"))
    src_seed = int(source_m.get("random_seed", 42)) if args.seed is None else int(args.seed)
    src_exp_id_base = str(source_m.get("exp_id", "transfer_src"))
    src_exp_id = f"{src_exp_id_base}_{args.tag}_src_s{src_seed}"
    src_model_name = model_save_name(src_exp_id)

    source_m["random_seed"] = src_seed
    source_m["exp_id"] = src_exp_id
    source_m["exp_name"] = f"{source_m.get('exp_name', 'BYSJ_TRANSFER')}_{args.tag}_src"
    source_m["save_model"] = 1
    source_m["only_test"] = 0
    source_m["model_name_save"] = ""
    source_m["transfer_non_strict"] = 0
    if args.source_epoch is not None:
        source_m["epoch"] = int(args.source_epoch)

    source_stage = f"{args.stage_root}/source_train"
    source_meta["stage"] = source_stage
    source_meta["model_tag"] = source_meta.get("model_tag", "MEAformer") + "-transfer-src"

    source_tmp = tmp_root / f"source_{source_cfg_path.stem}.yaml"
    dump_yaml(source_tmp, source_cfg)

    report = {
        "timestamp": ts,
        "tag": args.tag,
        "source": {
            "config": str(source_cfg_path),
            "tmp_config": str(source_tmp),
            "stage": source_stage,
            "data_choice": src_choice,
            "data_split": src_split,
            "seed": src_seed,
            "exp_id": src_exp_id,
            "model_name_save": src_model_name,
        },
        "targets": [],
    }

    source_ckpt_name = args.source_model_name.strip()
    source_cmd = [
        args.runner_python,
        args.runner_script,
        "--config",
        str(source_tmp),
        "--stage",
        source_stage,
        "--python",
        args.runner_python,
    ]
    if source_ckpt_name:
        print(f"[INFO] use existing source checkpoint: {source_ckpt_name}.pkl")
    else:
        print(f"[INFO] source checkpoint base name: {src_model_name}.pkl")
    if args.dry_run:
        if not source_ckpt_name:
            print(f"[DRY_RUN] {' '.join(source_cmd)}")
    else:
        if not source_ckpt_name:
            run_cmd(source_cmd)

    if not source_ckpt_name:
        ckpt = find_saved_checkpoint(src_exp_id)
        if not args.dry_run and ckpt is None:
            raise FileNotFoundError(
                f"source checkpoint not found under data/mmkg/MEAformer/save for exp_id={src_exp_id}"
            )
        if ckpt is not None:
            source_ckpt_name = ckpt.stem
            report["source"]["resolved_checkpoint"] = str(ckpt)
            report["source"]["resolved_model_name_save"] = source_ckpt_name
            print(f"[INFO] resolved source checkpoint: {ckpt}")
    else:
        ckpt = Path("data/mmkg/MEAformer/save") / f"{source_ckpt_name}.pkl"
        if not args.dry_run and not ckpt.exists():
            raise FileNotFoundError(f"source checkpoint not found: {ckpt}")
        report["source"]["resolved_checkpoint"] = str(ckpt)
        report["source"]["resolved_model_name_save"] = source_ckpt_name

    for target_cfg_path in target_paths:
        target_cfg = load_yaml(target_cfg_path)
        target_m = target_cfg["meaformer"]
        target_meta = target_cfg.setdefault("meta", {})
        tgt_choice = str(target_m.get("data_choice", "TGT"))
        tgt_split = str(target_m.get("data_split", "split"))
        tgt_seed = int(target_m.get("random_seed", 42)) if args.seed is None else int(args.seed)

        target_exp_id_base = str(target_m.get("exp_id", "transfer_tgt"))
        target_exp_id = (
            f"{target_exp_id_base}_{args.tag}_from_{src_choice}_{src_split}_s{tgt_seed}"
        )

        target_m["random_seed"] = tgt_seed
        target_m["exp_id"] = target_exp_id
        target_m["exp_name"] = (
            f"{target_m.get('exp_name', 'BYSJ_TRANSFER')}_{args.tag}_from_{src_choice}_{src_split}"
        )
        target_m["only_test"] = int(args.target_only_test)
        target_m["save_model"] = int(args.target_save_model)
        target_m["model_name_save"] = source_ckpt_name
        target_m["transfer_non_strict"] = 1
        target_m["transfer_skip_keys"] = args.transfer_skip_keys
        target_m["transfer_verbose"] = 1
        if args.target_epoch is not None:
            target_m["epoch"] = int(args.target_epoch)

        target_stage = f"{args.stage_root}/target_eval"
        target_meta["stage"] = target_stage
        target_meta["model_tag"] = target_meta.get("model_tag", "MEAformer") + "-transfer-tgt"

        target_tmp = tmp_root / f"target_{target_cfg_path.stem}.yaml"
        dump_yaml(target_tmp, target_cfg)

        target_cmd = [
            args.runner_python,
            args.runner_script,
            "--config",
            str(target_tmp),
            "--stage",
            target_stage,
            "--python",
            args.runner_python,
        ]
        report["targets"].append(
            {
                "config": str(target_cfg_path),
                "tmp_config": str(target_tmp),
                "stage": target_stage,
                "data_choice": tgt_choice,
                "data_split": tgt_split,
                "seed": tgt_seed,
                "exp_id": target_exp_id,
                "command": " ".join(target_cmd),
            }
        )
        if args.dry_run:
            print(f"[DRY_RUN] {' '.join(target_cmd)}")
        else:
            run_cmd(target_cmd)

    report_path = Path("reports/transfer") / f"transfer_run_card_{ts}_{args.tag}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] report -> {report_path}")


if __name__ == "__main__":
    main()

