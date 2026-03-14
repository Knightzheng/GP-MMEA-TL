import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from transfer_adapt_utils import resolve_source_model_name


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PY = r"D:\Anaconda_envs\envs\bysj-main\python.exe"
MEAFORMER_PY = r"D:\Anaconda_envs\envs\bysj-meaformer\python.exe"


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_seeds(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def read_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def run_cmd(cmd: list[str]):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def build_exact_exp_id(base_exp_id: str, tag: str, seed: int) -> str:
    return f"{base_exp_id}_{tag}_src_s{seed}"


def ensure_formal_source(seed: int, source_config: Path, stage_root: str, tmmeada: bool, run_missing: int):
    existing = resolve_source_model_name(seed=seed, tmmeada=tmmeada, allow_nonformal_fallback=False)
    if existing:
        print(f"[SKIP] seed={seed} already has exact formal source checkpoint: {existing}")
        return existing
    if run_missing == 0:
        print(f"[INFO] seed={seed} missing exact formal source checkpoint, but run-missing=0 so skip training.")
        return ""

    cfg = read_yaml(source_config)
    meta = cfg.setdefault("meta", {})
    m = cfg["meaformer"]

    tag = ("tmmeada_transfer_formal" if tmmeada else "baseline_transfer_formal") + f"_s{seed}"
    exp_id = build_exact_exp_id(str(m.get("exp_id", "transfer_src_zh_en_epoch10")), tag, seed)
    exp_name = str(m.get("exp_name", "BYSJ_transfer_source"))

    m["random_seed"] = int(seed)
    m["save_model"] = 1
    m["only_test"] = 0
    m["model_name_save"] = ""
    m["transfer_non_strict"] = 0
    m["exp_id"] = exp_id
    m["exp_name"] = f"{exp_name}_{tag}"
    meta["stage"] = stage_root
    meta["model_tag"] = str(meta.get("model_tag", "MEAformer-transfer-source")) + "-formal-src"

    tmp_cfg = ROOT / "runs" / "system" / "transfer_source_tmp" / f"{now_ts()}_{source_config.stem}_s{seed}.yaml"
    write_yaml(tmp_cfg, cfg)
    run_cmd(
        [
            RUNNER_PY,
            "scripts/run_meaformer.py",
            "--config",
            str(tmp_cfg),
            "--stage",
            stage_root,
            "--python",
            MEAFORMER_PY,
        ]
    )

    resolved = resolve_source_model_name(seed=seed, tmmeada=tmmeada, allow_nonformal_fallback=False)
    if not resolved:
        raise FileNotFoundError(f"exact formal source checkpoint still missing after training for seed={seed}")
    print(f"[DONE] exact formal source checkpoint ready for seed={seed}: {resolved}")
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Ensure exact zh_en transfer-source formal checkpoints exist for requested seeds.")
    parser.add_argument("--seeds", default="42,3407,2026,7,123")
    parser.add_argument(
        "--source-config",
        default="configs/transfer/meaformer_source_zh_en_epoch10.yaml",
        help="source config for baseline source formal training",
    )
    parser.add_argument(
        "--tmmeada-source-config",
        default="configs/transfer/tmmeada_source_zh_en_epoch10.yaml",
        help="source config for tmmeada source formal training",
    )
    parser.add_argument("--tmmeada", type=int, default=0, choices=[0, 1])
    parser.add_argument("--run-missing", type=int, default=1, choices=[0, 1])
    parser.add_argument("--stage-root", default="transfer/transfer_formal/source_train")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    source_config = Path(args.tmmeada_source_config if args.tmmeada else args.source_config)
    if not source_config.exists():
        raise FileNotFoundError(f"source config not found: {source_config}")

    for seed in seeds:
        ensure_formal_source(
            seed=seed,
            source_config=source_config,
            stage_root=args.stage_root,
            tmmeada=bool(args.tmmeada),
            run_missing=int(args.run_missing),
        )


if __name__ == "__main__":
    main()
