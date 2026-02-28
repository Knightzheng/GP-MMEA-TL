import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def now_str():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def append_log(path, line):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def init_run_dir(stage, model, dataset, split, seed):
    run_id = f"{now_str()}-{model}-{dataset}-{split}-s{seed}"
    run_dir = Path("runs") / stage / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figs").mkdir(exist_ok=True)
    return run_id, run_dir


def run_cmd_and_stream(cmd, cwd, log_path, env=None):
    append_log(log_path, f"[RUN] {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in process.stdout:
        safe = line.encode("gbk", errors="replace").decode("gbk", errors="replace")
        sys.stdout.write(safe)
        append_log(log_path, line.rstrip())
    process.wait()
    return process.returncode


def write_run_card(path, run_id, cfg_path, stage, model_tag):
    content = (
        f"# run_card\n\n"
        f"- run_id: `{run_id}`\n"
        f"- stage: `{stage}`\n"
        f"- model: `{model_tag}`\n"
        f"- started_at: `{datetime.now().isoformat(timespec='seconds')}`\n"
        f"- command: `{' '.join(sys.argv)}`\n"
        f"- config: `{cfg_path}`\n"
    )
    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run MEAformer baseline with reproducible run logging.")
    parser.add_argument("--config", default="configs/baselines/meaformer_zh_en.yaml")
    parser.add_argument("--stage", default="")
    parser.add_argument(
        "--python",
        default=r"D:\Anaconda_envs\envs\bysj-meaformer\python.exe",
        help="Python executable of MEAformer env",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    model = "MEAformer"
    dataset = cfg["meaformer"]["data_choice"]
    split = cfg["meaformer"]["data_split"]
    seed = int(cfg["meaformer"]["random_seed"])
    meta = cfg.get("meta", {}) if isinstance(cfg, dict) else {}
    stage = args.stage if args.stage else str(meta.get("stage", "baseline"))
    model_tag = str(meta.get("model_tag", model))

    run_id, run_dir = init_run_dir(stage, model_tag, dataset, split, seed)
    log_path = run_dir / "log.txt"
    write_run_card(run_dir / "run_card.md", run_id, args.config, stage, model_tag)

    cfg["runtime"] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": args.python,
        "stage": stage,
        "model_tag": model_tag,
    }
    write_yaml(run_dir / "config.yaml", cfg)

    baseline_root = Path("baselines/MEAformer")
    main_py = baseline_root / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"missing baseline entry: {main_py}")

    required_dataset_dir = Path("data/mmkg") / dataset / split
    precheck = {
        "required_dataset_dir": str(required_dataset_dir),
        "exists": required_dataset_dir.exists(),
    }
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(precheck, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not required_dataset_dir.exists():
        append_log(log_path, f"[ERROR] dataset directory not found: {required_dataset_dir}")
        raise FileNotFoundError(f"dataset directory not found: {required_dataset_dir}")

    # Build command from config
    m = cfg["meaformer"]
    cmd = [
        args.python,
        "main.py",
        "--gpu", str(m["gpu"]),
        "--eval_epoch", str(m["eval_epoch"]),
        "--only_test", str(m["only_test"]),
        "--model_name", m["model_name"],
        "--data_choice", m["data_choice"],
        "--data_split", m["data_split"],
        "--data_rate", str(m["data_rate"]),
        "--epoch", str(m["epoch"]),
        "--lr", str(m["lr"]),
        "--hidden_units", str(m["hidden_units"]),
        "--save_model", str(m["save_model"]),
        "--batch_size", str(m["batch_size"]),
        "--csls_k", str(m["csls_k"]),
        "--random_seed", str(m["random_seed"]),
        "--exp_name", str(m["exp_name"]),
        "--exp_id", str(m["exp_id"]),
        "--workers", str(m["workers"]),
        "--dist", str(m["dist"]),
        "--accumulation_steps", str(m["accumulation_steps"]),
        "--scheduler", str(m["scheduler"]),
        "--attr_dim", str(m["attr_dim"]),
        "--img_dim", str(m["img_dim"]),
        "--name_dim", str(m["name_dim"]),
        "--char_dim", str(m["char_dim"]),
        "--hidden_size", str(m["hidden_size"]),
        "--tau", str(m["tau"]),
        "--structure_encoder", str(m["structure_encoder"]),
        "--num_attention_heads", str(m["num_attention_heads"]),
        "--num_hidden_layers", str(m["num_hidden_layers"]),
        "--use_surface", str(m["use_surface"]),
        "--use_intermediate", str(m["use_intermediate"]),
        "--replay", str(m["replay"]),
    ]
    if "use_domain_align" in m:
        cmd.extend(["--use_domain_align", str(m["use_domain_align"])])
    if "domain_align_weight" in m:
        cmd.extend(["--domain_align_weight", str(m["domain_align_weight"])])
    if "use_source_select" in m:
        cmd.extend(["--use_source_select", str(m["use_source_select"])])
    if "source_select_weight" in m:
        cmd.extend(["--source_select_weight", str(m["source_select_weight"])])
    if "source_select_temp" in m:
        cmd.extend(["--source_select_temp", str(m["source_select_temp"])])
    if "use_missing_gate" in m:
        cmd.extend(["--use_missing_gate", str(m["use_missing_gate"])])
    if "missing_align_weight" in m:
        cmd.extend(["--missing_align_weight", str(m["missing_align_weight"])])
    if m.get("csls", True):
        cmd.append("--csls")
    if m.get("enable_sota", True):
        cmd.append("--enable_sota")
    # store_false flags in MEAformer config: pass flag to disable the modality.
    if m.get("w_gcn") is False:
        cmd.append("--w_gcn")
    if m.get("w_rel") is False:
        cmd.append("--w_rel")
    if m.get("w_attr") is False:
        cmd.append("--w_attr")
    if m.get("w_name") is False:
        cmd.append("--w_name")
    if m.get("w_char") is False:
        cmd.append("--w_char")
    if m.get("w_img") is False:
        cmd.append("--w_img")

    if args.dry_run:
        append_log(log_path, f"[DRY_RUN] {' '.join(cmd)}")
        print(f"Dry run command prepared: {' '.join(cmd)}")
        return

    env = None
    if "env" in cfg and isinstance(cfg["env"], dict):
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in cfg["env"].items()})
    rc = run_cmd_and_stream(cmd, cwd=baseline_root, log_path=log_path, env=env)
    append_log(log_path, f"[DONE] return_code={rc}")
    if rc != 0:
        raise SystemExit(rc)
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
