import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def resolve_tmmeada_source_model_name(seed: int) -> str:
    save_dir = Path("data/mmkg/MEAformer/save")
    if not save_dir.exists():
        return ""
    exact = save_dir / (
        "MEAformer_DBP15K_zh_en_"
        f"tmmeada_transfer_src_zh_en_epoch10_tmmeada_transfer_formal_s{seed}_src_s{seed}_.pkl"
    )
    if exact.exists():
        return exact.stem
    pattern = f"*tmmeada_transfer_formal_s{seed}*src_s{seed}*.pkl"
    matches = sorted(save_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].stem if matches else ""


def infer_seed_target(run_dir: Path):
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        m = payload.get("meaformer", {})
        seed = int(m.get("random_seed", -1))
        data_choice = str(m.get("data_choice", ""))
        data_split = str(m.get("data_split", ""))
        target = data_split if data_choice == "DBP15K" else data_choice
        return seed, target
    except Exception:
        return None


def latest_run_for_seed_target(stage_root: str, seed: int, target: str):
    target_eval_dir = Path("runs") / stage_root / "target_eval"
    if not target_eval_dir.exists():
        return None
    cands = []
    for run_dir in target_eval_dir.iterdir():
        if not run_dir.is_dir():
            continue
        parsed = infer_seed_target(run_dir)
        if parsed is None:
            continue
        s, t = parsed
        if s == seed and t == target:
            cands.append(run_dir)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def read_fr_delta(compare_csv: Path):
    if not compare_csv.exists():
        return None
    with compare_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("target") == "fr_en":
            try:
                return float(row.get("delta_avg_mrr_mean", "nan"))
            except Exception:
                return None
    return None


def run_variant(seed: int, target_config: str, variant_name: str, runner_python: str, meaformer_python: str, stage_root: str):
    cmd = [
        runner_python,
        "scripts/run_transfer_train_eval.py",
        "--source-config",
        "configs/transfer/tmmeada_source_zh_en_epoch10.yaml",
        "--target-configs",
        target_config,
        "--tag",
        f"tmmeada_transfer_adapt_v10_fren_{variant_name}_s{seed}",
        "--stage-root",
        stage_root,
        "--seed",
        str(seed),
        "--runner-python",
        meaformer_python,
        "--target-only-test",
        "0",
        "--target-save-model",
        "0",
    ]
    ckpt = resolve_tmmeada_source_model_name(seed=seed)
    if ckpt:
        cmd.extend(["--source-model-name", ckpt])
    run_cmd(cmd)


def summarize(runner_python: str, baseline_target_dir: str, tmmeada_target_dir: str, prefix: str):
    run_cmd(
        [
            runner_python,
            "scripts/summarize_transfer_formal.py",
            "--baseline-target-dir",
            baseline_target_dir,
            "--tmmeada-target-dir",
            tmmeada_target_dir,
            "--baseline-out",
            f"{prefix}_baseline_ref_summary.csv",
            "--tmmeada-out",
            f"{prefix}_tmmeada_summary.csv",
            "--compare-out-csv",
            f"{prefix}_compare_vs_baseline.csv",
            "--compare-out-md",
            f"{prefix}_compare_vs_baseline.md",
        ]
    )


def summarize_vs_ref(runner_python: str, ref_target_dir: str, tmmeada_target_dir: str, prefix: str):
    run_cmd(
        [
            runner_python,
            "scripts/summarize_transfer_formal.py",
            "--baseline-target-dir",
            ref_target_dir,
            "--tmmeada-target-dir",
            tmmeada_target_dir,
            "--baseline-out",
            f"{prefix}_v9_ref_summary.csv",
            "--tmmeada-out",
            f"{prefix}_tmmeada_summary.csv",
            "--compare-out-csv",
            f"{prefix}_compare_vs_v9.csv",
            "--compare-out-md",
            f"{prefix}_compare_vs_v9.md",
        ]
    )


def copy_run_dir(src: Path, dst_root: Path):
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / src.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Auto optimize transfer-adapt on fr_en v10: 3 pilots -> pick best -> formal 2-seed compare."
    )
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--meaformer-python", default=r"D:\Anaconda_envs\envs\bysj-meaformer\python.exe")
    parser.add_argument("--pilot-seed", type=int, default=42)
    parser.add_argument("--formal-seed", type=int, default=3407)
    args = parser.parse_args()

    variants = [
        {"name": "v10a_unsup900", "target_config": "configs/transfer_adapt/tmmeada_target_fr_en_v10a_unsup900.yaml"},
        {"name": "v10b_da0025", "target_config": "configs/transfer_adapt/tmmeada_target_fr_en_v10b_da0025.yaml"},
        {"name": "v10c_da0035", "target_config": "configs/transfer_adapt/tmmeada_target_fr_en_v10c_da0035.yaml"},
    ]

    decision_rows = []
    for v in variants:
        stage_root = f"transfer/transfer_adapt_v10_fren_pilot_{v['name']}"
        run_variant(
            seed=args.pilot_seed,
            target_config=v["target_config"],
            variant_name=f"pilot_{v['name']}",
            runner_python=args.runner_python,
            meaformer_python=args.meaformer_python,
            stage_root=stage_root,
        )
        run_dir = latest_run_for_seed_target(stage_root=stage_root, seed=args.pilot_seed, target="fr_en")
        prefix = f"reports/transfer/transfer_adapt_v10_fren_pilot_{v['name']}"
        summarize(
            runner_python=args.runner_python,
            baseline_target_dir="runs/transfer/transfer_adapt_v8_expand_s42_baseline/target_eval",
            tmmeada_target_dir=f"runs/{stage_root}/target_eval",
            prefix=prefix,
        )
        delta = read_fr_delta(Path(f"{prefix}_compare_vs_baseline.csv"))
        decision_rows.append(
            {
                "name": v["name"],
                "target_config": v["target_config"],
                "pilot_seed": args.pilot_seed,
                "pilot_stage_root": stage_root,
                "pilot_run_dir": str(run_dir) if run_dir else "",
                "delta_avg_mrr_mean_vs_baseline": delta,
            }
        )

    valid = [x for x in decision_rows if x["delta_avg_mrr_mean_vs_baseline"] is not None]
    if not valid:
        raise RuntimeError("No valid fr_en pilot result.")
    best = sorted(valid, key=lambda x: x["delta_avg_mrr_mean_vs_baseline"], reverse=True)[0]

    formal_stage_root = f"transfer/transfer_adapt_v10_fren_formal_{best['name']}"
    run_variant(
        seed=args.formal_seed,
        target_config=best["target_config"],
        variant_name=f"formal_{best['name']}",
        runner_python=args.runner_python,
        meaformer_python=args.meaformer_python,
        stage_root=formal_stage_root,
    )
    formal_run_dir = latest_run_for_seed_target(
        stage_root=formal_stage_root,
        seed=args.formal_seed,
        target="fr_en",
    )
    if formal_run_dir is None:
        raise RuntimeError("Formal fr_en run not found.")

    pilot_run_dir = Path(best["pilot_run_dir"])
    if not pilot_run_dir.exists():
        raise RuntimeError(f"Best pilot run dir not found: {pilot_run_dir}")

    merged_target_root = (
        Path("runs")
        / "transfer"
        / f"transfer_adapt_v10_fren_2seed_{best['name']}"
        / "target_eval"
    )
    copy_run_dir(pilot_run_dir, merged_target_root)
    copy_run_dir(formal_run_dir, merged_target_root)

    final_prefix = "reports/transfer/transfer_adapt_v10_fren_2seed"
    summarize(
        runner_python=args.runner_python,
        baseline_target_dir="runs/transfer/transfer_adapt_v8_expand_2seed_baseline/target_eval",
        tmmeada_target_dir=str(merged_target_root),
        prefix=final_prefix,
    )
    summarize_vs_ref(
        runner_python=args.runner_python,
        ref_target_dir="runs/transfer/transfer_adapt_v9_fren_2seed_v9a_tm_src_mild_da/target_eval",
        tmmeada_target_dir=str(merged_target_root),
        prefix=final_prefix,
    )

    payload = {
        "timestamp": now_ts(),
        "pilot_seed": args.pilot_seed,
        "formal_seed": args.formal_seed,
        "variants": decision_rows,
        "best_variant": best["name"],
        "best_variant_target_config": best["target_config"],
        "best_delta_avg_mrr_mean_vs_baseline": best["delta_avg_mrr_mean_vs_baseline"],
        "pilot_run_dir": str(pilot_run_dir),
        "formal_run_dir": str(formal_run_dir),
        "merged_target_root": str(merged_target_root),
        "final_compare_vs_baseline_csv": f"{final_prefix}_compare_vs_baseline.csv",
        "final_compare_vs_v9_csv": f"{final_prefix}_compare_vs_v9.csv",
    }
    out_json = Path("reports/transfer/transfer_adapt_v10_fren_decision.json")
    out_md = Path("reports/transfer/transfer_adapt_v10_fren_decision.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Transfer Adapt v10 fr_en Decision",
        "",
        f"- pilot_seed: `{args.pilot_seed}`",
        f"- formal_seed: `{args.formal_seed}`",
        f"- best_variant: `{best['name']}`",
        f"- best_delta_avg_mrr_mean_vs_baseline: `{best['delta_avg_mrr_mean_vs_baseline']}`",
        f"- pilot_run_dir: `{pilot_run_dir}`",
        f"- formal_run_dir: `{formal_run_dir}`",
        "",
        "## Pilot Summary",
        "",
        "| variant | delta_avg_mrr_mean_vs_baseline |",
        "|---|---:|",
    ]
    for row in decision_rows:
        lines.append(f"| {row['name']} | {row['delta_avg_mrr_mean_vs_baseline']} |")
    lines.extend(
        [
            "",
            "## Final Outputs",
            "",
            f"- `{final_prefix}_compare_vs_baseline.csv`",
            f"- `{final_prefix}_compare_vs_v9.csv`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] decision -> {out_json}")
    print(f"[DONE] decision md -> {out_md}")


if __name__ == "__main__":
    main()
