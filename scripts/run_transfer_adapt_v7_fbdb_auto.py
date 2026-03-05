import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def resolve_baseline_source_model_name(seed: int) -> str:
    save_dir = Path("data/mmkg/MEAformer/save")
    if not save_dir.exists():
        return ""
    exact = save_dir / (
        "MEAformer_DBP15K_zh_en_"
        f"transfer_src_zh_en_epoch10_baseline_transfer_formal_s{seed}_src_s{seed}_.pkl"
    )
    if exact.exists():
        return exact.stem
    pattern = f"*baseline_transfer_formal_s{seed}*src_s{seed}*.pkl"
    matches = sorted(save_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].stem if matches else ""


def run_one(
    runner_python: str,
    meaformer_python: str,
    seed: int,
    target_config: str,
    tag: str,
    stage_root: str,
):
    cmd = [
        runner_python,
        "scripts/run_transfer_train_eval.py",
        "--source-config",
        "configs/transfer/meaformer_source_zh_en_epoch10.yaml",
        "--target-configs",
        target_config,
        "--tag",
        tag,
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
    ckpt = resolve_baseline_source_model_name(seed=seed)
    if ckpt:
        cmd.extend(["--source-model-name", ckpt])
    run_cmd(cmd)


def summarize_vs_baseline(runner_python: str, tmmeada_target_dir: str, out_prefix: str):
    run_cmd(
        [
            runner_python,
            "scripts/summarize_transfer_formal.py",
            "--baseline-target-dir",
            "runs/transfer/transfer_adapt_pilot/target_eval",
            "--tmmeada-target-dir",
            tmmeada_target_dir,
            "--baseline-out",
            f"{out_prefix}_baseline_ref_summary.csv",
            "--tmmeada-out",
            f"{out_prefix}_tmmeada_summary.csv",
            "--compare-out-csv",
            f"{out_prefix}_compare_vs_baseline.csv",
            "--compare-out-md",
            f"{out_prefix}_compare_vs_baseline.md",
        ]
    )


def summarize_vs_v6(runner_python: str, tmmeada_target_dir: str, out_prefix: str):
    run_cmd(
        [
            runner_python,
            "scripts/summarize_transfer_formal.py",
            "--baseline-target-dir",
            "runs/transfer/transfer_adapt_v6_mixed/target_eval",
            "--tmmeada-target-dir",
            tmmeada_target_dir,
            "--baseline-out",
            f"{out_prefix}_v6_ref_summary.csv",
            "--tmmeada-out",
            f"{out_prefix}_tmmeada_summary.csv",
            "--compare-out-csv",
            f"{out_prefix}_compare_vs_v6.csv",
            "--compare-out-md",
            f"{out_prefix}_compare_vs_v6.md",
        ]
    )


def read_fbdb_delta(compare_csv: Path):
    if not compare_csv.exists():
        return None
    with compare_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("target") == "FBDB15K":
            try:
                return float(row.get("delta_avg_mrr_mean", "nan"))
            except Exception:
                return None
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Auto optimize FBDB15K transfer adapt: pilot sweep -> choose best -> 2-seed formal + summarize."
    )
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--meaformer-python", default=r"D:\Anaconda_envs\envs\bysj-meaformer\python.exe")
    parser.add_argument("--pilot-seed", type=int, default=42)
    parser.add_argument("--formal-seeds", default="42,3407")
    args = parser.parse_args()

    variants = [
        (
            "v7a",
            "configs/transfer_adapt/tmmeada_target_fbdb15k_v7a_mild_da_unsup_il.yaml",
        ),
        (
            "v7b",
            "configs/transfer_adapt/tmmeada_target_fbdb15k_v7b_mild_da_unsup_il.yaml",
        ),
        (
            "v7c",
            "configs/transfer_adapt/tmmeada_target_fbdb15k_v7c_mild_da_unsup_il.yaml",
        ),
    ]
    formal_seeds = [int(x.strip()) for x in args.formal_seeds.split(",") if x.strip()]

    decision_rows = []
    for variant, cfg in variants:
        stage_root = f"transfer/transfer_adapt_v7_fbdb_pilot_{variant}"
        tag = f"tmmeada_transfer_adapt_v7_fbdb_pilot_{variant}_s{args.pilot_seed}"
        print(f"[PILOT] {variant} seed={args.pilot_seed}")
        run_one(
            runner_python=args.runner_python,
            meaformer_python=args.meaformer_python,
            seed=args.pilot_seed,
            target_config=cfg,
            tag=tag,
            stage_root=stage_root,
        )
        out_prefix = f"reports/transfer/transfer_adapt_v7_fbdb_pilot_{variant}"
        tmmeada_target_dir = f"runs/{stage_root}/target_eval"
        summarize_vs_baseline(
            runner_python=args.runner_python,
            tmmeada_target_dir=tmmeada_target_dir,
            out_prefix=out_prefix,
        )
        delta = read_fbdb_delta(Path(f"{out_prefix}_compare_vs_baseline.csv"))
        decision_rows.append(
            {
                "variant": variant,
                "config": cfg,
                "pilot_seed": args.pilot_seed,
                "delta_avg_mrr_mean_vs_baseline": delta,
                "tmmeada_target_dir": tmmeada_target_dir,
            }
        )

    valid = [x for x in decision_rows if x["delta_avg_mrr_mean_vs_baseline"] is not None]
    if not valid:
        raise RuntimeError("No valid pilot result found for FBDB15K.")

    best = sorted(valid, key=lambda x: x["delta_avg_mrr_mean_vs_baseline"], reverse=True)[0]
    best_variant = best["variant"]
    best_cfg = best["config"]
    print(f"[DECISION] best_variant={best_variant} delta_mrr={best['delta_avg_mrr_mean_vs_baseline']}")

    formal_stage_root = f"transfer/transfer_adapt_v7_fbdb_formal_{best_variant}"
    for seed in formal_seeds:
        print(f"[FORMAL] variant={best_variant} seed={seed}")
        run_one(
            runner_python=args.runner_python,
            meaformer_python=args.meaformer_python,
            seed=seed,
            target_config=best_cfg,
            tag=f"tmmeada_transfer_adapt_v7_fbdb_formal_{best_variant}_s{seed}",
            stage_root=formal_stage_root,
        )

    final_out_prefix = "reports/transfer/transfer_adapt_v7_fbdb"
    tmmeada_final_target_dir = f"runs/{formal_stage_root}/target_eval"
    summarize_vs_baseline(
        runner_python=args.runner_python,
        tmmeada_target_dir=tmmeada_final_target_dir,
        out_prefix=final_out_prefix,
    )
    summarize_vs_v6(
        runner_python=args.runner_python,
        tmmeada_target_dir=tmmeada_final_target_dir,
        out_prefix=final_out_prefix,
    )

    decision_payload = {
        "timestamp": now_ts(),
        "pilot_seed": args.pilot_seed,
        "formal_seeds": formal_seeds,
        "variants": decision_rows,
        "best_variant": best_variant,
        "best_config": best_cfg,
        "best_delta_avg_mrr_mean_vs_baseline": best["delta_avg_mrr_mean_vs_baseline"],
        "formal_stage_root": formal_stage_root,
        "final_compare_vs_baseline_csv": f"{final_out_prefix}_compare_vs_baseline.csv",
        "final_compare_vs_v6_csv": f"{final_out_prefix}_compare_vs_v6.csv",
    }
    decision_json = Path("reports/transfer/transfer_adapt_v7_fbdb_decision.json")
    decision_md = Path("reports/transfer/transfer_adapt_v7_fbdb_decision.md")
    decision_json.parent.mkdir(parents=True, exist_ok=True)
    decision_json.write_text(json.dumps(decision_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Transfer Adapt v7 FBDB Auto Decision",
        "",
        f"- pilot_seed: `{args.pilot_seed}`",
        f"- formal_seeds: `{formal_seeds}`",
        f"- best_variant: `{best_variant}`",
        f"- best_config: `{best_cfg}`",
        f"- best_delta_avg_mrr_mean_vs_baseline: `{best['delta_avg_mrr_mean_vs_baseline']}`",
        f"- formal_stage_root: `runs/{formal_stage_root}`",
        "",
        "## Pilot Summary",
        "",
        "| variant | delta_avg_mrr_mean_vs_baseline |",
        "|---|---:|",
    ]
    for row in decision_rows:
        lines.append(f"| {row['variant']} | {row['delta_avg_mrr_mean_vs_baseline']} |")
    lines.append("")
    lines.append("## Final Outputs")
    lines.append("")
    lines.append(f"- `{final_out_prefix}_compare_vs_baseline.csv`")
    lines.append(f"- `{final_out_prefix}_compare_vs_v6.csv`")
    decision_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[DONE] decision -> {decision_json}")
    print(f"[DONE] decision md -> {decision_md}")


if __name__ == "__main__":
    main()
