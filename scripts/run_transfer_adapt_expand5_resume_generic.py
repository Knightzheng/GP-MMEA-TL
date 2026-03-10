import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


L2R_RE = re.compile(
    r"Ep (?:Test|[0-9]+) \| l2r: acc of top \[1, 10, 50\] = \[(?P<h1>[0-9.]+)\s+(?P<h10>[0-9.]+)\s+(?P<h50>[0-9.]+)\s*\], mr = (?P<mr>[0-9.]+), mrr = (?P<mrr>[0-9.]+)"
)
R2L_RE = re.compile(
    r"Ep (?:Test|[0-9]+) \| r2l: acc of top \[1, 10, 50\] = \[(?P<h1>[0-9.]+)\s+(?P<h10>[0-9.]+)\s+(?P<h50>[0-9.]+)\s*\], mr = (?P<mr>[0-9.]+), mrr = (?P<mrr>[0-9.]+)"
)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


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


def has_complete_eval_metrics(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    l2r_ok = False
    r2l_ok = False
    for line in text.splitlines():
        if L2R_RE.search(line):
            l2r_ok = True
        if R2L_RE.search(line):
            r2l_ok = True
    return l2r_ok and r2l_ok


def is_complete_target_eval_run(run_dir: Path) -> bool:
    parsed = infer_seed_target(run_dir)
    if parsed is None:
        return False
    return has_complete_eval_metrics(run_dir / "log.txt")


def matching_runs_for_seed_target_in_target_eval(target_eval_dir: Path, seed: int, target: str):
    if not target_eval_dir.exists():
        return []
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
    return cands


def latest_run_for_seed_target_from_roots(target_eval_roots, seed: int, target: str):
    cands = []
    for root in target_eval_roots:
        p = Path(root)
        cands.extend(matching_runs_for_seed_target_in_target_eval(p, seed, target))
    complete = [x for x in cands if is_complete_target_eval_run(x)]
    if complete:
        complete.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return complete[0]
    if not cands:
        return None
    # There are matching runs, but none has complete eval metrics.
    return None


def resolve_source_model_name(seed: int, tmmeada: bool) -> str:
    save_dir = Path("data/mmkg/MEAformer/save")
    if not save_dir.exists():
        return ""

    if tmmeada:
        exact = save_dir / (
            "MEAformer_DBP15K_zh_en_"
            f"tmmeada_transfer_src_zh_en_epoch10_tmmeada_transfer_formal_s{seed}_src_s{seed}_.pkl"
        )
        if exact.exists():
            return exact.stem
        pattern = f"*tmmeada_transfer_formal_s{seed}*src_s{seed}*.pkl"
    else:
        exact = save_dir / (
            "MEAformer_DBP15K_zh_en_"
            f"transfer_src_zh_en_epoch10_baseline_transfer_formal_s{seed}_src_s{seed}_.pkl"
        )
        if exact.exists():
            return exact.stem
        pattern = f"*baseline_transfer_formal_s{seed}*src_s{seed}*.pkl"

    matches = sorted(save_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].stem if matches else ""


def run_transfer_one(
    seed: int,
    runner_python: str,
    meaformer_python: str,
    source_config: str,
    target_config: str,
    tag: str,
    stage_root: str,
    tmmeada: bool,
):
    ckpt = resolve_source_model_name(seed=seed, tmmeada=tmmeada)
    cmd = [
        runner_python,
        "scripts/run_transfer_train_eval.py",
        "--source-config",
        source_config,
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
    if ckpt:
        cmd.extend(["--source-model-name", ckpt])
    run_cmd(cmd)


def copy_run_dir(src: Path, dst_root: Path):
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / src.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def rebuild_merged_target_eval(seeds, candidate_roots, merged_target_eval: Path, target: str):
    if merged_target_eval.exists():
        shutil.rmtree(merged_target_eval)
    merged_target_eval.mkdir(parents=True, exist_ok=True)

    selected = {}
    missing = []
    for seed in seeds:
        run_dir = latest_run_for_seed_target_from_roots(
            target_eval_roots=candidate_roots,
            seed=seed,
            target=target,
        )
        if run_dir is None:
            missing.append(seed)
            continue
        copy_run_dir(run_dir, merged_target_eval)
        selected[str(seed)] = str(run_dir)
    return selected, missing


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


def main():
    parser = argparse.ArgumentParser(
        description="Generic resume/expand transfer-adapt to 5-seed and rebuild compare summaries."
    )
    parser.add_argument("--runner-python", default=r"D:\Anaconda_envs\envs\bysj-main\python.exe")
    parser.add_argument("--meaformer-python", default=r"D:\Anaconda_envs\envs\bysj-meaformer\python.exe")
    parser.add_argument("--seeds", default="42,3407,2026,7,123")
    parser.add_argument("--target", required=True)
    parser.add_argument("--status-title", default="Transfer Adapt expand5 Status")

    parser.add_argument("--baseline-source-config", required=True)
    parser.add_argument("--baseline-target-config", required=True)
    parser.add_argument("--tmmeada-source-config", required=True)
    parser.add_argument("--tmmeada-target-config", required=True)

    parser.add_argument("--baseline-stage-root", required=True)
    parser.add_argument("--tmmeada-stage-root", required=True)

    parser.add_argument("--baseline-fallback-target-eval", required=True)
    parser.add_argument("--tmmeada-fallback-target-eval", required=True)
    parser.add_argument("--merged-baseline-target-eval", required=True)
    parser.add_argument("--merged-tmmeada-target-eval", required=True)
    parser.add_argument("--report-prefix", required=True)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--status-md", required=True)
    parser.add_argument(
        "--run-missing",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: run missing seeds; 0: only rebuild summaries from existing runs",
    )
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    baseline_stage_target_eval = Path("runs") / args.baseline_stage_root / "target_eval"
    tmmeada_stage_target_eval = Path("runs") / args.tmmeada_stage_root / "target_eval"

    baseline_candidate_roots = [baseline_stage_target_eval, Path(args.baseline_fallback_target_eval)]
    tmmeada_candidate_roots = [tmmeada_stage_target_eval, Path(args.tmmeada_fallback_target_eval)]

    initial_baseline_selected, initial_baseline_missing = rebuild_merged_target_eval(
        seeds=seeds,
        candidate_roots=baseline_candidate_roots,
        merged_target_eval=Path(args.merged_baseline_target_eval),
        target=args.target,
    )
    initial_tmmeada_selected, initial_tmmeada_missing = rebuild_merged_target_eval(
        seeds=seeds,
        candidate_roots=tmmeada_candidate_roots,
        merged_target_eval=Path(args.merged_tmmeada_target_eval),
        target=args.target,
    )

    print(f"[INFO] initial baseline missing seeds: {initial_baseline_missing}")
    print(f"[INFO] initial tmmeada missing seeds: {initial_tmmeada_missing}")

    if args.run_missing == 1:
        for seed in seeds:
            if seed in initial_baseline_missing:
                print(f"[QUEUE] run baseline missing seed={seed}")
                run_transfer_one(
                    seed=seed,
                    runner_python=args.runner_python,
                    meaformer_python=args.meaformer_python,
                    source_config=args.baseline_source_config,
                    target_config=args.baseline_target_config,
                    tag=f"baseline_transfer_adapt_{args.target}_expand5_s{seed}",
                    stage_root=args.baseline_stage_root,
                    tmmeada=False,
                )
            else:
                print(f"[SKIP] baseline seed={seed} already available")

            if seed in initial_tmmeada_missing:
                print(f"[QUEUE] run tmmeada missing seed={seed}")
                run_transfer_one(
                    seed=seed,
                    runner_python=args.runner_python,
                    meaformer_python=args.meaformer_python,
                    source_config=args.tmmeada_source_config,
                    target_config=args.tmmeada_target_config,
                    tag=f"tmmeada_transfer_adapt_{args.target}_expand5_s{seed}",
                    stage_root=args.tmmeada_stage_root,
                    tmmeada=True,
                )
            else:
                print(f"[SKIP] tmmeada seed={seed} already available")
    else:
        print("[INFO] run-missing=0, skip training.")

    final_baseline_selected, final_baseline_missing = rebuild_merged_target_eval(
        seeds=seeds,
        candidate_roots=baseline_candidate_roots,
        merged_target_eval=Path(args.merged_baseline_target_eval),
        target=args.target,
    )
    final_tmmeada_selected, final_tmmeada_missing = rebuild_merged_target_eval(
        seeds=seeds,
        candidate_roots=tmmeada_candidate_roots,
        merged_target_eval=Path(args.merged_tmmeada_target_eval),
        target=args.target,
    )

    if final_baseline_selected and final_tmmeada_selected:
        summarize(
            runner_python=args.runner_python,
            baseline_target_dir=args.merged_baseline_target_eval,
            tmmeada_target_dir=args.merged_tmmeada_target_eval,
            prefix=args.report_prefix,
        )
    else:
        print("[WARN] skip summarize because one branch has no available runs.")

    status = {
        "timestamp": now_ts(),
        "target": args.target,
        "seeds": seeds,
        "run_missing": args.run_missing,
        "baseline": {
            "stage_target_eval": str(baseline_stage_target_eval),
            "fallback_target_eval": args.baseline_fallback_target_eval,
            "merged_target_eval": args.merged_baseline_target_eval,
            "initial_selected": initial_baseline_selected,
            "initial_missing_seeds": initial_baseline_missing,
            "final_selected": final_baseline_selected,
            "final_missing_seeds": final_baseline_missing,
        },
        "tmmeada": {
            "stage_target_eval": str(tmmeada_stage_target_eval),
            "fallback_target_eval": args.tmmeada_fallback_target_eval,
            "merged_target_eval": args.merged_tmmeada_target_eval,
            "initial_selected": initial_tmmeada_selected,
            "initial_missing_seeds": initial_tmmeada_missing,
            "final_selected": final_tmmeada_selected,
            "final_missing_seeds": final_tmmeada_missing,
        },
        "report_prefix": args.report_prefix,
    }

    status_json = Path(args.status_json)
    status_md = Path(args.status_md)
    status_json.parent.mkdir(parents=True, exist_ok=True)
    status_json.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# {args.status_title}",
        "",
        f"- timestamp: `{status['timestamp']}`",
        f"- target: `{args.target}`",
        f"- seeds: `{','.join(str(x) for x in seeds)}`",
        f"- run_missing: `{args.run_missing}`",
        "",
        "## Baseline",
        "",
        f"- initial_missing_seeds: `{initial_baseline_missing}`",
        f"- final_missing_seeds: `{final_baseline_missing}`",
        "",
        "## TMMEA-DA",
        "",
        f"- initial_missing_seeds: `{initial_tmmeada_missing}`",
        f"- final_missing_seeds: `{final_tmmeada_missing}`",
        "",
        "## Output",
        "",
        f"- status_json: `{args.status_json}`",
        f"- report_prefix: `{args.report_prefix}`",
    ]
    status_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[DONE] status json -> {status_json}")
    print(f"[DONE] status md -> {status_md}")


if __name__ == "__main__":
    main()
