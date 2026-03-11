import re
import shutil
from pathlib import Path

import yaml


L2R_RE = re.compile(
    r"Ep (?:Test|[0-9]+) \| l2r: acc of top \[1, 10, 50\] = \[(?P<h1>[0-9.]+)\s+(?P<h10>[0-9.]+)\s+(?P<h50>[0-9.]+)\s*\], mr = (?P<mr>[0-9.]+), mrr = (?P<mrr>[0-9.]+)"
)
R2L_RE = re.compile(
    r"Ep (?:Test|[0-9]+) \| r2l: acc of top \[1, 10, 50\] = \[(?P<h1>[0-9.]+)\s+(?P<h10>[0-9.]+)\s+(?P<h50>[0-9.]+)\s*\], mr = (?P<mr>[0-9.]+), mrr = (?P<mrr>[0-9.]+)"
)
RETURN_CODE_OK_MARKER = "[DONE] return_code=0"


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
    if RETURN_CODE_OK_MARKER not in text:
        return False
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
        run_seed, run_target = parsed
        if run_seed == seed and run_target == target:
            cands.append(run_dir)
    return cands


def latest_complete_run_for_seed_target_from_roots(target_eval_roots, seed: int, target: str):
    cands = []
    for root in target_eval_roots:
        cands.extend(matching_runs_for_seed_target_in_target_eval(Path(root), seed, target))
    complete = [run_dir for run_dir in cands if is_complete_target_eval_run(run_dir)]
    if not complete:
        return None
    complete.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return complete[0]


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
        run_dir = latest_complete_run_for_seed_target_from_roots(
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


def resolve_source_model_name(seed: int, tmmeada: bool = False) -> str:
    save_dir = Path("data/mmkg/MEAformer/save")
    if not save_dir.exists():
        return ""

    if tmmeada:
        exact = save_dir / (
            "MEAformer_DBP15K_zh_en_"
            f"tmmeada_transfer_src_zh_en_epoch10_tmmeada_transfer_formal_s{seed}_src_s{seed}_.pkl"
        )
        pattern = f"*tmmeada_transfer_formal_s{seed}*src_s{seed}*.pkl"
    else:
        exact = save_dir / (
            "MEAformer_DBP15K_zh_en_"
            f"transfer_src_zh_en_epoch10_baseline_transfer_formal_s{seed}_src_s{seed}_.pkl"
        )
        pattern = f"*baseline_transfer_formal_s{seed}*src_s{seed}*.pkl"

    if exact.exists():
        return exact.stem

    matches = sorted(save_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0].stem if matches else ""
