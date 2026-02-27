import argparse
import csv
import re
from pathlib import Path


L2R_RE = re.compile(r"l2r: acc of top \[1, 10, 50\] = \[([0-9.\s]+)\], mr = ([0-9.]+), mrr = ([0-9.]+)")
R2L_RE = re.compile(r"r2l: acc of top \[1, 10, 50\] = \[([0-9.\s]+)\], mr = ([0-9.]+), mrr = ([0-9.]+)")


def parse_top_vals(s):
    parts = [p for p in s.strip().split() if p]
    vals = [float(x) for x in parts[:3]]
    while len(vals) < 3:
        vals.append(0.0)
    return vals


def extract_metrics(log_text):
    l2r = None
    r2l = None
    for line in log_text.splitlines():
        if "Test result" in line:
            continue
        if "Ep" in line and "l2r: acc of top [1, 10, 50]" in line:
            m = L2R_RE.search(line)
            if m:
                tops = parse_top_vals(m.group(1))
                l2r = {"hits@1": tops[0], "hits@10": tops[1], "hits@50": tops[2], "mr": float(m.group(2)), "mrr": float(m.group(3))}
        if "Ep" in line and "r2l: acc of top [1, 10, 50]" in line:
            m = R2L_RE.search(line)
            if m:
                tops = parse_top_vals(m.group(1))
                r2l = {"hits@1": tops[0], "hits@10": tops[1], "hits@50": tops[2], "mr": float(m.group(2)), "mrr": float(m.group(3))}
    return l2r, r2l


def infer_lang(run_name, cfg_path):
    for tag in ("zh_en", "ja_en", "fr_en"):
        if f"-{tag}-" in run_name:
            return tag
    for tag in ("FBDB15K", "FBYG15K"):
        if f"-{tag}-" in run_name:
            return tag
    if cfg_path:
        if "zh_en" in cfg_path:
            return "zh_en"
        if "ja_en" in cfg_path:
            return "ja_en"
        if "fr_en" in cfg_path:
            return "fr_en"
        if "FBDB15K" in cfg_path:
            return "FBDB15K"
        if "FBYG15K" in cfg_path:
            return "FBYG15K"
    if "-zh_en-" in run_name:
        return "zh_en"
    if "-ja_en-" in run_name:
        return "ja_en"
    if "-fr_en-" in run_name:
        return "fr_en"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Collect MEAformer run metrics into a CSV summary.")
    parser.add_argument("--runs-dir", default="runs/baseline")
    parser.add_argument("--out", default="reports/meaformer_results_summary.csv")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []
    for run_dir in sorted(runs_dir.glob("*MEAformer*")):
        log_path = run_dir / "log.txt"
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        l2r, r2l = extract_metrics(text)
        if not l2r or not r2l:
            continue
        cfg_path = ""
        cfg_file = run_dir / "config.yaml"
        if cfg_file.exists():
            cfg_path = str(cfg_file)
        rows.append(
            {
                "run_id": run_dir.name,
                "lang_pair": infer_lang(run_dir.name, cfg_path),
                "l2r_hits@1": l2r["hits@1"],
                "l2r_hits@10": l2r["hits@10"],
                "l2r_mrr": l2r["mrr"],
                "r2l_hits@1": r2l["hits@1"],
                "r2l_hits@10": r2l["hits@10"],
                "r2l_mrr": r2l["mrr"],
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "lang_pair", "l2r_hits@1", "l2r_hits@10", "l2r_mrr", "r2l_hits@1", "r2l_hits@10", "r2l_mrr"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary: {out_path} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
