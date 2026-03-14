import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs" / "transfer"
REPORT_DIR = ROOT / "reports" / "transfer"


TARGET_SPECS = [
    {
        "target": "ja_en",
        "scenario": "cross_lingual",
        "variant": "v15_refresh4_da0025_expand5",
        "baseline_dir": RUNS_DIR / "transfer_adapt_ja_v15_full_baseline_matched_ref" / "target_eval",
        "method_dir": RUNS_DIR / "transfer_adapt_ja_v15_full_ref" / "target_eval",
    },
    {
        "target": "FBDB15K",
        "scenario": "cross_graph",
        "variant": "v18c_bipartite_late_il_skiprel_expand5",
        "baseline_dir": RUNS_DIR / "transfer_adapt_v18_fbdb_v18c_expand5_baseline_matched_ref" / "target_eval",
        "method_dir": RUNS_DIR / "transfer_adapt_v18_fbdb_v18c_expand5_ref" / "target_eval",
    },
    {
        "target": "fr_en",
        "scenario": "cross_lingual",
        "variant": "v14b_refresh4_da0025_expand5",
        "baseline_dir": RUNS_DIR / "transfer_adapt_v14_fren_expand5_merged_baseline" / "target_eval",
        "method_dir": RUNS_DIR / "transfer_adapt_v14_fren_expand5_merged_tmmeada" / "target_eval",
    },
    {
        "target": "FBYG15K",
        "scenario": "cross_graph",
        "variant": "v24b_strictsrc_staged_fresh_il_top400_expand5",
        "baseline_dir": RUNS_DIR / "transfer_adapt_v24_fbyg_v24b_expand5_baseline_matched_ref" / "target_eval",
        "method_dir": RUNS_DIR / "transfer_adapt_v24_fbyg_v24b_expand5_ref" / "target_eval",
    },
]


PER_SEED_OUT = REPORT_DIR / "transfer_adapt_significance_per_seed.csv"
SUMMARY_OUT_CSV = REPORT_DIR / "transfer_adapt_significance_summary.csv"
SUMMARY_OUT_MD = REPORT_DIR / "transfer_adapt_significance_summary.md"
WRITEUP_OUT_MD = REPORT_DIR / "transfer_adapt_significance_writeup.md"


FINAL_METRIC_RE = re.compile(
    r"Ep\s+\d+\s+\|\s+(l2r|r2l):\s+acc of top \[1, 10, 50\]\s*=\s*\[([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s*\],\s*mr\s*=\s*([0-9eE+\-.]+),\s*mrr\s*=\s*([0-9eE+\-.]+)"
)
SEED_RE = re.compile(r"-s(\d+)$")


def parse_seed(run_name: str) -> int:
    match = SEED_RE.search(run_name)
    if not match:
        raise ValueError(f"Cannot parse seed from run name: {run_name}")
    return int(match.group(1))


def parse_final_metrics(log_path: Path) -> Dict[str, float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = FINAL_METRIC_RE.findall(text)
    if not matches:
        raise RuntimeError(f"No metrics found in log: {log_path}")

    last_by_side: Dict[str, Tuple[float, float, float, float]] = {}
    for side, h1, h10, _, mr, mrr in matches:
        last_by_side[side] = (float(h1), float(h10), float(mrr), float(mr))

    if "l2r" not in last_by_side or "r2l" not in last_by_side:
        raise RuntimeError(f"Incomplete l2r/r2l metrics in log: {log_path}")

    l_h1, l_h10, l_mrr, l_mr = last_by_side["l2r"]
    r_h1, r_h10, r_mrr, r_mr = last_by_side["r2l"]
    return {
        "l2r_hits@1": l_h1,
        "l2r_hits@10": l_h10,
        "l2r_mrr": l_mrr,
        "l2r_mr": l_mr,
        "r2l_hits@1": r_h1,
        "r2l_hits@10": r_h10,
        "r2l_mrr": r_mrr,
        "r2l_mr": r_mr,
        "avg_hits@1": (l_h1 + r_h1) / 2.0,
        "avg_hits@10": (l_h10 + r_h10) / 2.0,
        "avg_mrr": (l_mrr + r_mrr) / 2.0,
        "avg_mr": (l_mr + r_mr) / 2.0,
    }


def load_run_metrics(run_root: Path) -> Dict[int, Dict[str, float]]:
    if not run_root.exists():
        raise FileNotFoundError(f"Missing run root: {run_root}")

    result: Dict[int, Dict[str, float]] = {}
    for run_dir in sorted(x for x in run_root.iterdir() if x.is_dir()):
        seed = parse_seed(run_dir.name)
        result[seed] = parse_final_metrics(run_dir / "log.txt")
    return result


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 20000, seed: int = 20260314) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[idx].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def sign_test_p_greater(deltas: np.ndarray) -> float:
    non_zero = deltas[np.abs(deltas) > 1e-12]
    n = len(non_zero)
    wins = int((non_zero > 0).sum())
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)


def wilcoxon_p_greater(method_vals: np.ndarray, baseline_vals: np.ndarray) -> Optional[float]:
    if wilcoxon is None:
        return None
    try:
        result = wilcoxon(method_vals, baseline_vals, alternative="greater", zero_method="wilcox", mode="exact")
        return float(result.pvalue)
    except Exception:
        return None


def mean_std_text(values: np.ndarray) -> str:
    return f"{values.mean():.4f} +- {values.std(ddof=1):.4f}"


def load_all() -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    per_seed_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for spec in TARGET_SPECS:
        baseline = load_run_metrics(spec["baseline_dir"])
        method = load_run_metrics(spec["method_dir"])
        seeds = sorted(set(baseline.keys()) & set(method.keys()))
        if len(seeds) != 5:
            raise RuntimeError(f"{spec['target']}: expected 5 paired seeds, got {seeds}")

        b_mrr = []
        m_mrr = []
        deltas = []
        for seed in seeds:
            b = baseline[seed]
            m = method[seed]
            delta_h1 = m["avg_hits@1"] - b["avg_hits@1"]
            delta_h10 = m["avg_hits@10"] - b["avg_hits@10"]
            delta_mrr = m["avg_mrr"] - b["avg_mrr"]
            delta_mr = m["avg_mr"] - b["avg_mr"]
            per_seed_rows.append(
                {
                    "target": spec["target"],
                    "scenario": spec["scenario"],
                    "variant": spec["variant"],
                    "seed": seed,
                    "baseline_avg_hits@1": b["avg_hits@1"],
                    "method_avg_hits@1": m["avg_hits@1"],
                    "delta_avg_hits@1": delta_h1,
                    "baseline_avg_hits@10": b["avg_hits@10"],
                    "method_avg_hits@10": m["avg_hits@10"],
                    "delta_avg_hits@10": delta_h10,
                    "baseline_avg_mrr": b["avg_mrr"],
                    "method_avg_mrr": m["avg_mrr"],
                    "delta_avg_mrr": delta_mrr,
                    "baseline_avg_mr": b["avg_mr"],
                    "method_avg_mr": m["avg_mr"],
                    "delta_avg_mr": delta_mr,
                }
            )
            b_mrr.append(b["avg_mrr"])
            m_mrr.append(m["avg_mrr"])
            deltas.append(delta_mrr)

        b_mrr_np = np.array(b_mrr, dtype=float)
        m_mrr_np = np.array(m_mrr, dtype=float)
        deltas_np = np.array(deltas, dtype=float)
        ci_lo, ci_hi = bootstrap_mean_ci(deltas_np)
        wins = int((deltas_np > 0).sum())
        sign_p = sign_test_p_greater(deltas_np)
        wilcoxon_p = wilcoxon_p_greater(m_mrr_np, b_mrr_np)
        summary_rows.append(
            {
                "target": spec["target"],
                "scenario": spec["scenario"],
                "variant": spec["variant"],
                "n_pairs": len(seeds),
                "baseline_avg_mrr_mean": float(b_mrr_np.mean()),
                "baseline_avg_mrr_std": float(b_mrr_np.std(ddof=1)),
                "method_avg_mrr_mean": float(m_mrr_np.mean()),
                "method_avg_mrr_std": float(m_mrr_np.std(ddof=1)),
                "delta_avg_mrr_mean": float(deltas_np.mean()),
                "delta_avg_mrr_std": float(deltas_np.std(ddof=1)),
                "delta_mrr_ci95_lo": ci_lo,
                "delta_mrr_ci95_hi": ci_hi,
                "positive_seed_wins": wins,
                "sign_test_p_one_sided": sign_p,
                "wilcoxon_p_one_sided": wilcoxon_p if wilcoxon_p is not None else "",
                "all_seeds_positive": wins == len(seeds),
            }
        )

    return per_seed_rows, summary_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_paper_paragraphs(summary_rows: List[Dict[str, object]]) -> Tuple[str, str]:
    lines = []
    defense_lines = []
    lines.append("## Paper-Ready Paragraph")
    lines.append("")
    lines.append(
        "To strengthen the stability claim of the transfer results, we further performed paired significance analysis on the final 5-seed results. "
        "Because each target uses matched baseline/method runs under the same random seeds and the sample size is small (`n=5`), we use paired bootstrap confidence intervals on the seed-wise `avg MRR` gain as the primary uncertainty estimate, and report an exact one-sided sign test as a robustness check. "
        "This choice is more appropriate than relying only on a paired t-test under such a small-sample setting."
    )
    lines.append("")
    for row in summary_rows:
        target = row["target"]
        delta = row["delta_avg_mrr_mean"]
        lo = row["delta_mrr_ci95_lo"]
        hi = row["delta_mrr_ci95_hi"]
        wins = row["positive_seed_wins"]
        p = row["sign_test_p_one_sided"]
        if row["all_seeds_positive"]:
            lines.append(
                f"On `{target}`, the proposed method improves `avg MRR` by `{delta:+.4f}` over the matched baseline, "
                f"with a paired bootstrap `95% CI [{lo:+.4f}, {hi:+.4f}]`. "
                f"All `{wins}/5` seeds show positive gains, and the exact one-sided sign test gives `p={p:.4f}`, "
                "indicating that the improvement is stable under the matched-seed setting."
            )
            defense_lines.append(
                f"`{target}` 这项我们不是只看均值，而是看了 5 个配对 seed。5/5 个 seed 都比 baseline 好，`avg MRR` 的 paired bootstrap 95% CI 也保持为正，exact one-sided sign test `p={p:.4f}`，所以可以说提升在当前 5-seed 配对设定下是稳定且有统计支持的。"
            )
        else:
            lines.append(
                f"On `{target}`, the mean `avg MRR` gain is `{delta:+.4f}`, with a paired bootstrap `95% CI [{lo:+.4f}, {hi:+.4f}]`. "
                f"Positive gains are observed on `{wins}/5` seeds, while the exact one-sided sign test yields `p={p:.4f}`. "
                "This suggests a positive trend, but the evidence should be interpreted as supportive rather than strongly conclusive."
            )
            defense_lines.append(
                f"`{target}` 的均值提升是正的，但不是 5/5 个 seed 全部获益，所以我会更克制地表述为：在当前 5-seed 配对结果下，方法呈现稳定正趋势，bootstrap CI 和 seed-level 对比支持这个方向，但统计证据强度弱于全正的目标域。"
            )

    return "\n".join(lines) + "\n", "\n".join(["## Defense-Ready Answers", ""] + [f"- {x}" for x in defense_lines]) + "\n"


def write_summary_md(path: Path, summary_rows: List[Dict[str, object]]) -> None:
    lines = []
    lines.append("# Transfer-Adapt Significance Summary")
    lines.append("")
    lines.append("## Recommended Statistical Setting")
    lines.append("")
    lines.append("- Primary uncertainty estimate: paired bootstrap `95% CI` on seed-wise `avg MRR` gain.")
    lines.append("- Primary small-sample significance check: exact one-sided sign test on paired seed deltas.")
    lines.append("- Supplementary check: exact one-sided Wilcoxon signed-rank test when available.")
    lines.append("- Not recommended as the only evidence: paired t-test, because `n=5` is too small to rely on normality assumptions.")
    lines.append("")
    lines.append("## Paper Table")
    lines.append("")
    lines.append("| target | scenario | baseline avg MRR (mean+-std) | method avg MRR (mean+-std) | delta avg MRR | bootstrap 95% CI | positive seeds | sign test p (one-sided) | Wilcoxon p (one-sided) |")
    lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|")
    for row in summary_rows:
        wilcoxon_str = f"{row['wilcoxon_p_one_sided']:.4f}" if row["wilcoxon_p_one_sided"] != "" else "NA"
        lines.append(
            f"| {row['target']} | {row['scenario']} | "
            f"{row['baseline_avg_mrr_mean']:.4f}+-{row['baseline_avg_mrr_std']:.4f} | "
            f"{row['method_avg_mrr_mean']:.4f}+-{row['method_avg_mrr_std']:.4f} | "
            f"{row['delta_avg_mrr_mean']:+.4f} | "
            f"[{row['delta_mrr_ci95_lo']:+.4f}, {row['delta_mrr_ci95_hi']:+.4f}] | "
            f"{row['positive_seed_wins']}/5 | {row['sign_test_p_one_sided']:.4f} | {wilcoxon_str} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_writeup_md(path: Path, summary_rows: List[Dict[str, object]]) -> None:
    paper_text, defense_text = build_paper_paragraphs(summary_rows)
    lines = [
        "# Transfer-Adapt Significance Writeup",
        "",
        "## Recommended Use",
        "",
        "- Main metric for significance discussion: `avg MRR`.",
        "- Preferred wording: use `stable under 5-seed paired setting` or `supported by paired bootstrap and seed-level tests`.",
        "- Avoid overclaiming weak targets as universally significant if not all 5 seeds win.",
        "",
        paper_text.rstrip(),
        "",
        defense_text.rstrip(),
        "",
        "## Suggested Thesis Footnote",
        "",
        "Because the final transfer table is organized as matched baseline/method results under the same five random seeds, we evaluate significance on paired seed-level deltas rather than on independent samples.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    per_seed_rows, summary_rows = load_all()
    write_csv(PER_SEED_OUT, per_seed_rows)
    write_csv(SUMMARY_OUT_CSV, summary_rows)
    write_summary_md(SUMMARY_OUT_MD, summary_rows)
    write_writeup_md(WRITEUP_OUT_MD, summary_rows)
    print(f"[DONE] per-seed csv: {PER_SEED_OUT}")
    print(f"[DONE] summary csv : {SUMMARY_OUT_CSV}")
    print(f"[DONE] summary md  : {SUMMARY_OUT_MD}")
    print(f"[DONE] writeup md  : {WRITEUP_OUT_MD}")


if __name__ == "__main__":
    main()
