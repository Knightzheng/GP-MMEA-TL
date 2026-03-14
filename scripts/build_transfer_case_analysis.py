import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "transfer"


@dataclass(frozen=True)
class CaseSpec:
    dataset: str
    seed: int
    idx: int
    case_type: str
    subtype: str
    baseline_pred: Path
    method_pred: Path
    ent_ids_1: Path
    ent_ids_2: Path
    attrs_1: Optional[Path] = None


CASES: List[CaseSpec] = [
    CaseSpec(
        dataset="ja_en",
        seed=123,
        idx=3275,
        case_type="failure",
        subtype="regression",
        baseline_pred=ROOT / "data/mmkg/MEAformer/MEAformer_DBP15K_ja_en_transfer_adapt_tgt_ja_en_unsup_il_baseline_transfer_adapt_ja_en_expand5_s123_from_DBP15K_zh_en_s123_il8_b2__test_ep10_pred/MEAformer_DBP15K_ja_en_0.3_ep2_pred.txt",
        method_pred=ROOT / "data/mmkg/MEAformer/MEAformer_DBP15K_ja_en_tmmeada_transfer_adapt_v15_tgt_ja_en_refresh4_da0025_tmmeada_transfer_adapt_ja_v15_s123_from_DBP15K_zh_en_s123_il8_b2__test_ep10_pred/MEAformer_DBP15K_ja_en_0.3_ep2_pred.txt",
        ent_ids_1=ROOT / "data/mmkg/DBP15K/ja_en/ent_ids_1",
        ent_ids_2=ROOT / "data/mmkg/DBP15K/ja_en/ent_ids_2",
    ),
    CaseSpec(
        dataset="ja_en",
        seed=123,
        idx=1201,
        case_type="failure",
        subtype="shared_near_miss",
        baseline_pred=ROOT / "data/mmkg/MEAformer/MEAformer_DBP15K_ja_en_transfer_adapt_tgt_ja_en_unsup_il_baseline_transfer_adapt_ja_en_expand5_s123_from_DBP15K_zh_en_s123_il8_b2__test_ep10_pred/MEAformer_DBP15K_ja_en_0.3_ep2_pred.txt",
        method_pred=ROOT / "data/mmkg/MEAformer/MEAformer_DBP15K_ja_en_tmmeada_transfer_adapt_v15_tgt_ja_en_refresh4_da0025_tmmeada_transfer_adapt_ja_v15_s123_from_DBP15K_zh_en_s123_il8_b2__test_ep10_pred/MEAformer_DBP15K_ja_en_0.3_ep2_pred.txt",
        ent_ids_1=ROOT / "data/mmkg/DBP15K/ja_en/ent_ids_1",
        ent_ids_2=ROOT / "data/mmkg/DBP15K/ja_en/ent_ids_2",
    ),
    CaseSpec(
        dataset="FBDB15K",
        seed=123,
        idx=2283,
        case_type="success",
        subtype="large_rank_recovery",
        baseline_pred=ROOT / "data/mmkg/MEAformer/MEAformer_FBDB15K_0.3_transfer_adapt_tgt_FBDB15K_unsup_il_baseline_transfer_adapt_FBDB15K_expand5_s123_from_DBP15K_zh_en_s123_il8_b2__test_ep10_pred/MEAformer_FBDB15K_norm_0.3_ep2_pred.txt",
        method_pred=ROOT / "data/mmkg/MEAformer/MEAformer_FBDB15K_0.3_tmmeada_transfer_adapt_v18c_tgt_FBDB15K_bipartite_late_il_skiprel_tmmeada_transfer_adapt_v18_fbdb_v18c_s123_from_DBP15K_zh_en_s123_il2_b8__test_ep10_pred/MEAformer_FBDB15K_norm_0.3_ep8_pred.txt",
        ent_ids_1=ROOT / "data/mmkg/FBDB15K/norm/ent_ids_1",
        ent_ids_2=ROOT / "data/mmkg/FBDB15K/norm/ent_ids_2",
        attrs_1=ROOT / "data/mmkg/FBDB15K/norm/training_attrs_1",
    ),
    CaseSpec(
        dataset="FBDB15K",
        seed=123,
        idx=7880,
        case_type="success",
        subtype="large_rank_recovery",
        baseline_pred=ROOT / "data/mmkg/MEAformer/MEAformer_FBDB15K_0.3_transfer_adapt_tgt_FBDB15K_unsup_il_baseline_transfer_adapt_FBDB15K_expand5_s123_from_DBP15K_zh_en_s123_il8_b2__test_ep10_pred/MEAformer_FBDB15K_norm_0.3_ep2_pred.txt",
        method_pred=ROOT / "data/mmkg/MEAformer/MEAformer_FBDB15K_0.3_tmmeada_transfer_adapt_v18c_tgt_FBDB15K_bipartite_late_il_skiprel_tmmeada_transfer_adapt_v18_fbdb_v18c_s123_from_DBP15K_zh_en_s123_il2_b8__test_ep10_pred/MEAformer_FBDB15K_norm_0.3_ep8_pred.txt",
        ent_ids_1=ROOT / "data/mmkg/FBDB15K/norm/ent_ids_1",
        ent_ids_2=ROOT / "data/mmkg/FBDB15K/norm/ent_ids_2",
        attrs_1=ROOT / "data/mmkg/FBDB15K/norm/training_attrs_1",
    ),
    CaseSpec(
        dataset="FBYG15K",
        seed=123,
        idx=4851,
        case_type="success",
        subtype="large_rank_recovery",
        baseline_pred=ROOT / "data/mmkg/MEAformer/MEAformer_FBYG15K_0.3_transfer_adapt_tgt_FBYG15K_unsup_il_baseline_transfer_adapt_fbyg_expand5_s123_from_DBP15K_zh_en_s123_il8_b2__test_ep10_pred/MEAformer_FBYG15K_norm_0.3_ep2_pred.txt",
        method_pred=ROOT / "data/mmkg/MEAformer/MEAformer_FBYG15K_0.3_tmmeada_transfer_adapt_v24b_tgt_FBYG15K_strictsrc_staged_fresh_il_top400_tmmeada_transfer_adapt_v24_fbyg_v24b_s123_from_DBP15K_zh_en_s123_il5_b5__test_ep10_pred/MEAformer_FBYG15K_norm_0.3_ep5_pred.txt",
        ent_ids_1=ROOT / "data/mmkg/FBYG15K/norm/ent_ids_1",
        ent_ids_2=ROOT / "data/mmkg/FBYG15K/norm/ent_ids_2",
        attrs_1=ROOT / "data/mmkg/FBYG15K/norm/training_attrs_1",
    ),
    CaseSpec(
        dataset="FBYG15K",
        seed=123,
        idx=2903,
        case_type="success",
        subtype="large_rank_recovery",
        baseline_pred=ROOT / "data/mmkg/MEAformer/MEAformer_FBYG15K_0.3_transfer_adapt_tgt_FBYG15K_unsup_il_baseline_transfer_adapt_fbyg_expand5_s123_from_DBP15K_zh_en_s123_il8_b2__test_ep10_pred/MEAformer_FBYG15K_norm_0.3_ep2_pred.txt",
        method_pred=ROOT / "data/mmkg/MEAformer/MEAformer_FBYG15K_0.3_tmmeada_transfer_adapt_v24b_tgt_FBYG15K_strictsrc_staged_fresh_il_top400_tmmeada_transfer_adapt_v24_fbyg_v24b_s123_from_DBP15K_zh_en_s123_il5_b5__test_ep10_pred/MEAformer_FBYG15K_norm_0.3_ep5_pred.txt",
        ent_ids_1=ROOT / "data/mmkg/FBYG15K/norm/ent_ids_1",
        ent_ids_2=ROOT / "data/mmkg/FBYG15K/norm/ent_ids_2",
        attrs_1=ROOT / "data/mmkg/FBYG15K/norm/training_attrs_1",
    ),
    CaseSpec(
        dataset="ja_en",
        seed=123,
        idx=9563,
        case_type="failure",
        subtype="boundary_drift",
        baseline_pred=ROOT / "data/mmkg/MEAformer/MEAformer_DBP15K_ja_en_transfer_adapt_tgt_ja_en_unsup_il_baseline_transfer_adapt_ja_en_expand5_s123_from_DBP15K_zh_en_s123_il8_b2__test_ep10_pred/MEAformer_DBP15K_ja_en_0.3_ep2_pred.txt",
        method_pred=ROOT / "data/mmkg/MEAformer/MEAformer_DBP15K_ja_en_tmmeada_transfer_adapt_v15_tgt_ja_en_refresh4_da0025_tmmeada_transfer_adapt_ja_v15_s123_from_DBP15K_zh_en_s123_il8_b2__test_ep10_pred/MEAformer_DBP15K_ja_en_0.3_ep2_pred.txt",
        ent_ids_1=ROOT / "data/mmkg/DBP15K/ja_en/ent_ids_1",
        ent_ids_2=ROOT / "data/mmkg/DBP15K/ja_en/ent_ids_2",
    ),
    CaseSpec(
        dataset="FBDB15K",
        seed=123,
        idx=1959,
        case_type="success",
        subtype="attribute_guided_recovery",
        baseline_pred=ROOT / "data/mmkg/MEAformer/MEAformer_FBDB15K_0.3_transfer_adapt_tgt_FBDB15K_unsup_il_baseline_transfer_adapt_FBDB15K_expand5_s123_from_DBP15K_zh_en_s123_il8_b2__test_ep10_pred/MEAformer_FBDB15K_norm_0.3_ep2_pred.txt",
        method_pred=ROOT / "data/mmkg/MEAformer/MEAformer_FBDB15K_0.3_tmmeada_transfer_adapt_v18c_tgt_FBDB15K_bipartite_late_il_skiprel_tmmeada_transfer_adapt_v18_fbdb_v18c_s123_from_DBP15K_zh_en_s123_il2_b8__test_ep10_pred/MEAformer_FBDB15K_norm_0.3_ep8_pred.txt",
        ent_ids_1=ROOT / "data/mmkg/FBDB15K/norm/ent_ids_1",
        ent_ids_2=ROOT / "data/mmkg/FBDB15K/norm/ent_ids_2",
        attrs_1=ROOT / "data/mmkg/FBDB15K/norm/training_attrs_1",
    ),
]


def load_pred(path: Path) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(line for line in handle if line.strip())
        for row in reader:
            rows[int(row["idx"])] = row
    return rows


def load_ent_map(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            idx, value = line.rstrip("\n").split("\t", 1)
            mapping[int(idx)] = value
    return mapping


def load_attr_map(path: Optional[Path]) -> Dict[str, List[str]]:
    if path is None:
        return {}
    mapping: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            ent = parts[0]
            attrs = [pretty_label(item) for item in parts[1:4]]
            mapping[ent] = [item for item in attrs if item]
    return mapping


def pretty_label(raw: str) -> str:
    value = raw.strip().strip("<>").strip()
    if not value:
        return ""
    value = value.rsplit("/", 1)[-1]
    value = unquote(value)
    return value.replace("_", " ")


def mechanism_text(dataset: str, subtype: str) -> str:
    if dataset == "ja_en" and subtype == "regression":
        return "目标域适应整体有效，但在同域音乐/作品实体的细粒度边界上仍可能发生过度迁移。"
    if dataset == "ja_en":
        return "同系列版本实体高度相似，表面特征和模态证据不足以完全分离近邻候选。"
    if dataset == "FBDB15K":
        return "伪种子质量控制与保守迁移共同降低了跨图谱噪声，帮助模型恢复正确目标实体。"
    return "strict source 与 staged fresh-IL 提升了候选质量，使正确目标在噪声较大的跨图谱场景中重新排到首位。"


def reason_text(dataset: str, subtype: str, baseline_rank: int, method_rank: int) -> str:
    if dataset == "ja_en" and subtype == "regression":
        return f"baseline 已命中 top-1，但方法将注意力转移到语义相近的音乐实体，rank 从 {baseline_rank} 升至 {method_rank}。"
    if dataset == "ja_en":
        return f"baseline 与方法都把正确答案排到第 2 位附近，说明该样本更像是细粒度歧义而非完全失效。"
    return f"baseline 的正确答案排位很靠后（rank={baseline_rank}），方法恢复到 top-1，属于典型的大幅纠错样本。"


def conclusion_text(case_type: str, dataset: str, baseline_rank: int, method_rank: int) -> str:
    if case_type == "success":
        return f"该样本说明方法在 {dataset} 上能把被 baseline 严重误排的目标实体拉回 top-1。"
    return f"该样本说明 {dataset} 上仍存在细粒度近邻歧义，当前方法的增益并不覆盖全部困难样本。"


def build_rows() -> List[Dict[str, str]]:
    ent_cache: Dict[Path, Dict[int, str]] = {}
    pred_cache: Dict[Path, Dict[int, Dict[str, str]]] = {}
    attr_cache: Dict[Path, Dict[str, List[str]]] = {}
    rows: List[Dict[str, str]] = []

    for spec in CASES:
        ent_cache.setdefault(spec.ent_ids_1, load_ent_map(spec.ent_ids_1))
        ent_cache.setdefault(spec.ent_ids_2, load_ent_map(spec.ent_ids_2))
        pred_cache.setdefault(spec.baseline_pred, load_pred(spec.baseline_pred))
        pred_cache.setdefault(spec.method_pred, load_pred(spec.method_pred))
        if spec.attrs_1 is not None:
            attr_cache.setdefault(spec.attrs_1, load_attr_map(spec.attrs_1))

        left_map = ent_cache[spec.ent_ids_1]
        right_map = ent_cache[spec.ent_ids_2]
        base_row = pred_cache[spec.baseline_pred][spec.idx]
        method_row = pred_cache[spec.method_pred][spec.idx]

        query_id = int(method_row["query_id"])
        gt_id = int(method_row["gt_id"])
        base_top1 = int(base_row["ret1"])
        method_top1 = int(method_row["ret1"])
        method_top2 = int(method_row["ret2"])

        source_entity_raw = left_map[query_id]
        source_entity = pretty_label(source_entity_raw)
        source_hint = ""
        if spec.attrs_1 is not None:
            attrs = attr_cache[spec.attrs_1].get(source_entity_raw, [])
            source_hint = ", ".join(attrs[:3])

        rows.append(
            {
                "dataset": spec.dataset,
                "seed": str(spec.seed),
                "idx": str(spec.idx),
                "case_type": spec.case_type,
                "subtype": spec.subtype,
                "source_entity": source_entity,
                "source_id": str(query_id),
                "source_hint": source_hint,
                "ground_truth": pretty_label(right_map[gt_id]),
                "gt_id": str(gt_id),
                "baseline_rank": base_row["rank"],
                "method_rank": method_row["rank"],
                "baseline_top1": pretty_label(right_map[base_top1]),
                "method_top1": pretty_label(right_map[method_top1]),
                "method_top2": pretty_label(right_map[method_top2]),
                "reason": reason_text(spec.dataset, spec.subtype, int(base_row["rank"]), int(method_row["rank"])),
                "mechanism": mechanism_text(spec.dataset, spec.subtype),
                "conclusion": conclusion_text(spec.case_type, spec.dataset, int(base_row["rank"]), int(method_row["rank"])),
            }
        )
    return rows


def write_csv(rows: List[Dict[str, str]]) -> None:
    out_path = REPORT_DIR / "transfer_case_analysis_examples.csv"
    fieldnames = [
        "dataset",
        "seed",
        "idx",
        "case_type",
        "subtype",
        "source_entity",
        "source_id",
        "source_hint",
        "ground_truth",
        "gt_id",
        "baseline_rank",
        "method_rank",
        "baseline_top1",
        "method_top1",
        "method_top2",
        "reason",
        "mechanism",
        "conclusion",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]]) -> None:
    out_path = REPORT_DIR / "transfer_case_analysis_examples.md"
    lines: List[str] = []
    lines.append("# Transfer Case Analysis Examples")
    lines.append("")
    lines.append("## Suggested Subsection Draft")
    lines.append("")
    lines.append(
        f"Table below lists {len(rows)} representative cases selected from current formal variants. "
        "We intentionally include both success and failure examples. "
        "For `ja_en`, we highlight boundary failures to avoid overstating the method. "
        "For `FBDB15K` and `FBYG15K`, we highlight samples where the transfer-enhanced model recovers the correct target from severe baseline ranking errors."
    )
    lines.append("")
    lines.append("| Dataset | Type | Source Entity | GT | Baseline | Ours | Brief Conclusion |")
    lines.append("|---|---|---|---|---:|---:|---|")
    for row in rows:
        source = row["source_entity"]
        if row["source_hint"]:
            source = f"{source} ({row['source_hint']})"
        lines.append(
            f"| {row['dataset']} | {row['case_type']} | {source} | {row['ground_truth']} | "
            f"{row['baseline_rank']} | {row['method_rank']} | {row['conclusion']} |"
        )
    lines.append("")
    lines.append("## Per-Case Notes")
    lines.append("")
    for i, row in enumerate(rows, start=1):
        lines.append(f"### Case {i}: {row['dataset']} / {row['case_type']} / idx={row['idx']}")
        lines.append("")
        lines.append(f"- Source entity: `{row['source_entity']}` (id={row['source_id']})")
        if row["source_hint"]:
            lines.append(f"- Source hint: `{row['source_hint']}`")
        lines.append(f"- Ground truth target: `{row['ground_truth']}` (id={row['gt_id']})")
        lines.append(
            f"- Baseline prediction: top-1=`{row['baseline_top1']}`, rank of GT=`{row['baseline_rank']}`"
        )
        lines.append(
            f"- Our method prediction: top-1=`{row['method_top1']}`, top-2=`{row['method_top2']}`, rank of GT=`{row['method_rank']}`"
        )
        lines.append(f"- Possible reason: {row['reason']}")
        lines.append(f"- Mechanism interpretation: {row['mechanism']}")
        lines.append(f"- One-line takeaway: {row['conclusion']}")
        lines.append("")
    lines.append("## Thesis-Ready Paragraph")
    lines.append("")
    lines.append(
        "The case study further shows that the proposed transfer-enhanced framework brings different types of evidence across datasets. "
        "On `FBDB15K` and `FBYG15K`, the main benefit is large-rank recovery: samples that were ranked hundreds or even thousands of positions away by the baseline are restored to top-1 after introducing target-domain adaptation, stricter pseudo-label control, and more conservative transfer loading. "
        "In contrast, `ja_en` still contains fine-grained boundary failures, especially among highly similar music, media, and product entities. "
        "Therefore, the current evidence supports that the proposed mechanisms improve transfer robustness and candidate quality, while also indicating that cross-lingual fine-grained ambiguity remains an open challenge."
    )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows)
    write_markdown(rows)
    print(f"[OK] wrote {len(rows)} cases")


if __name__ == "__main__":
    main()
