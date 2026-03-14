from pathlib import Path
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
THESIS_DIR = ROOT / "reports" / "thesis"
FIG_DIR = THESIS_DIR / "figures"
TRANSFER_DIR = ROOT / "reports" / "transfer"


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


BLUE = "#2F5D8C"
LIGHT_BLUE = "#8FB9E1"
ORANGE = "#D77A2B"
GREEN = "#4A8F5D"
RED = "#B14D4D"
GRAY = "#6A737D"
LIGHT_GRAY = "#D9DEE5"
BG = "#F6F8FB"


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _add_box(ax, xy, width, height, title, lines, facecolor, edgecolor=BLUE, fontsize=10):
    x, y = xy
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.8,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height - 0.12 * height,
        title,
        ha="center",
        va="center",
        fontsize=fontsize + 1,
        fontweight="bold",
        color=edgecolor,
    )
    ax.text(
        x + width / 2,
        y + height / 2 - 0.04,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1F2933",
        linespacing=1.45,
    )


def _arrow(ax, start, end, color=GRAY, style="-|>", lw=1.8, ls="-"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls),
    )


def create_fig_1_1():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _add_box(
        ax,
        (0.03, 0.2),
        0.17,
        0.58,
        "研究问题",
        [
            "多模态实体对齐模型",
            "跨数据集迁移后性能退化",
            "需要统一、可复现的",
            "迁移实验链路",
        ],
        "#EAF2FB",
    )
    _add_box(
        ax,
        (0.24, 0.2),
        0.17,
        0.58,
        "阶段一：基线复现",
        [
            "复现 MEAformer",
            "统一数据、配置、日志",
            "建立 5-seed 结果留痕",
            "形成可靠 baseline",
        ],
        "#EDF6EE",
        edgecolor=GREEN,
    )
    _add_box(
        ax,
        (0.45, 0.2),
        0.17,
        0.58,
        "阶段二：迁移构建",
        [
            "源域 zh_en 监督训练",
            "参数迁移到目标域",
            "构建 source-train",
            "to target-adapt 流程",
        ],
        "#FFF3E8",
        edgecolor=ORANGE,
    )
    _add_box(
        ax,
        (0.66, 0.2),
        0.17,
        0.58,
        "阶段三：目标域优化",
        [
            "目标域自适应",
            "伪标签迭代更新",
            "质量控制与保守注入",
            "场景化策略调优",
        ],
        "#FCEEEF",
        edgecolor=RED,
    )
    _add_box(
        ax,
        (0.84, 0.2),
        0.13,
        0.58,
        "阶段四：统一评估",
        [
            "4 个目标域主表",
            "统计显著性",
            "案例与误差分析",
            "效率与局限性",
        ],
        "#F1F4F8",
        edgecolor=GRAY,
    )

    for x0, x1 in [(0.20, 0.24), (0.41, 0.45), (0.62, 0.66), (0.83, 0.84)]:
        _arrow(ax, (x0, 0.49), (x1, 0.49))

    ax.text(0.5, 0.92, "GP-MMEA-TL 总体技术路线", ha="center", fontsize=17, fontweight="bold", color="#1F2933")
    ax.text(0.5, 0.08, "从 baseline 复现出发，逐步推进到统一迁移实验链路、目标域自适应优化及论文级证据整理。", ha="center", fontsize=10.5, color=GRAY)
    return _save(fig, "fig1_1_technical_route.png")


def create_fig_3_1():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _add_box(
        ax,
        (0.04, 0.62),
        0.18,
        0.22,
        "源域输入",
        [
            "G_s^1, G_s^2",
            "监督对齐链接 L_s",
            "结构/文本/属性/图像",
        ],
        "#EAF2FB",
    )
    _add_box(
        ax,
        (0.29, 0.58),
        0.18,
        0.30,
        "MEAformer 骨干",
        [
            "多模态表示学习",
            "共享实体编码空间",
            "源域监督训练",
        ],
        "#EDF6EE",
        edgecolor=GREEN,
    )
    _add_box(
        ax,
        (0.54, 0.62),
        0.18,
        0.22,
        "目标域初始化",
        [
            "加载源域参数",
            "生成初始实体表示",
            "建立迁移起点",
        ],
        "#FFF3E8",
        edgecolor=ORANGE,
    )
    _add_box(
        ax,
        (0.79, 0.62),
        0.17,
        0.22,
        "目标域输出",
        [
            "实体相似度排序",
            "最终对齐结果",
            "评估指标统计",
        ],
        "#F1F4F8",
        edgecolor=GRAY,
    )

    _add_box(
        ax,
        (0.31, 0.16),
        0.20,
        0.24,
        "目标域自适应",
        [
            "无监督 target adapt",
            "域对齐辅助约束",
            "逐步贴近目标域分布",
        ],
        "#FCEEEF",
        edgecolor=RED,
    )
    _add_box(
        ax,
        (0.57, 0.13),
        0.24,
        0.30,
        "伪标签质量控制",
        [
            "候选生成与高置信筛选",
            "保守注入与阶段刷新",
            "严格源模型匹配/一致性控制",
            "降低噪声传播风险",
        ],
        "#EEF3F7",
        edgecolor=BLUE,
    )
    _add_box(
        ax,
        (0.05, 0.18),
        0.18,
        0.20,
        "辅助探索模块",
        [
            "source_select",
            "missing_gate",
            "作为补充性探索",
        ],
        "#F6F8FB",
        edgecolor=GRAY,
    )

    _arrow(ax, (0.22, 0.73), (0.29, 0.73))
    _arrow(ax, (0.47, 0.73), (0.54, 0.73))
    _arrow(ax, (0.72, 0.73), (0.79, 0.73))
    _arrow(ax, (0.38, 0.58), (0.41, 0.40), color=GREEN)
    _arrow(ax, (0.63, 0.62), (0.67, 0.43), color=ORANGE)
    _arrow(ax, (0.81, 0.43), (0.87, 0.62), color=BLUE)
    _arrow(ax, (0.23, 0.28), (0.31, 0.28), color=GRAY, ls="--")
    _arrow(ax, (0.51, 0.28), (0.57, 0.28), color=RED)

    ax.text(0.5, 0.94, "本文方法整体框架", ha="center", fontsize=17, fontweight="bold", color="#1F2933")
    ax.text(0.5, 0.05, "骨干保持统一，重点增强目标域自适应与伪标签质量控制。", ha="center", fontsize=10.5, color=GRAY)
    return _save(fig, "fig3_1_method_framework.png")


def create_fig_4_1():
    df = pd.read_csv(TRANSFER_DIR / "transfer_adapt_significance_summary.csv")
    order = ["ja_en", "fr_en", "FBDB15K", "FBYG15K"]
    df["target"] = pd.Categorical(df["target"], categories=order, ordered=True)
    df = df.sort_values("target")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    y = range(len(df))
    colors = [BLUE if s == "cross_lingual" else ORANGE for s in df["scenario"]]
    deltas = df["delta_avg_mrr_mean"].to_numpy()
    lows = deltas - df["delta_mrr_ci95_lo"].to_numpy()
    highs = df["delta_mrr_ci95_hi"].to_numpy() - deltas
    ax.errorbar(
        deltas,
        list(y),
        xerr=[lows, highs],
        fmt="o",
        color=GRAY,
        ecolor=GRAY,
        elinewidth=2,
        capsize=4,
        ms=0,
    )
    ax.scatter(deltas, list(y), s=120, c=colors, zorder=3)
    for i, delta in enumerate(deltas):
        ax.text(delta + 0.00035, i, f"{delta:+.4f}", va="center", fontsize=10, color="#1F2933")
    ax.axvline(0, color=LIGHT_GRAY, linestyle="--", linewidth=1.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["target"].tolist(), fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("delta avg MRR", fontsize=11)
    ax.set_title("四个目标域 delta avg MRR 及 95% bootstrap CI", fontsize=14, pad=14)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_facecolor("white")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=9, label="跨语言"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=9, label="跨图谱"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left")
    return _save(fig, "fig4_1_delta_ci.png")


def create_fig_4_2():
    scenario = ["跨语言", "跨图谱"]
    delta_mrr = [0.0121, 0.00555]
    delta_hits1 = [0.0105, 0.00325]

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    x = range(len(scenario))
    width = 0.32
    ax.bar([i - width / 2 for i in x], delta_mrr, width=width, color=BLUE, label="avg delta MRR")
    ax.bar([i + width / 2 for i in x], delta_hits1, width=width, color=LIGHT_BLUE, label="avg delta Hits@1")

    for i, val in enumerate(delta_mrr):
        ax.text(i - width / 2, val + 0.00035, f"{val:.4f}", ha="center", fontsize=10)
    for i, val in enumerate(delta_hits1):
        ax.text(i + width / 2, val + 0.00035, f"{val:.4f}", ha="center", fontsize=10)

    ax.set_xticks(list(x))
    ax.set_xticklabels(scenario, fontsize=11)
    ax.set_ylabel("平均增益", fontsize=11)
    ax.set_title("跨语言与跨图谱场景下的平均增益对比", fontsize=14, pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper right")
    return _save(fig, "fig4_2_scenario_gain.png")


def create_fig_4_3():
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.2))

    fbdb_steps = [
        ("早期参考版\nv7 expand5", 0.0008, "早期迁移链路"),
        ("v18a pilot", 0.0075, "伪种子质量改造"),
        ("v18b pilot", 0.0070, "保持正增益"),
        ("v18c pilot", 0.0080, "skip rel transfer"),
        ("v18c formal 5-seed", 0.0083, "正式主表版本"),
    ]
    fbyg_steps = [
        ("v19c pilot", 0.0010, "晚启 IL 初现正增益"),
        ("v21a formal 5-seed", 0.0016, "staged fresh-IL"),
        ("v24b formal 5-seed", 0.0028, "strict source + top400"),
    ]

    for ax, title, steps, color in [
        (axes[0], "FBDB15K 阶段优化路径", fbdb_steps, ORANGE),
        (axes[1], "FBYG15K 阶段优化路径", fbyg_steps, BLUE),
    ]:
        ax.set_facecolor(BG)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_ylabel("delta avg MRR")
        xs = list(range(len(steps)))
        ys = [s[1] for s in steps]
        ax.plot(xs, ys, color=color, marker="o", linewidth=2.6, markersize=8)
        ax.fill_between(xs, ys, [0] * len(xs), color=color, alpha=0.08)
        ax.set_xticks(xs)
        ax.set_xticklabels([s[0] for s in steps], fontsize=10)
        for i, (_, yv, note) in enumerate(steps):
            ax.text(i, yv + max(ys) * 0.04 + 0.00015, f"{yv:+.4f}\n{note}", ha="center", fontsize=9.2)
        ax.set_title(title, fontsize=13, pad=10)

    axes[1].text(
        0.99,
        0.08,
        "注：FBYG15K 中 adaptive top-k 后续被验证可工作，但未超过当前 formal 主表版本。",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=9.2,
        color=GRAY,
    )
    fig.suptitle("跨图谱场景中的阶段优化路径示意", fontsize=15, y=0.98, fontweight="bold")
    return _save(fig, "fig4_3_stage_paths.png")


def create_fig_4_4():
    df = pd.read_csv(TRANSFER_DIR / "transfer_case_analysis_examples.csv")
    labels = ["ja-1", "ja-2", "fbdb-1", "fbdb-2", "fbyg-1", "fbyg-2"]
    baseline_pos = (df["baseline_rank"] + 1).tolist()
    method_pos = (df["method_rank"] + 1).tolist()
    y = list(range(len(df)))[::-1]

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    for yi, b, m, ctype in zip(y, baseline_pos, method_pos, df["case_type"]):
        line_color = GREEN if m < b else RED if m > b else GRAY
        ax.plot([b, m], [yi, yi], color=line_color, linewidth=2.2, alpha=0.85)
    ax.scatter(baseline_pos, y, color=GRAY, s=70, label="Baseline", zorder=3)
    ax.scatter(method_pos, y, color=BLUE, s=70, label="本文方法", zorder=3)

    for yi, b, m in zip(y, baseline_pos, method_pos):
        ax.text(b * 1.05, yi + 0.08, str(b), fontsize=9, color=GRAY)
        ax.text(max(m * 1.05, 1.12), yi - 0.22, str(m), fontsize=9, color=BLUE)

    ax.set_xscale("log")
    ax.set_xlabel("正确目标排序位置（对数坐标，越小越好）", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_title("代表性案例中正确目标排序位置对比", fontsize=14, pad=14)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.01, -0.13, "ja 表示 ja_en；fbdb 表示 FBDB15K；fbyg 表示 FBYG15K。", transform=ax.transAxes, fontsize=9.5, color=GRAY)
    return _save(fig, "fig4_4_case_rank_recovery.png")


def main():
    outputs = [
        create_fig_1_1(),
        create_fig_3_1(),
        create_fig_4_1(),
        create_fig_4_2(),
        create_fig_4_3(),
        create_fig_4_4(),
    ]
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
