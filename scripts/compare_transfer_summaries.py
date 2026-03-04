import argparse
import csv
from pathlib import Path


METRICS = [
    "l2r_hits@1",
    "l2r_hits@10",
    "l2r_mrr",
    "r2l_hits@1",
    "r2l_hits@10",
    "r2l_mrr",
]


def read_summary(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["lang_pair"]
            rows[lang] = row
    return rows


def to_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Compare transfer summary CSVs (method vs baseline).")
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--method-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--method-name", default="method")
    args = parser.parse_args()

    baseline = read_summary(Path(args.baseline_csv))
    method = read_summary(Path(args.method_csv))
    langs = sorted(set(baseline.keys()) | set(method.keys()))

    out_rows = []
    for lang in langs:
        b = baseline.get(lang, {})
        m = method.get(lang, {})
        row = {"lang_pair": lang}
        for metric in METRICS:
            b_val = to_float(b, metric) if b else 0.0
            m_val = to_float(m, metric) if m else 0.0
            row[f"{args.baseline_name}_{metric}"] = round(b_val, 6)
            row[f"{args.method_name}_{metric}"] = round(m_val, 6)
            row[f"delta_{metric}"] = round(m_val - b_val, 6)
        out_rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["lang_pair"]
    for metric in METRICS:
        fieldnames.extend(
            [
                f"{args.baseline_name}_{metric}",
                f"{args.method_name}_{metric}",
                f"delta_{metric}",
            ]
        )
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[DONE] compare csv -> {out_csv}")

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append(f"# Transfer Compare: {args.method_name} vs {args.baseline_name}")
        lines.append("")
        lines.append("| lang_pair | delta_l2r_hits@1 | delta_l2r_hits@10 | delta_l2r_mrr | delta_r2l_hits@1 | delta_r2l_hits@10 | delta_r2l_mrr |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in out_rows:
            lines.append(
                f"| {row['lang_pair']} | "
                f"{row['delta_l2r_hits@1']:+.6f} | {row['delta_l2r_hits@10']:+.6f} | {row['delta_l2r_mrr']:+.6f} | "
                f"{row['delta_r2l_hits@1']:+.6f} | {row['delta_r2l_hits@10']:+.6f} | {row['delta_r2l_mrr']:+.6f} |"
            )
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[DONE] compare md -> {out_md}")


if __name__ == "__main__":
    main()
