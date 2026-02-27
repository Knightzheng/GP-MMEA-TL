import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METRIC_FIELDS = [
    "l2r_hits@1",
    "l2r_hits@10",
    "l2r_mrr",
    "r2l_hits@1",
    "r2l_hits@10",
    "r2l_mrr",
]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def main():
    parser = argparse.ArgumentParser(description="Aggregate MEAformer summary csv into mean±std by lang pair.")
    parser.add_argument("--in-csv", default="reports/meaformer_results_summary.csv")
    parser.add_argument("--out-csv", default="reports/meaformer_results_mean_std.csv")
    args = parser.parse_args()

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"missing input csv: {in_path}")

    grouped = defaultdict(lambda: defaultdict(list))
    with open(in_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["lang_pair"]
            for k in METRIC_FIELDS:
                grouped[lang][k].append(float(row[k]))

    out_rows = []
    for lang in sorted(grouped.keys()):
        row = {"lang_pair": lang, "num_runs": len(grouped[lang][METRIC_FIELDS[0]])}
        for k in METRIC_FIELDS:
            row[f"{k}_mean"] = round(mean(grouped[lang][k]), 4)
            row[f"{k}_std"] = round(std(grouped[lang][k]), 4)
        out_rows.append(row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["lang_pair", "num_runs"]
    for k in METRIC_FIELDS:
        fields.extend([f"{k}_mean", f"{k}_std"])
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote aggregated csv: {out_path}")


if __name__ == "__main__":
    main()
