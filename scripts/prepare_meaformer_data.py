import argparse
import csv
import hashlib
import json
import pickle
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


def log(msg, log_path):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_required_files(src_split_dir, dst_dir):
    required = ["ent_ids_1", "ent_ids_2", "triples_1", "triples_2"]
    for name in required:
        src = src_split_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing source file: {src}")
        shutil.copy2(src, dst_dir / name)


def copy_attr_files(src_lang_root, dst_dir):
    for name in ["training_attrs_1", "training_attrs_2"]:
        src = src_lang_root / name
        if not src.exists():
            # fallback: create empty placeholder to keep pipeline unblocked
            (dst_dir / name).write_text("", encoding="utf-8")
            continue
        shutil.copy2(src, dst_dir / name)


def copy_ill(src_split_dir, dst_dir):
    sup = src_split_dir / "sup_ent_ids"
    ref_ = src_split_dir / "ref_ent_ids"
    if not sup.exists() or not ref_.exists():
        raise FileNotFoundError(f"missing sup/ref ent id files: {sup}, {ref_}")
    out = dst_dir / "ill_ent_ids"
    with open(out, "w", encoding="utf-8") as fw:
        for p in (sup, ref_):
            with open(p, "r", encoding="utf-8", errors="replace") as fr:
                for line in fr:
                    fw.write(line)


def parse_entity_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 1 and parts[0].isdigit():
                ids.append(int(parts[0]))
    return ids


def build_random_img_features(dst_mmkg_root, lang_pair, all_entity_ids, seed=3407, dim=300):
    random.seed(seed)
    np.random.seed(seed)
    pkl_dir = dst_mmkg_root / "pkls"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = pkl_dir / f"{lang_pair}_GA_id_img_feature_dict.pkl"

    # Keep feature file small but valid for code path.
    img_dict = {eid: np.random.normal(0, 1, dim).astype(np.float32) for eid in all_entity_ids}
    with open(pkl_path, "wb") as f:
        pickle.dump(img_dict, f)
    return pkl_path


def write_stats(path, dataset, split, ent_count, triple_1, triple_2, ill_count):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "split", "entity_count", "triples_1", "triples_2", "ill_count"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": dataset,
                "split": split,
                "entity_count": ent_count,
                "triples_1": triple_1,
                "triples_2": triple_2,
                "ill_count": ill_count,
            }
        )


def count_lines(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser(description="Prepare DBP15K files for MEAformer baseline with full logs.")
    parser.add_argument("--lang-pair", choices=["zh_en", "ja_en", "fr_en"], default="zh_en")
    parser.add_argument("--split", choices=["0_1", "0_2", "0_3", "0_4", "0_5"], default="0_3")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    src_root = Path("data/raw/dbp15k/dbp15k")
    src_lang_root = src_root / args.lang_pair
    src_split_dir = src_lang_root / args.split

    dst_mmkg_root = Path("data/mmkg")
    dst_dataset_dir = dst_mmkg_root / "DBP15K" / args.lang_pair
    dst_dataset_dir.mkdir(parents=True, exist_ok=True)

    log_path = dst_dataset_dir / "prepare_meaformer.log"
    log(f"start prepare MEAformer data lang_pair={args.lang_pair} split={args.split}", log_path)

    if not src_split_dir.exists():
        raise FileNotFoundError(f"source split directory not found: {src_split_dir}")

    copy_required_files(src_split_dir, dst_dataset_dir)
    copy_attr_files(src_lang_root, dst_dataset_dir)
    copy_ill(src_split_dir, dst_dataset_dir)

    ent_ids_1 = parse_entity_ids(dst_dataset_dir / "ent_ids_1")
    ent_ids_2 = parse_entity_ids(dst_dataset_dir / "ent_ids_2")
    all_ids = sorted(set(ent_ids_1 + ent_ids_2))
    pkl_path = build_random_img_features(dst_mmkg_root, args.lang_pair, all_ids, seed=args.seed, dim=300)

    stats_path = dst_dataset_dir / "meaformer_data_stats.csv"
    write_stats(
        stats_path,
        dataset=f"DBP15K/{args.lang_pair}",
        split=args.split,
        ent_count=len(all_ids),
        triple_1=count_lines(dst_dataset_dir / "triples_1"),
        triple_2=count_lines(dst_dataset_dir / "triples_2"),
        ill_count=count_lines(dst_dataset_dir / "ill_ent_ids"),
    )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "split_dir": str(src_split_dir),
            "ill_files": [str(src_split_dir / "sup_ent_ids"), str(src_split_dir / "ref_ent_ids")],
        },
        "target": str(dst_dataset_dir),
        "image_feature_pkl": str(pkl_path),
        "checksums": {
            "ill_ent_ids_sha256": sha256_file(dst_dataset_dir / "ill_ent_ids"),
            "img_pkl_sha256": sha256_file(pkl_path),
        },
        "note": "Image features are random placeholders for pipeline bootstrap. Replace with official pkls for final results.",
    }
    (dst_dataset_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log("prepared MEAformer-compatible DBP15K files", log_path)
    log("finish prepare", log_path)


if __name__ == "__main__":
    main()
