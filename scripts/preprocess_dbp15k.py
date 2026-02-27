import argparse
import csv
import hashlib
import json
import tarfile
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve


JAPE_REPO_ARCHIVE = "https://github.com/nju-websoft/JAPE/archive/refs/heads/master.tar.gz"
JAPE_DBP15K_ARCHIVE_NAME = "dbp15k.tar.gz"


def log(msg, log_path):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_dirs():
    raw = Path("data/raw")
    processed = Path("data/processed")
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    return raw, processed


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_if_needed(url, dest, log_path):
    if dest.exists():
        log(f"archive exists, skip download: {dest}", log_path)
        return
    log(f"downloading: {url}", log_path)
    urlretrieve(url, str(dest))
    log(f"downloaded: {dest}", log_path)


def extract_tar_gz(archive_path, extract_to, log_path):
    if extract_to.exists() and any(extract_to.iterdir()):
        log(f"already extracted, skip: {extract_to}", log_path)
        return
    extract_to.mkdir(parents=True, exist_ok=True)
    log(f"extracting {archive_path} -> {extract_to}", log_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    log("extract done", log_path)


def find_jape_dbp15k_archive(raw_dir):
    candidates = list(raw_dir.glob("JAPE-*/data/dbp15k.tar.gz"))
    if candidates:
        return candidates[0]
    return None


def ensure_dbp15k_available(raw_dir, log_path):
    jape_tar = raw_dir / "JAPE-master.tar.gz"
    jape_extract_root = raw_dir / "JAPE-master-src"

    download_if_needed(JAPE_REPO_ARCHIVE, jape_tar, log_path)
    extract_tar_gz(jape_tar, jape_extract_root, log_path)

    dbp15k_tar = find_jape_dbp15k_archive(jape_extract_root)
    if dbp15k_tar is None:
        raise FileNotFoundError("Cannot locate dbp15k.tar.gz under extracted JAPE archive.")

    dbp15k_extract_root = raw_dir / "dbp15k"
    extract_tar_gz(dbp15k_tar, dbp15k_extract_root, log_path)

    dbp15k_root = dbp15k_extract_root / "dbp15k"
    if not dbp15k_root.exists():
        raise FileNotFoundError(f"dbp15k root not found: {dbp15k_root}")
    return dbp15k_root, dbp15k_tar


def count_lines(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def read_pair_file(path):
    pairs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def write_pairs(path, pairs):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(pairs)


def write_manifest(path, dataset, split, seed, train_ratio, source_archive, source_sha256):
    manifest = {
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "train_ratio": train_ratio,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "name": "JAPE DBP15K",
            "archive": str(source_archive),
            "archive_sha256": source_sha256,
            "url": JAPE_REPO_ARCHIVE,
            "license": "MIT (repository); underlying DBpedia-derived resources should be cited in thesis",
        },
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def write_stats(path, dataset, triples_1_count, triples_2_count, train_links, test_links):
    rows = [
        {
            "dataset": dataset,
            "triples_1": triples_1_count,
            "triples_2": triples_2_count,
            "train_links": train_links,
            "test_links": test_links,
            "total_triples": triples_1_count + triples_2_count,
            "total_links": train_links + test_links,
        }
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "triples_1", "triples_2", "train_links", "test_links", "total_triples", "total_links"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_processed_dataset(dbp15k_root, dataset, split, out_dir, log_path):
    src = dbp15k_root / dataset / split
    if not src.exists():
        raise FileNotFoundError(f"split directory not found: {src}")

    triples_1 = src / "triples_1"
    triples_2 = src / "triples_2"
    ref_ent_ids = src / "ref_ent_ids"

    for p in (triples_1, triples_2, ref_ent_ids):
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")

    log("counting triples and alignment pairs", log_path)
    triples_1_count = count_lines(triples_1)
    triples_2_count = count_lines(triples_2)
    pairs = read_pair_file(ref_ent_ids)

    # JAPE split folder provides supervised links as ref_ent_ids.
    # For baseline scaffold we store them into train_links and keep test empty.
    train_links = pairs
    test_links = []

    write_pairs(out_dir / "train_links.tsv", train_links)
    write_pairs(out_dir / "test_links.tsv", test_links)

    # Keep raw triples as canonical input for downstream loaders.
    (out_dir / "triples_1.tsv").write_text(triples_1.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    (out_dir / "triples_2.tsv").write_text(triples_2.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    write_stats(
        out_dir / "data_stats.csv",
        dataset=f"dbp15k_{dataset}",
        triples_1_count=triples_1_count,
        triples_2_count=triples_2_count,
        train_links=len(train_links),
        test_links=len(test_links),
    )
    log(
        f"stats triples_1={triples_1_count} triples_2={triples_2_count} train_links={len(train_links)}",
        log_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess DBP15K from JAPE with experiment logs.")
    parser.add_argument("--lang-pair", choices=["zh_en", "ja_en", "fr_en"], default="zh_en")
    parser.add_argument("--split", choices=["0_1", "0_2", "0_3", "0_4", "0_5"], default="0_3")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-ratio", type=float, default=0.3)
    args = parser.parse_args()

    raw, processed = ensure_dirs()
    dataset_key = f"dbp15k_{args.lang_pair}"
    out_dir = processed / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "preprocess.log"

    log(f"start preprocess dataset={dataset_key} split={args.split}", log_path)
    dbp15k_root, dbp15k_tar = ensure_dbp15k_available(raw, log_path)
    source_hash = sha256_file(dbp15k_tar)
    build_processed_dataset(dbp15k_root, args.lang_pair, args.split, out_dir, log_path)
    write_manifest(
        out_dir / "split_manifest.json",
        dataset=dataset_key,
        split=args.split,
        seed=args.seed,
        train_ratio=args.train_ratio,
        source_archive=dbp15k_tar,
        source_sha256=source_hash,
    )
    log("wrote split_manifest.json and data_stats.csv", log_path)
    log("finish preprocess", log_path)


if __name__ == "__main__":
    main()
