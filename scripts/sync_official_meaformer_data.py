import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copytree_overwrite(src: Path, dst: Path):
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            copytree_overwrite(item, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def main():
    parser = argparse.ArgumentParser(description="Sync official MEAformer mmkg data into project data/mmkg.")
    parser.add_argument("--src", default="data/raw/MEAformer_data/mmkg")
    parser.add_argument("--dst", default="data/mmkg")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(f"source not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)
    copytree_overwrite(src, dst)

    manifest = {
        "synced_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(src.resolve()),
        "target": str(dst.resolve()),
        "checksums": {},
    }

    key_files = [
        dst / "pkls" / "zh_en_GA_id_img_feature_dict.pkl",
        dst / "pkls" / "ja_en_GA_id_img_feature_dict.pkl",
        dst / "pkls" / "fr_en_GA_id_img_feature_dict.pkl",
        dst / "pkls" / "FBDB15K_id_img_feature_dict.pkl",
        dst / "pkls" / "FBYG15K_id_img_feature_dict.pkl",
        dst / "embedding" / "glove.6B.300d.txt",
    ]
    for p in key_files:
        if p.exists():
            manifest["checksums"][str(p.relative_to(dst))] = sha256_file(p)

    out = Path("data/official_data_manifest.json")
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Synced official data to: {dst}")
    print(f"Wrote manifest: {out}")


if __name__ == "__main__":
    main()
