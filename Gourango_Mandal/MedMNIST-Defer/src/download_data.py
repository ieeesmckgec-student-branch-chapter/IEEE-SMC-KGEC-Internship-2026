#!/usr/bin/env python3
"""Download the full MedMNIST v2 2D archives and write the metadata manifest.

The 12 archives at 28x28 resolution total roughly 574 MB. Files are fetched
from the official Zenodo record and verified against the published MD5 sums.

Usage:
    python scripts/download_data.py                  # all 12 datasets
    python scripts/download_data.py --datasets pathmnist bloodmnist
    python scripts/download_data.py --metadata-only   # refresh metadata.json only
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from medmnist import INFO

from common import load_json, save_json
from config import DATASETS, METADATA_PATH, MODALITY_NAMES, RAW_DIR

CHUNK_BYTES = 1 << 20


def md5_of_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_md5: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and md5_of_file(destination) == expected_md5:
        print(f"  already present and verified: {destination.name}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(temporary, "wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if total:
                percent = 100 * written / total
                print(f"\r  {destination.name}: {percent:5.1f}%", end="", flush=True)
    print()

    actual = md5_of_file(temporary)
    if actual != expected_md5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {destination.name}: {actual} != {expected_md5}")
    temporary.replace(destination)


def write_metadata(datasets: tuple[str, ...]) -> None:
    """Refresh manifest entries, preserving records for datasets not requested."""
    records = {}
    if METADATA_PATH.exists():
        records = load_json(METADATA_PATH).get("datasets", {})

    for name in datasets:
        info = INFO[name]
        records[name] = {
            "task": info["task"],
            "n_channels": info["n_channels"],
            "n_classes": len(info["label"]),
            "labels": info["label"],
            "modality": MODALITY_NAMES.get(name, name),
            "n_samples": info["n_samples"],
            "license": info.get("license", ""),
            "md5": info["MD5"],
            "url": info["url"],
        }
    payload = {
        "collection": "MedMNIST v2 (2D subsets)",
        "resolution": "28x28",
        "source": "https://medmnist.com/",
        "zenodo_record": "https://zenodo.org/records/10519652",
        "citation": "Yang et al., MedMNIST v2, Scientific Data 10:41 (2023)",
        "datasets": records,
    }
    save_json(METADATA_PATH, payload)
    print(f"Manifest at {METADATA_PATH} now covers {len(records)} datasets")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    unknown = [name for name in args.datasets if name not in INFO]
    if unknown:
        print(f"Unknown dataset names: {unknown}", file=sys.stderr)
        return 1

    write_metadata(tuple(args.datasets))
    if args.metadata_only:
        return 0

    print(f"Downloading {len(args.datasets)} archives into {RAW_DIR}")
    for index, name in enumerate(args.datasets, start=1):
        info = INFO[name]
        print(f"[{index}/{len(args.datasets)}] {name}")
        download(info["url"], RAW_DIR / f"{name}.npz", info["MD5"])

    total_bytes = sum(path.stat().st_size for path in RAW_DIR.glob("*.npz"))
    print(f"\nDone. {RAW_DIR} now holds {total_bytes / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
