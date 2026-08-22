#!/usr/bin/env python3
"""Check that the environment and data files are ready before a benchmark run.

Usage:
    python scripts/verify_setup.py
    python scripts/verify_setup.py --mode full
"""
import argparse
import platform
import sys

import numpy as np
import torch

from config import DATASETS, METADATA_PATH, PROFILES
from data import load_metadata, npz_path

CHECK = "OK  "
FAIL = "FAIL"


def report(label: str, ok: bool, detail: str = "") -> bool:
    status = CHECK if ok else FAIL
    print(f"[{status}] {label}{f' - {detail}' if detail else ''}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(PROFILES), default="sample")
    args = parser.parse_args()

    print("Environment")
    print(f"  python {platform.python_version()}")
    print(f"  torch  {torch.__version__}")
    print(f"  numpy  {np.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device {device}")
    if device == "cuda":
        print(f"  gpu    {torch.cuda.get_device_name(0)}")
    print()

    all_ok = True
    all_ok &= report("metadata manifest present", METADATA_PATH.exists(), str(METADATA_PATH))
    if not METADATA_PATH.exists():
        print("\nRun: python scripts/download_data.py --metadata-only")
        return 1

    metadata = load_metadata()
    all_ok &= report("metadata covers 12 datasets", len(metadata) == len(DATASETS),
                     f"{len(metadata)} entries")

    print(f"\nData files for mode '{args.mode}'")
    missing = []
    total_bytes = 0
    for name in DATASETS:
        path = npz_path(name, args.mode)
        if path.exists():
            size = path.stat().st_size
            total_bytes += size
            with np.load(path) as archive:
                n_train = len(archive["train_images"])
                n_test = len(archive["test_images"])
            print(f"  [{CHECK}] {name:15s} {n_train:6d} train / {n_test:6d} test "
                  f"({size / 1e6:6.2f} MB)")
        else:
            missing.append(name)
            print(f"  [{FAIL}] {name:15s} missing")

    print(f"\nTotal on disk: {total_bytes / 1e6:.1f} MB")
    if missing:
        all_ok = False
        hint = (
            "python scripts/build_sample_data.py"
            if args.mode == "sample"
            else "python scripts/download_data.py"
        )
        print(f"Missing {len(missing)} archives. Run: {hint}")

    print("\nAll checks passed." if all_ok else "\nSome checks failed.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
