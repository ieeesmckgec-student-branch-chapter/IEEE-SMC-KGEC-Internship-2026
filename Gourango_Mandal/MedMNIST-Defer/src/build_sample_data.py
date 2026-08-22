#!/usr/bin/env python3
"""Build the small, class-stratified sample subsets committed to the repository.

The sample archives let anyone run the entire pipeline end to end without
downloading the full collection. They are produced from the verified full
archives, so this script is only needed when regenerating them.

Usage:
    python scripts/build_sample_data.py
"""
import argparse
import sys

import numpy as np

from common import save_json, set_seed
from config import (
    DATASETS,
    RAW_DIR,
    SAMPLE_DIR,
    SAMPLE_TEST_PER_DATASET,
    SAMPLE_TRAIN_PER_DATASET,
    SEED,
)
from data import load_metadata


def stratified_indices(labels: np.ndarray, n_target: int, rng: np.random.Generator) -> np.ndarray:
    """Pick ``n_target`` indices, spreading them evenly across classes.

    Multi-label targets have no single class to stratify on, so those fall back
    to a uniform random draw.
    """
    n_available = len(labels)
    if n_target >= n_available:
        return np.arange(n_available)

    flat = labels.reshape(n_available, -1)
    if flat.shape[1] > 1:
        return rng.choice(n_available, size=n_target, replace=False)

    classes = np.unique(flat[:, 0])
    per_class = max(1, n_target // len(classes))
    chosen: list[np.ndarray] = []
    for class_id in classes:
        pool = np.flatnonzero(flat[:, 0] == class_id)
        take = min(per_class, len(pool))
        chosen.append(rng.choice(pool, size=take, replace=False))
    selected = np.concatenate(chosen)

    # Top up with random leftovers if integer division left us short.
    if len(selected) < n_target:
        remaining = np.setdiff1d(np.arange(n_available), selected)
        extra = rng.choice(remaining, size=min(n_target - len(selected), len(remaining)), replace=False)
        selected = np.concatenate([selected, extra])
    return np.sort(selected)


def build_one(name: str, rng: np.random.Generator) -> dict:
    source = RAW_DIR / f"{name}.npz"
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}. Run: python scripts/download_data.py")

    with np.load(source) as archive:
        train_images = archive["train_images"]
        train_labels = archive["train_labels"]
        test_images = archive["test_images"]
        test_labels = archive["test_labels"]

    train_idx = stratified_indices(train_labels, SAMPLE_TRAIN_PER_DATASET, rng)
    test_idx = stratified_indices(test_labels, SAMPLE_TEST_PER_DATASET, rng)

    destination = SAMPLE_DIR / f"{name}_sample.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        train_images=train_images[train_idx],
        train_labels=train_labels[train_idx],
        test_images=test_images[test_idx],
        test_labels=test_labels[test_idx],
    )
    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "bytes": int(destination.stat().st_size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()

    set_seed(SEED)
    load_metadata()  # fail fast if the manifest has not been written yet
    rng = np.random.default_rng(SEED)

    manifest = {}
    for index, name in enumerate(args.datasets, start=1):
        stats = build_one(name, rng)
        manifest[name] = stats
        print(
            f"[{index}/{len(args.datasets)}] {name}: "
            f"{stats['n_train']} train / {stats['n_test']} test "
            f"({stats['bytes'] / 1e6:.2f} MB)"
        )

    total = sum(entry["bytes"] for entry in manifest.values())
    save_json(
        SAMPLE_DIR / "sample_manifest.json",
        {
            "description": "Class-stratified subsets of MedMNIST v2 for end-to-end smoke runs",
            "seed": SEED,
            "train_target_per_dataset": SAMPLE_TRAIN_PER_DATASET,
            "test_target_per_dataset": SAMPLE_TEST_PER_DATASET,
            "total_bytes": total,
            "datasets": manifest,
        },
    )
    print(f"\nTotal sample size: {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
