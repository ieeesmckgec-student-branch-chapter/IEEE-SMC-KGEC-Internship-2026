"""Dataset loading for both the committed sample subsets and the full download.

Both run modes read plain ``.npz`` archives with the MedMNIST v2 key layout
(``train_images``, ``train_labels``, ``test_images``, ``test_labels``), so the
sample pipeline has no dependency on the ``medmnist`` package at all.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import METADATA_PATH, RAW_DIR, SAMPLE_DIR


class DatasetMetadata:
    """Static description of one MedMNIST subset."""

    def __init__(self, name: str, record: dict):
        self.name = name
        self.task = record["task"]
        self.n_channels = int(record["n_channels"])
        self.n_classes = int(record["n_classes"])
        self.labels = record["labels"]
        self.modality = record.get("modality", name)
        self.n_samples = record.get("n_samples", {})

    @property
    def is_multilabel(self) -> bool:
        return "multi-label" in self.task

    @property
    def n_outputs(self) -> int:
        """Number of output units the classification head needs."""
        return self.n_classes


def load_metadata() -> dict[str, DatasetMetadata]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {METADATA_PATH}. Run: python scripts/download_data.py --metadata-only"
        )
    with open(METADATA_PATH) as handle:
        raw = json.load(handle)
    return {name: DatasetMetadata(name, record) for name, record in raw["datasets"].items()}


def npz_path(name: str, mode: str) -> Path:
    """Location of the archive for a dataset under the given run mode."""
    if mode == "sample":
        return SAMPLE_DIR / f"{name}_sample.npz"
    return RAW_DIR / f"{name}.npz"


class NpzImageDataset(Dataset):
    """In-memory image dataset normalised to the [-1, 1] range."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, n_channels: int):
        self.images = images
        self.labels = labels
        self.n_channels = n_channels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index].astype(np.float32) / 255.0
        if image.ndim == 2:
            image = image[None, :, :]
        else:
            image = np.transpose(image, (2, 0, 1))
        image = (image - 0.5) / 0.5
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(
            np.asarray(self.labels[index])
        )


def load_split(name: str, mode: str, metadata: DatasetMetadata):
    """Return (train_dataset, test_dataset) for one MedMNIST subset."""
    path = npz_path(name, mode)
    if not path.exists():
        hint = (
            "python scripts/build_sample_data.py"
            if mode == "sample"
            else "python scripts/download_data.py"
        )
        raise FileNotFoundError(f"Missing {path}. Create it with: {hint}")

    with np.load(path) as archive:
        train_images = archive["train_images"]
        train_labels = archive["train_labels"]
        test_images = archive["test_images"]
        test_labels = archive["test_labels"]

    train = NpzImageDataset(train_images, train_labels, metadata.n_channels)
    test = NpzImageDataset(test_images, test_labels, metadata.n_channels)
    return train, test


def available_datasets(mode: str, candidates: tuple[str, ...]) -> list[str]:
    """Subset of ``candidates`` whose archives are present on disk."""
    return [name for name in candidates if npz_path(name, mode).exists()]
