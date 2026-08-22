"""Shared utilities: seeding, metrics, calibration, bootstrap and result I/O."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

from config import ECE_BINS, SEED


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row of a categorical probability matrix."""
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def binary_entropy(probs: np.ndarray) -> np.ndarray:
    """Element-wise entropy of independent Bernoulli probabilities."""
    p = np.clip(probs, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def accuracy_at_coverage(
    correct: np.ndarray,
    confidence: np.ndarray,
    coverages: tuple[float, ...],
) -> dict[str, float]:
    """Accuracy on the most confident fraction of samples, per coverage level.

    The least confident samples are deferred, mirroring how a triage system
    would hand ambiguous cases to a clinician.
    """
    n = len(correct)
    order = np.argsort(confidence)[::-1]
    results = {}
    for coverage in coverages:
        k = max(1, int(round(n * coverage)))
        results[f"acc@{coverage:.0%}"] = float(np.mean(correct[order[:k]]))
    return results


def risk_coverage_auc(correct: np.ndarray, confidence: np.ndarray) -> float:
    """Area under the risk-coverage curve; lower is better."""
    order = np.argsort(confidence)[::-1]
    ordered = correct[order]
    cumulative_risk = 1.0 - np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.mean(cumulative_risk))


def expected_calibration_error(
    correct: np.ndarray,
    confidence: np.ndarray,
    n_bins: int = ECE_BINS,
) -> float:
    """Standard binned ECE using the model's own confidence as the estimate."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(correct)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        weight = mask.sum() / n
        ece += weight * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def bootstrap_accuracy_at_coverage(
    correct: np.ndarray,
    confidence: np.ndarray,
    coverage: float,
    n_boot: int,
    seed: int = SEED,
) -> dict[str, float]:
    """Percentile bootstrap CI for accuracy at a fixed coverage level.

    Samples are resampled with replacement and the deferral threshold is
    recomputed inside each resample, so the interval reflects the uncertainty
    of the whole selective-prediction procedure rather than a fixed threshold.
    """
    rng = np.random.default_rng(seed)
    n = len(correct)
    k = max(1, int(round(n * coverage)))
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        order = np.argsort(confidence[idx])[::-1]
        boots[i] = np.mean(correct[idx[order[:k]]])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean": float(boots.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
    }


def paired_bootstrap_delta(
    correct: np.ndarray,
    confidence: np.ndarray,
    coverage: float,
    n_boot: int,
    seed: int = SEED,
) -> dict[str, float]:
    """Bootstrap the accuracy gain of deferral over full coverage.

    Both quantities are computed on the same resample, giving a paired test of
    whether the gain is distinguishable from zero.
    """
    rng = np.random.default_rng(seed + 1)
    n = len(correct)
    k = max(1, int(round(n * coverage)))
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        resampled = correct[idx]
        order = np.argsort(confidence[idx])[::-1]
        deltas[i] = np.mean(resampled[order[:k]]) - np.mean(resampled)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # Two-sided bootstrap p-value for the null hypothesis of zero gain.
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "delta_mean": float(deltas.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "p_value": float(min(p, 1.0)),
    }


def wilcoxon_pvalue(a: np.ndarray, b: np.ndarray) -> float | None:
    """Wilcoxon signed-rank p-value across datasets, or None if undefined."""
    if len(a) < 5:
        return None
    try:
        _, p = wilcoxon(a, b)
    except ValueError:
        return None
    return float(p)


def sanitize(obj):
    """Replace non-finite floats with None so the result is valid JSON."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    return obj


def save_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(sanitize(payload), handle, indent=2)
    return path


def load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)
