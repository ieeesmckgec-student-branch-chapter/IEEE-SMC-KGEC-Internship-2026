#!/usr/bin/env python3
"""MedMNIST-Defer: cross-modality selective-prediction benchmark.

For every MedMNIST v2 2D subset the script trains one classifier, then ranks
the test set by five different confidence signals and measures accuracy as the
least confident predictions are progressively deferred.

Usage:
    python scripts/run_benchmark.py --mode sample            # ~3 min, committed data
    python scripts/run_benchmark.py --mode full              # paper numbers
    python scripts/run_benchmark.py --mode full --backbone simplecnn
"""
import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from common import (
    accuracy_at_coverage,
    binary_entropy,
    bootstrap_accuracy_at_coverage,
    entropy_from_probs,
    expected_calibration_error,
    get_device,
    paired_bootstrap_delta,
    risk_coverage_auc,
    save_json,
    set_seed,
    wilcoxon_pvalue,
)
from config import (
    BOOTSTRAP_COVERAGES,
    COVERAGES,
    DATASETS,
    DEFAULT_BACKBONE,
    DEFERRAL_METHODS,
    MC_DROPOUT_PASSES,
    MODALITY_NAMES,
    PROFILES,
    RESULTS_DIR,
    SEED,
    VAL_FRACTION,
)
from data import DatasetMetadata, load_metadata, load_split
from models import build_model, enable_dropout_only

REFERENCE_COVERAGE = 0.5


def split_train_val(dataset, seed: int) -> tuple[Subset, Subset]:
    """Hold out a slice of the training split for temperature calibration."""
    n = len(dataset)
    n_val = max(2, int(round(n * VAL_FRACTION)))
    generator = np.random.default_rng(seed)
    order = generator.permutation(n)
    return Subset(dataset, order[n_val:].tolist()), Subset(dataset, order[:n_val].tolist())


def loss_for_task(logits: torch.Tensor, target: torch.Tensor, multilabel: bool) -> torch.Tensor:
    if multilabel:
        return F.binary_cross_entropy_with_logits(logits, target.float())
    return F.cross_entropy(logits, target)


def prepare_target(y: torch.Tensor, multilabel: bool) -> torch.Tensor:
    if multilabel:
        return y.view(y.size(0), -1).float()
    return y.reshape(-1).long()


def train_model(model, loader, optimizer, device, multilabel: bool, epochs: int) -> list[float]:
    history = []
    for _ in range(epochs):
        model.train()
        total, loss_sum = 0, 0.0
        for x, y in loader:
            # A trailing batch of one breaks batch-norm statistics.
            if x.size(0) < 2:
                continue
            x = x.to(device)
            target = prepare_target(y, multilabel).to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_for_task(logits, target, multilabel)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
        history.append(loss_sum / max(total, 1))
    return history


@torch.no_grad()
def collect_logits(model, loader, device, multilabel: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, targets_all = [], []
    for x, y in loader:
        logits_all.append(model(x.to(device)).cpu().numpy())
        targets_all.append(prepare_target(y, multilabel).numpy())
    return np.concatenate(logits_all), np.concatenate(targets_all)


@torch.no_grad()
def collect_mc_probabilities(model, loader, device, multilabel: bool, passes: int) -> np.ndarray:
    """Average predictive distribution over stochastic forward passes."""
    enable_dropout_only(model)
    accumulator = None
    for _ in range(passes):
        batch_probs = []
        for x, _ in loader:
            logits = model(x.to(device))
            probs = torch.sigmoid(logits) if multilabel else F.softmax(logits, dim=1)
            batch_probs.append(probs.cpu().numpy())
        stacked = np.concatenate(batch_probs)
        accumulator = stacked if accumulator is None else accumulator + stacked
    model.eval()
    return accumulator / passes


def fit_temperature(logits: np.ndarray, targets: np.ndarray, multilabel: bool) -> float:
    """Single-parameter temperature scaling fitted by NLL on the held-out split."""
    logit_tensor = torch.from_numpy(logits).float()
    if multilabel:
        target_tensor = torch.from_numpy(targets).float()
    else:
        target_tensor = torch.from_numpy(targets).long()

    log_temperature = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        scaled = logit_tensor / torch.exp(log_temperature)
        loss = loss_for_task(scaled, target_tensor, multilabel)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).item())


def probabilities_from_logits(logits: np.ndarray, multilabel: bool, temperature: float = 1.0):
    scaled = torch.from_numpy(logits).float() / temperature
    if multilabel:
        return torch.sigmoid(scaled).numpy()
    return F.softmax(scaled, dim=1).numpy()


def predictions_from_probabilities(probs: np.ndarray, multilabel: bool) -> np.ndarray:
    if multilabel:
        return (probs > 0.5).astype(np.int32)
    return probs.argmax(axis=1)


def correctness(targets: np.ndarray, predictions: np.ndarray, multilabel: bool) -> np.ndarray:
    """Per-sample correctness; multi-label uses exact match across all labels."""
    if multilabel:
        return np.all(targets.astype(np.int32) == predictions, axis=1).astype(float)
    return (targets.astype(np.int64) == predictions).astype(float)


def confidence_signal(probs: np.ndarray, method: str, multilabel: bool) -> np.ndarray:
    """Higher values mean the model is more confident and less likely deferred."""
    if multilabel:
        if method in ("entropy", "temp_entropy", "mc_dropout"):
            return -binary_entropy(probs).mean(axis=1)
        if method == "max_prob":
            return np.maximum(probs, 1 - probs).mean(axis=1)
        if method == "margin":
            return (2.0 * np.abs(probs - 0.5)).mean(axis=1)
        raise ValueError(f"Unknown deferral method: {method}")

    if method in ("entropy", "temp_entropy", "mc_dropout"):
        return -entropy_from_probs(probs)
    if method == "max_prob":
        return probs.max(axis=1)
    if method == "margin":
        top2 = np.partition(probs, -2, axis=1)[:, -2:]
        return top2.max(axis=1) - top2.min(axis=1)
    raise ValueError(f"Unknown deferral method: {method}")


def calibration_confidence(probs: np.ndarray, multilabel: bool) -> np.ndarray:
    if multilabel:
        return np.maximum(probs, 1 - probs).mean(axis=1)
    return probs.max(axis=1)


def evaluate_policy(
    probs: np.ndarray,
    targets: np.ndarray,
    multilabel: bool,
    method: str,
    bootstrap_n: int,
) -> dict:
    predictions = predictions_from_probabilities(probs, multilabel)
    correct = correctness(targets, predictions, multilabel)
    confidence = confidence_signal(probs, method, multilabel)

    curve = accuracy_at_coverage(correct, confidence, COVERAGES)
    intervals = {
        f"acc@{coverage:.0%}": bootstrap_accuracy_at_coverage(
            correct, confidence, coverage, bootstrap_n
        )
        for coverage in BOOTSTRAP_COVERAGES
    }
    deltas = {
        f"delta@{coverage:.0%}": paired_bootstrap_delta(
            correct, confidence, coverage, bootstrap_n
        )
        for coverage in BOOTSTRAP_COVERAGES
    }

    return {
        "accuracy_full_coverage": float(correct.mean()),
        "acc_at_coverage": curve,
        "confidence_intervals": intervals,
        "deferral_gain": deltas,
        "risk_coverage_auc": risk_coverage_auc(correct, confidence),
        "ece": expected_calibration_error(correct, calibration_confidence(probs, multilabel)),
    }


def run_dataset(
    name: str,
    metadata: DatasetMetadata,
    mode: str,
    backbone: str,
    device: torch.device,
    profile,
) -> dict:
    # Re-seed per dataset so a result never depends on which other datasets ran
    # before it in the same invocation.
    set_seed(SEED)

    train_full, test = load_split(name, mode, metadata)
    train_part, val_part = split_train_val(train_full, SEED)

    train_loader = DataLoader(
        train_part, batch_size=profile.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_part, batch_size=profile.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=profile.batch_size, shuffle=False)

    model = build_model(backbone, metadata.n_channels, metadata.n_outputs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=profile.lr)
    loss_history = train_model(
        model, train_loader, optimizer, device, metadata.is_multilabel, profile.epochs
    )

    val_logits, val_targets = collect_logits(model, val_loader, device, metadata.is_multilabel)
    temperature = fit_temperature(val_logits, val_targets, metadata.is_multilabel)

    test_logits, test_targets = collect_logits(model, test_loader, device, metadata.is_multilabel)
    plain_probs = probabilities_from_logits(test_logits, metadata.is_multilabel)
    scaled_probs = probabilities_from_logits(test_logits, metadata.is_multilabel, temperature)
    mc_probs = collect_mc_probabilities(
        model, test_loader, device, metadata.is_multilabel, MC_DROPOUT_PASSES
    )

    probs_for_method = {
        "entropy": plain_probs,
        "max_prob": plain_probs,
        "margin": plain_probs,
        "temp_entropy": scaled_probs,
        "mc_dropout": mc_probs,
    }

    policies = {
        method: evaluate_policy(
            probs_for_method[method],
            test_targets,
            metadata.is_multilabel,
            method,
            profile.bootstrap_n,
        )
        for method in DEFERRAL_METHODS
    }

    best_method = max(
        DEFERRAL_METHODS,
        key=lambda m: policies[m]["acc_at_coverage"][f"acc@{REFERENCE_COVERAGE:.0%}"],
    )
    entropy_policy = policies["entropy"]
    return {
        "dataset": name,
        "modality": MODALITY_NAMES.get(name, name),
        "task": metadata.task,
        "n_channels": metadata.n_channels,
        "n_classes": metadata.n_classes,
        "n_train": len(train_part),
        "n_val": len(val_part),
        "n_test": len(test),
        "accuracy_full_coverage": entropy_policy["accuracy_full_coverage"],
        "ece_uncalibrated": entropy_policy["ece"],
        "ece_temperature_scaled": policies["temp_entropy"]["ece"],
        "temperature": temperature,
        "final_train_loss": loss_history[-1] if loss_history else None,
        "policies": policies,
        "best_policy_at_50pct": best_method,
        "entropy_delta_at_50pct": entropy_policy["deferral_gain"]["delta@50%"]["delta_mean"],
        "entropy_p_value_at_50pct": entropy_policy["deferral_gain"]["delta@50%"]["p_value"],
    }


def aggregate(per_dataset: dict, names: list[str]) -> dict:
    def entropy_curve(name: str, coverage: float) -> float:
        return per_dataset[name]["policies"]["entropy"]["acc_at_coverage"][f"acc@{coverage:.0%}"]

    acc_full = np.array([per_dataset[n]["accuracy_full_coverage"] for n in names])
    acc_90 = np.array([entropy_curve(n, 0.9) for n in names])
    acc_50 = np.array([entropy_curve(n, 0.5) for n in names])
    gains_50 = acc_50 - acc_full
    p_values = np.array([per_dataset[n]["entropy_p_value_at_50pct"] for n in names])

    policy_means = {}
    for method in DEFERRAL_METHODS:
        values = [
            per_dataset[n]["policies"][method]["acc_at_coverage"]["acc@50%"]
            - per_dataset[n]["policies"][method]["accuracy_full_coverage"]
            for n in names
        ]
        rc_auc = [per_dataset[n]["policies"][method]["risk_coverage_auc"] for n in names]
        policy_means[method] = {
            "mean_gain_at_50pct": float(np.mean(values)),
            "mean_risk_coverage_auc": float(np.mean(rc_auc)),
            "n_datasets_won_at_50pct": sum(
                1 for n in names if per_dataset[n]["best_policy_at_50pct"] == method
            ),
        }

    ranked = sorted(names, key=lambda n: per_dataset[n]["entropy_delta_at_50pct"], reverse=True)
    ece_values = np.array([per_dataset[n]["ece_uncalibrated"] for n in names])
    correlation = (
        float(np.corrcoef(ece_values, gains_50)[0, 1]) if len(names) > 2 else None
    )

    return {
        "n_datasets": len(names),
        "mean_accuracy_full_coverage": float(acc_full.mean()),
        "mean_gain_at_90pct_entropy": float((acc_90 - acc_full).mean()),
        "mean_gain_at_50pct_entropy": float(gains_50.mean()),
        "median_gain_at_50pct_entropy": float(np.median(gains_50)),
        "min_gain_at_50pct_entropy": float(gains_50.min()),
        "max_gain_at_50pct_entropy": float(gains_50.max()),
        "datasets_with_significant_gain_at_50pct": int(np.sum(p_values < 0.05)),
        "datasets_with_gain_above_5pp": int(np.sum(gains_50 > 0.05)),
        "datasets_with_gain_above_10pp": int(np.sum(gains_50 > 0.10)),
        "accuracy_spread_at_50pct": float(acc_50.max() - acc_50.min()),
        "best_gain_dataset": ranked[0],
        "worst_gain_dataset": ranked[-1],
        "wilcoxon_acc50_vs_full_entropy": wilcoxon_pvalue(acc_50, acc_full),
        "wilcoxon_acc90_vs_full_entropy": wilcoxon_pvalue(acc_90, acc_full),
        "correlation_ece_vs_gain_at_50pct": correlation,
        "mean_ece_uncalibrated": float(ece_values.mean()),
        "mean_ece_temperature_scaled": float(
            np.mean([per_dataset[n]["ece_temperature_scaled"] for n in names])
        ),
        "per_policy": policy_means,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(PROFILES), default="sample")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--epochs", type=int, default=None, help="override profile epochs")
    parser.add_argument("--output", default=None, help="output JSON path")
    args = parser.parse_args()

    profile = PROFILES[args.mode]
    if args.epochs is not None:
        profile.epochs = args.epochs

    set_seed(SEED)
    device = get_device()
    metadata = load_metadata()

    print(f"MedMNIST-Defer | mode={args.mode} backbone={args.backbone} device={device}")
    print(f"Datasets: {len(args.datasets)} | epochs={profile.epochs} seed={SEED}\n")

    started = time.time()
    per_dataset = {}
    for index, name in enumerate(args.datasets, start=1):
        dataset_started = time.time()
        per_dataset[name] = run_dataset(
            name, metadata[name], args.mode, args.backbone, device, profile
        )
        per_dataset[name]["runtime_sec"] = round(time.time() - dataset_started, 1)
        record = per_dataset[name]
        print(
            f"[{index:2d}/{len(args.datasets)}] {name:15s} "
            f"acc={record['accuracy_full_coverage']:.3f} "
            f"acc@50%={record['policies']['entropy']['acc_at_coverage']['acc@50%']:.3f} "
            f"gain={record['entropy_delta_at_50pct']:+.3f} "
            f"p={record['entropy_p_value_at_50pct']:.4f} "
            f"({record['runtime_sec']}s)"
        )

    summary = aggregate(per_dataset, list(args.datasets))
    runtime = round(time.time() - started, 1)

    payload = {
        "benchmark": "MedMNIST-Defer v1",
        "mode": args.mode,
        "backbone": args.backbone,
        "resolution": "28x28",
        "seed": SEED,
        "epochs": profile.epochs,
        "batch_size": profile.batch_size,
        "learning_rate": profile.lr,
        "bootstrap_resamples": profile.bootstrap_n,
        "deferral_methods": list(DEFERRAL_METHODS),
        "coverages": list(COVERAGES),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "runtime_sec": runtime,
        "per_dataset": per_dataset,
        "aggregate": summary,
    }

    output = Path(args.output) if args.output else RESULTS_DIR / f"benchmark_{args.mode}.json"
    save_json(output, payload)

    print(f"\nSaved {output}")
    print(f"Runtime: {runtime}s")
    print(f"Mean gain @50% coverage (entropy): {summary['mean_gain_at_50pct_entropy']:+.3f}")
    print(
        "Datasets with significant gain: "
        f"{summary['datasets_with_significant_gain_at_50pct']}/{summary['n_datasets']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
