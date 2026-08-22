#!/usr/bin/env python3
"""Render the benchmark figures from a results JSON file.

Usage:
    python scripts/make_figures.py --results results/golden/benchmark_full.json
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from common import load_json
from config import COVERAGES, DEFERRAL_METHODS, FIGURES_DIR

PANEL_COLUMNS = 4
POLICY_COLOURS = {
    "entropy": "#1f77b4",
    "max_prob": "#ff7f0e",
    "margin": "#2ca02c",
    "temp_entropy": "#9467bd",
    "mc_dropout": "#d62728",
}


def coverage_axis() -> list[float]:
    return [coverage * 100 for coverage in COVERAGES]


def curve_values(record: dict, policy: str) -> list[float]:
    curve = record["policies"][policy]["acc_at_coverage"]
    return [curve[f"acc@{coverage:.0%}"] * 100 for coverage in COVERAGES]


def figure_coverage_panels(results: dict, output: Path) -> None:
    """One coverage-accuracy panel per dataset, all policies overlaid."""
    names = list(results["per_dataset"])
    rows = int(np.ceil(len(names) / PANEL_COLUMNS))
    fig, axes = plt.subplots(rows, PANEL_COLUMNS, figsize=(4 * PANEL_COLUMNS, 3.1 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, name in zip(axes, names):
        record = results["per_dataset"][name]
        for policy in DEFERRAL_METHODS:
            axis.plot(
                coverage_axis(),
                curve_values(record, policy),
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=policy,
                color=POLICY_COLOURS.get(policy),
            )
        axis.set_title(f"{name}\n{record['modality']}", fontsize=9)
        axis.set_xlabel("Coverage (%)", fontsize=8)
        axis.set_ylabel("Accuracy (%)", fontsize=8)
        axis.invert_xaxis()
        axis.grid(alpha=0.3, linewidth=0.5)
        axis.tick_params(labelsize=7)

    for axis in axes[len(names):]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(DEFERRAL_METHODS), fontsize=9)
    fig.suptitle(
        "MedMNIST-Defer: accuracy under progressive deferral of low-confidence cases",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def figure_gain_ranking(results: dict, output: Path) -> None:
    """Horizontal bar chart of the accuracy gain at 50% coverage."""
    names = list(results["per_dataset"])
    gains, errors_low, errors_high = [], [], []
    for name in names:
        gain = results["per_dataset"][name]["policies"]["entropy"]["deferral_gain"]["delta@50%"]
        gains.append(gain["delta_mean"] * 100)
        errors_low.append((gain["delta_mean"] - gain["ci95_lo"]) * 100)
        errors_high.append((gain["ci95_hi"] - gain["delta_mean"]) * 100)

    order = np.argsort(gains)
    sorted_names = [names[i] for i in order]
    sorted_gains = [gains[i] for i in order]
    errors = [[errors_low[i] for i in order], [errors_high[i] for i in order]]

    fig, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(sorted_names))
    axis.barh(positions, sorted_gains, xerr=errors, color="#1f77b4", alpha=0.85, capsize=3)
    axis.set_yticks(positions)
    axis.set_yticklabels(sorted_names, fontsize=9)
    axis.set_xlabel("Accuracy gain at 50% coverage (percentage points)")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.grid(axis="x", alpha=0.3)
    axis.set_title("Entropy deferral gain per modality (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def figure_calibration_vs_gain(results: dict, output: Path) -> None:
    """Scatter of pre-calibration ECE against the deferral gain."""
    names = list(results["per_dataset"])
    ece = [results["per_dataset"][n]["ece_uncalibrated"] for n in names]
    gains = [results["per_dataset"][n]["entropy_delta_at_50pct"] * 100 for n in names]
    base = [results["per_dataset"][n]["accuracy_full_coverage"] * 100 for n in names]

    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    scatter = axis.scatter(ece, gains, c=base, cmap="viridis", s=90, edgecolor="black")
    for name, x, y in zip(names, ece, gains):
        axis.annotate(name.replace("mnist", ""), (x, y), fontsize=7,
                      textcoords="offset points", xytext=(5, 4))
    axis.set_xlabel("Expected calibration error (uncalibrated)")
    axis.set_ylabel("Accuracy gain at 50% coverage (pp)")
    correlation = results["aggregate"].get("correlation_ece_vs_gain_at_50pct")
    subtitle = f" (Pearson r = {correlation:.2f})" if correlation is not None else ""
    axis.set_title(f"Calibration error versus deferral benefit{subtitle}")
    axis.grid(alpha=0.3)
    fig.colorbar(scatter, ax=axis, label="Accuracy at full coverage (%)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def figure_policy_comparison(results: dict, output: Path) -> None:
    """Mean gain and risk-coverage AUC per deferral policy."""
    policies = list(DEFERRAL_METHODS)
    per_policy = results["aggregate"]["per_policy"]
    gains = [per_policy[p]["mean_gain_at_50pct"] * 100 for p in policies]
    rc_auc = [per_policy[p]["mean_risk_coverage_auc"] for p in policies]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.bar(policies, gains, color=[POLICY_COLOURS.get(p) for p in policies], alpha=0.85)
    left.set_ylabel("Mean gain at 50% coverage (pp)")
    left.set_title("Average benefit per policy")
    left.tick_params(axis="x", rotation=20, labelsize=9)
    left.grid(axis="y", alpha=0.3)

    right.bar(policies, rc_auc, color=[POLICY_COLOURS.get(p) for p in policies], alpha=0.85)
    right.set_ylabel("Mean risk-coverage AUC (lower is better)")
    right.set_title("Ranking quality per policy")
    right.tick_params(axis="x", rotation=20, labelsize=9)
    right.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--outdir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    results = load_json(Path(args.results))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    figures = {
        "fig1_coverage_curves.png": figure_coverage_panels,
        "fig2_gain_ranking.png": figure_gain_ranking,
        "fig3_calibration_vs_gain.png": figure_calibration_vs_gain,
        "fig4_policy_comparison.png": figure_policy_comparison,
    }
    for filename, renderer in figures.items():
        path = outdir / filename
        renderer(results, path)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
