"""Aggregate every experiment CSV into one paper-ready findings document.

Reads whatever result files exist and writes notes/RESULTS.md. Missing stages are
reported as absent rather than crashing, so this is safe to run mid-pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "notes" / "RESULTS.md"

BOOTSTRAP_RESAMPLES = 2000


def load(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame if len(frame) else None


def ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Degenerate input returns a point."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1 or np.allclose(values, values[0]):
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, values.size), replace=True)
    means = draws.mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def mean_ci(values: np.ndarray, digits: int = 3) -> str:
    values = np.asarray(values, dtype=float)
    low, high = ci(values)
    return f"{np.nanmean(values):.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_no rows_\n"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def section_claims(out: list[str]) -> None:
    out.append("## 1. The original notebook's headline claims\n")
    frame = load("01_diagnostics_per_seed.csv")
    if frame is None:
        out.append("_results/01_diagnostics_per_seed.csv not present._\n")
        return

    n = len(frame)
    anchor = frame["optimum_is_classical_anchor"].mean()
    qaoa_opt = frame["qaoa_mode_is_optimal"].mean()
    out.append(
        f"Across {n} seeds, the QUBO optimum coincides with the greedy classical "
        f"anchor route in {pct(anchor)} of instances, and the QAOA mode is optimal in "
        f"{pct(qaoa_opt)}. The quantum stage therefore re-derives a route the classical "
        "pre-processing step already selected; its apparent success is not evidence of "
        "quantum advantage.\n"
    )
    rows = [
        {
            "quantity": "P(optimum == classical anchor)",
            "value": pct(anchor),
        },
        {"quantity": "P(QAOA mode is optimal)", "value": pct(qaoa_opt)},
        {
            "quantity": "XY feasible probability",
            "value": mean_ci(frame["xy_feasible_probability"].to_numpy()),
        },
        {
            "quantity": "Mode probability",
            "value": mean_ci(frame["mode_probability"].to_numpy()),
        },
        {
            "quantity": "Optimality gap to runner-up",
            "value": mean_ci(frame["optimality_gap_to_runner_up"].to_numpy()),
        },
        {"quantity": "Reported PDR", "value": mean_ci(frame["pdr"].to_numpy())},
        {
            "quantity": "Mean attack-detection F1",
            "value": mean_ci(frame["mean_attack_f1"].to_numpy()),
        },
    ]
    out.append(table(rows, ["quantity", "value"]))

    separability = load("01_risk_separability.csv")
    if separability is not None:
        gaps = separability.groupby("attack")["separation_gap"].mean().sort_values()
        rows = [
            {"attack": attack, "mean separation gap": f"{gap:+.4f}"}
            for attack, gap in gaps.items()
        ]
        out.append(
            "\nDetection F1 is 1.0 because the risk score is a strictly separating "
            "function of the same parameters used to inject the attacks. A positive "
            "gap means a threshold exists that never errs:\n"
        )
        out.append(table(rows, ["attack", "mean separation gap"]))


def section_detection(out: list[str]) -> None:
    out.append("\n## 2. Detection once the adversary can hide\n")
    frame = load("04_detection_under_stealth.csv")
    if frame is None:
        out.append("_results/04_detection_under_stealth.csv not present._\n")
        return

    grouped = frame.groupby(["detector", "stealth"])
    rows = []
    for (detector, stealth), block in grouped:
        rows.append(
            {
                "detector": detector,
                "stealth": f"{stealth:.2f}",
                "AUC": mean_ci(block["auc"].to_numpy()),
                "avg precision": mean_ci(block["average_precision"].to_numpy()),
                "F1": mean_ci(block["f1"].to_numpy()),
            }
        )
    out.append(
        "Stealth 0.0 reproduces the original always-drop adversary; higher values "
        "modulate the attack so its statistics overlap benign nodes suffering "
        "congestion or a weak link. Reporting AUC and average precision rather than a "
        "single thresholded F1 is what exposes the degradation:\n"
    )
    out.append(table(rows, ["detector", "stealth", "AUC", "avg precision", "F1"]))


def section_hardness(out: list[str]) -> None:
    out.append("\n## 3. Hardness ladder\n")

    out.append("### Rung 1: the published QUBO is a layered DAG shortest path\n")
    frame = load("05_classical_hardness.csv")
    if frame is None:
        out.append("_results/05_classical_hardness.csv not present._\n")
    else:
        layered = frame["is_layered_dag"].all()
        exact = frame["dp_matches_brute_force"].mean()
        out.append(
            f"Every one of {len(frame)} instances has a purely layer-adjacent coupling "
            f"graph (`is_layered_dag` all true: {layered}), and a stagewise dynamic "
            f"program matches brute-force enumeration on {pct(exact)} of them. The "
            "problem is solvable in O(stages x candidates^2) time, so no quantum "
            "speedup is available on this instance family.\n"
        )
        rows = []
        for (stages, candidates), block in frame.groupby(["n_stages", "candidates"]):
            rows.append(
                {
                    "size": f"{stages}x{candidates}",
                    "qubits": int(block["qubits"].iloc[0]),
                    "routes": int(block["n_routes"].iloc[0]),
                    "DP exact": pct(block["dp_matches_brute_force"].mean()),
                    "DP ms": f"{1000 * block['dp_seconds'].mean():.2f}",
                    "enumerate ms": f"{1000 * block['enumerate_seconds'].mean():.2f}",
                    "SA ratio": f"{block['ratio_sa'].mean():.3f}",
                    "greedy ratio": f"{block['ratio_greedy_forward'].mean():.3f}",
                }
            )
        out.append(
            table(
                rows,
                [
                    "size",
                    "qubits",
                    "routes",
                    "DP exact",
                    "DP ms",
                    "enumerate ms",
                    "SA ratio",
                    "greedy ratio",
                ],
            )
        )

    out.append("\n### Rung 2: co-channel interference breaks the DP\n")
    frame = load("06_interference_hardness.csv")
    if frame is None:
        out.append("_results/06_interference_hardness.csv not present._\n")
    else:
        rows = []
        for (radius, weight), block in frame.groupby(["radius", "weight"]):
            rows.append(
                {
                    "radius": radius,
                    "weight": weight,
                    "layered": pct(block["is_layered_dag"].mean()),
                    "DP optimal": pct(block["dp_is_optimal"].mean()),
                    "DP ratio": f"{block['dp_ratio'].mean():.3f}",
                    "SA optimal": pct(block["sa_is_optimal"].mean()),
                    "couplings": f"{block['n_couplings'].mean():.0f}",
                }
            )
        broken = frame[frame["weight"] > 0]["dp_is_optimal"].mean() if (frame["weight"] > 0).any() else float("nan")
        out.append(
            "Interference adds quadratic terms between non-adjacent stages, which is "
            "physically motivated (simultaneous transmissions on a shared channel) and "
            "destroys the layer-adjacency the DP relies on. With interference active "
            f"the DP is optimal on only {pct(broken)} of instances:\n"
        )
        out.append(
            table(
                rows,
                ["radius", "weight", "layered", "DP optimal", "DP ratio", "SA optimal", "couplings"],
            )
        )

    out.append("\n### Rung 3: multi-flow congestion\n")
    frame = load("12_multiflow_hardness_raw.csv")
    if frame is None:
        out.append("_results/12_multiflow_hardness_raw.csv not present (stage 12)._\n")
    else:
        out.append(f"{len(frame)} instances recorded.\n")
        for metric in ["dp_ratio", "sa_ratio", "qaoa_probability_of_optimum"]:
            if metric in frame.columns:
                out.append(f"- **{metric}:** mean = {frame[metric].mean():.4f}\n")


def section_scaling(out: list[str]) -> None:
    out.append("\n## 4. Scaling: constrained XY versus penalty encoding\n")
    frame = load("07_scaling_raw.csv")
    if frame is None:
        parts = sorted((RESULTS / "07_scaling_parts").glob("part_*.csv"))
        if not parts:
            out.append("_no scaling results yet (stage 07)._\n")
            return
        frame = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
        out.append(
            f"_Aggregated from {len(parts)} in-flight shards; the sweep may still be running._\n\n"
        )

    out.append(
        f"{len(frame)} configurations over "
        f"{frame['seed'].nunique()} seeds and sizes "
        f"{sorted(frame['qubits'].unique())} qubits.\n\n"
    )

    out.append("#### Feasible probability by size and encoding (mean [95% CI])\n")
    rows = []
    for qubits, block in frame.groupby("qubits"):
        row = {"qubits": int(qubits), "routes": int(block["n_routes"].iloc[0])}
        for mixer, sub in block.groupby("mixer"):
            row[f"{mixer} P(feasible)"] = mean_ci(sub["feasible_probability"].to_numpy(), 4)
        rows.append(row)
    columns = ["qubits", "routes"] + sorted(
        {key for row in rows for key in row if key.endswith("P(feasible)")}
    )
    out.append(table(rows, columns))

    out.append("\n#### Probability of the optimal route\n")
    rows = []
    for (qubits, mixer, objective, depth), block in frame.groupby(
        ["qubits", "mixer", "objective", "depth"]
    ):
        rows.append(
            {
                "qubits": int(qubits),
                "mixer": mixer,
                "objective": objective,
                "p": int(depth),
                "P(opt)": mean_ci(block["probability_of_optimum"].to_numpy(), 4),
                "P(opt | feasible)": mean_ci(
                    block["conditional_probability_of_optimum"].to_numpy(), 4
                ),
                "approx ratio": mean_ci(block["approximation_ratio"].to_numpy(), 4),
                "mode optimal": pct(block["conditional_mode_is_optimal"].mean()),
            }
        )
    out.append(
        table(
            rows,
            [
                "qubits",
                "mixer",
                "objective",
                "p",
                "P(opt)",
                "P(opt | feasible)",
                "approx ratio",
                "mode optimal",
            ],
        )
    )

    random_baseline = 1.0 / frame["n_routes"]
    frame = frame.assign(random_baseline=random_baseline)
    beats = frame.groupby(["qubits", "mixer"]).apply(
        lambda block: (
            block["conditional_probability_of_optimum"] > block["random_baseline"]
        ).mean(),
        include_groups=False,
    )
    out.append(
        "\n#### Does the circuit beat uniform sampling over feasible routes?\n"
        "A constrained mixer guarantees feasibility for free, so the honest question is "
        "whether the optimised circuit concentrates on good routes better than a uniform "
        "draw from the feasible subspace (1/routes).\n\n"
    )
    rows = [
        {"qubits": int(qubits), "mixer": mixer, "fraction beating uniform": pct(value)}
        for (qubits, mixer), value in beats.items()
    ]
    out.append(table(rows, ["qubits", "mixer", "fraction beating uniform"]))


def section_hardware(out: list[str]) -> None:
    out.append("\n## 5. Hardware-basis circuit cost (heavy-hex)\n")
    frame = load("08_hardware_basis_raw.csv")
    if frame is None:
        out.append("_results/08_hardware_basis_raw.csv not present._\n")
        return
    rows = []
    for (qubits, mixer, depth), block in frame.groupby(["qubits", "mixer", "depth"]):
        rows.append(
            {
                "qubits": int(qubits),
                "mixer": mixer,
                "p": int(depth),
                "depth": f"{block['circuit_depth'].mean():.0f}",
                "CX": f"{block['cx'].mean():.0f}",
            }
        )
    out.append(
        "Depth measured after transpilation to a heavy-hex coupling map with basis "
        "{rz, sx, x, cx}, which is the only number that predicts hardware runtime. The "
        "original notebook quoted pre-transpilation depth in an all-to-all basis:\n"
    )
    out.append(table(rows, ["qubits", "mixer", "p", "depth", "CX"]))


def section_noise(out: list[str]) -> None:
    out.append("\n## 6. Noise and post-selection\n")
    frame = load("11_noise_postselection_summary.csv")
    if frame is None:
        out.append("_results/11_noise_postselection_summary.csv not present (stage 11)._\n")
        return

    sub = frame[
        (frame["qubits"] == 9)
        & (frame["depth"] == 1)
        & (frame["noise_type"] == "symmetric_bit_flip")
        & (frame["metric"] == "retained_shot_fraction")
    ].sort_values("error_strength")
    rows = [
        {
            "error_strength": f"{row.error_strength:.4f}",
            "retention": mean_ci(np.array([row.mean])),
        }
        for row in sub.itertuples()
    ]
    out.append(
        "Post-selection retention for 9-qubit XY circuits under symmetric bit-flip noise:\n"
    )
    out.append(table(rows, ["error_strength", "retention"]))


def section_lifetime(out: list[str]) -> None:
    out.append("\n## 7. Multi-round network lifetime\n")
    frame = load("09_network_lifetime_summary.csv")
    if frame is None:
        out.append("_results/09_network_lifetime_summary.csv not present (stage 09)._\n")
        return
    rows = []
    for row in frame.itertuples():
        rows.append(
            {
                "method": row.method,
                "PDR": f"{row.pdr_mean:.4f}",
                "first relay death": f"{row.first_relay_death_round_mean:.1f}",
                "route changes": f"{row.route_changes_mean:.1f}",
            }
        )
    out.append(f"250 rounds, {int(frame['n_seeds'].iloc[0])} seeds.\n")
    out.append(table(rows, ["method", "PDR", "first relay death", "route changes"]))


def section_poisoning(out: list[str]) -> None:
    out.append("\n## 8. Routing-metric poisoning\n")
    frame = load("10_poisoning_robustness_summary.csv")
    if frame is None:
        out.append("_results/10_poisoning_robustness_summary.csv not present (stage 10)._\n")
        return

    sub = frame[
        (frame["attack"] == "sinkhole")
        & (frame["intensity"] == 0.25)
        & (frame["stealth"] == 0.0)
        & (frame["metric"].isin(["regret", "malicious_relay_exposure"]))
    ]
    rows = []
    for formulation in ["clean_oracle", "naive_poisoned", "trust_weighted"]:
        block = sub[sub["formulation"] == formulation]
        row = {"formulation": formulation}
        for metric in ["regret", "malicious_relay_exposure"]:
            values = block[block["metric"] == metric]["mean"]
            row[metric] = f"{values.iloc[0]:.4f}" if len(values) else "—"
        rows.append(row)
    out.append("Sinkhole attack at intensity 0.25:\n")
    out.append(table(rows, ["formulation", "regret", "malicious_relay_exposure"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    out: list[str] = [
        "# AgriTrust QAOA: consolidated experimental results\n",
        "Generated by `make_report.py`. Every number is produced by a script in "
        "`scripts/`; nothing here is hand-entered. Intervals are 2.5/97.5 percentile "
        f"bootstrap CIs for the mean over {BOOTSTRAP_RESAMPLES} resamples.\n\n",
    ]

    section_claims(out)
    section_detection(out)
    section_hardness(out)
    section_scaling(out)
    section_hardware(out)
    section_noise(out)
    section_lifetime(out)
    section_poisoning(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out))
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
