"""Shared utilities for statistically rigorous QAOA scaling sweeps."""

from __future__ import annotations

import numpy as np


def one_hot_violation(n_stages: int, candidates_per_stage: int) -> np.ndarray:
    """Return sum_s (sum_k x_sk - 1)^2 for every computational basis state."""
    n_qubits = n_stages * candidates_per_stage
    index_dtype = np.uint32 if n_qubits <= 32 else np.uint64
    indices = np.arange(1 << n_qubits, dtype=index_dtype)
    violation = np.zeros(len(indices), dtype=np.uint16)

    for stage in range(n_stages):
        count = np.zeros(len(indices), dtype=np.uint8)
        for candidate in range(candidates_per_stage):
            qubit = stage * candidates_per_stage + candidate
            count += ((indices >> qubit) & 1).astype(np.uint8)
        difference = count.astype(np.int16) - 1
        violation += (difference * difference).astype(np.uint16)

    return violation


def minimum_feasibility_penalty(
    base_costs: np.ndarray,
    violation: np.ndarray,
    margin: float = 1.05,
) -> float:
    """Smallest penalty weight that makes every ground state feasible.

    This is a stronger and fairer penalty-X baseline than choosing a magic
    coefficient: each instance receives the minimally sufficient exact penalty,
    with a small strict-separation margin.
    """
    if margin <= 1.0:
        raise ValueError("Penalty margin must be greater than one.")

    base_costs = np.asarray(base_costs, dtype=float)
    violation = np.asarray(violation, dtype=float)
    feasible = violation == 0
    if not np.any(feasible):
        raise ValueError("At least one feasible state is required.")

    feasible_optimum = float(np.min(base_costs[feasible]))
    infeasible = ~feasible
    if not np.any(infeasible):
        return 0.0

    required = (feasible_optimum - base_costs[infeasible]) / violation[infeasible]
    threshold = float(max(0.0, np.max(required)))
    return margin * threshold


SCORE_KEYS: tuple[str, ...] = (
    "feasible_probability",
    "probability_of_optimum",
    "conditional_probability_of_optimum",
    "conditional_mode_probability",
    "conditional_mode_is_optimal",
    "approximation_ratio",
    "n_routes",
    "n_optimal_routes",
    "best_cost",
    "worst_cost",
    "uniform_cost",
    "conditional_expected_cost",
    "normalized_approximation_ratio",
    "random_normalized_gap",
    "enrichment_over_uniform",
    "top_decile_mass",
    "mode_cost_percentile",
)


def score_route_distribution(
    route_probabilities: np.ndarray,
    route_costs: np.ndarray,
) -> dict[str, float | bool]:
    """Score feasible-route probability mass without hiding post-selection cost.

    Three families of number come out of this, and they answer different questions.

    Post-selection cost: ``feasible_probability`` is the fraction of shots that
    survive the one-hot filter. A constrained mixer makes this 1.0 by construction;
    a penalty encoding pays here and the payment grows with qubit count.

    Absolute quality: ``probability_of_optimum`` and ``conditional_mode_is_optimal``
    are the strict questions, and they are the ones that degrade honestly with size.

    Quality relative to doing nothing clever: ``random_normalized_gap`` and
    ``enrichment_over_uniform`` compare the circuit against a uniform draw from the
    feasible subspace, which is what a constrained mixer gives you for free before
    any optimisation. These exist because the raw ``approximation_ratio``
    (optimum / mode cost) is nearly 1 whenever route costs sit in a narrow band,
    so on its own it flatters the method regardless of whether it works.
    """
    probabilities = np.asarray(route_probabilities, dtype=float)
    costs = np.asarray(route_costs, dtype=float)
    if probabilities.shape != costs.shape:
        raise ValueError("Route probabilities and costs must have the same shape.")
    if probabilities.size == 0:
        raise ValueError("At least one feasible route is required.")

    n_routes = int(costs.size)
    feasible_probability = float(np.sum(probabilities))
    optimum = float(np.min(costs))
    worst = float(np.max(costs))
    uniform_cost = float(np.mean(costs))
    optimum_mask = np.isclose(costs, optimum, rtol=1e-10, atol=1e-12)
    n_optimal = int(np.count_nonzero(optimum_mask))
    probability_of_optimum = float(np.sum(probabilities[optimum_mask]))

    # The best 10% of routes by cost, used as a coarser and more robust signal than
    # the optimum alone. Uniform sampling would put ~0.1 mass here.
    decile_size = max(1, int(np.ceil(0.1 * n_routes)))
    decile_threshold = float(np.sort(costs)[decile_size - 1])
    decile_mask = costs <= decile_threshold

    if feasible_probability <= 0.0:
        return {
            "feasible_probability": 0.0,
            "probability_of_optimum": 0.0,
            "conditional_probability_of_optimum": 0.0,
            "conditional_mode_probability": 0.0,
            "conditional_mode_is_optimal": False,
            "approximation_ratio": float("nan"),
            "n_routes": n_routes,
            "n_optimal_routes": n_optimal,
            "best_cost": optimum,
            "worst_cost": worst,
            "uniform_cost": uniform_cost,
            "conditional_expected_cost": float("nan"),
            "normalized_approximation_ratio": float("nan"),
            "random_normalized_gap": float("nan"),
            "enrichment_over_uniform": 0.0,
            "top_decile_mass": 0.0,
            "mode_cost_percentile": float("nan"),
        }

    conditional = probabilities / feasible_probability
    mode_index = int(np.argmax(probabilities))
    mode_cost = float(costs[mode_index])
    expected_cost = float(np.dot(conditional, costs))

    # Both normalisations are undefined when every route costs the same, which makes
    # the instance vacuous rather than solved; report perfect and let the caller see
    # cost_spread to distinguish the cases.
    cost_spread = worst - optimum
    if cost_spread > 0:
        normalized_ratio = (worst - expected_cost) / cost_spread
    else:
        normalized_ratio = 1.0

    random_spread = uniform_cost - optimum
    if random_spread > 0:
        random_gap = (expected_cost - optimum) / random_spread
    else:
        random_gap = 0.0

    uniform_optimum_probability = n_optimal / n_routes
    conditional_optimum = probability_of_optimum / feasible_probability

    return {
        "feasible_probability": feasible_probability,
        "probability_of_optimum": probability_of_optimum,
        "conditional_probability_of_optimum": conditional_optimum,
        "conditional_mode_probability": float(conditional[mode_index]),
        "conditional_mode_is_optimal": bool(optimum_mask[mode_index]),
        "approximation_ratio": optimum / mode_cost if mode_cost != 0.0 else float("nan"),
        "n_routes": n_routes,
        "n_optimal_routes": n_optimal,
        "best_cost": optimum,
        "worst_cost": worst,
        "uniform_cost": uniform_cost,
        "conditional_expected_cost": expected_cost,
        "normalized_approximation_ratio": float(normalized_ratio),
        "random_normalized_gap": float(random_gap),
        "enrichment_over_uniform": conditional_optimum / uniform_optimum_probability,
        "top_decile_mass": float(np.sum(conditional[decile_mask])),
        "mode_cost_percentile": float(np.mean(costs >= mode_cost)),
    }
