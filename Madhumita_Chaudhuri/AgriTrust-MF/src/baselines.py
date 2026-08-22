"""Classical solvers for the layered routing QUBO.

A structural observation drives this file. In `build_routing_qubo` the quadratic
terms connect stage i only to stage i+1, and there are no couplings within a stage
or between non-adjacent stages. Minimising over stage-wise one-hot assignments is
therefore exactly a shortest-path problem on a layered DAG, which a dynamic
programme solves exactly in O(s * k^2) time for any size.

That has to be measured, not assumed, so `layered_dp` is checked against exhaustive
enumeration wherever enumeration is affordable.
"""

from __future__ import annotations

import numpy as np


def layered_dp(linear, quadratic, n_stages: int, candidates_per_stage: int):
    """Exact optimum of the stage-wise one-hot QUBO in O(s * k^2).

    Returns (cost, choices). Correct whenever the quadratic terms only couple
    consecutive stages, which `assert_layered` verifies.
    """
    k = candidates_per_stage
    linear = np.asarray(linear, dtype=float)

    def coupling(stage):
        """k x k matrix of stage->stage+1 costs."""
        matrix = np.zeros((k, k))
        for a in range(k):
            for b in range(k):
                i = stage * k + a
                j = (stage + 1) * k + b
                matrix[a, b] = quadratic.get((min(i, j), max(i, j)), 0.0)
        return matrix

    best = linear[0:k].copy()
    parents = []
    for stage in range(n_stages - 1):
        matrix = coupling(stage)
        totals = best[:, None] + matrix + linear[(stage + 1) * k : (stage + 2) * k][None, :]
        parent = np.argmin(totals, axis=0)
        best = totals[parent, np.arange(k)]
        parents.append(parent)

    final = int(np.argmin(best))
    choices = [final]
    for parent in reversed(parents):
        choices.append(int(parent[choices[-1]]))
    choices.reverse()
    return float(best[final]), tuple(choices)


def assert_layered(quadratic, n_stages: int, candidates_per_stage: int) -> bool:
    """True if every quadratic term couples only consecutive stages."""
    k = candidates_per_stage
    for i, j in quadratic:
        if i // k == j // k:
            return False
        if abs(i // k - j // k) != 1:
            return False
    return True


def greedy_stagewise(linear, quadratic, n_stages: int, candidates_per_stage: int):
    """Pick the cheapest candidate per stage using linear terms only.

    The natural naive heuristic: ignores inter-stage link costs entirely.
    """
    k = candidates_per_stage
    linear = np.asarray(linear, dtype=float)
    choices = tuple(int(np.argmin(linear[stage * k : (stage + 1) * k])) for stage in range(n_stages))
    return _cost_of(choices, linear, quadratic, k), choices


def greedy_forward(linear, quadratic, n_stages: int, candidates_per_stage: int):
    """Left-to-right greedy: commit each stage given the previous commitment."""
    k = candidates_per_stage
    linear = np.asarray(linear, dtype=float)
    choices = [int(np.argmin(linear[0:k]))]
    for stage in range(1, n_stages):
        previous = choices[-1]
        i = (stage - 1) * k + previous
        scores = []
        for b in range(k):
            j = stage * k + b
            link = quadratic.get((min(i, j), max(i, j)), 0.0)
            scores.append(linear[j] + link)
        choices.append(int(np.argmin(scores)))
    choices = tuple(choices)
    return _cost_of(choices, linear, quadratic, k), choices


def random_feasible(linear, quadratic, n_stages: int, candidates_per_stage: int, rng):
    choices = tuple(int(rng.integers(candidates_per_stage)) for _ in range(n_stages))
    return _cost_of(choices, np.asarray(linear, dtype=float), quadratic, candidates_per_stage), choices


def expected_random_cost(feasible_costs):
    return float(np.mean(feasible_costs))


def simulated_annealing(
    linear,
    quadratic,
    n_stages: int,
    candidates_per_stage: int,
    rng,
    iterations: int = 2000,
    initial_temperature: float = 2.0,
):
    """Single-flip simulated annealing over the feasible one-hot subspace."""
    k = candidates_per_stage
    linear = np.asarray(linear, dtype=float)
    current = [int(rng.integers(k)) for _ in range(n_stages)]
    current_cost = _cost_of(tuple(current), linear, quadratic, k)
    best, best_cost = tuple(current), current_cost

    for step in range(iterations):
        temperature = initial_temperature * (1.0 - step / iterations) + 1e-9
        stage = int(rng.integers(n_stages))
        previous = current[stage]
        proposal = int(rng.integers(k))
        if proposal == previous:
            continue
        current[stage] = proposal
        proposal_cost = _cost_of(tuple(current), linear, quadratic, k)
        delta = proposal_cost - current_cost
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            current_cost = proposal_cost
            if current_cost < best_cost:
                best, best_cost = tuple(current), current_cost
        else:
            current[stage] = previous

    return best_cost, best


def _cost_of(choices, linear, quadratic, k):
    active = [stage * k + choice for stage, choice in enumerate(choices)]
    value = float(linear[active].sum())
    active_set = set(active)
    for (i, j), coefficient in quadratic.items():
        if i in active_set and j in active_set:
            value += coefficient
    return value
