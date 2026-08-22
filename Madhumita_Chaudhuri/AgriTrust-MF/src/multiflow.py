"""Coupled multi-flow routing for the third AgriTrust hardness rung.

The base AgriTrust objective is a shortest path through a layered DAG: with one
one-hot block per stage and only consecutive-stage couplings, dynamic programming
is polynomial.  Here every flow has its own copy of those routing blocks, while
all copies refer to the *same physical relay identifiers*.  Quadratic terms charge
two conflicting flows for selecting the same capacity-one relay or nearby relays.
Those cross-flow terms prevent decomposition into independent layered paths.

The general formulation is NP-hard by a direct reduction from graph colouring.
Given a graph, create one single-stage flow per vertex and one common physical
relay per colour.  Set routing costs to zero and add a same-relay penalty exactly
for flow pairs corresponding to graph edges.  A zero-cost assignment exists iff
the graph is k-colourable.  Thus deciding whether this routing objective has value
zero is NP-complete for k >= 3.  This reduction is a complexity statement about
the general family; small simulations in this module do not demonstrate quantum
or computational advantage.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from agritrust_core import Config as AgriTrustConfig
from agritrust_core import build_instance as build_single_flow_instance
from agritrust_core import cost_scale, edge_routing_cost
from baselines import layered_dp, simulated_annealing
from fast_sim import SubspaceXYSimulator, feasible_costs_from_qubo


@dataclass(frozen=True)
class MultiFlowConfig:
    """Deterministic controls for a shared-relay multi-flow instance."""

    seed: int = 7
    n_flows: int = 2
    n_stages: int = 4
    candidates_per_stage: int = 3
    n_sensor_nodes: int = 150
    congestion_weight: float = 2.0
    nearby_weight: float = 1.0
    conflict_radius: float = 12.0
    stage_conflict_window: int = 0
    flow_demands: tuple[float, ...] | None = None
    flow_conflicts: tuple[tuple[int, int], ...] | None = None
    disconnect_penalty: float = 6.0
    sa_iterations: int = 2000
    qaoa_depth: int = 1
    qaoa_maxiter: int = 25
    max_exact_states: int = 2_000_000

    def __post_init__(self):
        if self.n_flows < 1 or self.n_stages < 1 or self.candidates_per_stage < 2:
            raise ValueError("Need at least one flow/stage and two candidates per stage.")
        if self.flow_demands is not None and len(self.flow_demands) != self.n_flows:
            raise ValueError("flow_demands must contain one positive value per flow.")
        if any(demand <= 0 for demand in self.resolved_demands()):
            raise ValueError("Flow demands must be positive.")
        if self.conflict_radius < 0 or self.stage_conflict_window < 0:
            raise ValueError("Conflict radius and stage window must be non-negative.")

    @property
    def n_blocks(self) -> int:
        return self.n_flows * self.n_stages

    @property
    def n_qubits(self) -> int:
        return self.n_blocks * self.candidates_per_stage

    @property
    def n_feasible(self) -> int:
        return self.candidates_per_stage**self.n_blocks

    def resolved_demands(self) -> tuple[float, ...]:
        if self.flow_demands is not None:
            return tuple(float(value) for value in self.flow_demands)
        return tuple(1.0 + 0.15 * flow for flow in range(self.n_flows))

    def resolved_conflicts(self) -> tuple[tuple[int, int], ...]:
        if self.flow_conflicts is None:
            return tuple(combinations(range(self.n_flows), 2))
        conflicts = []
        for left, right in self.flow_conflicts:
            if left == right or not (0 <= left < self.n_flows and 0 <= right < self.n_flows):
                raise ValueError(f"Invalid flow conflict ({left}, {right}).")
            conflicts.append((min(left, right), max(left, right)))
        return tuple(sorted(set(conflicts)))


@dataclass(frozen=True)
class SolverResult:
    cost: float
    choices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class DPDiagnostic:
    cross_flow_couplings: int
    objective_is_coupled: bool
    independent_dp_cost: float
    independent_dp_true_cost: float
    independent_dp_choices: tuple[tuple[int, ...], ...]
    exact_cost: float | None
    independent_dp_is_optimal: bool | None


@dataclass(frozen=True)
class QAOAResult:
    probabilities: np.ndarray
    probability_of_optimum: float
    approximation_ratio: float
    expectation_ratio: float
    mode_cost: float
    expectation: float
    optimizer_objective: float
    optimizer_success: bool
    evaluations: int


@dataclass(frozen=True)
class MultiFlowInstance:
    cfg: MultiFlowConfig
    wsn: dict
    candidates: list[list[int]]
    flow_sources: tuple[int, ...]
    flow_sinks: tuple[int, ...]
    linear: np.ndarray
    quadratic: dict[tuple[int, int], float]
    route_costs: np.ndarray
    diagnostic: DPDiagnostic


def qindex(
    flow: int,
    stage: int,
    candidate: int,
    cfg: MultiFlowConfig,
) -> int:
    """Qubit index for the one-hot block identified by (flow, stage)."""
    if not 0 <= flow < cfg.n_flows:
        raise IndexError("flow index out of range")
    if not 0 <= stage < cfg.n_stages:
        raise IndexError("stage index out of range")
    if not 0 <= candidate < cfg.candidates_per_stage:
        raise IndexError("candidate index out of range")
    return (flow * cfg.n_stages + stage) * cfg.candidates_per_stage + candidate


def _add_quadratic(
    quadratic: dict[tuple[int, int], float],
    left: int,
    right: int,
    coefficient: float,
):
    if coefficient == 0.0:
        return
    key = (min(left, right), max(left, right))
    quadratic[key] = quadratic.get(key, 0.0) + float(coefficient)


def _validate_model_inputs(
    candidates: list[list[int]],
    node_cost: np.ndarray,
    flow_sources: tuple[int, ...],
    flow_sinks: tuple[int, ...],
    cfg: MultiFlowConfig,
):
    if len(candidates) != cfg.n_stages:
        raise ValueError("candidates must contain one physical relay layer per stage.")
    if any(len(layer) != cfg.candidates_per_stage for layer in candidates):
        raise ValueError("Every stage must contain candidates_per_stage relays.")
    if len(flow_sources) != cfg.n_flows or len(flow_sinks) != cfg.n_flows:
        raise ValueError("Each flow needs a source and sink.")
    if max(node for layer in candidates for node in layer) >= len(node_cost):
        raise ValueError("node_cost does not cover every relay candidate.")


def build_multiflow_qubo(
    wsn: dict,
    candidates: list[list[int]],
    node_cost: np.ndarray,
    flow_sources: tuple[int, ...],
    flow_sinks: tuple[int, ...],
    cfg: MultiFlowConfig,
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Build the one-hot QUBO with flow-local routing and cross-flow congestion.

    The same ``candidates`` list is used by every flow.  Consequently, selecting
    candidate ``c`` at stage ``s`` in two flows selects the same WSN node, not two
    synthetic copies that merely happen to receive a coupling.
    """
    node_cost = np.asarray(node_cost, dtype=float)
    _validate_model_inputs(candidates, node_cost, flow_sources, flow_sinks, cfg)
    graph = wsn["graph"]
    positions = np.asarray(wsn["positions"], dtype=float)
    scale = cost_scale(graph)
    demands = cfg.resolved_demands()
    linear = np.zeros(cfg.n_qubits, dtype=float)
    quadratic: dict[tuple[int, int], float] = {}

    for flow in range(cfg.n_flows):
        demand = demands[flow]
        source = flow_sources[flow]
        sink = flow_sinks[flow]
        for stage, layer in enumerate(candidates):
            for local, node in enumerate(layer):
                index = qindex(flow, stage, local, cfg)
                linear[index] += demand * float(node_cost[node])
                if stage == 0:
                    linear[index] += demand * edge_routing_cost(
                        graph,
                        source,
                        node,
                        scale,
                        cfg.disconnect_penalty,
                    )
                if stage == cfg.n_stages - 1:
                    linear[index] += demand * edge_routing_cost(
                        graph,
                        node,
                        sink,
                        scale,
                        cfg.disconnect_penalty,
                    )

        for stage in range(cfg.n_stages - 1):
            for left_local, left_node in enumerate(candidates[stage]):
                for right_local, right_node in enumerate(candidates[stage + 1]):
                    _add_quadratic(
                        quadratic,
                        qindex(flow, stage, left_local, cfg),
                        qindex(flow, stage + 1, right_local, cfg),
                        demand
                        * edge_routing_cost(
                            graph,
                            left_node,
                            right_node,
                            scale,
                            cfg.disconnect_penalty,
                        ),
                    )

    for left_flow, right_flow in cfg.resolved_conflicts():
        demand_scale = min(demands[left_flow], demands[right_flow])
        for left_stage, left_layer in enumerate(candidates):
            first_right_stage = max(0, left_stage - cfg.stage_conflict_window)
            last_right_stage = min(
                cfg.n_stages - 1,
                left_stage + cfg.stage_conflict_window,
            )
            for right_stage in range(first_right_stage, last_right_stage + 1):
                right_layer = candidates[right_stage]
                for left_local, left_node in enumerate(left_layer):
                    for right_local, right_node in enumerate(right_layer):
                        coefficient = 0.0
                        if left_node == right_node:
                            coefficient = cfg.congestion_weight * demand_scale
                        elif cfg.conflict_radius > 0.0 and cfg.nearby_weight != 0.0:
                            separation = float(
                                np.linalg.norm(positions[left_node] - positions[right_node])
                            )
                            if separation < cfg.conflict_radius:
                                coefficient = (
                                    cfg.nearby_weight
                                    * demand_scale
                                    * (1.0 - separation / cfg.conflict_radius)
                                )
                        _add_quadratic(
                            quadratic,
                            qindex(left_flow, left_stage, left_local, cfg),
                            qindex(right_flow, right_stage, right_local, cfg),
                            coefficient,
                        )

    return linear, quadratic


def _reshape_choices(
    choices: tuple[int, ...],
    cfg: MultiFlowConfig,
) -> tuple[tuple[int, ...], ...]:
    array = np.asarray(choices, dtype=int).reshape(cfg.n_flows, cfg.n_stages)
    return tuple(tuple(int(value) for value in row) for row in array)


def _flatten_choices(
    choices: tuple[tuple[int, ...], ...],
    cfg: MultiFlowConfig,
) -> tuple[int, ...]:
    array = np.asarray(choices, dtype=int)
    if array.shape != (cfg.n_flows, cfg.n_stages):
        raise ValueError("choices must have shape (n_flows, n_stages).")
    if np.any(array < 0) or np.any(array >= cfg.candidates_per_stage):
        raise ValueError("A candidate choice is out of range.")
    return tuple(int(value) for value in array.ravel())


def feasible_costs(
    linear: np.ndarray,
    quadratic: dict[tuple[int, int], float],
    cfg: MultiFlowConfig,
) -> np.ndarray:
    """Cost of every assignment with one relay per (flow, stage) block."""
    if cfg.n_feasible > cfg.max_exact_states:
        raise ValueError(
            f"Exact feasible space has {cfg.n_feasible:,} states; "
            f"limit is {cfg.max_exact_states:,}."
        )
    return feasible_costs_from_qubo(
        linear,
        quadratic,
        cfg.n_blocks,
        cfg.candidates_per_stage,
    )


def cost_of_choices(
    choices: tuple[tuple[int, ...], ...],
    linear: np.ndarray,
    quadratic: dict[tuple[int, int], float],
    cfg: MultiFlowConfig,
) -> float:
    flat = _flatten_choices(choices, cfg)
    active = [
        block * cfg.candidates_per_stage + choice
        for block, choice in enumerate(flat)
    ]
    value = float(np.asarray(linear, dtype=float)[active].sum())
    active_set = set(active)
    for (left, right), coefficient in quadratic.items():
        if left in active_set and right in active_set:
            value += coefficient
    return float(value)


def exact_feasible_optimum(
    linear: np.ndarray,
    quadratic: dict[tuple[int, int], float],
    cfg: MultiFlowConfig,
) -> SolverResult:
    costs = feasible_costs(linear, quadratic, cfg)
    flat_index = int(np.argmin(costs))
    flat_choices = tuple(
        int(value)
        for value in np.unravel_index(
            flat_index,
            (cfg.candidates_per_stage,) * cfg.n_blocks,
        )
    )
    return SolverResult(float(costs[flat_index]), _reshape_choices(flat_choices, cfg))


def random_feasible_baseline(
    linear: np.ndarray,
    quadratic: dict[tuple[int, int], float],
    cfg: MultiFlowConfig,
    samples: int = 100,
) -> tuple[SolverResult, float]:
    """Best and mean costs from deterministic uniformly random feasible routes."""
    if samples < 1:
        raise ValueError("samples must be positive.")
    rng = np.random.default_rng(cfg.seed + 17_003)
    results = []
    for _ in range(samples):
        flat = tuple(int(rng.integers(cfg.candidates_per_stage)) for _ in range(cfg.n_blocks))
        choices = _reshape_choices(flat, cfg)
        results.append(SolverResult(cost_of_choices(choices, linear, quadratic, cfg), choices))
    return min(results, key=lambda result: result.cost), float(
        np.mean([result.cost for result in results])
    )


def greedy_multiflow(
    linear: np.ndarray,
    quadratic: dict[tuple[int, int], float],
    cfg: MultiFlowConfig,
) -> SolverResult:
    """Commit blocks in flow-major order using costs exposed by prior choices."""
    linear = np.asarray(linear, dtype=float)
    active = []
    flat_choices = []
    for block in range(cfg.n_blocks):
        scores = []
        for candidate in range(cfg.candidates_per_stage):
            index = block * cfg.candidates_per_stage + candidate
            score = float(linear[index])
            for previous in active:
                score += quadratic.get((min(previous, index), max(previous, index)), 0.0)
            scores.append(score)
        choice = int(np.argmin(scores))
        flat_choices.append(choice)
        active.append(block * cfg.candidates_per_stage + choice)
    choices = _reshape_choices(tuple(flat_choices), cfg)
    return SolverResult(cost_of_choices(choices, linear, quadratic, cfg), choices)


def simulated_annealing_multiflow(
    linear: np.ndarray,
    quadratic: dict[tuple[int, int], float],
    cfg: MultiFlowConfig,
) -> SolverResult:
    """Single-block-flip annealing that always preserves all one-hot constraints."""
    rng = np.random.default_rng(cfg.seed + 29_011)
    cost, flat_choices = simulated_annealing(
        linear,
        quadratic,
        cfg.n_blocks,
        cfg.candidates_per_stage,
        rng,
        iterations=cfg.sa_iterations,
    )
    return SolverResult(float(cost), _reshape_choices(flat_choices, cfg))


def _independent_dp_choices(
    linear: np.ndarray,
    quadratic: dict[tuple[int, int], float],
    cfg: MultiFlowConfig,
) -> tuple[float, tuple[tuple[int, ...], ...]]:
    k = cfg.candidates_per_stage
    flow_width = cfg.n_stages * k
    total = 0.0
    choices = []
    for flow in range(cfg.n_flows):
        offset = flow * flow_width
        local_linear = np.asarray(linear[offset : offset + flow_width], dtype=float)
        local_quadratic = {}
        for (left, right), coefficient in quadratic.items():
            if offset <= left < offset + flow_width and offset <= right < offset + flow_width:
                local_quadratic[(left - offset, right - offset)] = coefficient
        flow_cost, flow_choices = layered_dp(
            local_linear,
            local_quadratic,
            cfg.n_stages,
            k,
        )
        total += flow_cost
        choices.append(tuple(int(value) for value in flow_choices))
    return float(total), tuple(choices)


def diagnose_independent_dp(
    linear: np.ndarray,
    quadratic: dict[tuple[int, int], float],
    cfg: MultiFlowConfig,
) -> DPDiagnostic:
    """Prove structural non-separability and, when affordable, score its effect.

    A nonzero cross-flow quadratic coefficient is a direct witness that summing
    independent single-flow layered-DP objectives omits part of the true objective.
    Whether the omitted term changes a particular tiny instance's optimum is a
    separate empirical diagnostic.
    """
    flow_width = cfg.n_stages * cfg.candidates_per_stage
    cross_flow = sum(
        1
        for (left, right), coefficient in quadratic.items()
        if coefficient != 0.0 and left // flow_width != right // flow_width
    )
    independent_cost, choices = _independent_dp_choices(linear, quadratic, cfg)
    true_cost = cost_of_choices(choices, linear, quadratic, cfg)
    exact_cost = None
    is_optimal = None
    if cfg.n_feasible <= cfg.max_exact_states:
        exact_cost = exact_feasible_optimum(linear, quadratic, cfg).cost
        is_optimal = bool(np.isclose(true_cost, exact_cost, rtol=1e-10, atol=1e-12))
    return DPDiagnostic(
        cross_flow_couplings=cross_flow,
        objective_is_coupled=cross_flow > 0,
        independent_dp_cost=independent_cost,
        independent_dp_true_cost=true_cost,
        independent_dp_choices=choices,
        exact_cost=exact_cost,
        independent_dp_is_optimal=is_optimal,
    )


def run_subspace_qaoa(
    linear: np.ndarray,
    quadratic: dict[tuple[int, int], float],
    cfg: MultiFlowConfig,
) -> QAOAResult:
    """Optimize a small XY-QAOA state in the exact feasible one-hot subspace."""
    costs = feasible_costs(linear, quadratic, cfg)
    simulator = SubspaceXYSimulator(
        costs,
        n_stages=cfg.n_blocks,
        candidates_per_stage=cfg.candidates_per_stage,
    )
    rng = np.random.default_rng(cfg.seed + 41_021)
    initial = np.concatenate(
        [
            rng.uniform(0.0, np.pi, size=cfg.qaoa_depth),
            rng.uniform(0.0, np.pi / 2.0, size=cfg.qaoa_depth),
        ]
    )

    def objective(parameters):
        gammas = np.mod(parameters[: cfg.qaoa_depth], 2.0 * np.pi)
        betas = np.mod(parameters[cfg.qaoa_depth :], 2.0 * np.pi)
        probabilities = simulator.probabilities(gammas, betas)
        return float(np.dot(probabilities, costs))

    result = minimize(
        objective,
        x0=initial,
        method="COBYLA",
        options={"maxiter": cfg.qaoa_maxiter, "rhobeg": 0.7, "tol": 1e-5},
    )
    gammas = np.mod(result.x[: cfg.qaoa_depth], 2.0 * np.pi)
    betas = np.mod(result.x[cfg.qaoa_depth :], 2.0 * np.pi)
    probabilities = simulator.probabilities(gammas, betas)
    optimum = float(np.min(costs))
    optimum_mask = np.isclose(costs, optimum, rtol=1e-10, atol=1e-12)
    probability_of_optimum = float(np.sum(probabilities[optimum_mask]))
    mode_cost = float(costs[int(np.argmax(probabilities))])
    expectation = float(np.dot(probabilities, costs))

    def ratio(cost):
        if np.isclose(cost, 0.0):
            return 1.0 if np.isclose(optimum, 0.0) else float("nan")
        return float(optimum / cost)

    return QAOAResult(
        probabilities=probabilities,
        probability_of_optimum=probability_of_optimum,
        approximation_ratio=ratio(mode_cost),
        expectation_ratio=ratio(expectation),
        mode_cost=mode_cost,
        expectation=expectation,
        optimizer_objective=float(result.fun),
        optimizer_success=bool(result.success),
        evaluations=int(result.nfev),
    )


def _select_flow_sources(
    wsn: dict,
    candidates: list[list[int]],
    cfg: MultiFlowConfig,
) -> tuple[int, ...]:
    graph = wsn["graph"]
    positions = np.asarray(wsn["positions"], dtype=float)
    relay_nodes = {node for layer in candidates for node in layer}
    selected = [int(wsn["source"])]
    eligible = [
        node
        for node in range(cfg.n_sensor_nodes)
        if node not in relay_nodes and node not in selected
    ]
    connected = [
        node
        for node in eligible
        if any(graph.has_edge(node, relay) for relay in candidates[0])
    ]
    ranked = sorted(connected, key=lambda node: (positions[node, 0], node))
    ranked.extend(
        node
        for node in sorted(eligible, key=lambda node: (positions[node, 0], node))
        if node not in ranked
    )
    for node in ranked:
        if len(selected) == cfg.n_flows:
            break
        selected.append(int(node))
    if len(selected) != cfg.n_flows:
        raise RuntimeError("Could not select distinct physical sources for all flows.")
    return tuple(selected)


def build_multiflow_instance(cfg: MultiFlowConfig) -> MultiFlowInstance:
    """Reuse AgriTrust's WSN and relay pruning, then couple flow copies physically."""
    single_cfg = AgriTrustConfig(
        seed=cfg.seed,
        n_sensor_nodes=cfg.n_sensor_nodes,
        n_stages=cfg.n_stages,
        candidates_per_stage=cfg.candidates_per_stage,
        disconnect_penalty=cfg.disconnect_penalty,
    )
    base = build_single_flow_instance(single_cfg, full_space=False)
    candidates = [list(layer) for layer in base.candidates]
    sources = _select_flow_sources(base.wsn, candidates, cfg)
    sinks = tuple(int(base.wsn["sink"]) for _ in range(cfg.n_flows))
    linear, quadratic = build_multiflow_qubo(
        base.wsn,
        candidates,
        base.node_cost,
        sources,
        sinks,
        cfg,
    )
    costs = feasible_costs(linear, quadratic, cfg)
    diagnostic = diagnose_independent_dp(linear, quadratic, cfg)
    return MultiFlowInstance(
        cfg=cfg,
        wsn=base.wsn,
        candidates=candidates,
        flow_sources=sources,
        flow_sinks=sinks,
        linear=linear,
        quadratic=quadratic,
        route_costs=costs,
        diagnostic=diagnostic,
    )
