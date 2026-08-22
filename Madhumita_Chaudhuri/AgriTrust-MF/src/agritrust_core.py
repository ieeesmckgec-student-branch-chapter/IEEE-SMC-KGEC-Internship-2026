"""Faithful, parameterised port of AgriTrust_9Qubit_Hybrid_QAOA_Colab.

The original Colab script hard-codes the problem size (3 stages x 3 candidates = 9
qubits), the `/content` output paths and a single seed. This module keeps the exact
same numerics and the exact same RNG draw order so that seed 7 reproduces the
published notebook output bit-for-bit, while exposing the knobs needed for
scaling, ablation and baseline studies.

Reference output for seed 7 (from the original notebook):
    source=12, sink=150, communication_range=30.0
    candidates=[[6, 127, 63], [7, 51, 14], [61, 70, 91]]
    exact_cost=4.164461, exact_choices=(0, 0, 0), mode_probability=0.77002
    pdr=0.236667, natural_drops=97, malicious_drops=0, energy=0.352529 J
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

ATTACK_TYPES = [
    "blackhole",
    "selective_forwarding",
    "sinkhole",
    "wormhole",
    "sybil",
]

PACKET_BITS = 4000
E_ELEC = 50e-9  # J/bit
E_AMP = 100e-12  # J/bit/m^2
FIELD_X = 100.0
FIELD_Y = 100.0
SINK_POSITION = np.array([105.0, 50.0])

MALICIOUS_DROP_PROBABILITY = {
    "normal": 0.00,
    "blackhole": 0.98,
    "selective_forwarding": 0.45,
    "sinkhole": 0.18,
    "wormhole": 0.20,
    "sybil": 0.22,
}


@dataclass
class Config:
    """All knobs. Defaults reproduce the original seed-7 notebook exactly."""

    seed: int = 7
    n_sensor_nodes: int = 150
    n_stages: int = 3
    candidates_per_stage: int = 3
    qaoa_depth: int = 1
    maxiter: int = 60
    shots: int = 2048
    packets_to_simulate: int = 300
    attackers_per_type: int = 3

    # Extensions (all defaults are no-ops relative to the original script).
    mixer: str = "xy"  # "xy" (W-state init) or "x" (penalty + Hadamard init)
    penalty_weight: float = 0.0  # only used when mixer == "x"
    objective: str = "expectation"  # "expectation" or "cvar"
    cvar_alpha: float = 0.25
    optimizer: str = "COBYLA"
    n_restarts: int = 1
    wrap_parameters: bool = True  # original behaviour: np.mod(theta, 2*pi)
    communication_ranges: tuple[float, ...] = (30.0, 34.0, 38.0, 42.0, 48.0)
    stage_windows: tuple[tuple[float, float], ...] | None = None
    pool_size: int = 35
    disconnect_penalty: float = 6.0

    # Co-channel interference between relays that are active in the same pipelined
    # transmission window. Physically standard in multi-hop WSNs and, unlike the
    # consecutive-stage link costs, it couples non-adjacent stages, which destroys
    # the layered-DAG structure that makes the base problem polynomial.
    interference_radius: float = 0.0  # metres; 0 disables
    interference_weight: float = 0.0

    # The original hard-coded windows, kept verbatim so 3 stages reproduces exactly.
    ORIGINAL_THREE_STAGE_WINDOWS = ((0.08, 0.42), (0.30, 0.70), (0.58, 0.94))

    @property
    def n_qubits(self) -> int:
        return self.n_stages * self.candidates_per_stage

    @property
    def n_feasible(self) -> int:
        return self.candidates_per_stage**self.n_stages

    def resolved_stage_windows(self) -> tuple[tuple[float, float], ...]:
        """Overlapping progress windows, one per relay stage.

        For three stages this returns the original notebook constants unchanged.
        For other stage counts it generalises the same pattern: centres evenly
        spaced strictly between source and sink, with windows wide enough to
        overlap their neighbours.
        """
        if self.stage_windows is not None:
            return self.stage_windows
        if self.n_stages == 3:
            return self.ORIGINAL_THREE_STAGE_WINDOWS
        centres = np.linspace(1.0, self.n_stages, self.n_stages) / (self.n_stages + 1)
        half_width = 0.55 / self.n_stages
        return tuple(
            (float(max(0.02, c - half_width)), float(min(0.98, c + half_width))) for c in centres
        )


# ----------------------------------------------------------------------------
# 1. WSN construction and attack simulation
# ----------------------------------------------------------------------------


def minmax(values, eps=1e-12):
    values = np.asarray(values, dtype=float)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < eps:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def radio_tx_energy(distance_m):
    return PACKET_BITS * (E_ELEC + E_AMP * distance_m**2)


def radio_rx_energy():
    return PACKET_BITS * E_ELEC


def build_graph(positions, environment, communication_range):
    """Undirected WSN graph. The sink is node index len(positions)."""
    all_positions = np.vstack([positions, SINK_POSITION])
    graph = nx.Graph()

    for node_id in range(len(all_positions)):
        graph.add_node(node_id, pos=all_positions[node_id])

    for i in range(len(all_positions)):
        for j in range(i + 1, len(all_positions)):
            distance = float(np.linalg.norm(all_positions[i] - all_positions[j]))
            if distance > communication_range:
                continue

            env_i = environment[i] if i < len(environment) else 0.0
            env_j = environment[j] if j < len(environment) else 0.0
            env_factor = 0.5 * (env_i + env_j)

            natural_loss = np.clip(
                0.015 + 0.18 * (distance / communication_range) ** 2 + 0.08 * env_factor,
                0.01,
                0.35,
            )
            etx = 1.0 / max(1.0 - natural_loss, 1e-6)

            graph.add_edge(
                i,
                j,
                distance=distance,
                natural_loss=float(natural_loss),
                etx=float(etx),
                tx_energy=float(radio_tx_energy(distance)),
                delay=float(0.004 + 0.002 * etx + distance / 2.0e8),
            )

    return graph


def create_wsn(cfg: Config):
    n = cfg.n_sensor_nodes
    rng = np.random.default_rng(cfg.seed)
    positions = rng.uniform([0.0, 0.0], [FIELD_X, FIELD_Y], size=(n, 2))
    environment = rng.uniform(0.0, 1.0, size=n)
    initial_energy = rng.uniform(0.075, 0.125, size=n)

    source_target = np.array([0.0, 50.0])
    source = int(np.argmin(np.linalg.norm(positions - source_target, axis=1)))
    sink = n

    graph = None
    selected_range = None
    for communication_range in cfg.communication_ranges:
        candidate_graph = build_graph(positions, environment, communication_range)
        if nx.has_path(candidate_graph, source, sink):
            graph = candidate_graph
            selected_range = communication_range
            break

    if graph is None:
        raise RuntimeError("Could not create a connected WSN instance.")

    labels = np.full(n, "normal", dtype=object)
    eligible = np.array([i for i in range(n) if i != source])
    malicious = rng.choice(
        eligible,
        size=len(ATTACK_TYPES) * cfg.attackers_per_type,
        replace=False,
    )

    cursor = 0
    for attack in ATTACK_TYPES:
        labels[malicious[cursor : cursor + cfg.attackers_per_type]] = attack
        cursor += cfg.attackers_per_type

    return {
        "rng": rng,
        "positions": positions,
        "environment": environment,
        "initial_energy": initial_energy,
        "energy": initial_energy.copy(),
        "load": np.zeros(n, dtype=float),
        "source": source,
        "sink": sink,
        "graph": graph,
        "communication_range": selected_range,
        "attack_labels": labels,
    }


def node_natural_loss(graph, node):
    losses = [graph[node][nbr]["natural_loss"] for nbr in graph.neighbors(node)]
    return float(np.mean(losses)) if losses else 0.35


def create_telemetry(wsn, cfg: Config):
    """Watchdog/routing telemetry. Labels drive behaviour; risk uses telemetry only."""
    rng = wsn["rng"]
    graph = wsn["graph"]
    labels = wsn["attack_labels"]

    rows = []
    for node in range(cfg.n_sensor_nodes):
        natural_loss = node_natural_loss(graph, node)
        expected_forward = 1.0 - natural_loss

        forwarding_ratio = np.clip(expected_forward + rng.normal(0, 0.025), 0, 1)
        advert_anomaly = np.clip(rng.beta(1.2, 14.0), 0, 1)
        timing_anomaly = np.clip(rng.beta(1.2, 14.0), 0, 1)
        identity_similarity = np.clip(rng.beta(1.1, 16.0), 0, 1)

        label = labels[node]
        if label == "blackhole":
            forwarding_ratio = rng.uniform(0.00, 0.07)
        elif label == "selective_forwarding":
            forwarding_ratio = rng.uniform(0.35, 0.65)
        elif label == "sinkhole":
            advert_anomaly = rng.uniform(0.82, 1.00)
            forwarding_ratio = min(forwarding_ratio, rng.uniform(0.65, 0.88))
        elif label == "wormhole":
            timing_anomaly = rng.uniform(0.82, 1.00)
        elif label == "sybil":
            identity_similarity = rng.uniform(0.82, 1.00)

        rows.append(
            {
                "node": node,
                "natural_loss": natural_loss,
                "expected_forward": expected_forward,
                "forwarding_ratio": forwarding_ratio,
                "advert_anomaly": advert_anomaly,
                "timing_anomaly": timing_anomaly,
                "identity_similarity": identity_similarity,
            }
        )

    return pd.DataFrame(rows).set_index("node")


def estimate_five_attack_risk(telemetry):
    """Telemetry-only risk estimator with natural-loss correction."""
    expected = telemetry["expected_forward"].to_numpy()
    observed = telemetry["forwarding_ratio"].to_numpy()
    excess_drop = np.clip((expected - observed) / np.maximum(expected, 1e-6), 0, 1)

    blackhole = sigmoid(14.0 * (excess_drop - 0.72))
    selective = np.exp(-0.5 * ((excess_drop - 0.43) / 0.17) ** 2) * (1.0 - blackhole)

    component = pd.DataFrame(
        {
            "blackhole": blackhole,
            "selective_forwarding": selective,
            "sinkhole": telemetry["advert_anomaly"].to_numpy(),
            "wormhole": telemetry["timing_anomaly"].to_numpy(),
            "sybil": telemetry["identity_similarity"].to_numpy(),
        },
        index=telemetry.index,
    )

    combined = 1.0 - np.prod(1.0 - np.clip(component.to_numpy(), 0, 1), axis=1)
    component["combined_risk"] = np.clip(combined, 0, 1)
    component["trust"] = 1.0 - component["combined_risk"]
    return component


# ----------------------------------------------------------------------------
# 2. Classical candidate pruning
# ----------------------------------------------------------------------------


def adaptive_weights(wsn, risks):
    alive = wsn["energy"] > 0
    mean_energy_ratio = float(
        np.mean(wsn["energy"][alive] / np.maximum(wsn["initial_energy"][alive], 1e-12))
    )
    mean_risk = float(risks.loc[alive, "combined_risk"].mean())

    raw = {
        "energy": 0.30 + 0.35 * (1.0 - mean_energy_ratio),
        "risk": 0.35 + 0.40 * mean_risk,
        "load": 0.15,
        "quality": 0.20,
    }
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def calculate_node_costs(wsn, risks, cfg: Config):
    graph = wsn["graph"]
    energy = wsn["energy"]
    weights = adaptive_weights(wsn, risks)

    energy_penalty = 1.0 - np.clip(energy / np.max(wsn["initial_energy"]), 0, 1)
    load_penalty = minmax(wsn["load"])

    mean_etx = np.zeros(cfg.n_sensor_nodes)
    for node in range(cfg.n_sensor_nodes):
        incident = [graph[node][nbr]["etx"] for nbr in graph.neighbors(node)]
        mean_etx[node] = np.mean(incident) if incident else 10.0
    quality_penalty = minmax(mean_etx)

    node_cost = (
        weights["energy"] * energy_penalty
        + weights["risk"] * risks["combined_risk"].to_numpy()
        + weights["load"] * load_penalty
        + weights["quality"] * quality_penalty
    )
    node_cost[energy <= 0] = 10.0
    return node_cost, weights


def edge_routing_cost(graph, u, v, scale, disconnect_penalty=6.0):
    if not graph.has_edge(u, v):
        return disconnect_penalty
    edge = graph[u][v]
    return (
        0.50 * edge["tx_energy"] / max(scale["tx"], 1e-12)
        + 0.35 * edge["etx"] / max(scale["etx"], 1e-12)
        + 0.15 * edge["delay"] / max(scale["delay"], 1e-12)
    )


def cost_scale(graph):
    edge_values = list(graph.edges(data=True))
    return {
        "tx": max(data["tx_energy"] for _, _, data in edge_values),
        "etx": max(data["etx"] for _, _, data in edge_values),
        "delay": max(data["delay"] for _, _, data in edge_values),
    }


def select_layered_candidates(wsn, node_cost, cfg: Config):
    """Progress-based relay layers, seeded by a physically connected anchor route."""
    graph = wsn["graph"]
    source, sink = wsn["source"], wsn["sink"]
    positions = wsn["positions"]

    source_pos = positions[source]
    direction = SINK_POSITION - source_pos
    total_length = float(np.linalg.norm(direction))
    direction = direction / max(total_length, 1e-12)
    progress = ((positions - source_pos) @ direction) / max(total_length, 1e-12)

    pools = []
    for low, high in cfg.resolved_stage_windows():
        pool = [
            node
            for node in range(cfg.n_sensor_nodes)
            if node != source and wsn["energy"][node] > 0 and low <= progress[node] <= high
        ]
        pool = sorted(pool, key=lambda node: node_cost[node])[: cfg.pool_size]
        if len(pool) < cfg.candidates_per_stage:
            raise RuntimeError("A routing stage does not contain enough live nodes.")
        pools.append(pool)

    best_anchor, best_score = _find_anchor(graph, pools, node_cost, source, sink, cfg)

    if best_anchor is None:
        best_anchor = _fallback_anchor(graph, node_cost, source, sink, cfg)
        anchor_path = [source, *best_anchor, sink]
    else:
        anchor_path = [source, *best_anchor, sink]

    selected = []
    used = {source}
    target_progress = np.linspace(0.25, 0.75, cfg.n_stages)
    for stage, anchor in enumerate(best_anchor):
        anchor_position = positions[anchor]

        def ranking_score(node, anchor_position=anchor_position, stage=stage):
            spatial = (
                np.linalg.norm(positions[node] - anchor_position) / wsn["communication_range"]
            )
            stage_error = abs(progress[node] - target_progress[stage])
            return 0.60 * node_cost[node] + 0.25 * spatial + 0.15 * stage_error

        available = [node for node in pools[stage] if node not in used]
        ranked = sorted(available, key=ranking_score)
        candidates = [anchor] + [node for node in ranked if node != anchor]
        candidates = candidates[: cfg.candidates_per_stage]

        if len(candidates) != cfg.candidates_per_stage:
            raise RuntimeError(f"Could not select candidates for stage {stage + 1}.")

        selected.append(candidates)
        used.update(candidates)

    return selected, anchor_path, best_score


def _find_anchor(graph, pools, node_cost, source, sink, cfg: Config, forbidden=None):
    """Cheapest physically connected chain source -> stage_1 -> ... -> stage_s -> sink.

    Layered dynamic programme over (stage, node), which is O(s * |pool|^2) rather
    than the original nested loops that were exponential in the number of stages.
    Node distinctness is not expressible in the DP state, so a repeated relay is
    detected afterwards and forbidden on a retry.
    """
    forbidden = forbidden or set()

    best_cost = {}
    parent = {}
    for node in pools[0]:
        if (0, node) in forbidden or not graph.has_edge(source, node):
            continue
        best_cost[(0, node)] = node_cost[node] + graph[source][node]["etx"]
        parent[(0, node)] = None

    for stage in range(1, cfg.n_stages):
        for node in pools[stage]:
            if (stage, node) in forbidden:
                continue
            best, best_previous = float("inf"), None
            for previous in pools[stage - 1]:
                if (stage - 1, previous) not in best_cost or previous == node:
                    continue
                if not graph.has_edge(previous, node):
                    continue
                total = (
                    best_cost[(stage - 1, previous)]
                    + node_cost[node]
                    + graph[previous][node]["etx"]
                )
                if total < best:
                    best, best_previous = total, previous
            if best_previous is not None:
                best_cost[(stage, node)] = best
                parent[(stage, node)] = best_previous

    last = cfg.n_stages - 1
    best_total, best_final = float("inf"), None
    for node in pools[last]:
        if (last, node) not in best_cost or not graph.has_edge(node, sink):
            continue
        total = best_cost[(last, node)] + graph[node][sink]["etx"]
        if total < best_total:
            best_total, best_final = total, node

    if best_final is None:
        return None, float("inf")

    chain = [best_final]
    for stage in range(last, 0, -1):
        chain.append(parent[(stage, chain[-1])])
    chain.reverse()

    duplicates = [node for node in set(chain) if chain.count(node) > 1]
    if duplicates and len(forbidden) < 8:
        # Forbid the later occurrence of one repeated relay and retry.
        repeated = duplicates[0]
        latest_stage = max(stage for stage, node in enumerate(chain) if node == repeated)
        return _find_anchor(
            graph, pools, node_cost, source, sink, cfg, forbidden | {(latest_stage, repeated)}
        )

    return tuple(chain), best_total


def _fallback_anchor(graph, node_cost, source, sink, cfg: Config):
    for u, v, data in graph.edges(data=True):
        node_term = 0.0
        if u < cfg.n_sensor_nodes:
            node_term += node_cost[u]
        if v < cfg.n_sensor_nodes:
            node_term += node_cost[v]
        data["anchor_weight"] = data["etx"] + 0.5 * node_term

    anchor_path = None
    for path_index, path in enumerate(
        nx.shortest_simple_paths(graph, source, sink, weight="anchor_weight")
    ):
        if len(path) >= cfg.n_stages + 2:
            anchor_path = path
            break
        if path_index >= 100:
            break
    if anchor_path is None:
        raise RuntimeError("Could not construct a connected multi-relay route.")

    internal = anchor_path[1:-1]
    indices = np.linspace(0, len(internal) - 1, cfg.n_stages).round().astype(int)
    return tuple(internal[index] for index in indices)


# ----------------------------------------------------------------------------
# 3. QUBO / Ising
# ----------------------------------------------------------------------------


def build_routing_qubo(wsn, candidates, node_cost, cfg: Config):
    graph = wsn["graph"]
    source, sink = wsn["source"], wsn["sink"]
    scale = cost_scale(graph)

    linear = np.zeros(cfg.n_qubits, dtype=float)
    quadratic = {}

    def qindex(stage, candidate_index):
        return stage * cfg.candidates_per_stage + candidate_index

    for stage in range(cfg.n_stages):
        for local_index, node in enumerate(candidates[stage]):
            linear[qindex(stage, local_index)] += float(node_cost[node])

    for local_index, node in enumerate(candidates[0]):
        linear[qindex(0, local_index)] += edge_routing_cost(
            graph, source, node, scale, cfg.disconnect_penalty
        )

    for local_index, node in enumerate(candidates[-1]):
        linear[qindex(cfg.n_stages - 1, local_index)] += edge_routing_cost(
            graph, node, sink, scale, cfg.disconnect_penalty
        )

    for stage in range(cfg.n_stages - 1):
        for left_index, left_node in enumerate(candidates[stage]):
            for right_index, right_node in enumerate(candidates[stage + 1]):
                i = qindex(stage, left_index)
                j = qindex(stage + 1, right_index)
                quadratic[(min(i, j), max(i, j))] = edge_routing_cost(
                    graph, left_node, right_node, scale, cfg.disconnect_penalty
                )

    if cfg.interference_radius > 0.0 and cfg.interference_weight != 0.0:
        positions = wsn["positions"]
        for left_stage in range(cfg.n_stages):
            for right_stage in range(left_stage + 2, cfg.n_stages):
                for left_index, left_node in enumerate(candidates[left_stage]):
                    for right_index, right_node in enumerate(candidates[right_stage]):
                        separation = float(
                            np.linalg.norm(positions[left_node] - positions[right_node])
                        )
                        if separation >= cfg.interference_radius:
                            continue
                        # Interference grows as the relays get closer.
                        strength = cfg.interference_weight * (
                            1.0 - separation / cfg.interference_radius
                        )
                        i = qindex(left_stage, left_index)
                        j = qindex(right_stage, right_index)
                        key = (min(i, j), max(i, j))
                        quadratic[key] = quadratic.get(key, 0.0) + strength

    return linear, quadratic


def add_one_hot_penalty(linear, quadratic, cfg: Config, weight: float):
    """Standard penalty encoding: weight * sum_stage (sum_k x_k - 1)^2.

    Expands to  weight * (-x_k + 2 x_k x_l)  per block, plus a constant that is
    irrelevant for optimisation but tracked by the caller if needed.
    """
    linear = linear.copy()
    quadratic = dict(quadratic)
    for stage in range(cfg.n_stages):
        block = [stage * cfg.candidates_per_stage + k for k in range(cfg.candidates_per_stage)]
        for i in block:
            linear[i] -= weight
        for a in range(len(block)):
            for b in range(a + 1, len(block)):
                key = (block[a], block[b])
                quadratic[key] = quadratic.get(key, 0.0) + 2.0 * weight
    return linear, quadratic, weight * cfg.n_stages


def qubo_cost(bits, linear, quadratic):
    bits = np.asarray(bits, dtype=int)
    value = float(np.dot(linear, bits))
    for (i, j), coefficient in quadratic.items():
        value += coefficient * bits[i] * bits[j]
    return float(value)


def qubo_to_ising(linear, quadratic):
    """For x_i=(1-Z_i)/2, return constant, h_i and J_ij."""
    constant = 0.5 * float(np.sum(linear))
    h = -0.5 * np.asarray(linear, dtype=float)
    J = {}

    for (i, j), q in quadratic.items():
        constant += q / 4.0
        h[i] -= q / 4.0
        h[j] -= q / 4.0
        J[(i, j)] = q / 4.0

    return constant, h, J


def feasible_bits(choice_tuple, cfg: Config):
    bits = np.zeros(cfg.n_qubits, dtype=int)
    for stage, local_choice in enumerate(choice_tuple):
        bits[stage * cfg.candidates_per_stage + local_choice] = 1
    return bits


def exact_feasible_optimum(linear, quadratic, cfg: Config):
    records = []
    for choices in product(range(cfg.candidates_per_stage), repeat=cfg.n_stages):
        bits = feasible_bits(choices, cfg)
        records.append((qubo_cost(bits, linear, quadratic), choices, bits))
    return min(records, key=lambda row: row[0]), sorted(records, key=lambda row: row[0])


def all_basis_bits(n_qubits):
    """(2^n, n) int8 matrix of basis-state bits, little-endian (bits[:, q] = qubit q)."""
    index = np.arange(2**n_qubits, dtype=np.uint32)
    return ((index[:, None] >> np.arange(n_qubits)) & 1).astype(np.int8)


def all_basis_costs(linear, quadratic, cfg: Config):
    """Vectorised replacement for the original O(2^n) Python loop."""
    bits = all_basis_bits(cfg.n_qubits)
    costs = bits @ np.asarray(linear, dtype=float)
    for (i, j), coefficient in quadratic.items():
        costs += coefficient * (bits[:, i] * bits[:, j])

    blocks = bits.reshape(-1, cfg.n_stages, cfg.candidates_per_stage).sum(axis=2)
    feasible_mask = np.all(blocks == 1, axis=1)
    return costs, feasible_mask


def verify_qubo_ising_mapping(linear, quadratic, constant, h, J, cfg: Config):
    bits = all_basis_bits(cfg.n_qubits)
    z = 1 - 2 * bits.astype(float)
    qubo_values = bits @ np.asarray(linear, dtype=float)
    ising_values = np.full(bits.shape[0], constant) + z @ np.asarray(h, dtype=float)
    for (i, j), coefficient in quadratic.items():
        qubo_values += coefficient * (bits[:, i] * bits[:, j])
    for (i, j), coefficient in J.items():
        ising_values += coefficient * (z[:, i] * z[:, j])

    maximum_error = float(np.max(np.abs(qubo_values - ising_values)))
    if maximum_error > 1e-9:
        raise AssertionError(f"QUBO-to-Ising verification failed: {maximum_error}")
    return maximum_error


# ----------------------------------------------------------------------------
# 4. Circuits
# ----------------------------------------------------------------------------


def w_state_preparation_gate(k: int):
    """Uniform superposition over the k Hamming-weight-1 states of k qubits."""
    amplitudes = np.zeros(2**k, dtype=complex)
    amplitudes[[1 << q for q in range(k)]] = 1.0 / np.sqrt(k)
    return StatePreparation(amplitudes, label=f"W{k}")


def apply_cost_layer(circuit, gamma, h, J):
    """U_C(gamma)=exp(-i gamma H_C). The Ising constant is a global phase only."""
    for qubit, coefficient in enumerate(h):
        if abs(coefficient) > 1e-12:
            circuit.rz(2.0 * gamma * coefficient, qubit)

    for (i, j), coefficient in J.items():
        if abs(coefficient) > 1e-12:
            circuit.rzz(2.0 * gamma * coefficient, i, j)


def apply_xy_mixer_layer(circuit, beta, cfg: Config):
    """First-order Trotterisation of the ring XY mixer within each one-hot block.

    Each factor exp[-i beta/2 (X_i X_j + Y_i Y_j)] preserves the Hamming weight of
    its block, so the product preserves feasibility exactly even though it only
    approximates exp of the summed ring Hamiltonian.
    """
    k = cfg.candidates_per_stage
    for stage in range(cfg.n_stages):
        block = [stage * k + q for q in range(k)]
        ring_pairs = [(block[q], block[(q + 1) % k]) for q in range(k)] if k > 2 else [(block[0], block[1])]
        for i, j in ring_pairs:
            circuit.rxx(beta, i, j)
            circuit.ryy(beta, i, j)


def apply_x_mixer_layer(circuit, beta, cfg: Config):
    for qubit in range(cfg.n_qubits):
        circuit.rx(2.0 * beta, qubit)


def build_parameterized_qaoa(h, J, cfg: Config):
    gammas = ParameterVector("gamma", cfg.qaoa_depth)
    betas = ParameterVector("beta", cfg.qaoa_depth)
    circuit = QuantumCircuit(cfg.n_qubits, name=f"AgriTrust-{cfg.mixer.upper()}QAOA")

    if cfg.mixer == "xy":
        w_gate = w_state_preparation_gate(cfg.candidates_per_stage)
        for stage in range(cfg.n_stages):
            block = [stage * cfg.candidates_per_stage + q for q in range(cfg.candidates_per_stage)]
            circuit.append(w_gate, block)
    elif cfg.mixer == "x":
        circuit.h(range(cfg.n_qubits))
    else:
        raise ValueError(f"Unknown mixer: {cfg.mixer}")

    circuit.barrier()
    for layer in range(cfg.qaoa_depth):
        apply_cost_layer(circuit, gammas[layer], h, J)
        circuit.barrier()
        if cfg.mixer == "xy":
            apply_xy_mixer_layer(circuit, betas[layer], cfg)
        else:
            apply_x_mixer_layer(circuit, betas[layer], cfg)
        circuit.barrier()

    return circuit, gammas, betas


# ----------------------------------------------------------------------------
# 5. Variational optimisation
# ----------------------------------------------------------------------------


def cvar_of_distribution(probabilities, costs, alpha):
    """Conditional value at risk of the lower alpha tail of the cost distribution."""
    order = np.argsort(costs)
    sorted_costs = costs[order]
    sorted_probs = probabilities[order]
    cumulative = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cumulative, alpha, side="left"))
    cutoff = min(cutoff, len(sorted_costs) - 1)
    weights = sorted_probs[: cutoff + 1].copy()
    excess = cumulative[cutoff] - alpha
    if excess > 0:
        weights[cutoff] -= excess
    total = weights.sum()
    if total <= 0:
        return float(sorted_costs[0])
    return float(np.dot(weights, sorted_costs[: cutoff + 1]) / total)


def optimize_qaoa(ansatz, gammas, betas, basis_costs, feasible_mask, cfg: Config):
    state_backend = AerSimulator(method="statevector")

    state_circuit = ansatz.copy()
    state_circuit.save_statevector(label="statevector")
    compiled = transpile(state_circuit, state_backend, optimization_level=1)

    def parameter_map(parameters):
        parameters = np.asarray(parameters, dtype=float)
        mapping = {}
        for layer, parameter in enumerate(gammas):
            value = parameters[layer]
            mapping[parameter] = float(np.mod(value, 2.0 * np.pi)) if cfg.wrap_parameters else float(value)
        for layer, parameter in enumerate(betas):
            value = parameters[len(gammas) + layer]
            mapping[parameter] = float(np.mod(value, 2.0 * np.pi)) if cfg.wrap_parameters else float(value)
        return mapping

    def probabilities_for(parameters):
        bound = compiled.assign_parameters(parameter_map(parameters), inplace=False)
        result = state_backend.run(bound, seed_simulator=cfg.seed).result()
        statevector = np.asarray(result.data(0)["statevector"], dtype=complex)
        return np.abs(statevector) ** 2

    best = None
    all_traces = []
    for restart in range(cfg.n_restarts):
        rng = np.random.default_rng(cfg.seed + 1000 * restart)
        initial = np.concatenate(
            [
                rng.uniform(0.0, np.pi, size=len(gammas)),
                rng.uniform(0.0, np.pi / 2.0, size=len(betas)),
            ]
        )
        trace = []

        def objective(parameters):
            probabilities = probabilities_for(parameters)
            expectation = float(np.dot(probabilities, basis_costs))
            feasible_probability = float(np.sum(probabilities[feasible_mask]))
            if cfg.objective == "cvar":
                value = cvar_of_distribution(probabilities, basis_costs, cfg.cvar_alpha)
            else:
                value = expectation
            trace.append((value, expectation, feasible_probability))
            return value

        result = minimize(
            objective,
            x0=initial,
            method=cfg.optimizer,
            options={"maxiter": cfg.maxiter, "rhobeg": 0.7, "tol": 1e-5}
            if cfg.optimizer == "COBYLA"
            else {"maxiter": cfg.maxiter},
        )
        trace_frame = pd.DataFrame(
            trace, columns=["objective", "expectation", "feasible_probability"]
        ).assign(restart=restart)
        all_traces.append(trace_frame)

        if best is None or result.fun < best[0].fun:
            best = (result, parameter_map(result.x), probabilities_for(result.x))

    result, final_map, final_probabilities = best
    return result, final_map, final_probabilities, pd.concat(all_traces, ignore_index=True)


def sample_optimized_circuit(ansatz, parameter_values, cfg: Config, noise_model=None):
    backend = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()
    measured = ansatz.assign_parameters(parameter_values, inplace=False)
    measured.measure_all()
    compiled = transpile(measured, backend, optimization_level=2)
    counts = backend.run(compiled, shots=cfg.shots, seed_simulator=cfg.seed).result().get_counts()
    return counts, compiled


def bitstring_to_bits(bitstring):
    clean = bitstring.replace(" ", "")
    # Counts are reported q_{n-1}...q_0; reverse so array index == qubit index.
    return np.array([int(character) for character in clean[::-1]], dtype=int)


def decode_choices(bits, cfg: Config):
    choices = []
    for stage in range(cfg.n_stages):
        block = bits[stage * cfg.candidates_per_stage : (stage + 1) * cfg.candidates_per_stage]
        if block.sum() != 1:
            return None
        choices.append(int(np.argmax(block)))
    return tuple(choices)


def analyse_counts(counts, linear, quadratic, cfg: Config):
    feasible_rows = []
    total_shots = sum(counts.values())

    for bitstring, frequency in counts.items():
        bits = bitstring_to_bits(bitstring)
        choices = decode_choices(bits, cfg)
        if choices is None:
            continue
        feasible_rows.append(
            {
                "bitstring": bitstring,
                "shots": frequency,
                "probability": frequency / total_shots,
                "cost": qubo_cost(bits, linear, quadratic),
                "choices": choices,
            }
        )

    if not feasible_rows:
        return None, None, pd.DataFrame(columns=["bitstring", "shots", "probability", "cost", "choices"])

    mode = max(feasible_rows, key=lambda row: row["shots"])
    best_sampled = min(feasible_rows, key=lambda row: row["cost"])
    table = pd.DataFrame(feasible_rows).sort_values(["shots", "cost"], ascending=[False, True])
    return mode, best_sampled, table


# ----------------------------------------------------------------------------
# 6. Route evaluation
# ----------------------------------------------------------------------------


def choices_to_route(wsn, candidates, choices):
    relays = [candidates[stage][choice] for stage, choice in enumerate(choices)]
    return [wsn["source"], *relays, wsn["sink"]]


def route_is_connected(graph, route):
    return all(graph.has_edge(u, v) for u, v in zip(route[:-1], route[1:]))


def simulate_packets(wsn, route, cfg: Config, packets=None, rng=None, energy=None, load=None):
    """Packet-level evaluation.

    The original mutated wsn['energy'], wsn['load'] and consumed wsn['rng'], which
    makes any two-route comparison in one process unfair. Callers can now pass
    isolated copies; defaults preserve the original in-place behaviour.
    """
    packets = cfg.packets_to_simulate if packets is None else packets
    rng = wsn["rng"] if rng is None else rng
    energy = wsn["energy"] if energy is None else energy
    load = wsn["load"] if load is None else load

    graph = wsn["graph"]
    labels = wsn["attack_labels"]
    n = cfg.n_sensor_nodes

    delivered = 0
    natural_drops = 0
    malicious_drops = 0
    energy_drops = 0
    disconnect_drops = 0
    energy_before = float(np.sum(energy))

    for _ in range(packets):
        packet_ok = True
        for hop_index, (u, v) in enumerate(zip(route[:-1], route[1:])):
            if not graph.has_edge(u, v):
                disconnect_drops += 1
                packet_ok = False
                break

            edge = graph[u][v]
            tx = edge["tx_energy"]
            rx = radio_rx_energy()

            if u < n:
                if energy[u] < tx:
                    energy_drops += 1
                    packet_ok = False
                    break
                energy[u] -= tx

            if v < n:
                if energy[v] < rx:
                    energy[v] = 0.0
                    energy_drops += 1
                    packet_ok = False
                    break
                energy[v] -= rx

            if rng.random() < edge["natural_loss"]:
                natural_drops += 1
                packet_ok = False
                break

            if v < n and hop_index < len(route) - 2:
                if rng.random() < MALICIOUS_DROP_PROBABILITY[labels[v]]:
                    malicious_drops += 1
                    packet_ok = False
                    break

        if packet_ok:
            delivered += 1
            for relay in route[1:-1]:
                load[relay] += 1.0

    energy_after = float(np.sum(energy))
    return {
        "packets_sent": packets,
        "packets_delivered": delivered,
        "pdr": delivered / packets,
        "natural_drops": natural_drops,
        "malicious_drops": malicious_drops,
        "energy_drops": energy_drops,
        "disconnect_drops": disconnect_drops,
        "energy_consumed_j": energy_before - energy_after,
        "energy_per_delivered_packet_j": (
            (energy_before - energy_after) / delivered if delivered else np.inf
        ),
        "alive_nodes": int(np.sum(energy > 0)),
        "malicious_relays": int(sum(1 for r in route[1:-1] if labels[r] != "normal")),
    }


def classification_metrics(true_labels, risks, threshold=0.50):
    component = risks[ATTACK_TYPES]
    max_score = component.max(axis=1)
    predicted = component.idxmax(axis=1).astype(object)
    predicted[max_score < threshold] = "normal"

    rows = []
    for attack in ATTACK_TYPES:
        truth = np.asarray(true_labels == attack)
        pred = np.asarray(predicted == attack)
        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "attack": attack,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return pd.DataFrame(rows), predicted


# ----------------------------------------------------------------------------
# 7. Pipeline
# ----------------------------------------------------------------------------


@dataclass
class Instance:
    cfg: Config
    wsn: dict
    telemetry: pd.DataFrame
    risks: pd.DataFrame
    node_cost: np.ndarray
    weights: dict
    candidates: list
    anchor_path: list
    linear: np.ndarray
    quadratic: dict
    basis_costs: np.ndarray = field(default=None)
    feasible_mask: np.ndarray = field(default=None)


MAX_EAGER_FULL_SPACE_QUBITS = 20


def build_instance(cfg: Config, full_space: bool | None = None) -> Instance:
    """Build one routing instance.

    The full 2**n cost table is only needed by the penalty + X-mixer baseline. It
    costs 2**n floats (268 MB at 25 qubits) and a pass per quadratic term, so it is
    skipped above `MAX_EAGER_FULL_SPACE_QUBITS` unless explicitly requested.
    """
    wsn = create_wsn(cfg)
    telemetry = create_telemetry(wsn, cfg)
    risks = estimate_five_attack_risk(telemetry)
    node_cost, weights = calculate_node_costs(wsn, risks, cfg)
    candidates, anchor_path, _ = select_layered_candidates(wsn, node_cost, cfg)
    linear, quadratic = build_routing_qubo(wsn, candidates, node_cost, cfg)

    if full_space is None:
        full_space = cfg.n_qubits <= MAX_EAGER_FULL_SPACE_QUBITS
    if full_space:
        basis_costs, feasible_mask = all_basis_costs(linear, quadratic, cfg)
    else:
        basis_costs, feasible_mask = None, None

    return Instance(
        cfg=cfg,
        wsn=wsn,
        telemetry=telemetry,
        risks=risks,
        node_cost=node_cost,
        weights=weights,
        candidates=candidates,
        anchor_path=anchor_path,
        linear=linear,
        quadratic=quadratic,
        basis_costs=basis_costs,
        feasible_mask=feasible_mask,
    )
