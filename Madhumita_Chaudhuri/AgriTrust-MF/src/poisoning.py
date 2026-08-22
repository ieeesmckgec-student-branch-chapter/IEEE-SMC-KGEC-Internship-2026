"""Stage-B routing-metric poisoning experiments for AgriTrust.

The attacker changes only a copied routing view. Route quality is always evaluated
against the untouched physical graph, so a cheap advertisement cannot make a bad
or nonexistent radio hop cheap in the reported outcome.

Threat-model controls have separate meanings:

* ``intensity`` controls how strongly routing metrics are falsified and how often
  selected malicious relays drop packets.
* ``stealth`` controls how much attack evidence is available to the trust-aware
  formulation. It does not make the underlying attack less effective.

Sybil identities are logical graph nodes. ``logical_to_physical`` records their
physical owner, and all physical evaluation collapses aliases through that map.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Hashable, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from agritrust_core import (
    MALICIOUS_DROP_PROBABILITY,
    Config,
    build_instance,
    radio_rx_energy,
)
from baselines import layered_dp

SUPPORTED_ATTACKS = ("sinkhole", "wormhole", "sybil")
DEFAULT_SUMMARY_METRICS = (
    "true_route_cost",
    "regret",
    "pdr",
    "physical_connected",
    "malicious_relays",
    "malicious_relay_exposure",
    "tunnel_used",
    "natural_drops",
    "malicious_drops",
    "energy_drops",
    "disconnect_drops",
)


@dataclass
class PoisonedRoutingState:
    """Copied routing state plus the physical interpretation of logical nodes."""

    wsn: dict
    candidates: list[list[Hashable]]
    logical_to_physical: dict[Hashable, int]
    malicious_physical_nodes: frozenset[int]
    tunnel_edges: frozenset[frozenset[Hashable]]
    observed_risk: dict[Hashable, float]
    attack: str
    intensity: float
    stealth: float
    metadata: dict

    @property
    def graph(self) -> nx.Graph:
        return self.wsn["graph"]


def _validate_control(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _copy_wsn_state(clean_wsn: dict, seed: int) -> dict:
    copied = {}
    for key, value in clean_wsn.items():
        if key == "graph":
            copied[key] = copy.deepcopy(value)
        elif key == "rng":
            copied[key] = np.random.default_rng(seed)
        elif isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, pd.DataFrame):
            copied[key] = value.copy(deep=True)
        else:
            copied[key] = copy.deepcopy(value)
    return copied


def _unique_physical_candidates(candidates: Iterable[Iterable[Hashable]]) -> list[int]:
    unique = []
    for node in (node for layer in candidates for node in layer):
        if not isinstance(node, (int, np.integer)):
            continue
        physical = int(node)
        if physical not in unique:
            unique.append(physical)
    return unique


def _choose_nodes(
    candidates: list[list[Hashable]],
    rng: np.random.Generator,
    count: int,
    preferred_layer: int | None = None,
) -> tuple[int, ...]:
    if preferred_layer is None:
        eligible = _unique_physical_candidates(candidates)
    else:
        eligible = _unique_physical_candidates([candidates[preferred_layer]])
    if len(eligible) < count:
        raise ValueError(f"Need {count} distinct candidate actors, found {len(eligible)}")
    indices = rng.choice(len(eligible), size=count, replace=False)
    return tuple(eligible[int(index)] for index in np.atleast_1d(indices))


def _node_position(graph: nx.Graph, node: int) -> np.ndarray:
    position = graph.nodes[node].get("pos")
    if position is None:
        return np.array([float(node), 0.0])
    return np.asarray(position, dtype=float)


def _choose_wormhole_pair(
    graph: nx.Graph,
    candidates: list[list[Hashable]],
    rng: np.random.Generator,
) -> tuple[int, int]:
    pairs = []
    for stage in range(len(candidates) - 1):
        left_nodes = _unique_physical_candidates([candidates[stage]])
        right_nodes = _unique_physical_candidates([candidates[stage + 1]])
        for left in left_nodes:
            for right in right_nodes:
                if left == right:
                    continue
                distance = float(
                    np.linalg.norm(_node_position(graph, left) - _node_position(graph, right))
                )
                pairs.append((not graph.has_edge(left, right), distance, left, right))
    if not pairs:
        raise ValueError("Wormhole attack requires candidates in two adjacent stages")

    nonphysical = [pair for pair in pairs if pair[0]]
    ranked = sorted(nonphysical or pairs, key=lambda row: row[1], reverse=True)
    top_count = max(1, math.ceil(len(ranked) * 0.25))
    selected = ranked[int(rng.integers(top_count))]
    return int(selected[2]), int(selected[3])


def _resolve_actors(
    graph: nx.Graph,
    candidates: list[list[Hashable]],
    attack: str,
    intensity: float,
    rng: np.random.Generator,
    actor_nodes: Iterable[int] | None,
) -> tuple[int, ...]:
    if actor_nodes is not None:
        actors = tuple(int(node) for node in actor_nodes)
    elif attack == "wormhole":
        actors = _choose_wormhole_pair(graph, candidates, rng)
    elif attack == "sybil":
        actors = _choose_nodes(candidates, rng, count=1, preferred_layer=-1)
    else:
        count = max(1, min(2, math.ceil(2.0 * intensity)))
        actors = _choose_nodes(candidates, rng, count=count)

    expected_count = 2 if attack == "wormhole" else None
    if expected_count is not None and len(actors) != expected_count:
        raise ValueError("wormhole actor_nodes must contain exactly two colluding nodes")
    if not actors:
        raise ValueError(f"{attack} requires at least one actor node")

    physical_candidates = set(_unique_physical_candidates(candidates))
    invalid = [node for node in actors if node not in physical_candidates]
    if invalid:
        raise ValueError(f"Actor nodes must be routing candidates: {invalid}")
    if len(set(actors)) != len(actors):
        raise ValueError("Actor nodes must be distinct")
    return actors


def _poison_sinkhole(
    graph: nx.Graph,
    actors: tuple[int, ...],
    intensity: float,
) -> tuple[frozenset[frozenset[Hashable]], dict]:
    factor = round(max(0.05, 1.0 - 0.95 * intensity), 12)
    touched = set()
    for actor in actors:
        for neighbor in list(graph.neighbors(actor)):
            edge_key = frozenset((actor, neighbor))
            if edge_key in touched:
                continue
            touched.add(edge_key)
            edge = graph[actor][neighbor]
            edge["true_advertised_etx_before_poisoning"] = float(edge["etx"])
            edge["etx"] = float(edge["etx"]) * factor
            edge["poisoned_by"] = "sinkhole"
    return frozenset(), {
        "actor_nodes": list(actors),
        "advertised_etx_factor": factor,
        "poisoned_edge_count": len(touched),
    }


def _poison_wormhole(
    graph: nx.Graph,
    actors: tuple[int, ...],
    intensity: float,
) -> tuple[frozenset[frozenset[Hashable]], dict]:
    left, right = actors
    factor = round(max(0.05, 1.0 - 0.95 * intensity), 12)
    minimum_etx = min(float(data["etx"]) for _, _, data in graph.edges(data=True))
    physical_distance = float(
        np.linalg.norm(_node_position(graph, left) - _node_position(graph, right))
    )
    advertised_etx = minimum_etx * factor
    graph.add_edge(
        left,
        right,
        etx=advertised_etx,
        natural_loss=0.0,
        distance=physical_distance,
        tx_energy=0.0,
        delay=0.0001,
        poisoned_by="wormhole",
        poisoned_tunnel=True,
    )
    tunnel = frozenset((left, right))
    return frozenset((tunnel,)), {
        "actor_nodes": [left, right],
        "tunnel_pair": [left, right],
        "physical_tunnel_distance": physical_distance,
        "advertised_tunnel_etx": advertised_etx,
    }


def _poison_sybil(
    graph: nx.Graph,
    candidates: list[list[Hashable]],
    actor: int,
    intensity: float,
    max_sybil_identities: int,
    logical_to_physical: dict[Hashable, int],
) -> tuple[frozenset[frozenset[Hashable]], dict]:
    if max_sybil_identities < 1:
        raise ValueError("max_sybil_identities must be at least 1")
    count = max(1, math.ceil(max_sybil_identities * intensity))
    factor = round(max(0.05, 1.0 - 0.95 * intensity), 12)
    aliases = [f"sybil:{actor}:{index}" for index in range(count)]

    actor_attributes = copy.deepcopy(graph.nodes[actor])
    incident_edges = [
        (neighbor, copy.deepcopy(attributes))
        for neighbor, attributes in graph[actor].items()
    ]
    for alias in aliases:
        graph.add_node(
            alias,
            **actor_attributes,
            physical_node=actor,
            poisoned_by="sybil",
        )
        logical_to_physical[alias] = actor
        for neighbor, attributes in incident_edges:
            attributes = copy.deepcopy(attributes)
            attributes["true_advertised_etx_before_poisoning"] = float(attributes["etx"])
            attributes["etx"] = float(attributes["etx"]) * factor
            attributes["poisoned_by"] = "sybil"
            graph.add_edge(alias, neighbor, **attributes)

    actor_stage = next(
        stage for stage, layer in enumerate(candidates) if actor in layer
    )
    actor_index = candidates[actor_stage].index(actor)
    replacement_indices = [actor_index] + [
        index for index in range(len(candidates[actor_stage])) if index != actor_index
    ]
    for alias, index in zip(aliases, replacement_indices):
        candidates[actor_stage][index] = alias

    return frozenset(), {
        "actor_nodes": [actor],
        "sybil_identities": aliases,
        "physical_identity": actor,
        "advertised_etx_factor": factor,
    }


def poison_routing_inputs(
    clean_wsn: dict,
    candidates: list[list[Hashable]],
    attack: str,
    intensity: float,
    stealth: float = 0.0,
    seed: int = 0,
    actor_nodes: Iterable[int] | None = None,
    max_sybil_identities: int = 3,
) -> PoisonedRoutingState:
    """Create an isolated poisoned routing view without mutating clean inputs."""

    if attack not in SUPPORTED_ATTACKS:
        raise ValueError(f"attack must be one of {SUPPORTED_ATTACKS}, got {attack!r}")
    intensity = _validate_control("intensity", intensity)
    stealth = _validate_control("stealth", stealth)
    rng = np.random.default_rng(seed)
    poisoned_wsn = _copy_wsn_state(clean_wsn, seed)
    poisoned_candidates = [list(layer) for layer in candidates]
    graph = poisoned_wsn["graph"]
    logical_to_physical = {
        node: int(node)
        for node in graph.nodes
        if isinstance(node, (int, np.integer))
    }
    actors = _resolve_actors(
        graph,
        poisoned_candidates,
        attack,
        intensity,
        rng,
        actor_nodes,
    )

    if intensity == 0.0:
        tunnel_edges = frozenset()
        metadata = {
            "actor_nodes": list(actors),
            "advertised_etx_factor": 1.0,
            "poisoned_edge_count": 0,
        }
        if attack == "wormhole":
            metadata.update(
                {
                    "tunnel_pair": list(actors),
                    "tunnel_injected": False,
                }
            )
        elif attack == "sybil":
            metadata.update(
                {
                    "sybil_identities": [],
                    "physical_identity": actors[0],
                }
            )
    elif attack == "sinkhole":
        tunnel_edges, metadata = _poison_sinkhole(graph, actors, intensity)
    elif attack == "wormhole":
        tunnel_edges, metadata = _poison_wormhole(graph, actors, intensity)
    else:
        tunnel_edges, metadata = _poison_sybil(
            graph,
            poisoned_candidates,
            actors[0],
            intensity,
            max_sybil_identities,
            logical_to_physical,
        )

    evidence = float(intensity * (1.0 - stealth))
    observed_risk = {
        logical: evidence
        for logical, physical in logical_to_physical.items()
        if physical in actors
    }
    metadata = {
        **metadata,
        "attack": attack,
        "intensity": intensity,
        "stealth": stealth,
        "observed_attack_evidence": evidence,
        "selection_seed": int(seed),
    }
    return PoisonedRoutingState(
        wsn=poisoned_wsn,
        candidates=poisoned_candidates,
        logical_to_physical=logical_to_physical,
        malicious_physical_nodes=frozenset(actors),
        tunnel_edges=tunnel_edges,
        observed_risk=observed_risk,
        attack=attack,
        intensity=intensity,
        stealth=stealth,
        metadata=metadata,
    )


def _edge_selection_cost(
    graph: nx.Graph,
    left: Hashable,
    right: Hashable,
    missing_penalty: float,
) -> float:
    if not graph.has_edge(left, right):
        return float(missing_penalty)
    return float(graph[left][right]["etx"])


def build_layered_metric_qubo(
    graph: nx.Graph,
    candidates: list[list[Hashable]],
    source: Hashable,
    sink: Hashable,
    node_penalties: dict[Hashable, float] | None = None,
    missing_penalty: float = 100.0,
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Build the consecutive-stage metric objective consumed by exact DP."""

    if not candidates or not all(candidates):
        raise ValueError("candidates must contain at least one nonempty stage")
    width = len(candidates[0])
    if any(len(layer) != width for layer in candidates):
        raise ValueError("all candidate stages must have equal width")
    node_penalties = node_penalties or {}
    n_stages = len(candidates)
    linear = np.zeros(n_stages * width, dtype=float)
    quadratic = {}

    for stage, layer in enumerate(candidates):
        for local_index, node in enumerate(layer):
            index = stage * width + local_index
            linear[index] += float(node_penalties.get(node, 0.0))
            if stage == 0:
                linear[index] += _edge_selection_cost(
                    graph, source, node, missing_penalty
                )
            if stage == n_stages - 1:
                linear[index] += _edge_selection_cost(
                    graph, node, sink, missing_penalty
                )

    for stage in range(n_stages - 1):
        for left_index, left in enumerate(candidates[stage]):
            for right_index, right in enumerate(candidates[stage + 1]):
                i = stage * width + left_index
                j = (stage + 1) * width + right_index
                quadratic[(i, j)] = _edge_selection_cost(
                    graph, left, right, missing_penalty
                )
    return linear, quadratic


def _solve_layered_route(
    graph: nx.Graph,
    candidates: list[list[Hashable]],
    source: Hashable,
    sink: Hashable,
    node_penalties: dict[Hashable, float] | None = None,
) -> tuple[float, tuple[int, ...], list[Hashable]]:
    linear, quadratic = build_layered_metric_qubo(
        graph,
        candidates,
        source,
        sink,
        node_penalties=node_penalties,
    )
    selection_cost, choices = layered_dp(
        linear,
        quadratic,
        n_stages=len(candidates),
        candidates_per_stage=len(candidates[0]),
    )
    route = [source] + [
        candidates[stage][choice] for stage, choice in enumerate(choices)
    ] + [sink]
    return selection_cost, choices, route


def _physical_route(
    route: list[Hashable],
    logical_to_physical: dict[Hashable, int],
) -> list[int]:
    physical = []
    for node in route:
        if node in logical_to_physical:
            physical.append(logical_to_physical[node])
        else:
            physical.append(int(node))
    return physical


def _true_route_cost(
    clean_graph: nx.Graph,
    route: list[int],
    missing_penalty: float | None = None,
) -> float:
    maximum_etx = max(float(data["etx"]) for _, _, data in clean_graph.edges(data=True))
    missing_penalty = 10.0 * maximum_etx if missing_penalty is None else missing_penalty
    total = 0.0
    for left, right in zip(route[:-1], route[1:]):
        if clean_graph.has_edge(left, right):
            total += float(clean_graph[left][right]["etx"])
        else:
            total += float(missing_penalty)
    return total


def _simulate_true_packets(
    clean_wsn: dict,
    route: list[int],
    malicious_nodes: frozenset[int],
    attack: str,
    intensity: float,
    packets: int,
    seed: int,
) -> dict:
    graph = clean_wsn["graph"]
    physical_connected = all(
        graph.has_edge(left, right) for left, right in zip(route[:-1], route[1:])
    )
    if not physical_connected:
        return {
            "packets_sent": packets,
            "packets_delivered": 0,
            "pdr": 0.0,
            "natural_drops": 0,
            "malicious_drops": 0,
            "energy_drops": 0,
            "disconnect_drops": packets,
            "energy_consumed_j": 0.0,
        }

    rng = np.random.default_rng(seed)
    energy = np.asarray(clean_wsn["energy"], dtype=float).copy()
    energy_before = float(energy.sum())
    n_sensor_nodes = len(energy)
    malicious_drop_probability = MALICIOUS_DROP_PROBABILITY[attack] * intensity
    delivered = 0
    natural_drops = 0
    malicious_drops = 0
    energy_drops = 0

    for _ in range(packets):
        packet_ok = True
        for hop_index, (left, right) in enumerate(zip(route[:-1], route[1:])):
            edge = graph[left][right]
            tx_energy = float(edge.get("tx_energy", 0.0))
            rx_energy = float(radio_rx_energy())

            if left < n_sensor_nodes:
                if energy[left] < tx_energy:
                    energy_drops += 1
                    packet_ok = False
                    break
                energy[left] -= tx_energy
            if right < n_sensor_nodes:
                if energy[right] < rx_energy:
                    energy[right] = 0.0
                    energy_drops += 1
                    packet_ok = False
                    break
                energy[right] -= rx_energy

            if rng.random() < float(edge.get("natural_loss", 0.0)):
                natural_drops += 1
                packet_ok = False
                break
            if (
                right in malicious_nodes
                and hop_index < len(route) - 2
                and rng.random() < malicious_drop_probability
            ):
                malicious_drops += 1
                packet_ok = False
                break
        if packet_ok:
            delivered += 1

    return {
        "packets_sent": packets,
        "packets_delivered": delivered,
        "pdr": delivered / packets if packets else float("nan"),
        "natural_drops": natural_drops,
        "malicious_drops": malicious_drops,
        "energy_drops": energy_drops,
        "disconnect_drops": 0,
        "energy_consumed_j": energy_before - float(energy.sum()),
    }


def compare_routing_formulations(
    clean_wsn: dict,
    clean_candidates: list[list[Hashable]],
    poisoned: PoisonedRoutingState,
    packets: int = 300,
    packet_seed: int = 0,
    trust_weight: float = 12.0,
) -> pd.DataFrame:
    """Solve all three formulations exactly and evaluate on clean physics."""

    source = clean_wsn["source"]
    sink = clean_wsn["sink"]
    robust_penalties = {
        node: trust_weight * risk for node, risk in poisoned.observed_risk.items()
    }
    formulations = (
        (
            "naive_poisoned",
            poisoned.graph,
            poisoned.candidates,
            poisoned.logical_to_physical,
            None,
        ),
        (
            "trust_weighted",
            poisoned.graph,
            poisoned.candidates,
            poisoned.logical_to_physical,
            robust_penalties,
        ),
        (
            "clean_oracle",
            clean_wsn["graph"],
            [list(layer) for layer in clean_candidates],
            {node: int(node) for node in clean_wsn["graph"].nodes},
            None,
        ),
    )

    solved = []
    for name, graph, candidates, mapping, penalties in formulations:
        selection_cost, choices, logical_route = _solve_layered_route(
            graph,
            candidates,
            source,
            sink,
            node_penalties=penalties,
        )
        physical_route = _physical_route(logical_route, mapping)
        true_cost = _true_route_cost(clean_wsn["graph"], physical_route)
        tunnel_used = name != "clean_oracle" and any(
            frozenset((left, right)) in poisoned.tunnel_edges
            for left, right in zip(logical_route[:-1], logical_route[1:])
        )
        packet_metrics = _simulate_true_packets(
            clean_wsn,
            physical_route,
            poisoned.malicious_physical_nodes,
            poisoned.attack,
            poisoned.intensity,
            packets,
            packet_seed,
        )
        relay_nodes = physical_route[1:-1]
        malicious_relays = sum(
            node in poisoned.malicious_physical_nodes for node in relay_nodes
        )
        sybil_identities = set(poisoned.metadata.get("sybil_identities", ()))
        solved.append(
            {
                "formulation": name,
                "selection_cost": selection_cost,
                "choices": json.dumps(choices),
                "logical_route": json.dumps(logical_route),
                "physical_route": json.dumps(physical_route),
                "selection_connected": all(
                    graph.has_edge(left, right)
                    for left, right in zip(logical_route[:-1], logical_route[1:])
                ),
                "physical_connected": all(
                    clean_wsn["graph"].has_edge(left, right)
                    for left, right in zip(physical_route[:-1], physical_route[1:])
                ),
                "true_route_cost": true_cost,
                "malicious_relays": int(malicious_relays),
                "malicious_relay_exposure": float(
                    malicious_relays / max(len(relay_nodes), 1)
                ),
                "selected_malicious_relay": bool(malicious_relays),
                "selected_sybil_identities": sum(
                    node in sybil_identities for node in logical_route[1:-1]
                ),
                "tunnel_used": bool(tunnel_used),
                **packet_metrics,
            }
        )

    oracle_cost = next(
        row["true_route_cost"]
        for row in solved
        if row["formulation"] == "clean_oracle"
    )
    for row in solved:
        row["regret"] = float(row["true_route_cost"] - oracle_cost)
    return pd.DataFrame(solved)


def run_poisoning_sweep(
    seeds: Iterable[int],
    attacks: Iterable[str],
    intensities: Iterable[float],
    stealth_levels: Iterable[float] = (0.0,),
    packets: int = 300,
    n_sensor_nodes: int = 150,
    trust_weight: float = 12.0,
    max_sybil_identities: int = 3,
) -> pd.DataFrame:
    """Build each clean WSN once, then run every attack setting on that instance."""

    rows = []
    for seed in seeds:
        cfg = Config(
            seed=int(seed),
            n_sensor_nodes=n_sensor_nodes,
            packets_to_simulate=packets,
        )
        instance = build_instance(cfg, full_space=False)
        for attack in attacks:
            for intensity in intensities:
                for stealth in stealth_levels:
                    attack_seed = (
                        int(seed) * 1_000_003
                        + SUPPORTED_ATTACKS.index(attack) * 10_007
                        + round(float(intensity) * 1_000)
                        + round(float(stealth) * 100)
                    )
                    poisoned = poison_routing_inputs(
                        instance.wsn,
                        instance.candidates,
                        attack=attack,
                        intensity=float(intensity),
                        stealth=float(stealth),
                        seed=attack_seed,
                        max_sybil_identities=max_sybil_identities,
                    )
                    comparison = compare_routing_formulations(
                        instance.wsn,
                        instance.candidates,
                        poisoned,
                        packets=packets,
                        packet_seed=int(seed) + 50_000,
                        trust_weight=trust_weight,
                    )
                    for record in comparison.to_dict("records"):
                        rows.append(
                            {
                                "seed": int(seed),
                                "attack": attack,
                                "intensity": float(intensity),
                                "stealth": float(stealth),
                                "actor_nodes": json.dumps(
                                    poisoned.metadata["actor_nodes"]
                                ),
                                "attack_seed": attack_seed,
                                **record,
                            }
                        )
    return pd.DataFrame(rows)


def bootstrap_summary(
    raw: pd.DataFrame,
    metrics: Iterable[str] = DEFAULT_SUMMARY_METRICS,
    resamples: int = 5000,
    seed: int = 20260805,
) -> pd.DataFrame:
    """Return deterministic seed-bootstrap confidence intervals in long form."""

    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    group_columns = ["attack", "intensity", "stealth", "formulation"]
    missing = [column for column in group_columns if column not in raw]
    if missing:
        raise ValueError(f"raw results are missing grouping columns: {missing}")

    rng = np.random.default_rng(seed)
    rows = []
    for key, group in raw.groupby(group_columns, sort=True):
        base = dict(zip(group_columns, key))
        for metric in metrics:
            if metric not in group:
                raise ValueError(f"raw results are missing metric {metric!r}")
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                samples = rng.choice(
                    values,
                    size=(resamples, len(values)),
                    replace=True,
                ).mean(axis=1)
                mean = float(values.mean())
                low = float(np.quantile(samples, 0.025))
                high = float(np.quantile(samples, 0.975))
            else:
                mean = low = high = float("nan")
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_seeds": int(group["seed"].nunique())
                    if "seed" in group
                    else len(values),
                }
            )
    return pd.DataFrame(rows)
