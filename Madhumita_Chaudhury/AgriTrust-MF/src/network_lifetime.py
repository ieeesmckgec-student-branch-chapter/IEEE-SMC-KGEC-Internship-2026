"""Multi-round wireless-sensor-network lifetime evaluation.

Each routing method receives its own energy/load arrays and a newly constructed
packet RNG with the same seed. Energy and load persist between that method's
rounds, but no state is shared between methods.

Network lifetime is the first round after whose packet batch the source and sink
are disconnected in the graph induced by nodes with positive residual energy.
If that never occurs, the lifetime is right-censored at ``rounds + 1``.

The trust-aware shortest-path cost for an edge ``u -> v`` is

    0.60 * ETX(u, v) + 0.25 * risk(v) + 0.15 * energy_pressure(v)

where ``energy_pressure(v) = 1 - residual(v) / initial(v)``. The sink has zero
risk and energy pressure. Dead nodes are isolated in a temporary graph.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from agritrust_core import (
    Config,
    build_routing_qubo,
    calculate_node_costs,
    choices_to_route,
    route_is_connected,
    select_layered_candidates,
    simulate_packets,
)
from baselines import assert_layered, layered_dp

ROUTING_METHODS = (
    "static_layered",
    "adaptive_layered_dp",
    "etx_shortest_path",
    "trust_aware_shortest_path",
)

TRUST_ETX_COEFFICIENT = 0.60
TRUST_RISK_COEFFICIENT = 0.25
TRUST_ENERGY_COEFFICIENT = 0.15

DROP_COLUMNS = (
    "natural_drops",
    "malicious_drops",
    "energy_drops",
    "disconnect_drops",
)

SUMMARY_METRICS = (
    "pdr",
    "network_lifetime_round",
    "first_relay_death_round",
    "first_route_failure_round",
    "first_partition_round",
    "cumulative_energy_j",
    "route_changes",
    "malicious_relay_rounds",
)


def build_alive_graph(wsn, energy):
    """Return an isolated graph copy with dead sensor nodes disconnected."""
    energy = np.asarray(energy, dtype=float)
    if energy.shape != np.asarray(wsn["initial_energy"]).shape:
        raise ValueError("Energy must contain one value per sensor node.")

    graph = wsn["graph"].copy()
    dead_nodes = [node for node, remaining in enumerate(energy) if remaining <= 0.0]
    edges_to_remove = []
    for node in dead_nodes:
        edges_to_remove.extend(list(graph.edges(node)))
    graph.remove_edges_from(edges_to_remove)
    return graph


def _path_exists(graph, source, sink):
    return graph.has_node(source) and graph.has_node(sink) and nx.has_path(graph, source, sink)


def _routing_state(wsn, energy, load):
    state = dict(wsn)
    state["energy"] = energy
    state["load"] = load
    state["graph"] = build_alive_graph(wsn, energy)
    return state


def layered_exact_dp_path(wsn, risks, cfg, energy, load):
    """Select live layered candidates and solve their QUBO exactly by DP."""
    state = _routing_state(wsn, energy, load)
    source, sink = state["source"], state["sink"]
    if energy[source] <= 0.0 or not _path_exists(state["graph"], source, sink):
        return None

    try:
        node_cost, _ = calculate_node_costs(state, risks, cfg)
        candidates, _, _ = select_layered_candidates(state, node_cost, cfg)
        linear, quadratic = build_routing_qubo(state, candidates, node_cost, cfg)
    except (KeyError, nx.NetworkXException, RuntimeError, ValueError):
        return None

    if not assert_layered(quadratic, cfg.n_stages, cfg.candidates_per_stage):
        raise ValueError("Adaptive layered DP requires a strictly layered QUBO.")

    _, choices = layered_dp(
        linear,
        quadratic,
        cfg.n_stages,
        cfg.candidates_per_stage,
    )
    route = choices_to_route(state, candidates, choices)
    return route if route_is_connected(state["graph"], route) else None


def etx_shortest_path(wsn, energy):
    """Return the minimum-ETX source-to-sink path through live nodes."""
    graph = build_alive_graph(wsn, energy)
    source, sink = wsn["source"], wsn["sink"]
    if energy[source] <= 0.0 or not _path_exists(graph, source, sink):
        return None
    return nx.shortest_path(graph, source, sink, weight="etx")


def trust_aware_shortest_path(
    wsn,
    risks,
    energy,
    etx_coefficient=TRUST_ETX_COEFFICIENT,
    risk_coefficient=TRUST_RISK_COEFFICIENT,
    energy_coefficient=TRUST_ENERGY_COEFFICIENT,
):
    """Return a live-node path balancing ETX, destination risk, and energy."""
    coefficients = np.array(
        [etx_coefficient, risk_coefficient, energy_coefficient],
        dtype=float,
    )
    if np.any(coefficients < 0.0) or not np.isclose(coefficients.sum(), 1.0):
        raise ValueError("Trust-aware coefficients must be nonnegative and sum to one.")

    graph = build_alive_graph(wsn, energy)
    source, sink = wsn["source"], wsn["sink"]
    if energy[source] <= 0.0 or not _path_exists(graph, source, sink):
        return None

    initial_energy = np.asarray(wsn["initial_energy"], dtype=float)
    residual_energy = np.asarray(energy, dtype=float)
    n_sensor_nodes = len(initial_energy)

    def edge_weight(_u, v, data):
        if v == sink:
            destination_risk = 0.0
            energy_pressure = 0.0
        elif v < n_sensor_nodes:
            destination_risk = float(risks.loc[v, "combined_risk"])
            ratio = residual_energy[v] / max(initial_energy[v], 1e-12)
            energy_pressure = 1.0 - float(np.clip(ratio, 0.0, 1.0))
        else:
            destination_risk = 0.0
            energy_pressure = 0.0
        return (
            etx_coefficient * float(data["etx"])
            + risk_coefficient * destination_risk
            + energy_coefficient * energy_pressure
        )

    return nx.shortest_path(graph, source, sink, weight=edge_weight)


def _plan_route(method, wsn, risks, cfg, energy, load):
    if method in {"static_layered", "adaptive_layered_dp"}:
        return layered_exact_dp_path(wsn, risks, cfg, energy, load)
    if method == "etx_shortest_path":
        return etx_shortest_path(wsn, energy)
    if method == "trust_aware_shortest_path":
        return trust_aware_shortest_path(wsn, risks, energy)
    raise ValueError(f"Unknown routing method: {method}")


def _disconnected_packet_stats(packets, energy):
    return {
        "packets_sent": packets,
        "packets_delivered": 0,
        "pdr": 0.0,
        "natural_drops": 0,
        "malicious_drops": 0,
        "energy_drops": 0,
        "disconnect_drops": packets,
        "energy_consumed_j": 0.0,
        "energy_per_delivered_packet_j": np.inf,
        "alive_nodes": int(np.sum(energy > 0.0)),
        "malicious_relays": 0,
    }


def _event_value(event_round, rounds):
    return event_round if event_round is not None else rounds + 1


def simulate_lifetime_method(
    wsn,
    cfg: Config,
    risks,
    method,
    rounds,
    packets_per_round,
    packet_seed,
    initial_route=None,
):
    """Run one method with persistent private state over multiple rounds.

    Event rounds use one-based indexing. Unobserved events are represented as
    ``rounds + 1`` and accompanied by a censoring flag.
    """
    if method not in ROUTING_METHODS:
        raise ValueError(f"Unknown routing method: {method}")
    if rounds <= 0 or packets_per_round <= 0:
        raise ValueError("Rounds and packets per round must both be positive.")
    if initial_route is not None and method != "static_layered":
        raise ValueError("An initial route can only be supplied to the static method.")

    energy = np.asarray(wsn["initial_energy"], dtype=float).copy()
    load = np.zeros_like(np.asarray(wsn["load"], dtype=float))
    packet_rng = np.random.default_rng(packet_seed)
    labels = np.asarray(wsn["attack_labels"])
    source, sink = wsn["source"], wsn["sink"]

    static_route = (
        list(initial_route)
        if initial_route is not None
        else _plan_route("static_layered", wsn, risks, cfg, energy, load)
    )
    previous_route = None
    route_changes = 0
    first_relay_death = None
    first_route_failure = None
    first_partition = None
    cumulative_energy = 0.0
    cumulative_sent = 0
    cumulative_delivered = 0
    cumulative_drops = {column: 0 for column in DROP_COLUMNS}
    unique_malicious_relays = set()
    malicious_relay_rounds = 0
    rows = []

    simulation_state = dict(wsn)
    simulation_state["energy"] = energy
    simulation_state["load"] = load

    for round_number in range(1, rounds + 1):
        alive_before = energy > 0.0
        alive_graph = build_alive_graph(wsn, energy)
        partitioned_before = not _path_exists(alive_graph, source, sink)
        if partitioned_before and first_partition is None:
            first_partition = round_number

        if method == "static_layered":
            route = static_route
        else:
            route = _plan_route(method, wsn, risks, cfg, energy, load)

        route_available = (
            route is not None
            and energy[source] > 0.0
            and all(energy[node] > 0.0 for node in route[:-1] if node < len(energy))
            and route_is_connected(alive_graph, route)
        )
        if not route_available and first_route_failure is None:
            first_route_failure = round_number

        route_tuple = tuple(route) if route_available else None
        route_changed = (
            previous_route is not None
            and route_tuple is not None
            and route_tuple != previous_route
        )
        if route_changed:
            route_changes += 1
        if route_tuple is not None:
            previous_route = route_tuple

        if route_available:
            relays = route[1:-1]
            malicious_relays = [node for node in relays if labels[node] != "normal"]
            unique_malicious_relays.update(malicious_relays)
            malicious_relay_rounds += len(malicious_relays)
            stats = simulate_packets(
                simulation_state,
                route,
                cfg,
                packets=packets_per_round,
                rng=packet_rng,
                energy=energy,
                load=load,
            )
        else:
            relays = []
            malicious_relays = []
            stats = _disconnected_packet_stats(packets_per_round, energy)

        newly_dead_relays = [
            node
            for node in relays
            if node < len(energy) and alive_before[node] and energy[node] <= 0.0
        ]
        if newly_dead_relays and first_relay_death is None:
            first_relay_death = round_number

        partitioned_after = not _path_exists(build_alive_graph(wsn, energy), source, sink)
        if partitioned_after and first_partition is None:
            first_partition = round_number

        cumulative_sent += int(stats["packets_sent"])
        cumulative_delivered += int(stats["packets_delivered"])
        cumulative_energy += float(stats["energy_consumed_j"])
        for column in DROP_COLUMNS:
            cumulative_drops[column] += int(stats[column])

        rows.append(
            {
                "seed": cfg.seed,
                "method": method,
                "round": round_number,
                "route": "->".join(str(node) for node in route) if route_available else "",
                "route_available": route_available,
                "route_changed": route_changed,
                "route_changes": route_changes,
                "malicious_relays": ",".join(str(node) for node in malicious_relays),
                "malicious_relay_count": len(malicious_relays),
                **stats,
                "cumulative_packets_sent": cumulative_sent,
                "cumulative_packets_delivered": cumulative_delivered,
                "cumulative_pdr": cumulative_delivered / cumulative_sent,
                "cumulative_energy_j": cumulative_energy,
                **{
                    f"cumulative_{column}": cumulative_drops[column]
                    for column in DROP_COLUMNS
                },
            }
        )

    first_relay_death_value = _event_value(first_relay_death, rounds)
    first_route_failure_value = _event_value(first_route_failure, rounds)
    first_partition_value = _event_value(first_partition, rounds)
    lifetime_round = first_partition_value
    event_columns = {
        "first_relay_death_round": first_relay_death_value,
        "relay_death_censored": first_relay_death is None,
        "first_route_failure_round": first_route_failure_value,
        "route_failure_censored": first_route_failure is None,
        "first_partition_round": first_partition_value,
        "partition_censored": first_partition is None,
        "network_lifetime_round": lifetime_round,
        "lifetime_censored": first_partition is None,
    }
    for row in rows:
        row.update(event_columns)

    final_alive = int(np.sum(energy > 0.0))
    aggregate = {
        "seed": cfg.seed,
        "method": method,
        "rounds": rounds,
        "packets_per_round": packets_per_round,
        "packets_sent": cumulative_sent,
        "packets_delivered": cumulative_delivered,
        "pdr": cumulative_delivered / cumulative_sent,
        **cumulative_drops,
        "cumulative_energy_j": cumulative_energy,
        "energy_per_delivered_packet_j": (
            cumulative_energy / cumulative_delivered
            if cumulative_delivered
            else np.inf
        ),
        "alive_nodes": final_alive,
        "alive_fraction": final_alive / len(energy),
        "route_changes": route_changes,
        "malicious_relay_rounds": malicious_relay_rounds,
        "unique_malicious_relay_count": len(unique_malicious_relays),
        "unique_malicious_relays": ",".join(
            str(node) for node in sorted(unique_malicious_relays)
        ),
        **event_columns,
    }
    return pd.DataFrame(rows), aggregate


def evaluate_lifetimes(
    wsn,
    cfg: Config,
    risks,
    rounds,
    packets_per_round,
    packet_seed=None,
    methods=ROUTING_METHODS,
):
    """Evaluate methods with equally seeded RNGs and independent mutable state."""
    packet_seed = cfg.seed * 1_000_003 + 17 if packet_seed is None else packet_seed
    round_frames = []
    aggregates = []
    for method in methods:
        frame, aggregate = simulate_lifetime_method(
            wsn,
            cfg,
            risks,
            method=method,
            rounds=rounds,
            packets_per_round=packets_per_round,
            packet_seed=packet_seed,
        )
        round_frames.append(frame)
        aggregates.append(aggregate)
    return pd.concat(round_frames, ignore_index=True), pd.DataFrame(aggregates)


def _bootstrap_mean_interval(values, rng, resamples):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) == 1 or resamples == 0:
        return mean, mean, mean
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return mean, min(float(low), mean), max(float(high), mean)


def bootstrap_summary(raw, metrics=SUMMARY_METRICS, resamples=5000, seed=2025):
    """Summarize seed-level outcomes with percentile bootstrap 95% CIs."""
    if resamples < 0:
        raise ValueError("Bootstrap resamples cannot be negative.")
    if "method" not in raw:
        raise ValueError("Raw results must include a method column.")

    rng = np.random.default_rng(seed)
    rows = []
    available_metrics = [metric for metric in metrics if metric in raw]
    for method, group in raw.groupby("method", sort=True):
        row = {
            "method": method,
            "n_seeds": int(group["seed"].nunique()) if "seed" in group else len(group),
        }
        for metric in available_metrics:
            mean, low, high = _bootstrap_mean_interval(
                group[metric].to_numpy(),
                rng,
                resamples,
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        for flag in (
            "relay_death_censored",
            "route_failure_censored",
            "partition_censored",
            "lifetime_censored",
        ):
            if flag in group:
                row[f"{flag}_fraction"] = float(group[flag].mean())
        rows.append(row)
    return pd.DataFrame(rows)
