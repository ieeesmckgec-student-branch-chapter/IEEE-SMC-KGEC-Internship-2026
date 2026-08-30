from simulation.metrics import metrics


def min_max_normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def fitness(tasks, allocation):
    """
    Multi-objective Fitness Function

    Objectives:
        Minimize:
            - Latency
            - Energy Consumption
            - Execution Cost
            - SLA Violations
            - Load Imbalance

        Maximize:
            - Throughput

    Lower Fitness Score = Better Solution
    """

    result = metrics(tasks, allocation)

    latency = result["latency"]
    energy = result["energy"]
    cost = result["cost"]
    throughput = result["throughput"]
    sla = result["sla_violation"]
    load = result["load_balance"]

    total_tasks = max(len(tasks), 1)

    # -------------------------------------------------
    # Normalization
    # -------------------------------------------------

    latency_norm = min_max_normalize(
        latency,
        0,
        total_tasks * 600
    )

    energy_norm = min_max_normalize(
        energy,
        0,
        total_tasks * 500
    )

    cost_norm = min_max_normalize(
        cost,
        0,
        total_tasks * 3000
    )

    sla_norm = min_max_normalize(
        sla,
        0,
        total_tasks
    )

    load_norm = min_max_normalize(
        load,
        0,
        total_tasks
    )

    # -------------------------------------------------
    # Throughput (Higher is Better)
    # -------------------------------------------------

    max_throughput = total_tasks * 120

    throughput_norm = 1 - min(
        throughput / max(max_throughput, 1),
        1.0
    )

    # -------------------------------------------------
    # Additional SLA Penalty
    # -------------------------------------------------

    beta = 0.10

    sla_penalty = beta * sla_norm

    # -------------------------------------------------
    # Final Fitness Score
    # -------------------------------------------------

    fitness_score = (

        0.25 * latency_norm +

        0.20 * energy_norm +

        0.15 * cost_norm +

        0.15 * sla_norm +

        0.15 * load_norm +

        0.10 * throughput_norm +

        sla_penalty

    )

    return round(max(fitness_score, 0.0), 6)


if __name__ == "__main__":

    from simulation.task_generator import generate_tasks
    from simulation.offloading import offload

    tasks = generate_tasks(20)

    allocation = offload(tasks)

    print("Fitness :", fitness(tasks, allocation))