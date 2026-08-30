import time
import statistics
import pandas as pd
import random
import numpy as np

from simulation.task_generator import generate_tasks
from simulation.metrics import metrics
from simulation.fitness import fitness

from optimizers.baseline import baseline
from optimizers.pso import pso
from optimizers.de import de
from optimizers.ga import ga
from optimizers.lion import lion
from optimizers.hybrid_ga_lion import hybrid_ga_lion

# ==========================================================
# CONFIGURATION
# ==========================================================

RUNS = 20
WORKLOADS = list(range(5000, 50001, 5000))

ALGORITHMS = {
    "Baseline": baseline,
    "PSO": pso,
    "DE": de,
    "GA": ga,
    "Lion": lion,
    "Hybrid": hybrid_ga_lion
}

results = []

print("=" * 90)
print("EDGE-CLOUD TASK OFFLOADING PERFORMANCE EVALUATION")
print("=" * 90)

# ==========================================================
# SIMULATION
# ==========================================================

for workload in WORKLOADS:

    print(f"\nProcessing Workload : {workload}")

    if workload <= 15000:
        category = "Low"
    elif workload <= 35000:
        category = "Medium"
    else:
        category = "High"

    row = {
        "Workload": category,
        "Tasks": workload
    }

    # Create storage for every algorithm
    algorithm_results = {}

    for algo_name in ALGORITHMS.keys():
        algorithm_results[algo_name] = {
            "latency": [],
            "energy": [],
            "throughput": [],
            "response": [],
            "cost": [],
            "load": [],
            "sla": [],
            "fitness": [],
            "runtime": []
        }

    # ======================================================
    # Repeat experiment
    # ======================================================

    for run in range(RUNS):

        # Reproducible but different workload every run
        seed = 42 + run
        random.seed(seed)
        np.random.seed(seed)

        # Same workload for every algorithm in this run
        tasks = generate_tasks(max(50, workload // 100))

        for algo_name, algo in ALGORITHMS.items():

            start = time.perf_counter()

            allocation = algo(tasks)

            runtime = time.perf_counter() - start

            m = metrics(tasks, allocation)
            f = fitness(tasks, allocation)

            algorithm_results[algo_name]["latency"].append(m["latency"])
            algorithm_results[algo_name]["energy"].append(m["energy"])
            algorithm_results[algo_name]["throughput"].append(m["throughput"])
            algorithm_results[algo_name]["response"].append(m["response_time"])
            algorithm_results[algo_name]["cost"].append(m["cost"])
            algorithm_results[algo_name]["load"].append(m["load_balance"])
            algorithm_results[algo_name]["sla"].append(m["sla_violation"])
            algorithm_results[algo_name]["fitness"].append(f)
            algorithm_results[algo_name]["runtime"].append(runtime)

    # ======================================================
    # Store Statistics
    # ======================================================

    for algo_name in ALGORITHMS.keys():

        data = algorithm_results[algo_name]

        row[f"{algo_name} Latency"] = statistics.mean(data["latency"])
        row[f"{algo_name} Energy"] = statistics.mean(data["energy"])
        row[f"{algo_name} Throughput"] = statistics.mean(data["throughput"])
        row[f"{algo_name} Response"] = statistics.mean(data["response"])
        row[f"{algo_name} Cost"] = statistics.mean(data["cost"])
        row[f"{algo_name} Load"] = statistics.mean(data["load"])
        row[f"{algo_name} SLA"] = statistics.mean(data["sla"])
        row[f"{algo_name} Fitness"] = statistics.mean(data["fitness"])
        row[f"{algo_name} Runtime"] = statistics.mean(data["runtime"])

        row[f"{algo_name} Latency Std"] = statistics.stdev(data["latency"])
        row[f"{algo_name} Energy Std"] = statistics.stdev(data["energy"])
        row[f"{algo_name} Throughput Std"] = statistics.stdev(data["throughput"])
        row[f"{algo_name} Fitness Std"] = statistics.stdev(data["fitness"])

        row[f"{algo_name} Median Fitness"] = statistics.median(data["fitness"])
        row[f"{algo_name} Best Fitness"] = min(data["fitness"])
        row[f"{algo_name} Worst Fitness"] = max(data["fitness"])

        print(
            f"{algo_name:10}"
            f" Fitness={statistics.mean(data['fitness']):8.3f}"
            f" Runtime={statistics.mean(data['runtime']):.4f}s"
        )

    results.append(row)

# ==========================================================
# SAVE RESULTS
# ==========================================================

df = pd.DataFrame(results)

df.to_csv("results.csv", index=False)

print("\n" + "=" * 90)
print("RESULTS SAVED SUCCESSFULLY")
print("=" * 90)

print(df)

print("\nresults.csv generated successfully.")