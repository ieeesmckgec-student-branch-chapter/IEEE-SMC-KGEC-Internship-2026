import pandas as pd
import random

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
random.seed(42)

rows = []

# -------------------------------------------------
# Task Sizes (5000 to 50000)
# -------------------------------------------------
task_sizes = list(range(5000, 50001, 5000))

# 500 samples for each task size
samples_per_size = 500

for tasks in task_sizes:

    for _ in range(samples_per_size):

        # -----------------------------
        # Cloud Features
        # -----------------------------
        cpu = random.randint(10, 100)                  # CPU Utilization (%)
        memory = random.randint(256, 8192)             # Memory (MB)
        bandwidth = random.randint(50, 1000)           # Mbps
        priority = random.randint(1, 5)                # Task Priority
        deadline = random.randint(50, 500)             # Deadline (ms)

        task_size = random.randint(100, 5000)          # Task Size (MB)
        server_load = random.randint(10, 100)          # Server Load (%)
        network_latency = random.randint(5, 120)       # Network Latency (ms)
        vm_utilization = random.randint(20, 100)       # VM Utilization (%)
        edge_distance = random.randint(1, 50)          # Distance to Edge (km)

        # -----------------------------
        # Derived Metrics
        # -----------------------------
        execution_time = round(
            tasks / 1000 +
            cpu * 0.20 +
            memory / 1024 +
            network_latency * 0.10 +
            random.uniform(-5, 5),
            2
        )

        energy = round(
            cpu * 0.55 +
            vm_utilization * 0.45 +
            random.uniform(-3, 3),
            2
        )

        cost = round(
            execution_time * 0.05 +
            memory * 0.0004 +
            bandwidth * 0.002 +
            random.uniform(-0.5, 0.5),
            2
        )

        response_time = round(
            execution_time +
            network_latency * 0.50,
            2
        )

        sla_violation = 1 if response_time > deadline else 0

        # -----------------------------
        # Workload Score
        # -----------------------------
        workload = (
            cpu * 0.25 +
            (memory / 8192) * 20 +
            (bandwidth / 1000) * 10 +
            priority * 8 +
            server_load * 0.20 +
            vm_utilization * 0.15 +
            tasks / 2500
        )

        # -----------------------------
        # Optimizer Selection
        # -----------------------------
        if workload < 45:
            optimizer = "GA"

        elif workload < 65:
            optimizer = "PSO"

        elif workload < 85:
            optimizer = "Lion"

        else:
            optimizer = "Hybrid"

        # Add small randomness (8%)
        if random.random() < 0.08:
            optimizer = random.choice(
                ["GA", "PSO", "Lion", "Hybrid"]
            )

        rows.append([
            tasks,
            cpu,
            memory,
            bandwidth,
            priority,
            deadline,
            task_size,
            server_load,
            network_latency,
            vm_utilization,
            edge_distance,
            execution_time,
            energy,
            cost,
            response_time,
            sla_violation,
            optimizer
        ])

# -------------------------------------------------
# Create DataFrame
# -------------------------------------------------
columns = [
    "tasks",
    "cpu",
    "memory",
    "bandwidth",
    "priority",
    "deadline",
    "task_size",
    "server_load",
    "network_latency",
    "vm_utilization",
    "edge_distance",
    "execution_time",
    "energy",
    "cost",
    "response_time",
    "sla_violation",
    "optimizer"
]

df = pd.DataFrame(rows, columns=columns)

# Shuffle Dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save Dataset
df.to_csv("optimizer_data.csv", index=False)

# -------------------------------------------------
# Output
# -------------------------------------------------
print("=" * 60)
print("Cloud Optimization Dataset Created Successfully")
print("=" * 60)

print(f"\nDataset Shape: {df.shape}")

print("\nOptimizer Distribution:")
print(df["optimizer"].value_counts())

print("\nFirst 5 Rows:")
print(df.head())

print("\noptimizer_data.csv saved successfully.")