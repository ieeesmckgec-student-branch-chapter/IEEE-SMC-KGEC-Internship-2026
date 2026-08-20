import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Read Results
# ==========================================================

df = pd.read_csv("results.csv")

# ==========================================================
# Create Output Folder
# ==========================================================

OUTPUT_FOLDER = "graphs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# Publication Style
# ==========================================================

plt.style.use("ggplot")

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

# ==========================================================
# Algorithms
# ==========================================================

algorithms = [
    "Baseline",
    "PSO",
    "DE",
    "GA",
    "Lion",
    "Hybrid"
]

colors = [
    "black",
    "blue",
    "green",
    "orange",
    "purple",
    "red"
]

markers = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "*"
]

# ==========================================================
# Metrics
# ==========================================================

metrics = {
    "Latency": "Latency Comparison",
    "Energy": "Energy Consumption",
    "Fitness": "Fitness Comparison",
    "Throughput": "Throughput Comparison",
    "Response": "Response Time Comparison",
    "Cost": "Execution Cost Comparison",
    "Load": "Load Balance Comparison",
    "SLA": "SLA Violations Comparison",
    "Runtime": "Execution Time Comparison"
}

# ==========================================================
# Plot Function
# ==========================================================

def plot_metric(metric, title):

    plt.figure(figsize=(11, 6))

    for i, algo in enumerate(algorithms):

        std_column = f"{algo} {metric} Std"

        if std_column in df.columns:

            plt.errorbar(
                df["Tasks"],
                df[f"{algo} {metric}"],
                yerr=df[std_column],
                marker=markers[i],
                markersize=6,
                linewidth=2.5,
                capsize=4,
                color=colors[i],
                label=algo
            )

        else:

            plt.plot(
                df["Tasks"],
                df[f"{algo} {metric}"],
                marker=markers[i],
                markersize=6,
                linewidth=2.5,
                color=colors[i],
                label=algo
            )

    plt.title(title, fontweight="bold")

    plt.xlabel("Number of Tasks")

    plt.ylabel(metric)

    plt.xticks(df["Tasks"])

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_FOLDER}/{metric.lower()}_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ==========================================================
# Generate Line Graphs
# ==========================================================

for metric, title in metrics.items():

    plot_metric(metric, title)

# ==========================================================
# Average Fitness Bar Chart
# ==========================================================

average_fitness = [
    df[f"{algo} Fitness"].mean()
    for algo in algorithms
]

plt.figure(figsize=(10,6))

bars = plt.bar(
    algorithms,
    average_fitness
)

plt.title(
    "Average Fitness Comparison",
    fontweight="bold"
)

plt.ylabel("Average Fitness")

plt.grid(axis="y", alpha=0.4)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x()+bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/average_fitness_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Runtime Bar Chart
# ==========================================================

average_runtime = [
    df[f"{algo} Runtime"].mean()
    for algo in algorithms
]

plt.figure(figsize=(10,6))

bars = plt.bar(
    algorithms,
    average_runtime
)

plt.title(
    "Average Runtime Comparison",
    fontweight="bold"
)

plt.ylabel("Runtime (seconds)")

plt.grid(axis="y", alpha=0.4)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x()+bar.get_width()/2,
        height,
        f"{height:.4f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/runtime_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Fitness Box Plot
# ==========================================================

fitness_data = [
    df[f"{algo} Fitness"]
    for algo in algorithms
]

plt.figure(figsize=(11,6))

plt.boxplot(
    fitness_data,
    labels=algorithms,
    showmeans=True
)

plt.title(
    "Fitness Distribution",
    fontweight="bold"
)

plt.ylabel("Fitness")

plt.grid(True, alpha=0.5)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/fitness_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Improvement Chart
# ==========================================================

baseline = df["Baseline Fitness"].mean()

improvements = []

for algo in algorithms:

    if algo == "Baseline":

        improvements.append(0)

    else:

        value = df[f"{algo} Fitness"].mean()

        improvements.append(
            ((baseline - value) / baseline) * 100
        )

plt.figure(figsize=(10,6))

bars = plt.bar(
    algorithms,
    improvements
)

plt.title(
    "Improvement over Baseline (%)",
    fontweight="bold"
)

plt.ylabel("Improvement (%)")

plt.grid(axis="y", alpha=0.4)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x()+bar.get_width()/2,
        height,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/improvement.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 70)
print("ALL PUBLICATION-QUALITY GRAPHS GENERATED SUCCESSFULLY")
print("=" * 70)
print(f"\nGraphs saved inside '{OUTPUT_FOLDER}' folder.\n")