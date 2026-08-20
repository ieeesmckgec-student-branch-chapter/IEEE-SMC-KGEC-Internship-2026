import pandas as pd
import statistics
import math

# ==========================================================
# Read Results
# ==========================================================

df = pd.read_csv("results.csv")

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

# ==========================================================
# Metrics
# ==========================================================

metrics = [
    "Fitness",
    "Latency",
    "Energy",
    "Throughput",
    "Response",
    "Cost",
    "Load",
    "SLA",
    "Runtime"
]

summary = []

# ==========================================================
# Statistical Analysis
# ==========================================================

for algo in algorithms:

    for metric in metrics:

        column = f"{algo} {metric}"

        values = df[column]

        mean = values.mean()

        median = values.median()

        std = values.std()

        variance = values.var()

        minimum = values.min()

        maximum = values.max()

        best = minimum

        worst = maximum

        # 95% Confidence Interval
        confidence = 1.96 * (std / math.sqrt(len(values)))

        # Improvement over Baseline
        if algo == "Baseline":

            improvement = 0

        else:

            baseline_mean = df[f"Baseline {metric}"].mean()

            if baseline_mean != 0:

                if metric in [
                    "Latency",
                    "Energy",
                    "Fitness",
                    "Response",
                    "Cost",
                    "Load",
                    "SLA",
                    "Runtime"
                ]:

                    improvement = (

                        (baseline_mean - mean)

                        /

                        baseline_mean

                    ) * 100

                else:

                    improvement = (

                        (mean - baseline_mean)

                        /

                        baseline_mean

                    ) * 100

            else:

                improvement = 0

        summary.append({

            "Algorithm": algo,

            "Metric": metric,

            "Mean": round(mean,4),

            "Median": round(median,4),

            "Std Dev": round(std,4),

            "Variance": round(variance,4),

            "Minimum": round(minimum,4),

            "Maximum": round(maximum,4),

            "Best": round(best,4),

            "Worst": round(worst,4),

            "95% CI": round(confidence,4),

            "Improvement (%)": round(improvement,2)

        })

# ==========================================================
# Summary DataFrame
# ==========================================================

summary_df = pd.DataFrame(summary)

# ==========================================================
# Ranking
# ==========================================================

fitness_table = []

for algo in algorithms:

    fitness_table.append({

        "Algorithm": algo,

        "Average Fitness":

        df[f"{algo} Fitness"].mean()

    })

fitness_df = pd.DataFrame(fitness_table)

fitness_df = fitness_df.sort_values(

    by="Average Fitness"

)

fitness_df["Rank"] = range(

    1,

    len(fitness_df)+1

)

# ==========================================================
# Save CSV Files
# ==========================================================

summary_df.to_csv(

    "statistical_summary.csv",

    index=False

)

fitness_df.to_csv(

    "algorithm_ranking.csv",

    index=False

)

# ==========================================================
# Print
# ==========================================================

print("="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

print(summary_df)

print("\n")

print("="*80)
print("ALGORITHM RANKING")
print("="*80)

print(fitness_df)

print("\n")

print("Files Generated Successfully")

print("1. statistical_summary.csv")

print("2. algorithm_ranking.csv")