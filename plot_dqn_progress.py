import os
import re
import matplotlib.pyplot as plt
import numpy as np

def plot_dqn_learning():
    log_path = "dqn_lra_log.txt"
    if not os.path.exists(log_path):
        print("Log file dqn_lra_log.txt not found.")
        return

    evals = []
    complexities = []
    constraints = []
    
    d1_vals = []
    d2_vals = []
    alpha_vals = []
    gamma_vals = []

    with open(log_path, "r") as f:
        # Skip header
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Match state list, target_value, and constraint_ok
            # Example: [5.0, 0.0, 0.0, -0.7999999999999998] 1.38000 True
            match = re.match(r"\[(.*?)\]\s+([\d\.]+)\s+(\w+)", line)
            if match:
                state_str, complexity_str, constraint_str = match.groups()
                state = [float(x) for x in state_str.split(",")]
                complexity = float(complexity_str)
                constraint = constraint_str == "True"

                # Decode parameters (incorporating the clipping behavior of black_box_function)
                D1 = max(1, int(round(state[0] * 2)))
                D2 = max(1, int(round(state[1] * 1)))
                alpha = max(0.05, float(state[2]) / 4.0)
                gamma_lra = max(0.1, float(state[3]) * 1.0)

                d1_vals.append(D1)
                d2_vals.append(D2)
                alpha_vals.append(alpha)
                gamma_vals.append(gamma_lra)

                complexities.append(complexity)
                constraints.append(constraint)
                evals.append(len(evals) + 1)

    if not evals:
        print("No evaluations to plot yet.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Complexity over evaluations
    axes[0].plot(evals, complexities, color="royalblue", label="Average SCL Runs", linewidth=1.5)
    
    # Mark constraint failures
    fail_indices = [i for i, c in enumerate(constraints) if not c]
    if fail_indices:
        fail_evals = [evals[i] for i in fail_indices]
        fail_comps = [complexities[i] for i in fail_indices]
        axes[0].scatter(fail_evals, fail_comps, color="crimson", marker="x", s=50, label="Constraint Fail (FER Exceeded)")
        
    axes[0].set_title("DQN Parameter Tuning: Decoding Complexity Optimization", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Average Decoding Attempts", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Plot 2: Parameter trajectories
    axes[1].plot(evals, d1_vals, label="D1 (FLIPSKIP window)", color="darkorange", alpha=0.8)
    axes[1].plot(evals, d2_vals, label="D2 (FLIPSKIP threshold)", color="forestgreen", alpha=0.8)
    axes[1].plot(evals, alpha_vals, label=r"$\alpha$ (SCL-RE threshold)", color="purple", alpha=0.8)
    axes[1].plot(evals, gamma_vals, label=r"$\gamma_{lra}$ (reliability exponent)", color="teal", alpha=0.8)

    axes[1].set_title("Parameter Exploration Trajectories", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Evaluation Steps", fontsize=10)
    axes[1].set_ylabel("Parameter Value", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    output_img = "dqn_tuning_progress.png"
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved successfully to: {os.path.abspath(output_img)}")

if __name__ == "__main__":
    plot_dqn_learning()
