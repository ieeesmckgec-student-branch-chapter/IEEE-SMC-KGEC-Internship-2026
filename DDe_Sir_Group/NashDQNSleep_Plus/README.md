# NashDQNSleep+: Non-Linear AoI Extension for IIoT Sleep Scheduling

## Summer Research Internship 2026

### IEEE SMC KGEC Student Branch

---

## Project Title

**NashDQNSleep+: Nash-Based Deep Q-Network Adaptive Sleep Scheduling with a Non-Linear Age of Information Model for Industrial Internet of Things**

---

## Team Members

- **Swastik Basu**

---

## Project Overview

Industrial Internet of Things (IIoT) networks consist of sensor nodes that continuously monitor industrial environments and communicate information to a central sink or monitoring system.

Continuous operation of these sensor nodes can lead to significant energy consumption. Sleep scheduling is therefore used to reduce energy usage by allowing sensor nodes to switch between active and sleep modes.

However, excessive sleeping can cause information to become outdated.

This creates a trade-off between:

- **Energy efficiency**
- **Information freshness**

The **Age of Information (AoI)** metric is used to quantify the freshness of information at the receiver.

The original **NashDQNSleep** framework addresses this problem using decentralized decision-making with Deep Q-Networks (DQN), while incorporating AoI and energy-related factors into the reward formulation.

This project investigates one specific limitation identified through the research-gap analysis of the NashDQNSleep framework:

> **The use of a linear AoI growth model.**

The proposed **NashDQNSleep+** extension introduces a non-linear AoI growth model in which the rate of information ageing increases as the information becomes older.

---

# Research Gap Addressed

The research-gap analysis identified several potential limitations in the original NashDQNSleep framework.

This implementation focuses specifically on:

### Gap 1 — Linear AoI Growth

In the original formulation, when a successful transmission does not occur, AoI increases linearly:

    ```text
    Δn(k+1) = Δn(k) + 1
    ````

This assumes that every additional time step contributes the same amount to information staleness.

For practical industrial monitoring systems, older information may become increasingly undesirable as the system continues operating without a successful update.

---

# Proposed Solution

The proposed NashDQNSleep+ model replaces the linear AoI growth with an exponential non-linear formulation:

    ```text
    Δn(k+1) = Δn(k) + exp(λ(Δn(k) - 1))
    ```

where:

* `Δn(k)` represents the current Age of Information of node `n`
* `λ` is the non-linearity growth parameter

When:

    ```text
    λ = 0
    ```

the model reduces to the original linear AoI formulation.

Increasing `λ` causes the AoI growth to become progressively more aggressive as the information becomes older.

---

# Non-Linear AoI Cost

The proposed model also introduces a non-linear AoI cost function:

    ```text
    φ(Δ) = (exp(λΔ) - 1) / λ
    ```

for:

    ```text
    λ > 0
    ```

and:

    ```text
    φ(Δ) = Δ
    ```

for:

```text
λ = 0
```

This cost is incorporated into the reward function so that increasingly stale information receives a progressively larger penalty.

---

# Reward Function

The implementation evaluates the original and proposed reward structures.

### Original NashDQNSleep

The original reward considers:

* Battery level
* AoI
* AoI threshold violation
* Collision/interference
* Successful transmission

### NashDQNSleep+

The proposed extension replaces the linear AoI term with the non-linear AoI cost:

```text
R = w1·B
    - w2·φ(Δ)
    - w3·max(0, φ(Δ) - Δth)
    - w4·collision
    + w5·success
```

This allows the reward to place an increasingly strong penalty on stale information.

---

# Experimental Setup

The simulation evaluates the original linear AoI model and the proposed non-linear AoI model.

| Parameter                 | Value |
| ------------------------- | ----: |
| Number of sensor nodes    |    10 |
| Episodes                  |    30 |
| Learning rate             | 0.001 |
| Discount factor (γ)       |  0.95 |
| DQN hidden layers         |     2 |
| Neurons per hidden layer  |    64 |
| Packet loss probability   |  0.05 |
| Maximum battery level     |   100 |
| Active energy consumption |   1.0 |
| Sleep energy consumption  |  0.05 |
| AoI threshold             |     5 |
| Random seed               |    42 |

### Reward Weights

```text
w1 = 0.4
w2 = 0.4
w3 = 0.1
w4 = 0.1
w5 = 0.1
```

---

# Experiments

Four experimental configurations are evaluated.

### Experiment A — Original NashDQNSleep

```text
λ = 0
Linear AoI
```

This serves as the baseline.

### Experiment B — NashDQNSleep+

```text
λ = 0.05
Non-linear AoI
```

### Experiment C — NashDQNSleep+

```text
λ = 0.15
Non-linear AoI
```

### Experiment D — NashDQNSleep+

```text
λ = 0.30
Non-linear AoI
```

The different λ values are used to study how increasing the non-linearity of the AoI model affects information freshness, battery behaviour, transmission success, and reward.


# IIoT Environment

The simulation models a network of sensor nodes.

Each node maintains information related to:

* Battery level
* Current operating mode
* Age of Information
* Neighbour AoI context

The nodes communicate over a small-world network topology.

The simulation supports two AoI modes:

```text
linear
```

and

```text
nonlinear
```

The linear mode represents the original AoI behaviour, while the non-linear mode represents the proposed NashDQNSleep+ extension.

---

# Topology

The simulation uses a **Watts-Strogatz small-world topology** with:

```text
Number of nodes = 10
k = 4
Rewiring probability = 0.3
```

This provides local connectivity together with long-range connections between sensor nodes.

---

# Generated Results

The simulation generates five main visual analyses.

### Figure 1 — AoI Growth Comparison

Compares:

* Linear AoI
* λ = 0.05
* λ = 0.15
* λ = 0.30

It also visualizes the non-linear AoI cost function `φ(Δ)`.

---

### Figure 2 — Effect of λ

Examines how the non-linearity parameter affects:

* Active versus sleep reward
* Wake-up incentive
* AoI penalty sensitivity

---

### Figure 3 — Training Performance

Compares the original NashDQNSleep configuration with the three non-linear AoI configurations across:

* Average AoI
* Average battery level
* Cumulative reward
* Transmission success rate

---

### Figure 4 — Final Training Performance

Compares the average performance over the final 10 training episodes for:

* Original NashDQNSleep
* NashDQNSleep+ λ = 0.05
* NashDQNSleep+ λ = 0.15
* NashDQNSleep+ λ = 0.30

---

### Figure 5 — Summary Dashboard

Provides a combined view of:

* AoI growth
* Non-linear AoI cost
* Active/sleep reward difference
* AoI training performance
* Battery training performance
* Transmission success rate

---

# Project Structure

```text
NashDQNSleep+/
│
├── README.md
├── requirements.txt
│
├── src/
│   └── research_gap.py
│
├── outputs/
│   ├── fig1_aoi_growth.png
│   ├── fig2_lambda_effect.png
│   ├── fig3_training_comparison.png
│   ├── fig4_bar_comparison.png
│   └── fig5_dashboard.png
│
└── docs/
    ├── Research_Gap_Analysis.pdf
    └── Presentation.pdf
```

---

# Technologies Used

The project is implemented using Python and the following libraries:

* **Python**
* **PyTorch** — DQN implementation and neural-network training
* **NumPy** — Numerical computation and simulation
* **Matplotlib** — Visualization and result generation
* **NetworkX** — Small-world network topology
* **Python Standard Library** — Randomization, replay buffer, file handling, and utilities

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/ieeesmckgec-student-branch-chapter/IEEE-SMC-KGEC-Internship-2026.git
```

Navigate to the project directory:

```bash
cd IEEE-SMC-KGEC-Internship-2026
```

Then navigate to the assigned team folder and project directory.

---

## 2. Install Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

---

# Running the Simulation

From the project directory, run:

```bash
python src/research_gap.py
```

The program executes the four experimental configurations:

```text
Experiment A
Original NashDQNSleep — Linear AoI

Experiment B
NashDQNSleep+ — λ = 0.05

Experiment C
NashDQNSleep+ — λ = 0.15

Experiment D
NashDQNSleep+ — λ = 0.30
```

After the experiments, the program generates the corresponding analytical and training-performance visualizations.

---

# Reproducibility

A fixed random seed is used:

```text
SEED = 42
```

The seed is applied to:

* NumPy
* PyTorch
* Python's random module

This provides a consistent experimental starting point for the simulation.

---

# Scope of the Work

The research-gap analysis identified multiple potential research directions in the NashDQNSleep framework.

However, the present project intentionally focuses on **one primary research gap**:

> **The linear Age of Information growth model.**

The implementation therefore concentrates on:

```text
Linear AoI
      ↓
Non-Linear AoI
      ↓
Modified AoI Cost
      ↓
Modified Reward
      ↓
DQN-Based Simulation
      ↓
Performance Comparison
```

Other identified research gaps are treated as potential directions for future research and are outside the scope of the present implementation.

---

# Limitations

The current project is a simulation-based research prototype.

The study is focused on evaluating the proposed non-linear AoI formulation and does not constitute a complete physical deployment of an IIoT network.

The implementation also uses a simplified DQN-based learning architecture for experimental evaluation rather than reproducing every implementation detail of the original NashDQNSleep system.

---

# Research Contribution

The primary contribution of this project is the investigation of a **non-linear AoI formulation** for the NashDQNSleep sleep-scheduling framework.

The proposed formulation aims to represent the increasing importance of information freshness as data becomes progressively older.

The study evaluates the effect of different non-linearity parameters and compares the proposed approach against the original linear AoI formulation.

---

# Documentation

The `docs/` directory contains the supporting research documentation and presentation associated with this project.

```text
docs/
├── Research_Gap_Analysis.pdf
└── Presentation.pdf
```

These documents provide the detailed research-gap analysis, motivation, methodology, proposed formulation, and discussion of the work.

---

# Authors

**Swastik Basu**
Department of Computer Science and Engineering
Ramakrishna Mission Vivekananda Centenary College

**Summer Research Internship 2026**
**IEEE SMC KGEC Student Branch**

---

## Acknowledgement

This work was carried out as part of the Summer Research Internship Programme 2026 under the IEEE SMC KGEC Student Branch.

The authors acknowledge the guidance and support received during the research internship and the development of this project.

```