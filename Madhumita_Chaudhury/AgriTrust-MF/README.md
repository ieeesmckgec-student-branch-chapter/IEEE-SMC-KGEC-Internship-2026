# AgriTrust-MF: Constraint-Preserving QAOA for Interference-Aware Multi-Flow WSN Routing

## IEEE SMC Student Branch Chapter, KGEC — Summer Research Internship Programme 2026

### Team Details

| Field | Details |
|-------|---------|
| **Student** | Madhumita Chaudhury |
| **Department** | Computer Science & Engineering |
| **Mentor** | Kaustabh Ganguly |
| **Mentor Affiliation** | Senior AI/ML Engineer, Gracenote |
| **Domain** | Quantum Computing & IoT |
| **Duration** | Started 14 August 2026 |

---

## Project Description

A 150-node smart-agriculture WSN carries several simultaneous source-to-sink flows. Each flow has multiple topology-feasible candidate paths. Selecting one path changes the cost of other flows through cross-flow wireless interference and congestion, so the task is to choose one path per flow jointly.

The proposed quantum method represents each flow by a one-hot block of route choices. A W/Dicke-1 initial state starts each block with exactly one active route, and a block-wise XY mixer allows the algorithm to explore different routes while preserving that exactly-one condition. The method is compared against exact classical ground truth, simulated annealing, and conventional penalty-based X-QAOA.

### Central Research Question

For an interference-coupled multi-flow WSN routing problem, does constraint-preserving W-state + XY-QAOA keep more probability inside valid routing solutions and produce better-quality feasible solutions than conventional penalty-X QAOA, without making a quantum-speedup claim?

### Key Findings

- XY-QAOA achieves **46x higher probability** of sampling the optimal route vs penalty-X at 9 qubits
- Penalty-X feasible probability collapses to **0.44%** at 25 qubits; XY maintains **100% feasibility**
- CVaR at depth p=2 raises mode optimality from **46.7% to 76.7%**
- Scaling tested across 9, 12, 16, 20, and 25 qubits with 20 pre-declared seeds

---

## Technologies Used

- Python 3.x
- Qiskit >= 2.0
- Qiskit Aer >= 0.17
- NumPy >= 2.0
- SciPy >= 1.11
- Pandas >= 2.0
- Matplotlib >= 3.8
- NetworkX

---

## Project Structure

```
Madhumita_Chaudhury/AgriTrust-MF/
├── README.md
├── requirements.txt
├── .gitignore
├── AgriTrust_MultiFlow_FINAL_REVISED.ipynb   (latest experimental notebook)
├── src/
│   ├── agritrust_core.py
│   ├── attacks.py
│   ├── baselines.py
│   ├── fast_sim.py
│   ├── make_report.py
│   ├── multiflow.py
│   ├── network_lifetime.py
│   ├── noise_study.py
│   ├── poisoning.py
│   ├── run_all.py
│   └── sweep_utils.py
├── experiments/results/         (CSV results from 1,800+ configurations)
└── docs/
    └── IEEE-SMC-SBC-KGEC-AgriTrust-Report.docx
```

---

## Setup / Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run the latest notebook (Google Colab recommended):
Open `AgriTrust_MultiFlow_FINAL_REVISED.ipynb` in Colab or Jupyter.

### Run the experiment suite locally:
```bash
cd src
python run_all.py --smoke          # fast sanity check (~5 min)
python make_report.py              # rebuild results summary
```

Full experiments need a multi-core CPU and optionally 1-4 NVIDIA GPUs.

---

## Source Repository

Full project repository: [https://github.com/stabgan/agritrust-constrained-qaoa-wsn](https://github.com/stabgan/agritrust-constrained-qaoa-wsn)

---

## References

- E. Farhi, J. Goldstone, S. Gutmann, "A quantum approximate optimization algorithm," arXiv:1411.4028, 2014.
- S. Hadfield et al., "From the quantum approximate optimization algorithm to a quantum alternating operator ansatz," Algorithms, vol. 12, no. 2, 2019.
- P. Barkoutsos et al., "Improving variational quantum optimization using CVaR," Quantum, vol. 4, p. 256, 2020.
- Qiskit Development Team, "Qiskit: An open-source framework for quantum computing," 2024.
