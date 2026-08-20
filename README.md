# Edge-Cloud Task Offloading Performance Evaluation

## IEEE SMC SBC KGEC Internship Programme 2026

### Project Overview

This project focuses on the performance evaluation of task offloading strategies in an Edge-Cloud Computing Environment. The objective is to efficiently allocate computational tasks between edge and cloud servers while minimizing latency and energy consumption and improving overall system performance.

The project implements and compares classical scheduling techniques and population-based optimization algorithms for task offloading. Different workloads are generated and evaluated under multiple performance metrics to analyze the effectiveness of each approach.

## Objectives

* Evaluate task offloading in an edge-cloud environment.
* Compare classical scheduling and optimization algorithms.
* Minimize task latency and energy consumption.
* Improve response time and system throughput.
* Analyze server resource utilization and load balancing.
* Evaluate SLA-related performance.
* Perform statistical comparison of the implemented algorithms.

## Algorithms Implemented

The project includes the following scheduling and optimization approaches:

* First Come First Serve (FCFS)
* Round Robin (RR)
* Earliest Deadline First (EDF)
* Particle Swarm Optimization (PSO)
* Differential Evolution (DE)
* Genetic Algorithm (GA)
* Lion Optimization Algorithm
* Hybrid GA-Lion Optimization

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* CSV-based datasets and result analysis

## Project Structure

```text
IEEE-SMC-KGEC-Internship-2026/
│
├── analysis/              # Statistical and performance analysis
├── dataset/               # Dataset and workload-related files
├── graphs/                # Generated performance graphs
├── ml/                    # Machine learning components
├── optimizers/            # Optimization algorithms
├── report/                # Final internship report
├── simulation/            # Task, server and simulation modules
│
├── add_offloading.py      # Offloading-related implementation
├── algorithm_ranking.csv  # Algorithm ranking results
├── conclusion.txt         # Project conclusions
├── main.csv               # Main experimental results
├── main.py                # Main simulation program
├── optimizer_data.csv     # Optimizer-related dataset
├── results.csv            # Experimental results
└── statistical_summary.csv # Statistical analysis results
```

## Performance Metrics

The implemented approaches are evaluated using several performance indicators:

* Latency
* Energy Consumption
* Response Time
* Throughput
* Resource Utilization
* Load Balancing
* SLA Performance
* Edge and Cloud Task Distribution
* Computational Cost

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ieeesmckgec-student-branch-chapter/IEEE-SMC-KGEC-Internship-2026.git
```

### 2. Navigate to the project directory

```bash
cd IEEE-SMC-KGEC-Internship-2026
```

### 3. Install the required Python packages

```bash
pip install -r requirements.txt
```

## Usage

Run the main simulation using:

```bash
python main.py
```

The program performs the task offloading simulation and generates the required experimental results.

The generated CSV files can then be used for statistical analysis and performance comparison.

## Results

The project provides comparative analysis of the implemented algorithms using workload sizes and multiple performance metrics. The generated results and graphs are available in the corresponding `results.csv`, `statistical_summary.csv`, and `graphs/` files.

## Internship Information

**Programme:** IEEE SMC SBC KGEC Internship Programme 2026
**Domain:** Data Science and Artificial Intelligence
**Project Area:** Edge-Cloud Computing and Task Offloading
**Mentor/Supervisor:** Gopa Mandal

## Team Members

Soumyadeep Panja
Mohit Kumar Ray

## Disclaimer

This project was developed as part of the IEEE SMC SBC KGEC Internship Programme 2026 for academic and research purposes.
