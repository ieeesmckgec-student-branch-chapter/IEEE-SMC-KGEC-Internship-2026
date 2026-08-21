# Federated Learning-Based Intrusion Detection Using LSTM

**IEEE SMC SBC KGEC Summer Research Internship Programme 2026**

## Project Description

This project evaluates **Federated Learning (FL)** for intrusion detection using an
**LSTM**-based model, across two datasets — a drone communication dataset and the
NSL-KDD network intrusion benchmark. A centralized LSTM baseline is compared against
three federated aggregation strategies, **FedAvg**, **FedProx**, and **FedNova**, each
trained under both **IID** and **non-IID** client data partitions.

The central question: FedProx and FedNova are theoretically designed to outperform
plain FedAvg once client data becomes heterogeneous (non-IID). This project tests that
claim empirically, and reports the result including a case where the
theoretical expectation did not hold, along with a follow-up hyperparameter trial run to
probe why.

## Experiment Phases

| Script | Description |
|---|---|
| `phase1_lstm_ids.py` | Centralized LSTM baseline |
| `phase2_federated_lstm.py` | FedAvg, IID client partitioning |
| `phase2b_fedavg_noniid.py` | FedAvg, non-IID client partitioning |
| `phase4_fedprox.py` | FedProx, IID partitioning |
| `phase4b_fedprox_noniid.py` | FedProx, non-IID partitioning |
| `phase5_fednova.py` | FedNova, IID partitioning (heterogeneous local epochs) |
| `phase5b_fednova_noniid.py` | FedNova, non-IID partitioning + heterogeneous local epochs |
| `phase6_full_comparison.py` | FedAvg vs. FedProx vs. FedNova, and best strategy vs. Centralized |
| `phase7_iid_vs_noniid.py` | Full IID vs. non-IID comparison across all algorithms and both datasets |

Each script trains an LSTM (128 units, LayerNorm, dropout), saves the trained model(s)
and evaluation plots to an `outputs/` folder, and reports accuracy, F1, precision,
recall, FPR, RMSE, and ROC-AUC.

## Key Results

| Setting | Centralized | FedAvg | FedProx | FedNova |
|---|---|---|---|---|
| Drone, IID | 78.64% | 72.94% | 72.94% | 73.00% |
| NSL-KDD, IID | 99.19% | 98.97% | 99.17% | 99.09% |
| Drone, Non-IID | — | 64.96% | 64.49% | 63.04% |
| NSL-KDD, Non-IID | — | 98.87% | 98.35% | 98.61% |

Under IID data, all three FL strategies perform almost identically. Under non-IID data,
the gap between them widens as expected — but **FedAvg remained the strongest FL
strategy in both settings**, the opposite of what FedProx/FedNova's design intent would
predict. A follow-up trial with stronger FedProx/FedNova hyperparameters (μ = 0.1,
extreme local-epoch spread) still underperformed FedAvg, which strengthens rather than
undermines this finding. 

## Technologies Used

- Python 3, TensorFlow / Keras (LSTM model and training)
- scikit-learn (preprocessing, metrics, train/validation split)
- pandas, NumPy (data handling)
- Matplotlib (ROC curves, confusion matrices, comparison charts)

## Project Structure

```
Federated_Learning_LSTM/
├── README.md
├── requirements.txt
└── src/
    ├── phase1_lstm_ids.py
    ├── phase2_federated_lstm.py
    ├── phase2b_fedavg_noniid.py
    ├── phase4_fedprox.py
    ├── phase4b_fedprox_noniid.py
    ├── phase5_fednova.py
    ├── phase5b_fednova_noniid.py
    ├── phase6_full_comparison.py
    └── phase7_iid_vs_noniid.py
```

> **Datasets are not included in this repository**. Download them and place both files in `src/` before running:
> - **NSL-KDD** (`Train_data.csv`): M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani,
>   "A detailed analysis of the KDD CUP 99 data set," in *Proc. IEEE CISDA*, 2009.
> - **Drone Communication Dataset** (`drone_clean.csv`): DatasetEngineer, "Drone
>   Communication Dataset," Kaggle, 2025.
>   [kaggle.com/datasets/datasetengineer/drone-communication-dataset](https://www.kaggle.com/datasets/datasetengineer/drone-communication-dataset)
>
> If your local copy uses different filenames, update the paths at the top of each script.

## Setup

```bash
git clone https://github.com/ieeesmckgec-student-branch-chapter/IEEE-SMC-KGEC-Internship-2026.git
cd IEEE-SMC-KGEC-Internship-2026/Group3/Federated_Learning_LSTM

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Place `drone_clean.csv` and `Train_data.csv` inside `src/`.

## Usage

Run from inside `src/`, in order — later phases load models saved by earlier ones:

```bash
cd src
python phase1_lstm_ids.py
python phase2_federated_lstm.py
python phase2b_fedavg_noniid.py
python phase4_fedprox.py
python phase4b_fedprox_noniid.py
python phase5_fednova.py
python phase5b_fednova_noniid.py
python phase6_full_comparison.py
python phase7_iid_vs_noniid.py
```

Each run updates `outputs/` with trained model checkpoints (`.keras`), evaluation plots,
and CSV summaries of accuracy/F1/AUC across methods and datasets.

## Notes

- Random seed fixed at `42` throughout, for reproducibility.
- No credentials, API keys, or confidential data are included in this repository.

## Author

**Hrishita Ray**

Mentor: Dr. Anwesha Mukherjee.
