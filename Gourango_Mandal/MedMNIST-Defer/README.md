# MedMNIST-Defer: A Cross-Modality Benchmark for Selective Prediction in Biomedical Imaging

## IEEE SMC Student Branch Chapter, KGEC — Summer Research Internship Programme 2026

### Team Details

| Field | Details |
|-------|---------|
| **Student** | Gourango Mandal |
| **Department** | Computer Science & Engineering |
| **Mentor** | Kaustabh Ganguly |
| **Mentor Affiliation** | Senior AI/ML Engineer, Gracenote |
| **Domain** | Healthcare / AI-ML |
| **Duration** | Started 14 August 2026 |

---

## Project Description

MedMNIST-Defer is a cross-modality benchmark that evaluates selective prediction (the "reject option") across all 12 two-dimensional MedMNIST v2 datasets under a single fixed experimental protocol.

A ResNet-18 model is trained from scratch for 8 epochs with random seed 42 and no data augmentation or per-dataset tuning. Five confidence measures are assessed:
- Predictive entropy
- Maximum softmax probability
- Top-two probability margin
- Temperature-scaled entropy
- Monte Carlo dropout

The benchmark evaluates accuracy at 100%, 90%, 80%, 70%, 60%, and 50% coverages along with risk-coverage AUC and expected calibration error (ECE).

### Key Results

| Metric | Value |
|--------|------:|
| Image types benchmarked | 12 |
| Total images | 707,962 |
| Mean accuracy gain at 50% coverage | **+16.0 points** |
| Datasets with statistically significant gain | **12 / 12** |
| Statistical test (Wilcoxon signed-rank) | **p = 4.9 x 10⁻⁴** |
| Range of gains across image types | +9.6 to +20.9 points |
| Spread between 5 confidence methods | 0.18 points |
| Full benchmark runtime (1 GPU) | 326 seconds |

---

## Technologies Used

- Python 3.13
- PyTorch 2.11.0 (CUDA 12.8)
- torchvision
- NumPy 2.4
- SciPy
- Matplotlib
- MedMNIST v2

---

## Project Structure

```
Gourango_Mandal/MedMNIST-Defer/
├── README.md
├── requirements.txt
├── REPRODUCE.md
├── run_all.sh
├── .gitignore
├── src/
│   ├── config.py
│   ├── common.py
│   ├── data.py
│   ├── models.py
│   ├── download_data.py
│   ├── build_sample_data.py
│   ├── run_benchmark.py
│   ├── make_figures.py
│   ├── verify_setup.py
│   └── compare_to_golden.py
├── data/
│   ├── metadata.json
│   └── sample/              (13 MB sample datasets)
├── results/
│   ├── figures/             (generated figures)
│   └── golden/              (reference benchmark results)
└── docs/
    └── IEEE-SMC-SBC-KGEC-MedMNIST-Defer-Report.docx
```

---

## Setup / Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Quick run (sample data, ~10 seconds on GPU):
```bash
./run_all.sh
```

### Full benchmark (downloads 574 MB, reproduces all paper numbers):
```bash
./run_all.sh full
```

---

## Source Repository

Full project repository: [https://github.com/stabgan/medmnist-defer](https://github.com/stabgan/medmnist-defer)

---

## References

- Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., Ni, B. *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data 10:41 (2023).
- He, K., Zhang, X., Ren, S., Sun, J. *Deep residual learning for image recognition.* CVPR, 2016.
- Gal, Y., Ghahramani, Z. *Dropout as a Bayesian approximation.* ICML, 2016.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. *On calibration of modern neural networks.* ICML, 2017.
