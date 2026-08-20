# NashDQNSleep+: Non-Linear AoI Extension

## Project Title

NashDQNSleep+: Non-Linear AoI Extension for Energy-Efficient
Sleep Scheduling in Industrial IoT

## Team Members

- Swastik Basu

## Project Description

This project presents a research-gap analysis and experimental
extension of the NashDQNSleep framework for Industrial Internet
of Things (IIoT) networks.

The work focuses specifically on the limitation of using a linear
Age of Information (AoI) model. The proposed extension introduces
a non-linear exponential AoI formulation in which the penalty
associated with stale information increases with the current AoI.

The study evaluates the proposed model for different values of
the non-linearity parameter λ and compares its behaviour with the
original linear AoI formulation.

## Research Focus

The primary research gap addressed in this implementation is:

- Linear AoI growth model in NashDQNSleep

The following λ values are evaluated:

- λ = 0.00 — Original linear AoI
- λ = 0.05
- λ = 0.15
- λ = 0.30

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- NetworkX

## Project Structure

```text
NashDQNSleep+/
├── README.md
├── requirements.txt
├── src/
│   └── research_gap.py
├── outputs/
│   └── generated figures
└── docs/
    ├── Research_Gap_Analysis.pdf
    └── Presentation.pdf