# ALAS-SCLF: Adaptive-L Asymmetric-Segmented SCLF Polar Decoder

**Authors**: Deep Shekhar Halder (Adamas University) & Prasun Majumder  
**Mentor**: Dr. Dwaipayan Ghosh (Assistant Professor, Kalyani Government Engineering College)  
**Program**: IEEE SMC Student Branch Chapter, KGEC – Summer Research Internship 2026  
**Group**: Group 12  

---

## 📁 Official GitHub Repository Structure

```text
Project Name (Group_12_ALAS_SCLF) /
├── README.md             <-- Main project overview & documentation index
├── requirements.txt      <-- Python environment dependencies
├── src/                  <-- Source Code & Simulators
│   ├── polar_simulator.py      (Core Polar Code Decoder implementation)
│   ├── rigorous_polar.py      (Monte Carlo simulation engine)
│   ├── dqn_alas_optimizer.py   (DQN Reinforcement Learning optimizer)
│   ├── main.py                (Main entry script)
│   ├── sweeps/                (Parameter sweep scripts)
│   └── build_scripts/         (Document & Figure Generators)
└── docs/                 <-- Documentation & Figures
    ├── figures/               (High-resolution plots & system flowchart)
    ├── source_paper/          (TeX, HTML, and paper sources)
    └── templates/             (Official templates & admin files)
```

---

## 🚀 Setup & Installation Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ieeesmckgec-student-branch-chapter/IEEE-SMC-KGEC-Internship-2026.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Monte Carlo Polar Code Simulation**:
   ```bash
   python src/main.py
   ```

4. **Run Deep Q-Network Optimization**:
   ```bash
   python src/dqn_alas_optimizer.py
   ```

---

## 📊 Performance Summary

| SNR (dB) | Baseline TS-SCLF FER | Proposed ALAS-SCLF FER | Performance Gain |
| :--- | :--- | :--- | :--- |
| **0.5 dB** | 0.12800 | **0.11500** | 21.0% Latency Saving |
| **1.0 dB** | 0.05800 | **0.04750** | 5.7% Latency Saving |
| **1.5 dB** | 0.02050 | **0.01300** | **36.6% Relative FER Reduction** |
| **2.0 dB** | 0.00600 | **0.00450** | **25.0% Relative FER Reduction** |
