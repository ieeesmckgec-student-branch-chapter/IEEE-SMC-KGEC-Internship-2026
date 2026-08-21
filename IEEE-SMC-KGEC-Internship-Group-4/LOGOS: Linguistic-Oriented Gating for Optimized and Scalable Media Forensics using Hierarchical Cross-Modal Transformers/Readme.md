```markdown
# LOGOS: Linguistic-Oriented Gating for Optimized and Scalable Media Forensics using Hierarchical Cross-Modal Transformers

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Tiered%20Middleware-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LOGOS** is a tiered, multi-modal media forensics middleware framework designed to address the trade-off between forensic accuracy and computational scalability in real-time streaming environments (e.g., Telegram, WhatsApp, Discord). By combining lightweight linguistic gating with asynchronous, decoupled feature extraction and hierarchical cross-modal attention, LOGOS delivers sub-second detection without saturating GPU resources.

---

## 📌 Key Innovations

* **Linguistic-Oriented Early Gating (Tier 1):** Employs a quantized INT8 NLP transformer to screen semantic and metadata context, allowing over 60% of benign chat traffic to exit early without triggering expensive visual/audio pipelines.
* **Decoupled Asynchronous Workers (Tier 2):** Media payloads are queued via Redis and dispatched concurrently to:
  * **Acoustic Worker:** Codec-invariant phonetic feature extraction using fine-tuned `Wav2Vec2`.
  * **Visual Worker:** 16-frame spatial-temporal facial sequences using `Vision Transformers (ViT)` / `3D-CNN`.
  * **Kinetic Tracker:** Dynamic mouth-region cropping and phoneme-to-viseme kinetic synchrony mapping.
* **Hierarchical Cross-Modal Transformer (Tier 3):** Performs intra-modal token refinement followed by inter-modal cross-attention to detect subtle speech-to-lip desynchronization and multimodal manipulation artifacts.
* **Glass-Box Diagnostic Telemetry:** Generates Explainable AI (XAI) threat diagnostics (Grad-CAM heatmaps for visual zones and LIME/SHAP for acoustic/text contributions) with closed-loop threshold adaptation.

---

## 🏗️ System Architecture


```

```
                  [ Telegram / WhatsApp / Discord Webhook ]
                                     │
                                     ▼

```

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION & INTERCEPT PLANE                           │
│   ┌──────────────────────────┐               ┌──────────────────────────────┐   │
│   │   API Webhook Gateway    │──────────────>│   Asynchronous Handshake     │   │
│   │  (Platform Invariant)    │               │       (HTTP 200 OK)          │   │
│   └──────────────────────────┘               └──────────────────────────────┘   │
└────────────────────────────────────────┬────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      TIER 1: LIGHTWEIGHT SEMANTIC SHIELD                        │
│                      ┌────────────────────────────────┐                         │
│                      │   Quantized INT8 Transformer   │                         │
│                      │        (Text Gatekeeper)       │                         │
│                      └────────────────────────────────┘                         │
│                                       │                                         │
│                     ┌─────────────────┴─────────────────┐                       │
│                     ▼                                   ▼                       │
│           [ Malicious Indicators ]            [ Benign Chat Noise ]             │
│              (Trigger Tier 2)                 (Purge Traffic / Idle)            │
└─────────────────────┬───────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     BUFFERING & RESOURCE MANAGEMENT LAYER                       │
│                    ┌───────────────────────────────────────┐                    │
│                    │     Asynchronous Redis Message Broker │                    │
│                    │       (Decoupled Worker Queues)       │                    │
│                    └───────────────────────────────────────┘                    │
└─────────────────────┬───────────────────────────────────┬───────────────────────┘
│ (Audio Payload)                   │ (Video Payload)
▼                                   ▼
┌────────────────────────────────────────┐ ┌──────────────────────────────────────┐
│  TIER 2: ACOUSTIC VERTICAL FORENSICS   │ │   TIER 2: VISUAL VERTICAL FORENSICS  │
│  ┌──────────────────────────────────┐  │ │  ┌─────────────────────────────────┐ │
│  │     Wav2Vec2 Feature Module      │  │ │  │   Mouth-Region Bounding Box     │ │
│  │ (Multi-lingual Phoneme Vectors)  │  │ │  │      Extraction Pipeline        │ │
│  └──────────────────────────────────┘  │ │  └─────────────────────────────────┘ │
│                  │                     │ │                  │                   │
│                  ▼                     │ │                  ▼                   │
│  ┌──────────────────────────────────┐  │ │  ┌─────────────────────────────────┐ │
│  │   Codec-Invariant Macro-Acoustic │  │ │  │    Vision Transformer (ViT)     │ │
│  │          Tracking Blocks         │  │ │  │   Lip Muscular Kinetic Map      │ │
│  └──────────────────────────────────┘  │ │  └─────────────────────────────────┘ │
└─────────────────────┬──────────────────┘ └──────────────────┬───────────────────┘
│                                       │
└───────────────────┬───────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  FUSION PLANE & CROSS-MODAL FORENSICS LAYER                     │
│                    ┌─────────────────────────────────────────┐                  │
│                    │       Unified Latent Space Mapping      │                  │
│                    └─────────────────────────────────────────┘                  │
│                                         │                                       │
│                                         ▼                                       │
│                    ┌─────────────────────────────────────────┐                  │
│                    │   Cross-Modal Token Consistency Check   │                  │
│                    │  (Phoneme-to-Lip Trajectory Alignment)  │                  │
│                    └─────────────────────────────────────────┘                  │
└────────────────────────────────────────┬────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DIAGNOSTIC TELEMETRY PLANE (GLASS-BOX)                       │
│     ┌────────────────────────┐                   ┌────────────────────────┐     │
│     │     LIME / SHAP Plots  │                   │     Grad-CAM Heatmaps  │     │
│     │ (Acoustic/Text Weights)│                   │(Visual Attention Zones)│     │
│     └────────────────────────┘                   └────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────┘

```

---

## 📊 Benchmark Results

| Model / Architecture | Detection AUC | Avg. Latency | Linguistic Awareness | Codec Invariance | GPU Efficiency | XAI Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Frame-Level 2D-CNN | 82.3% | ~220 ms | ❌ None | Low | Moderate | Minimal |
| Monolithic 3D-CNN | 91.2% | ~1800 ms | ❌ None | Moderate | ❌ Saturated | Black-box |
| SyncNet + Audio | 85.7% | ~620 ms | ❌ None | Low | Moderate | None |
| **LOGOS (Proposed)** | **96.1%** | **< 180 ms** | **✅ INT8 Gated** | **✅ High (Wav2Vec2+ViT)** | **✅ High (Dynamic)** | **Grad-CAM + SHAP** |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Redis Server (v6.0+)
* CUDA-enabled GPU (NVIDIA RTX 3060 / T4 or better recommended)

### 1. Clone Repository & Install Dependencies
```bash
git clone [https://github.com/your-username/LOGOS-Forensics.git](https://github.com/your-username/LOGOS-Forensics.git)
cd LOGOS-Forensics
pip install -r requirements.txt

```

### 2. Environment Setup

Create a `.env` file in the root directory:

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
MODEL_WEIGHTS_PATH=./weights/
GATE_CONFIDENCE_THRESHOLD=0.75
DEVICE=cuda

```

### 3. Start Redis Message Broker

```bash
redis-server --daemonize yes

```


```

---



---

## 👥 Contributors & Acknowledgements

* **Author / Presenter:** Aasish Shrestha
* **Mentor / Advisor:** Professor Moirangthem Marjit Singh
* **Affiliation:** IEEE SMC Student Branch Chapter • Kalyani Government Engineering College





```