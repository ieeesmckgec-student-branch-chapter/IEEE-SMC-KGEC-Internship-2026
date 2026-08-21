# Weakly Supervised EMCAD for Polyp Segmentation

This repository contains the current EMCAD-based polyp segmentation pipeline used in this workspace. The main entry points are:

- `train_polyp_bp.py` for bounded polygon supervision
- `train_polyp_FS.py` for a full-supervision baseline
-  `train_polyp_hup.py` for a HUP annotation weak supervision
- `test_polyp.py` for evaluation and export of metrics

## Architecture

<p align="center">
  <img src="model_architecture.png" width="100%" height="40%" class="center" alt="EMCAD architecture overview">
</p>

## Qualitative Results

<p align="center">
  <img src="qualitative_results_clinicdb.jpeg" width="80%" height="25%" class="center" alt="Qualitative polyp segmentation results">
</p>

## HUPAnnot

<img width="1600" height="1093" alt="image" src="https://github.com/user-attachments/assets/20af24c4-e1bf-483b-9682-6edb7aa21bdd" />

## Current project summary

The updated code currently includes:

- weak-supervision HUPannotation training in `train_polyp_hup.py` with multi-scale loss terms and checkpoint logging;
- a full-supervision baseline in `train_polyp_FS.py` for comparison;
- evaluation in `test_polyp.py` that writes per-image metrics and spreadsheet summaries;
- corrected loader imports in `utils/dataloader.py` and `utils/dataloader_polyp.py` so the runtime resolves the weak-mask utility path properly.

Validation completed in this environment:

- `python train_polyp.py --help` starts successfully.
- `python test_polyp.py --help` starts successfully.
- `python -m compileall train_polyp.py train_polyp_FS.py test_polyp.py utils lib` passes.

## Environment setup

```bash
conda create -n emcadenv python=3.9
conda activate emcadenv
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

If dependency resolution is problematic on your machine, install the runtime packages used by the current scripts directly:

```bash
python -m pip install timm==0.6.12 opencv-python-headless albumentations medpy seaborn ptflops thop tensorboardx ml-collections loguru torchsummary torchsummaryx segmentation-mask-overlay
```

## Data layout

Place the polyp datasets in the following structure before training:

```text
./data/polyp/target/ClinicDB/train/images/
./data/polyp/target/ClinicDB/train/masks/
./data/polyp/target/ClinicDB/test/images/
./data/polyp/target/ClinicDB/test/masks/
```

## Pretrained encoder weights

Put the PVTv2 checkpoint files in `./pretrained_pth/pvt/` before running an encoder-based training job.

## Training

### Weakly supervised HUPannotation run

```bash
python train_polyp_hup.py \
  --encoder pvt_v2_b2 \
  --pretrained_dir ./pretrained_pth/pvt/ \
  --train_path ./data/polyp/target/ClinicDB/train/ \
  --test_path ./data/polyp/target/ClinicDB/ \
  --epoch 200 \
  --batchsize 8
```

### Full-supervision baseline

```bash
python train_polyp_FS.py \
  --train_path ./data/polyp/target/ClinicDB/train/ \
  --test_path ./data/polyp/target/ClinicDB/ \
  --epoch 200 \
  --batchsize 8
```

## Evaluation

```bash
python test_polyp.py \
  --run_id <run_id> \
  --encoder pvt_v2_b2 \
  --dataset_name ClinicDB
```
---

```markdown

---

## 1. Overview

HUPAnno (Hierarical Uncertinity Polygon Annotation) is a weakly-supervised annotation strategy that combines **global coarse polygon rings** (cheap, fast) with **Local Refinement Patches (LRPs)** at hard boundary segments (expensive but targeted). 

This document records two versions:

- **Previous HUPAnno** — the original theoretical design `.
- **Improved HUPAnno** — the current implementation that actually trains stably and achieves strong results (Dice ~0.90, HD95 ~13 px on Kvasir-SEG).

---

## 2. Previous HUPAnno (Original Design)

### 2.1 Core Philosophy
The annotator acts as a **hard-sample oracle**. Instead of letting the model noisily discover hard regions via entropy, the annotator explicitly marks the K hardest boundary segments. The model then receives near-dense supervision at exactly those locations.

### 2.2 Annotation Structure

| Zone | Symbol | Meaning |
|------|--------|---------|
| Certain FG (global) | Ω_I | Inside P_in |
| Certain BG (global) | Ω_O | Outside P_out |
| Global Uncertain | Ω_Δ | Between P_in and P_out |
| LRP Resolved FG | Ω_RF | Inside tight LRP inner ring |
| LRP Resolved BG | Ω_RB | Outside tight LRP outer ring (but inside patch) |
| LRP Uncertain | Ω_LRPΔ | Between tight LRP rings |

- **Hard segments:** Identified by human annotator or pre-highlighted via gradient saliency from a pretrained backbone.
- **LRP geometry:** Tight erosion (~6% radius) and dilation (~8% radius) around each hard segment.

### 2.3 Network Architecture

- **Backbone:** PVTv2-B2 + EMCAD decoder
- **CCG Head:** 4-class classifier  
  `0 = Certain BG | 1 = Global Uncertain | 2 = LRP Uncertain | 3 = Certain FG`
- **Embedding Head:** 128-dim L2-normalized features for Pixel-wise Contrastive Learning (PCL)
- **No explicit confidence head:** Spatially-varying entropy thresholds (μ_hard, μ_easy) are used only as fixed hyperparameters during anchor sampling.

### 2.4 Loss Formulation

| Loss | Mathematical Form |
|------|-------------------|
| **L_c** | `Dice(pred, y_in) + Dice(pred, y_out) + Dice(pred, lrp_fg) + Dice(pred, lrp_bg)` — **equal weights** |
| **L_ce** | `CrossEntropy(cls_logits, y_c)` |
| **L_PCL** | InfoNCE on uncertain anchors. Phase 2: easy only. Phase 3: all uncertain (global + LRP combined). |
| **L_patch** | `KL(p(x) ‖ p̄_category)` where `p̄_fg = EMA(mean(pred[Ω_I]))`, `p̄_bg = EMA(mean(pred[Ω_O]))` |

### 2.5 Training Curriculum

| Phase | Epochs | Active Losses |
|-------|--------|---------------|
| **Phase 1** | 0 – 30% | `L = L_c` |
| **Phase 2** | 30% – 70% | `L = L_c + λ₁·L_PCL(easy) + λ₂·L_ce` |
| **Phase 3** | 70% – 100% | `L = L_c + λ₁·L_PCL(all) + λ₂·L_ce + λ₃·L_patch` |

### 2.6 Why It Underperformed

1. **Pure Dice instability:** Equal-weight pure Dice on tiny LRP regions (Ω_RF, Ω_RB) collapses when overlap is near-zero, causing gradient starvation.
2. **Strict outer boundary:** Treating everything outside P_out as hard background (`Dice(pred, 1−y_out)`) over-suppresses the uncertain ring early in training.
3. **No learned uncertainty:** Fixed thresholds μ_hard/μ_easy cannot adapt as the feature space evolves.
4. **Annotator variance:** Human-identified hard segments (or noisy gradient saliency) introduce inconsistent LRP placements across the dataset.
5. **Prototype noise:** Early-training EMA prototypes from Ω_I/Ω_O are unreliable, making L_patch destabilize Phase 3.

---

## 3. Improved HUPAnno (Current Implementation)

### 3.1 Core Philosophy
Preserve the global+local supervision structure, but **automate hardness detection** and **stabilize every loss component** so the model converges reliably without manual tuning or human variance.

### 3.2 Annotation Structure

| Zone | Symbol | Meaning |
|------|--------|---------|
| Certain FG (global) | Ω_I | Inside P_in |
| Certain BG (global) | Ω_O | Outside P_out |
| Global Uncertain | Ω_Δ | Between P_in and P_out |
| LRP Resolved FG | Ω_RF | Inside tight LRP inner ring |
| LRP Resolved BG | Ω_RB | Outside tight LRP outer ring (inside patch bbox) |
| LRP Uncertain | Ω_LRPΔ | Between tight LRP rings |

**Key change — Hardness Detection:**
- Automated curvature-based detection (`find_hard_segments`).
- For each contour point: `curvature = 1.0 − cos(angle between successive tangents)`.
- Greedy non-overlapping peak selection (K=2).
- **Impact:** Fully reproducible, no human variance, selects genuine geometric hard points (sharp corners, narrow protrusions, concave indentations).

### 3.3 Network Architecture

- **Backbone:** PVTv2-B2 + EMCAD decoder (unchanged)
- **CCG Head:** 4-class classifier (unchanged)
- **Embedding Head:** 128-dim for PCL (unchanged)
- **NEW — Confidence Head (`conf_head`):**  
  Lightweight 2-conv head predicting `μ(x) ∈ (0,1)` per pixel.  
  Supervised to output **0.3** inside LRP patches (stricter) and **0.5** outside (tolerant).  
  Forces the network to explicitly learn where the annotation is tight vs. loose.

### 3.4 Loss Formulation

| Loss | Implementation | Change vs. Previous |
|------|----------------|---------------------|
| **L_c** | `structure_loss(pred, y_in)` + `structure_loss(pred, y_out)` + `structure_loss(pred, lrp_fg)` + `BCE(pred[lrp_bg], 0)` | **BCE + Dice** instead of pure Dice; softer outer-boundary supervision |
| **L_ce** | `CrossEntropy(cls_logits, y_c)` | Same |
| **L_conf** | `MSE(conf_map, target_μ)` where `target = 0.3` inside LRP, `0.5` outside | **NEW** — explicit spatial confidence supervision |
| **L_PCL** | InfoNCE. Phase 2: easy uncertain only. Phase 3: LRP anchors (ρ=0.85) + easy anchors (ρ=0.5). | LRP treated as **primary** hard samples; easy as secondary |
| **L_patch** | `KL(p(x) ‖ proto_fg/bg)` where assignment is decided by `cv2.distanceTransform` proximity to Ω_RF / Ω_RB within the LRP strip | **Distance-aware** prototype assignment instead of hard zone membership |

### 3.5 Training Curriculum

| Phase | Epochs | Active Losses |
|-------|--------|---------------|
| **Phase 1** | 0 – 30% | `L = L_c` |
| **Phase 2** | 30% – 70% | `L = L_c + λ₁·L_PCL(easy) + λ₂·L_ce + λ₃·L_conf` |
| **Phase 3** | 70% – 100% | `L = L_c + λ₁·L_PCL(LRP+easy) + λ₂·L_ce + λ₃·L_patch + λ₄·L_conf` |

**Hyperparameters (current):**
- `λ₁ (PCL) = 0.1`
- `λ₂ (CE) = 0.3`
- `λ₃ (Patch) = 0.2`
- `λ₄ (Conf) = 0.05`
- `μ_hard = 0.3` (LRP stricter)
- `μ_easy = 0.5` (global tolerant)
- `ρ_LRP = 0.85` (aggressive sampling of annotator-verified hard pixels)
- `ρ_easy = 0.5` (standard entropy-based sampling)
- `EMA_DECAY = 0.99`

### 3.6 Key Improvements & Rationale

| # | Improvement | Rationale |
|---|-------------|-----------|
| 1 | **Curvature-based hard detection** | Eliminates annotator variance; selects objectively hard geometric points (high curvature = sharp corners/protrusions). |
| 2 | **Structure Loss (BCE + Dice)** | BCE provides stable gradients even when Dice overlap is near-zero on tiny LRP zones. Prevents gradient collapse. |
| 3 | **Softer outer boundary** | `structure_loss(pred, y_out)` treats inside-P_out as foreground (BPAnno-style). Prevents the model from over-suppressing the boundary in Phase 1. |
| 4 | **Pure BCE for LRP background** | `lrp_bg` regions are very small; pure BCE is more stable than Dice for micro-zones. |
| 5 | **Explicit Confidence Head** | Network learns to predict its own spatial uncertainty threshold rather than relying on fixed values. Adapts to feature space evolution. |
| 6 | **Distance-aware L_patch** | Uses `cv2.distanceTransform` to softly assign each LRP-uncertain pixel to the FG or BG prototype based on geometric distance to the resolved rings. More accurate than hard binary assignment near the strip center. |

---

## 4. Side-by-Side Comparison

| Aspect | Previous HUPAnno | Improved HUPAnno |
|--------|------------------|------------------|
| **Hard Segment Source** | Human annotator / Gradient saliency | Contour curvature analysis (automated) |
| **L_c Formulation** | Pure Dice, equal weights | BCE + Dice (`structure_loss`); mixed formulation for stability |
| **Outer Boundary (L_out)** | Strict BG outside P_out | Soft FG inside P_out (prevents over-suppression) |
| **LRP BG Loss (L_rb)** | Dice | Pure BCE (stable on tiny masks) |
| **Confidence Mechanism** | Fixed thresholds μ_hard / μ_easy | Learned `conf_head` with MSE supervision |
| **L_patch Assignment** | Hard binary: pixel assigned to nearest prototype zone | Continuous: `distanceTransform` decides prototype |
| **Training Stability** | Unstable; Phase 3 often noisy | Stable convergence across all phases |
| **Kvasir-SEG Performance** | Did not converge reliably | **Dice ~0.90, HD95 ~13 px** |

---

## 5. Code Mapping

| Component | File | Key Function / Class |
|-----------|------|----------------------|
| **Curvature-based LRP** | `dataloader.py` | `find_hard_segments()`, `generate_lrp_masks()` |
| **4-Class CCG + Conf Head** | `networks.py` | `cls_head` (4 outputs), `conf_head` (1 output + Sigmoid) |
| **Stable Certain Loss** | `train_polyp.py` | `loss_certain()` — uses `structure_loss()` + BCE |
| **Explicit Confidence Loss** | `train_polyp.py` | `loss_conf()` — MSE against spatial μ target |
| **Distance-Aware Patch Loss** | `train_polyp.py` | `loss_patch()` — `cv2.distanceTransform` + KL to EMA prototypes |
| **Curriculum** | `train_polyp.py` | `get_phase()` — 3-phase logic with Phase 3 LRP-primary PCL |

---

## 6. Summary

The **HUPAnno** keeps the exact same information-theoretic goal but replaces every fragile component with a stable, differentiable, and reproducible alternative. The result is a weakly-supervised system that closes most of the gap to fully-supervised performance at ~35–45% of the dense annotation cost.
```

---
