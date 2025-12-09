# Exoplanet Transit Classification with Multi-View Self-Supervised Transformers

This repository contains the code and experiments for our graduate-level CSE project on classifying exoplanet transit candidates using phase–folded light curves from NASA’s TESS mission. We evaluate three architectures — a CNN baseline, a supervised transformer, and two self-supervised extensions — to leverage both labeled and unlabeled astronomical observations.

---

## 🚀 Project Overview

Exoplanet detection from stellar photometry is notoriously challenging due to:
- high class imbalance (confirmed planets are rare)
- stellar variability and instrumental noise
- subtle temporal patterns in transit events

Our approach uses **two synchronized representations** of each light curve:
- **Global view**: full orbital phase (captures context and baseline shape)
- **Local view**: zoomed window around centroid peak (captures transit structure)

Both flux and centroid signals are encoded as aligned channels.

---

## 🧠 Model Architectures

| Model | Input | Learning Type | Purpose |
|------|------|---------------|---------|
| **CNN Baseline** | Global flux only | Supervised | Local dip detection |
| **Transformer Encoder** | Dual-view flux + centroid | Supervised | Global + local temporal reasoning |
| **Pseudo-Labeled Transformer** | Same as above | Semi-supervised | Expand training with confident predictions |
| **SSL Reconstruction Transformer** | Same as above | Self-supervised → fine-tuned | Learn transit structure without labels |

Full methodology details are provided in the project report (`report/`).

---

## 📂 Repository Structure

```text
.
├── data/                         # Not included, stored in google drive
│
├── models/
│   ├── cnn.ipynb                 # CNN baseline implementation
│   ├── transformer.ipynb         # Supervised & pseudo-labeled transformer
│   ├── ssl_reconstruction.ipynb  # Masked reconstruction transformer
│   └── reading_bulk_data.ipynb   # Utility functions
│
├── report/
│   └── final_report.pdf
│
└── README.md
