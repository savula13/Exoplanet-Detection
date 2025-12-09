# Exoplanet Transit Classification with CNNs and Multi View Transformers

This repository contains the code and experiments for our graduate-level CSE project on classifying exoplanet transit candidates using phase–folded light curves from NASA’s TESS mission. We evaluate three architectures — a CNN baseline, a supervised transformer, and two self-supervised extensions — to leverage both labeled and unlabeled astronomical observations.

---

Disclaimer: All collaboration and code execution was done in a Google Colab environment, with the environment containing the necessary files and packages.

---

## 🚀 Project Overview

Exoplanet detection from light curves is challenging due to:
- high class imbalance (confirmed planets are very rare!)
- stellar variability and instrumental noise
- subtle temporal patterns in transit crossing events (TCEs)

Our approach uses **two views** of each light curve:
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
| **SSL Reconstruction Transformer** | Same as above | Self-supervised → fine-tuned | Learn transit curve structure without labels |

Full methodology details are provided in the project report (`report/`).

---

## 📂 Repository Structure

```text
.
├── data/                         # Not included, stored in google drive
│
├── models/
│   ├── cnn.ipynb                 # CNN baseline implementation
│   ├── transformer.ipynb         # Supervised transformer
    ├── pseudo_labeling.ipynb     # Psuedo-labeled transformer
│   ├── ssl_reconstruction.ipynb  # Masked reconstruction transformer (not included in results)
│   └── reading_bulk_data.ipynb   # Utility functions for reading data
│
├── report/
│   └── final_report.pdf
│
└── README.md
