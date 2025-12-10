# Design-Aware Poster Aesthetic Scoring using Deep Learning

This repository contains the preliminary implementation of a **design-aware and explainable poster aesthetic scoring system**. The model predicts an aesthetic score from a poster image using a Convolutional Neural Network (CNN), extracts explicit **design quality metrics** using classical computer vision, and fuses both to generate a final interpretable score. The system also provides **Grad-CAM visual explanations** highlighting image regions that influence the model’s decision.

---

## Project Overview

The system consists of four main components:

1. **CNN-based Aesthetic Scoring**
   A ResNet-18 model is trained as a regression network to predict a continuous aesthetic score from a poster image.

2. **Explainability using Grad-CAM**
   Heatmaps are generated to visualize the regions influencing the CNN prediction.

3. **Design Metric Extraction (OpenCV-based)**
   Explicit design features are computed, including:

   - Color harmony
   - Contrast
   - Clutter
   - Balance
   - Whitespace

4. **Score Fusion**
   The CNN score and the design score are combined using a weighted fusion strategy to obtain the final **design-aware aesthetic rating**.

---

## Current Implementation Status

The following components are already implemented and working:

- Poster dataset loading using custom PyTorch `Dataset` and `DataLoader`
- ResNet-18 training pipeline for regression
- Training and validation loss curve generation
- Grad-CAM explainability visualization
- Design metric extraction using OpenCV
- CNN + Design score fusion
- CSV-based result generation:

  - `design_scores.csv`
  - `final_scores.csv`

All training and inference can run on **CPU**, with optional GPU support if available.

---

## Folder Structure

```
project_root/
│
├── scripts/
│   ├── ava_dataset.py
│   ├── train_baseline.py
│   ├── test_loader.py
│   ├── gradcam.py
│   ├── design_metrics.py
│   └── predict_and_fuse.py
│
├── data/
│   └── poster_dataset/
│   ├── images/
│   └── labels.csv
│
├── results/
│   ├── resnet18_final.pt
│   ├── loss_curve.png
│   ├── gradcam_overlay.png
│   ├── design_scores.csv
│   └── final_scores.csv
│
├── demo_pipeline.ipynb
└── README.md
```

---

## System Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- NumPy
- Pandas
- Pillow
- Matplotlib

---

## Installation

```bash
pip install torch torchvision opencv-python numpy pandas pillow matplotlib
```

---

## Dataset Format

```
data/poster_dataset/
├── images/
│   ├── poster1.jpg
│   ├── poster2.jpg
│   └── poster3.jpg
└── labels.csv
```

`labels.csv` format:

```
image,score
poster1.jpg,7.5
poster2.jpg,4.0
poster3.jpg,8.2
```

Scores must be in the range **0–10**.

---

## How to Test the Data Loader

```bash
python scripts/test_loader.py
```

Expected output:

- Batch tensor shape: `(batch_size, 3, 224, 224)`
- Corresponding numeric label tensor

---

## How to Train the Baseline Model

```bash
python scripts/train_baseline.py --epochs 10 --batch 4
```

Outputs:

- Trained model saved as: `results/resnet18_final.pt`
- Training loss plot: `results/loss_curve.png`

---

## How to Generate Grad-CAM Visualizations

```bash
python scripts/gradcam.py
```

---

## How to Compute Design Metrics

```bash
python scripts/design_metrics.py
```

This generates:

- `results/design_scores.csv`

---

## How to Predict and Fuse Final Scores

```bash
python scripts/predict_and_fuse.py
```

This generates:

- `results/final_scores.csv`

---

## Demo Pipeline

A Jupyter notebook (/demo_pipeline.ipynb) is included to demonstrate the full flow: load a sample poster → run the trained CNN → compute design metrics → fuse scores → display Grad-CAM and loss curve.

Run Demo

```bash
jupyter notebook demo_pipeline.ipynb
```

Then click Run All to generate the outputs.

---

## Future Work

- Scaling the poster dataset to 500+ images
- Proper quantitative evaluation (MSE, MAE, Pearson correlation)
- Ablation studies (CNN-only vs Design-only vs Fused)
- Advanced explainability and failure-case analysis
- Optional web deployment for live poster evaluation

---
