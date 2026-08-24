# Transformer Condition Monitoring using Infrared Thermography

A deep learning and machine learning framework for detecting transformer faults from infrared thermographic (IRT) images. This repository implements and compares five classical ML classifiers (SVM, KNN, DT, LR, LS-SVM) and four CNN architectures (InceptionV3, MobileNetV2, DenseNet121, and a proposed **Modified VGG-16**) for classifying transformers as **Healthy** or **Defect (Remarkable Temperature Rise)**. The Modified VGG-16 model achieves **99% accuracy**, outperforming all baseline approaches.

> Official implementation accompanying the paper: *"A novel infrared thermography image analysis for transformer condition monitoring"*, published in **e-Prime – Advances in Electrical Engineering, Electronics and Energy**, 2024.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Approach 1: Machine Learning (MLA)](#approach-1-machine-learning-mla)
  - [Approach 2: Deep Learning (DLA)](#approach-2-deep-learning-dla)
  - [Proposed Modified VGG-16 Architecture](#proposed-modified-vgg-16-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## Overview

Electrical transformers are critical assets in power transmission and distribution networks, and unplanned failures can result in costly downtime and safety hazards. Traditional condition-monitoring techniques — such as Dissolved Gas Analysis (DGA), Low Voltage Impulse (LVI), and Frequency Response Analysis (FRA) — are often offline, time-consuming, and require power outages.

This project adopts **Infrared Thermography (IRT)** as a non-contact, non-intrusive alternative for real-time transformer condition monitoring. Thermal images were captured using a Fluke Ti480 PRO infrared camera at a 250 KVA distribution transformer over a two-month period, covering both healthy operating conditions and conditions with remarkable temperature rise. These images form the basis for two complementary detection pipelines: a classical machine learning pipeline using PCA-based feature extraction, and a deep learning pipeline based on transfer learning with pretrained CNNs.

## Key Features

- 📷 Real-world thermal image dataset of a 250 KVA distribution transformer (healthy vs. faulty states)
- 🧠 Two complete pipelines: classical ML (with PCA feature extraction) and deep learning (transfer learning)
- 🏆 A **Modified VGG-16** architecture with partial fine-tuning, dropout, and batch normalization, achieving state-of-the-art accuracy
- 📊 Comprehensive evaluation: accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
- ⚙️ Optimizer comparison (Adam, SGD, RMSProp, Adadelta) across all deep learning models
- 🔁 Reproducible data augmentation pipeline to address limited dataset size

## Repository Structure

```
Transformer-Condition-Monitoring/
├── code/          # Source code for MLA and DLA pipelines
├── README.md      # Project documentation (this file)
├── .gitignore     # Ignored files and directories
└── LICENSE        # License information
```

## Dataset

Thermal images were collected from a distribution transformer located near Bhubaneswar, Odisha, India, using a **Fluke Ti480 PRO** infrared camera over a two-month period, at fixed intervals between 5:30 AM and 7:00 PM, at ambient temperatures ranging from 25°C to 37°C.

**Transformer Specifications**

| Parameter | Value |
|---|---|
| Phase | 3 |
| Power | 250 KVA |
| Voltage | 11,000 / 430 V |
| Input Current | 13.12 A |
| Frequency | 50–60 Hz |

**Thermal Camera Specifications**

| Parameter | Value |
|---|---|
| Model | Fluke Ti480 PRO Infrared Camera |
| Detector Resolution | 640 × 480 |
| Measurement Accuracy | ±2°C (at 26°C nominal) |
| Thermal Sensitivity | ≤ 0.05°C at 30°C target temp (50 mK) |
| Temperature Range | ≤ −20°C to 1000°C |
| Frame Rate | 50/60 Hz |

**Original Dataset**

| Class | Total Images | Train | Test |
|---|---|---|---|
| Healthy | 250 | 200 | 50 |
| Remarkable Temperature Rise | 250 | 200 | 50 |

**Augmented Dataset** (rotation, rescaling, horizontal flip, width/height shift)

| Class | Total Images | Train | Test |
|---|---|---|---|
| Healthy | 500 | 400 | 100 |
| Remarkable Temperature Rise | 500 | 400 | 100 |

> **Note:** Due to confidentiality and file size constraints, the raw thermal image dataset is not included in this repository. It is available from the corresponding author upon reasonable request.

## Methodology

### Approach 1: Machine Learning (MLA)

Thermal images are resized to 224×224 pixels, augmented, and dimensionality-reduced using **Principal Component Analysis (PCA)** before being passed to five classical classifiers:

- Support Vector Machine (SVM)
- K-Nearest Neighbours (KNN)
- Decision Tree (DT)
- Logistic Regression (LR)
- Least Squares Support Vector Machine (LS-SVM)

```
Dataset → Pre-processing (Augmentation) → Feature Extraction (PCA) → ML Classifiers → Classification (Good / Defect)
```

### Approach 2: Deep Learning (DLA)

Four transfer-learning-based CNN architectures, pretrained on ImageNet, are fine-tuned for binary classification:

- InceptionV3
- MobileNetV2
- DenseNet121
- **Modified VGG-16 (proposed)**

```
Dataset → Normalization + Augmentation → Pretrained CNN Backbone → FC Layer with Softmax → Classification (Good / Defect)
```

### Proposed Modified VGG-16 Architecture

The proposed model builds on the standard VGG-16 backbone (16 layers, ~138M parameters, 5 convolutional blocks) with the following modifications:

- The first **three convolutional blocks retain pretrained ImageNet weights** (frozen)
- The **last two blocks are fine-tuned** via transfer learning on the thermal image dataset
- **Dropout layers (rate = 0.5)** inserted after dense layers to reduce overfitting
- **Batch normalization layers** for stable, faster convergence and implicit regularization
- Dense layers (256 → 128 units, ReLU) followed by a final Softmax output layer for classification

Training configuration: **30 epochs**, **Adam optimizer**, **learning rate = 0.01**, **batch size = 8**, input size **224×224**.

## Results

### Machine Learning Approach — Classification Performance

| Classifier | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| SVM | 0.88 | 0.95 | 0.91 | 91.25% |
| KNN | 0.98 | 0.93 | 0.95 | 95.00% |
| DT | 0.89 | 0.89 | 0.89 | 90.00% |
| LR | 0.93 | 0.91 | 0.92 | 91.25% |
| LS-SVM | 0.94 | 0.89 | 0.92 | 92.50% |

### Deep Learning Approach — Model & Optimizer Comparison

| Model | Optimizer | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| InceptionV3 | Adam | 95% | 0.94 | 0.96 | 0.95 |
| MobileNetV2 | Adam | 97% | 1.00 | 0.94 | 0.97 |
| DenseNet121 | Adam | 98% | 0.96 | 1.00 | 0.98 |
| **Modified VGG-16** | **Adam** | **99%** | **0.98** | **1.00** | **0.99** |

*(Full results across SGD, RMSProp, and Adadelta optimizers are provided in the paper.)*

### Comparison with Existing Literature

| Reference | Year | Method | Accuracy |
|---|---|---|---|
| [Fang et al.] | 2021 | Semi-supervised learning + GAN | 86.20% |
| [Xudong & Yi] | 2024 | LGC–Net | 98.92% |
| [Jin et al.] | 2023 | Improved BP Neural Network + SVM | 92.00% |
| [Li et al.] | 2023 | CNN + Few-shot learning | 98.90% |
| [Tang & Jian] | 2024 | DeDn-CNN, RetinaNet, DeeplabV3+ | 90.91% |
| **Proposed** | **2024** | **Modified VGG-16** | **99.00%** |

**Key Takeaway:** The Modified VGG-16 model consistently outperforms both classical ML classifiers and competing deep learning architectures, achieving the highest accuracy, precision, recall, and F1-score while maintaining stable convergence and lower training loss.

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Transformer-Condition-Monitoring.git
cd Transformer-Condition-Monitoring

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:** `tensorflow` / `keras`, `scikit-learn`, `numpy`, `pandas`, `opencv-python`, `matplotlib`, `seaborn`

## Usage

```bash
# Run the classical machine learning pipeline (SVM, KNN, DT, LR, LS-SVM)
python code/mla_pipeline.py

# Train and evaluate deep learning models (InceptionV3, MobileNetV2, DenseNet121, Modified VGG-16)
python code/dla_pipeline.py --model modified_vgg16 --optimizer adam --epochs 30
```

> Update dataset paths and hyperparameters in the corresponding config files before running.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{balabantaraya2024novel,
  title={A novel infrared thermography image analysis for transformer condition monitoring},
  author={Balabantaraya, Rupali and Sahoo, Ashwin Kumar and Sahoo, Prabodh Kumar and Abir, Chayan Mondal and Panda, Manoj Kumar},
  journal={e-Prime-Advances in Electrical Engineering, Electronics and Energy},
  volume={10},
  pages={100758},
  year={2024},
  publisher={Elsevier}
}
```

📄 **Full paper:** [ScienceDirect – doi.org/10.1016/j.prime.2024.100758](https://www.sciencedirect.com/science/article/pii/S2772671124003383)

## License

This project is licensed under the terms specified in the [LICENSE](./LICENSE) file. The associated paper is published under a **CC BY 4.0** license by Elsevier.

## Contact

**Chayan Mondal Abir**
📧 abirchayan2000@gmail.com

For questions, issues, or collaboration inquiries, please open a GitHub issue or reach out via email.

## Acknowledgments

This work is based on research conducted at the Department of Electrical Engineering, C. V. Raman Global University, Bhubaneswar, Odisha, India, in collaboration with the Department of Mechatronics Engineering, Parul Institute of Technology, Parul University, and the Department of Electronics and Communication Engineering, GIET University, Gunupur, Odisha, India.
