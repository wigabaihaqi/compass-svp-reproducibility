# EEG-Based Learning Style Classification Framework 
## COMPASS-SVP & Comprehensive Machine Learning Benchmarks

An advanced machine learning repository for classifying cognitive learning styles—specifically focusing on the **Verbal-Visual (VV)** and **Active-Reflective (AR)** dimensions—using Electroencephalography (EEG) functional connectivity features derived via **Phase Locking Value (PLV)**.

This repository implements the novel **COMPASS-SVP** framework alongside a comprehensive suite of baseline models, component ablation studies, and literature-adapted architectures (including deep learning models) evaluated under a strict **Leave-One-Subject-Out (LOSO)** cross-validation protocol. All evaluations and experiments for each cognitive dimension are centralized within dedicated Jupyter Notebooks.

---

## Key Features

* **Advanced Prototype-Based Classification (COMPASS-SVP)**: Utilizes Support Vector Machine (SVM) boundary sifting combined with K-Means prototype condensation (macro-anchors) and geometric distance metrics (Euclidean and Cosine) for robust manifold classification.
* **Hierarchical Majority Voting**: Implements a robust multi-tier aggregation pipeline scaling predictions from **Window-Level** to **Trial-Level** and finally to **Subject-Level** to mitigate EEG signal noise.
* **Rigorous Component Ablation Studies**: Evaluates individual components of the proposed system (e.g., removing K-Means, skipping SVM sifting) to prove architectural efficacy.
* **Extensive Literature & Deep Learning Benchmarks**: 
  * Traditional Machine Learning Baselines: SVM, Random Forest (RF), Multilayer Perceptron (MLP), and K-Nearest Neighbors (KNN).
  * Adapted Deep Learning Benchmarks: Custom CNN-LSTM architecture (adapted from Jawed et al., 2024).
  * Multiple Instance Learning (MIL) Benchmark: MIL-RF framework (inspired by Wijaya et al.).
  * Statistical Feature Selection: ANOVA-based feature selection (SelectPercentile).

---

## Repository Structure


```text
├── gabungan-FE-PLV-VV.csv              # Processed feature dataset for Verbal-Visual dimension
├── gabungan-FE-PLV-AR.csv              # Processed feature dataset for Active-Reflective dimension
├── verbal_visual_evaluation.ipynb      # Complete evaluation notebook for Verbal-Visual dimension
├── active_reflective_evaluation.ipynb  # Complete evaluation notebook for Active-Reflective dimension
└── README.md                           # Project documentation
```

## Dataset Format

## Dataset Format

The dataset files contain multi-channel EEG connectivity features extracted via Phase Locking Value (PLV).

- **subject**: Unique identifier for each participant.
- **trial**: Experimental session or trial identifier.
- **label**: Target category (verbal vs. visual for VV; aktif vs. reflektif for AR).
- **Feature Columns (-)**: Connectivity strength values across electrode pairs.

## Installation & Prerequisites

Ensure you have Python 3.8+ and Jupyter Notebook installed. The project relies on the following core libraries for scientific computing, machine learning, and deep learning workflows:

```bash
pip install pandas numpy scikit-learn tensorflow scipy matplotlib jupyter
```
## How to Run

* Place your dataset files (`gabungan-FE-PLV-VV.csv` and `gabungan-FE-PLV-AR.csv`) in the root directory.
* Launch Jupyter Notebook from your terminal:

```bash
jupyter notebook
```
* Open either ```verbal_visual_evaluation.ipynb ``` or ```active_reflective_evaluation.ipynb``` to execute all comprehensive benchmarks, ablation studies, and deep learning models interactively.

## Evaluation Protocol
* All experiments follow a strict Leave-One-Subject-Out (LOSO) cross-validation strategy:
* Subject-Wise Standardization: Features are scaled independently per subject using ```StandardScaler``` to remove inter-subject physiological variance.
* Instance Prediction: Models predict labels across temporal sliding windows.
* Hierarchical Aggregation:
  * Windows are aggregated into Trials via majority voting.
  * Trials are aggregated into final Subject-level classifications using strict frequency thresholds and controlled tie-breaking mechanisms.
