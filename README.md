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
├── gabungan-FE-PLV-VV.csv        # Processed feature dataset for Verbal-Visual dimension
├── gabungan-FE-PLV-AR.csv        # Processed feature dataset for Active-Reflective dimension
├── verbal_visual_evaluation.ipynb  # Complete evaluation notebook for Verbal-Visual dimension
├── active_reflective_evaluation.ipynb # Complete evaluation notebook for Active-Reflective dimension
└── README.md                     # Project documentation
