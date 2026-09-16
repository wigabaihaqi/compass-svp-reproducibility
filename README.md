# EEG-Based Learning Style Classification Framework 
## COMPASS-SVP & Comprehensive Machine Learning Benchmarks

An advanced machine learning repository for classifying cognitive learning styles—specifically focusing on the **Verbal-Visual (VV)** and **Active-Reflective (AR)** dimensions—using Electroencephalography (EEG) functional connectivity features derived via **Phase Locking Value (PLV)** and rigorously validated against zero-lag volume conduction using the **weighted Phase Lag Index (wPLI)**.

This repository implements the novel **COMPASS-SVP** framework alongside a comprehensive suite of baseline models, component ablation studies, and literature-adapted architectures (including deep learning models) evaluated under a strict **Leave-One-Subject-Out (LOSO)** cross-validation protocol. All evaluations, statistical testing, and experiments for each cognitive dimension are centralized within dedicated Jupyter Notebooks.

---

## Key Features

* **Advanced Prototype-Based Classification (COMPASS-SVP)**: Utilizes Support Vector Machine (SVM) boundary sifting combined with K-Means prototype condensation (macro-anchors) and geometric distance metrics (Euclidean and Cosine) for robust manifold classification.
* **Hierarchical Majority Voting**: Implements a robust multi-tier aggregation pipeline scaling predictions from **Window-Level** to **Trial-Level** and finally to **Subject-Level** to mitigate EEG signal noise.
* **Rigorous Statistical Uncertainty Analysis**: Implements exact subject-level statistical hypothesis testing, automatically computing **Clopper-Pearson 95% Exact Confidence Intervals**, **Exact McNemar Tests**, and **Paired Permutation Tests (10,000 iterations)** to transparently quantify performance deltas.
* **wPLI Control Analysis**: Integrates parallel benchmarking using magnitude-weighted Phase Lag Index (wPLI) features to mathematically isolate and validate the framework's robustness against zero-lag spatial leakage and volume conduction artifacts.
* **Rigorous Component Ablation Studies**: Evaluates individual components of the proposed system (e.g., removing K-Means, skipping SVM sifting) to prove architectural efficacy.
* **Extensive Literature & Deep Learning Benchmarks**: 
  * Traditional Machine Learning Baselines: SVM, Random Forest (RF), Multilayer Perceptron (MLP), and K-Nearest Neighbors (KNN).
  * Adapted Deep Learning Benchmarks: Custom CNN-LSTM architecture (adapted from Jawed et al., 2024).
  * Multiple Instance Learning (MIL) Benchmark: MIL-RF framework (inspired by Wijaya et al., 2026).
  * Statistical Feature Selection: ANOVA-based feature selection (SelectPercentile).

---

## Repository Structure

```text
├── combined_fe_plv_vv.csv                 # PLV feature dataset for Verbal-Visual
├── combined_fe_plv_ar.csv                 # PLV feature dataset for Active-Reflective
├── combined_fe_wpli_vv.csv                # wPLI feature dataset for VV (Control Analysis)
├── combined_fe_wpli_ar.csv                # wPLI feature dataset for AR (Control Analysis)
├── COMPASS_SVP_Evaluation_VV.ipynb        # Primary evaluation & statistical notebook for PLV VV
├── COMPASS_SVP_Evaluation_AR.ipynb        # Primary evaluation & statistical notebook for PLV AR
├── COMPASS_SVP_Evaluation_wPLI_VV.ipynb   # Control evaluation & statistical notebook for wPLI VV
├── COMPASS_SVP_Evaluation_wPLI_AR.ipynb   # Control evaluation & statistical notebook for wPLI AR
└── README.md                              # Project documentation
```

## Dataset Format

The dataset files contain multi-channel EEG connectivity features pre-extracted via Phase Locking Value (PLV) and weighted Phase Lag Index (wPLI).

- **subject**: Unique identifier for each participant.
- **trial**: Experimental session or trial identifier.
- **label**: Target category (verbal vs. visual for VV; aktif vs. reflektif for AR).
- **Feature Columns (-)**: Connectivity strength values across electrode pairs.

Preprocessing Pipeline: The evaluation scripts automatically handle cross-subject feature standardization using `StandardScaler` This Z-score normalization is strictly fitted on the training folds and transformed on the test folds during the Leave-One-Subject-Out (LOSO) cross-validation to prevent data leakage.

## Installation & Prerequisites

Ensure you have Python 3.8+ and Jupyter Notebook installed. The project relies on the following core libraries for scientific computing, machine learning, and deep learning workflows:

```bash
pip install pandas numpy scikit-learn tensorflow scipy matplotlib jupyter
```
## How to Run

* Place your dataset files (`combined_fe_plv_*.csv` and `combined_fe_wpli_*.csv`) in the root directory.
* Launch Jupyter Notebook from your terminal:

```bash
jupyter notebook
```
* Open any of the evaluation notebooks (`COMPASS_SVP_Evaluation_*.ipynb`) to execute the comprehensive benchmarks, ablation studies, and automated statistical rigor tests interactively for either PLV or wPLI feature spaces.

## Evaluation Protocol
* All experiments follow a strict Leave-One-Subject-Out (LOSO) cross-validation strategy:
* Subject-Wise Standardization: Features are scaled independently per subject using ```StandardScaler``` to remove inter-subject physiological variance.
* Instance Prediction: Models predict labels across temporal sliding windows.
* Hierarchical Aggregation:
  * Windows are aggregated into Trials via majority voting.
  * Trials are aggregated into final Subject-level classifications using strict frequency thresholds and controlled tie-breaking mechanisms.

## Note on Pretrained Weights & Reproducibility
Unlike deep neural networks, COMPASS-SVP is a deterministic geometric machine learning framework (SVM optimization + K-Means condensation). Model optimization occurs analytically and instantaneously during each LOSO fold. Consequently, loading external "pretrained model weights" (e.g., `.h5` or `.pth` files) is structurally inapplicable.

To guarantee exact reproducibility, the provided scripts enforce fixed random seeds (`seed=42`). Running the evaluation notebooks will instantaneously recompute the support vectors, extract the macroscopic anchors, and reproduce the exact classification metrics and statistical p-values reported in the manuscript.
