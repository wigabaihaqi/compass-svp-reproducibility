# COMPASS-SVP: Reproducibility Package

Code, connectivity features, and per-subject outputs for the paper

> W. M. Baihaqi, I. Hidayah, S. Kusrohmaniah, and N. A. Setiawan, "COMPASS-SVP: A Support Vector Prototype
> Framework via Functional Connectivity for Cross-Subject EEG Learning Style Identification",
> *International Journal of Intelligent Engineering and Systems* (under review).

COMPASS-SVP classifies Felder-Silverman learning styles (Verbal-Visual and Active-Reflective) from 8-channel
EEG Phase Locking Value (PLV) connectivity under strict Leave-One-Subject-Out (LOSO) validation:

1. **Support-vector sifting** - an RBF-SVM (C = 1, gamma = `scale`, balanced class weights) is fitted on the
   training subjects; only its support vectors (alpha_i > 0) are kept.
2. **Intra-class condensation** - K-Means with K = 2 per class turns the support vectors into four macro-anchors.
3. **Geometric inference** - each 1-s window of the held-out subject takes the label of its nearest
   macro-anchor (Euclidean or Cosine distance).
4. **Hierarchical majority voting** - windows -> trials (15 windows) -> subject (20 trials).

Every script reproduces one part of the manuscript, saves all per-subject predictions, and can be audited
independently.

---

## Repository structure

```
compass-svp-reproducibility/
├── README.md
├── LICENSE
├── requirements.txt
├── run_all.sh                          # steps 01-07 for both dimensions, in dependency order
├── .gitignore                          # excludes raw recordings and segmented EEG
│
├── data/                               # ALL input files (details: data/README.md)
│   ├── README.md
│   ├── subjects_manifest_template.csv  # template for step 00a (raw file + stimulus onset per subject)
│   ├── combined_fe_plv_vv.csv          # PLV features, Verbal-Visual        (released)
│   ├── combined_fe_plv_ar.csv          # PLV features, Active-Reflective    (released)
│   ├── combined_fe_wpli_vv.csv         # wPLI features, Verbal-Visual       (released)
│   ├── combined_fe_wpli_ar.csv         # wPLI features, Active-Reflective   (released)
│   ├── raw_source/                     # raw OpenBCI recordings             (not redistributed)
│   ├── raw_eeg_vv/                     # filtered 1-s EEG windows, VV       (not redistributed)
│   └── raw_eeg_ar/                     # filtered 1-s EEG windows, AR       (not redistributed)
│
├── scripts/
│   ├── common.py                       # shared paths, constants, normalization, voting, metrics
│   ├── 00a_segment_eeg.py              # band-pass filtering and 1-s windowing of raw recordings
│   ├── 00b_extract_connectivity.py     # PLV / wPLI, 28 upper-triangular pairs -> combined_fe_*.csv
│   ├── 01_wcss_profiles.py             # within-fold WCSS profiles used to fix K = 2
│   ├── 02_sota_anova_ml.py             # ANOVA + RF / KNN / MLP, adapted from Yuvaraj et al. [12]
│   ├── 03_sota_mil_rf.py               # MIL-RF, adapted from Wijaya et al. [8]
│   ├── 04_sota_lstm_cnn.py             # LSTM-CNN, adapted from Jawed et al. [11]
│   ├── 05_compass_svp_benchmark.py     # COMPASS-SVP + baselines + subject-level statistics
│   ├── 06_compass_svp_ablation.py      # component ablation (PLV) and wPLI control analysis
│   ├── 07_compass_svp_geometry.py      # t-SNE visualization of the fitted model
│   ├── 08_kmeans_robustness.py         # COMPASS-SVP over 20 K-Means seeds (initialization robustness)
│   └── 09_fusion_levels_figure.py      # supplementary figure: window / trial / subject accuracy (values in Table 1)
│
├── results_wcss/                       # output of step 01
├── results_anova_ml/                   # output of step 02
├── results_mil_rf/                     # output of step 03
├── results_lstmcnn/                    # output of step 04
├── results_compass/                    # output of steps 05, 08 and 09
├── results_ablation/                   # output of step 06
└── results_compass-svp_geometry/       # output of step 07
```

## Pipeline

```
raw OpenBCI recordings ──00a──> filtered 1-s windows ──00b──> PLV / wPLI features
                                        │                            │
                                        └──04 (LSTM-CNN)             ├──01  WCSS profiles
                                                                     ├──02  ANOVA + ML [12]
                                                                     ├──03  MIL-RF [8]
                                                                     ├──05  COMPASS-SVP + baselines + statistics (uses 02-04)
                                                                     ├──06  ablation (PLV) / control analysis (wPLI)
                                                                     ├──07  t-SNE geometry
                                                                     ├──08  robustness to K-Means initialization
                                                                     └──09  supplementary fusion-level figure
```

Steps 00a, 00b and 04 need the (non-redistributed) EEG recordings. Every other result can be reproduced
from the released feature files in `data/`; the LSTM-CNN can be audited from its per-subject predictions
in `results_lstmcnn/`.

## Manuscript items and where they come from

| Manuscript item | Script | Output files |
|---|---|---|
| Section 2.1.3 (filtering, 1-s windows) | `00a_segment_eeg.py` | `data/raw_eeg_<dim>/*.csv`, `data/segmentation_report.csv` |
| Section 2.2 (PLV, 28 features); Table 4 input (wPLI) | `00b_extract_connectivity.py` | `data/combined_fe_<plv\|wpli>_<dim>.csv` |
| Section 3.2, Fig. 3 (WCSS profiles, K = 2; VV top, AR bottom) | `01_wcss_profiles.py` | `results_wcss/<dim>_wcss_per_fold.csv`, `<dim>_wcss_summary.csv`, `<dim>_wcss_convergence.png/.pdf`; the two panels are stacked in `Figure3_WCSS_VV_AR_combined.png` |
| Table 1 (benchmark, window -> trial -> subject levels; Section 3.10) | `05_compass_svp_benchmark.py` | `results_compass/<dim>_benchmark_summary.csv` |
| Section 3.6 (robustness to K-Means initialization, 20 seeds) | `08_kmeans_robustness.py` | `results_compass/kmeans_initialization_robustness.csv`, `kmeans_initialization_robustness_summary.csv` |
| Supplementary figure of the window / trial / subject accuracy (not in the manuscript; values in Table 1) | `09_fusion_levels_figure.py` | `results_compass/fig6_hierarchical_fusion_levels.png/.pdf/.csv` |
| Section 3.6 (exact CI, McNemar, permutation tests) | `05_compass_svp_benchmark.py` | `results_compass/<dim>_subject_level_statistics.csv`, `<dim>_all_models_subject_accuracy_ci.csv` |
| Table 2 (component ablation, VV and AR) | `06_compass_svp_ablation.py --feature plv` | `results_ablation/<dim>_plv_ablation_summary.csv`, `<dim>_plv_ablation_statistics.csv` |
| Table 3 (state-of-the-art comparison, VV and AR) | `02`, `03`, `04`, then `05` | summaries in `results_anova_ml/`, `results_mil_rf/`, `results_lstmcnn/`; tests in `results_compass/` |
| Table 4 (wPLI control analysis) | `06_compass_svp_ablation.py --feature wpli` | `results_ablation/<dim>_wpli_ablation_summary.csv` |
| Fig. 4 (decision geometry; VV left, AR right) | `07_compass_svp_geometry.py` | `results_compass-svp_geometry/<dim>_tsne_geometry.png/.pdf`; the two panels are placed side by side in `Figure4_tSNE_VV_AR_combined.png` |

`<dim>` is `vv` (Verbal-Visual, n = 10 subjects) or `ar` (Active-Reflective, n = 18 subjects).

### Data correction (Active-Reflective, subject 8)

The window file of Active subject 8 (`data/raw_eeg_ar/active-subject8-preprocessed.csv`) originally
contained, after the subject's own 300 windows, a copy of the recording of subject 9 and a second copy of
its own windows. Feature extraction therefore assigned the subject-9 windows to subject 8, so that the
earlier AR feature files contained subject 9 twice and lacked subject 8. The window file was reduced to the
subject's own 300 windows, the AR features were re-extracted with `00b_extract_connectivity.py`, and all AR
results in the `results_*` folders were regenerated; the VV files are unaffected. In the released
`combined_fe_<plv|wpli>_ar.csv`, subject 8 has the trial indices 141-160 and subject 9 has 161-180.

### Audit files (per-subject predictions, confusion matrices, folds, choice of K)

Every reported number can be recomputed from the discrete LOSO predictions below.

| What | Where |
|---|---|
| Per-subject predictions of every model | `results_*/<dim>_<model>_subject_predictions.csv` (`subject`, `true_class`, `pred_subject`, `trials_class1`, `tie`, `correct`); LSTM-CNN: `results_lstmcnn/lstmcnn_<dim>_subject_predictions.csv` |
| Per-trial and per-window predictions | `..._trial_predictions.csv`, `..._window_predictions.csv` (MIL-RF: `..._instance_predictions.csv`) |
| Subject-level confusion matrix of every configuration | columns `TN`, `FP`, `FN`, `TP` in `results_compass/<dim>_benchmark_summary.csv`, `results_ablation/<dim>_<feature>_ablation_summary.csv`, `results_anova_ml/<dim>_anova_ml_summary.csv`, `results_mil_rf/<dim>_milrf_summary.csv`, `results_lstmcnn/lstmcnn_<dim>_summary.csv` |
| LOSO fold identifiers (fold number -> held-out subject) | `results_compass/<dim>_compass_fold_log.csv`, `results_ablation/<dim>_<feature>_fold_log.csv`, `results_anova_ml/<dim>_anova_selected_features_per_fold.csv`; the `fold` column of the window-prediction files |
| Paired 2x2 tables and p-values | `results_compass/<dim>_subject_level_statistics.csv`, `results_ablation/<dim>_<feature>_ablation_statistics.csv` |
| Outputs used to fix K | `results_wcss/<dim>_wcss_per_fold.csv`, `<dim>_wcss_summary.csv`, `<dim>_wcss_convergence.png` |
| Distance metric | not selected: Euclidean and Cosine are reported side by side in every table |

Each table and figure of the manuscript is produced by the script listed in the table above; the scripts
replace the Colab notebook cells used during development, so no notebook is needed to regenerate them.

---

## Installation

```bash
git clone https://github.com/wigabaihaqi/compass-svp-reproducibility.git
cd compass-svp-reproducibility
pip install -r requirements.txt
```

The versions in `requirements.txt` are those of the final run (Google Colab, Python 3.13.16).
TensorFlow is only needed for `04_sota_lstm_cnn.py`.

## Running

### From the released features (steps 01-09)

```bash
bash run_all.sh
```

or one step at a time:

```bash
python scripts/01_wcss_profiles.py         --dimension VV
python scripts/02_sota_anova_ml.py         --dimension VV
python scripts/03_sota_mil_rf.py           --dimension VV
python scripts/04_sota_lstm_cnn.py         --dimension VV      # needs data/raw_eeg_vv/
python scripts/05_compass_svp_benchmark.py --dimension VV
python scripts/06_compass_svp_ablation.py  --dimension VV --feature plv
python scripts/06_compass_svp_ablation.py  --dimension VV --feature wpli
python scripts/07_compass_svp_geometry.py  --dimension VV
python scripts/08_kmeans_robustness.py                       # both dimensions, 20 seeds
python scripts/09_fusion_levels_figure.py                    # supplementary fusion-level figure (after step 05 for VV and AR)
```

Replace `VV` by `AR` for the Active-Reflective dimension. Steps 02-04 must be run before step 05, which loads
their subject-level predictions for the paired tests (missing files are skipped with a message). Every script
prints its options with `--help`.

The `results_*` folders already contain the outputs reported in the manuscript. With the package versions
in `requirements.txt`, steps 01-03 and 05-09 reproduce them exactly (all random seeds are fixed to 42).
The LSTM-CNN (step 04) and the t-SNE layout of step 07 can differ slightly between machines, because
TensorFlow training and floating-point summation order are not fully deterministic across hardware.

### From the raw recordings (steps 00a-00b)

```bash
cp data/subjects_manifest_template.csv data/subjects_manifest.csv   # then add one row per subject
python scripts/00a_segment_eeg.py                                    # -> data/raw_eeg_vv/, data/raw_eeg_ar/
python scripts/00b_extract_connectivity.py --dimension VV --feature plv
python scripts/00b_extract_connectivity.py --dimension VV --feature wpli
```

`00b_extract_connectivity.py --verify <existing.csv>` compares newly extracted features with an existing
feature file and reports the largest difference and the subject-ID correspondence.

### Paths

All paths are defined once in `scripts/common.py` and can be overridden with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `COMPASS_DATA_DIR` | `data/` | folder with all input files |
| `COMPASS_RESULTS_DIR` | repository root | folder in which the `results_*` folders are created |
| `COMPASS_RAW_SOURCE_DIR` | `data/raw_source/` | raw OpenBCI recordings (step 00a only) |

### Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/wigabaihaqi/compass-svp-reproducibility.git
%cd compass-svp-reproducibility
!pip install -q -r requirements.txt

import os  # optional: keep inputs and results on Google Drive
os.environ['COMPASS_DATA_DIR'] = '/content/drive/MyDrive/<your-folder>/data'
os.environ['COMPASS_RESULTS_DIR'] = '/content/drive/MyDrive/<your-folder>'

!python scripts/05_compass_svp_benchmark.py --dimension VV
```

---

## Evaluation protocol (identical for all models)

| Setting | Value |
|---|---|
| Validation | strict Leave-One-Subject-Out; the held-out subject is never used for training or for choosing K |
| Normalization | subject-wise z-score of each feature (each channel for the LSTM-CNN) using only that subject's own unlabeled data; applied identically to training and held-out subjects |
| Random seed | 42 (LSTM-CNN: 42 + fold index) |
| Voting | window -> trial (15 windows, odd, so no ties) -> subject (20 trials); a tied vote is resolved to Class 1 (Visual / Reflective) |
| Number of prototypes | K = 2 per class, fixed for all folds (Section 3.2) |
| K-Means initialization | k-means++ with 10 restarts (`n_init=10`, `random_state=42`); results are invariant to the row order of the data |
| Distance metrics | Euclidean and Cosine are both reported as separate, complete pipelines |
| Metrics | accuracy; macro-averaged precision, recall and F1 (unweighted mean over both classes) |
| Uncertainty | Clopper-Pearson exact 95% CI on the number of correctly classified subjects |
| Paired tests | exact McNemar and fully enumerated paired permutation test on subject-level correctness, Holm-corrected |

Class 0 = Verbal / Active, Class 1 = Visual / Reflective.

### Signal processing and features

| Step | Setting |
|---|---|
| Acquisition | OpenBCI Cyton, 250 Hz, Fp1, Fp2, F3, F4, C3, C4, O1, O2, reference A1 |
| Filtering | 4th-order Butterworth band-pass 8-30 Hz, zero-phase (`filtfilt`), whole recording |
| Windowing | 10 RAPM questions x 2 phases x 15 consecutive 1-s windows = 300 windows per subject |
| Phase | Hilbert transform per 1-s window |
| PLV | \|(1/M) sum exp(j(phi_x - phi_y))\| |
| wPLI | \|sum Im(S_xy)\| / sum \|Im(S_xy)\|, with S_xy = z_x conj(z_y) (Vinck et al., 2011) |
| Features | upper triangle of the 8 x 8 matrix (i < j): 8 x 7 / 2 = 28 values per window |

### Model settings

| Model | Settings |
|---|---|
| COMPASS-SVP | RBF-SVM (C = 1.0, gamma = `scale`, balanced class weights) -> K-Means (K = 2, 10 restarts, `n_init=10`) -> nearest prototype |
| SVM baseline | the same RBF-SVM used directly as a classifier |
| Random Forest baseline | 100 trees, Gini criterion, unconstrained depth |
| MLP baseline | one hidden layer of 100 ReLU units, Adam optimizer, up to 300 iterations |
| KNN baseline | k = 5, Euclidean distance, uniform weights |
| ANOVA + RF/KNN/MLP [12] | ANOVA F-value selection of the top 30% of features, fitted on training subjects only |
| MIL-RF [8] | instance = one channel pair (15-window PLV time course), bag = trial; RF with 10 trees, max depth 16 |
| LSTM-CNN [11] | LSTM(400) -> Conv1D(256, k = 3) -> Dropout(0.2) -> MaxPool(3) -> Dense(2); Adam, lr 0.001, 15 epochs, batch 16 |

Reference numbers follow the revised manuscript. The model labels printed by the scripts and stored in the
result CSV files keep the numbers of the first submission: "[15]" = Yuvaraj et al. [12], "[11]" = Wijaya et al. [8],
and "[14]" = Jawed et al. [11].

Adaptations of published methods: [12] originally used time-domain descriptors and [8] adaptive-DWT
features; here both operate on the same PLV features as COMPASS-SVP. The post-hoc "inverted relabeling" of
[8] is not reproduced.

---

## Output files

Every model writes three prediction files plus a summary:

| File | Content |
|---|---|
| `*_window_predictions.csv` (`*_instance_predictions.csv` for MIL-RF) | `subject, trial, y, y_pred, fold` for every window / instance |
| `*_trial_predictions.csv` | trial-level vote counts, predicted class, tie flag |
| `*_subject_predictions.csv` | `subject, true_class, pred_subject, pred_subject_flat, trials_class1, n_trials, tie, correct` |
| `*_summary.csv` | accuracy at every level, macro precision / recall / F1, exact 95% CI, confusion matrix (TN, FP, FN, TP) |

`pred_subject` is the hierarchical (window -> trial -> subject) decision; `pred_subject_flat` is the flat
(window -> subject) decision used in the voting ablation. The `fold` column identifies the LOSO fold of every
prediction (the fold of a subject is the one in which it is held out).

---

## Data availability

The PLV and wPLI feature matrices in `data/` are sufficient to reproduce every result except the LSTM-CNN,
which requires the filtered EEG windows. The raw EEG recordings are part of an ongoing data-publication effort
and are available from the corresponding author upon reasonable request, subject to the ethical approval of the
study (Universitas Indonesia Maju, No. 2649/Sket/Ka-Dept/RE/UIMA/VI/2025). The per-subject LSTM-CNN predictions
are provided in `results_lstmcnn/`.

## Citation

```bibtex
@article{baihaqi_compass_svp,
  author  = {Baihaqi, Wiga Maulana and Hidayah, Indriana and Kusrohmaniah, Sri and Setiawan, Noor Akhmad},
  title   = {{COMPASS-SVP}: A Support Vector Prototype Framework via Functional Connectivity for
             Cross-Subject {EEG} Learning Style Identification},
  journal = {International Journal of Intelligent Engineering and Systems},
  note    = {under review}
}
```

## License

MIT License (see `LICENSE`).
