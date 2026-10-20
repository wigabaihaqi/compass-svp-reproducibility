"""
Shared configuration and utilities for all COMPASS-SVP scripts (00a-07).

Every script imports this module so that all models use exactly the same
data paths, labels, subject-wise normalization, voting rules, and metrics.

Paths can be overridden with environment variables:
    COMPASS_DATA_DIR     folder that contains all input files   (default: <repo>/data)
    COMPASS_RESULTS_DIR  folder in which the results_* folders are created (default: <repo>)
    COMPASS_RAW_SOURCE_DIR  raw OpenBCI recordings, step 00a only        (default: <data>/raw_source)
"""

import os
import random
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ==========================================
# PATHS
# ==========================================
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('COMPASS_DATA_DIR', os.path.join(REPO_DIR, 'data'))
RESULTS_DIR = os.environ.get('COMPASS_RESULTS_DIR', REPO_DIR)

FEATURE_FILES = {   # connectivity feature matrices (one row per 1-s window)
    ('VV', 'plv'):  'combined_fe_plv_vv.csv',
    ('AR', 'plv'):  'combined_fe_plv_ar.csv',
    ('VV', 'wpli'): 'combined_fe_wpli_vv.csv',
    ('AR', 'wpli'): 'combined_fe_wpli_ar.csv',
}
RAW_EEG_DIRS = {    # band-pass-filtered EEG windows, one CSV per subject (LSTM-CNN input)
    'VV': 'raw_eeg_vv',
    'AR': 'raw_eeg_ar',
}
OUTPUT_DIRS = {
    'wcss':     'results_wcss',
    'anova_ml': 'results_anova_ml',
    'mil_rf':   'results_mil_rf',
    'lstmcnn':  'results_lstmcnn',
    'compass':  'results_compass',
    'ablation': 'results_ablation',
    'geometry': 'results_compass-svp_geometry',
}

# ==========================================
# EXPERIMENTAL CONSTANTS (Section 2.6 of the manuscript)
# ==========================================
SEED = 42
K_PROTOTYPES = 2            # macro-anchors per class (fixed for all folds; see Section 3.2)
KMEANS_N_INIT = 10          # K-Means restarts; a single start ('auto') makes the prototypes depend on row order
WINDOWS_PER_TRIAL = 15      # 15 x 1-s windows per 15-s trial (odd -> no window-to-trial ties)
TRIALS_PER_SUBJECT = 20     # 10 RAPM questions x 2 phases
TIE_CLASS = 1               # any tied vote (e.g., 10-10 trials) is resolved to Class 1

CLASS_NAMES = {'VV': ('verbal', 'visual'), 'AR': ('active', 'reflective')}
LABEL_TO_INDEX = {          # accepted label spellings (English and Indonesian)
    'VV': {'verbal': 0, 'visual': 1},
    'AR': {'active': 0, 'aktif': 0, 'reflective': 1, 'reflektif': 1},
}
DIMENSION_NAMES = {'VV': 'Verbal-Visual', 'AR': 'Active-Reflective'}


# ==========================================
# GENERAL HELPERS
# ==========================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def output_dir(key):
    path = os.path.join(RESULTS_DIR, OUTPUT_DIRS[key])
    os.makedirs(path, exist_ok=True)
    return path


def clean_subject_id(subj):
    """Normalize subject IDs so that 'S01', 'subject_1', 1, 1.0 and '01' all become '1'."""
    s = str(subj).strip()
    try:
        return str(int(float(s)))
    except ValueError:
        digits = re.findall(r'\d+', s)
        return str(int(digits[0])) if digits else s.lower()


def subject_sort_key(s):
    s = str(s)
    return (0, int(s)) if s.isdigit() else (1, s)


def safe_name(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def encode_labels(labels, dimension):
    mapping = LABEL_TO_INDEX[dimension]
    lab = pd.Series(labels).astype(str).str.strip().str.lower()
    y = lab.map(mapping)
    if y.isna().any():
        raise ValueError(f"Unexpected class labels {sorted(lab[y.isna()].unique())}; "
                         f"expected one of {list(mapping)}.")
    return y.astype(int).values


# ==========================================
# DATA LOADING
# ==========================================
def load_features(dimension, feature='plv', zscore=True):
    """Load a connectivity feature file and apply subject-wise z-score normalization.

    Subject-wise z-scoring uses only each subject's own (unlabeled) data, so applying
    it once to the whole file is identical to re-applying it inside every LOSO fold.
    Returns (df, feature_cols, subjects); df has columns subject, trial, y, <features>.
    """
    path = os.path.join(DATA_DIR, FEATURE_FILES[(dimension, feature)])
    if not os.path.isfile(path):
        kind = 'a directory, not a CSV file' if os.path.isdir(path) else 'not found'
        raise FileNotFoundError(f"Input file is {kind}: {path}")

    df = pd.read_csv(path)
    for col in ('subject', 'trial', 'label'):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in {path}")
    df['subject'] = df['subject'].apply(clean_subject_id)
    df['y'] = encode_labels(df['label'], dimension)
    feature_cols = [c for c in df.columns if '-' in c]
    if not feature_cols:
        raise ValueError(f"No channel-pair feature columns (e.g. 'Fp1-Fp2') found in {path}")

    order = ['subject', 'trial'] + (['window'] if 'window' in df.columns else [])
    df = df.sort_values(order, kind='stable').reset_index(drop=True)
    subjects = sorted(df['subject'].unique(), key=subject_sort_key)

    if zscore:
        for s in subjects:
            m = df['subject'] == s
            df.loc[m, feature_cols] = StandardScaler().fit_transform(df.loc[m, feature_cols])
    return df, feature_cols, subjects


# ==========================================
# HIERARCHICAL MAJORITY VOTING
# ==========================================
def majority_vote(preds, tie_class=TIE_CLASS):
    preds = np.asarray(preds)
    n1 = int((preds == 1).sum())
    n0 = len(preds) - n1
    if n1 == n0:
        return tie_class
    return 1 if n1 > n0 else 0


def aggregate(units):
    """Hierarchical (unit -> trial -> subject) and flat (unit -> subject) voting.

    `units` must contain the columns subject, trial, y, y_pred, where a unit is a
    window (or a MIL instance). Returns (trials_df, subjects_df).
    """
    trials = (units.groupby(['subject', 'trial'], sort=True)
              .agg(true_class=('y', 'first'),
                   pred_trial=('y_pred', majority_vote),
                   n_units=('y_pred', 'size'),
                   units_class1=('y_pred', 'sum'))
              .reset_index())
    trials['tie'] = trials['units_class1'] * 2 == trials['n_units']

    subjects = (trials.groupby('subject', sort=False)
                .agg(true_class=('true_class', 'first'),
                     pred_subject=('pred_trial', majority_vote),
                     n_trials=('pred_trial', 'size'),
                     trials_class1=('pred_trial', 'sum'))
                .reset_index())
    flat = (units.groupby('subject', sort=False)
            .agg(pred_subject_flat=('y_pred', majority_vote)).reset_index())
    subjects = subjects.merge(flat, on='subject')
    subjects['tie'] = subjects['trials_class1'] * 2 == subjects['n_trials']
    subjects['correct'] = subjects['pred_subject'] == subjects['true_class']
    subjects = subjects.sort_values('subject', key=lambda c: c.map(subject_sort_key)).reset_index(drop=True)
    return trials, subjects


# ==========================================
# METRICS
# ==========================================
def exact_ci(n_correct, n_total):
    ci = binomtest(int(n_correct), n=int(n_total)).proportion_ci(confidence_level=0.95, method='exact')
    return ci.low * 100, ci.high * 100


def summarize(units, trials, subjects, unit_name='Win'):
    """Accuracy at every level plus macro-averaged subject-level metrics and exact 95% CI."""
    yt, yp = subjects['true_class'].values, subjects['pred_subject'].values
    n_correct = int((yt == yp).sum())
    lo, hi = exact_ci(n_correct, len(yt))
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    return {
        f'{unit_name} Acc (%)': round(accuracy_score(units['y'], units['y_pred']) * 100, 2),
        'Trial Acc (%)': round(accuracy_score(trials['true_class'], trials['pred_trial']) * 100, 2),
        'Trial Macro F1 (%)': round(f1_score(trials['true_class'], trials['pred_trial'],
                                             average='macro', zero_division=0) * 100, 2),
        'Subj Acc (%)': round(accuracy_score(yt, yp) * 100, 2),
        'Subj Correct': f'{n_correct}/{len(yt)}',
        '95% CI (%)': f'[{lo:.1f}, {hi:.1f}]',
        'Subj Macro Prec (%)': round(precision_score(yt, yp, average='macro', zero_division=0) * 100, 2),
        'Subj Macro Rec (%)': round(recall_score(yt, yp, average='macro', zero_division=0) * 100, 2),
        'Subj Macro F1 (%)': round(f1_score(yt, yp, average='macro', zero_division=0) * 100, 2),
        'Subj Acc Flat Vote (%)': round(accuracy_score(yt, subjects['pred_subject_flat']) * 100, 2),
        'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]), 'FN': int(cm[1, 0]), 'TP': int(cm[1, 1]),
        'Subject Ties': int(subjects['tie'].sum()),
    }


def print_subject_report(name, dimension, subjects, summary):
    c0, c1 = CLASS_NAMES[dimension]
    print(f"\n--- {name} | subject level ---")
    for _, r in subjects.iterrows():
        print(f"  Subject {r['subject']:>4} | true: {[c0, c1][r['true_class']]:<10} | "
              f"pred: {[c0, c1][r['pred_subject']]:<10} | trials voting {c1}: "
              f"{r['trials_class1']}/{r['n_trials']}{' (TIE)' if r['tie'] else ''} | "
              f"{'CORRECT' if r['correct'] else 'WRONG'}")
    print(f"  Accuracy {summary['Subj Acc (%)']:.2f}% ({summary['Subj Correct']}), 95% CI {summary['95% CI (%)']} | "
          f"Macro P/R/F1 {summary['Subj Macro Prec (%)']:.2f}/{summary['Subj Macro Rec (%)']:.2f}/"
          f"{summary['Subj Macro F1 (%)']:.2f} | CM [[{summary['TN']} {summary['FP']}] "
          f"[{summary['FN']} {summary['TP']}]]")


def export_predictions(out_dir, prefix, units, trials, subjects, unit_name='window'):
    units.to_csv(os.path.join(out_dir, f'{prefix}_{unit_name}_predictions.csv'), index=False)
    trials.to_csv(os.path.join(out_dir, f'{prefix}_trial_predictions.csv'), index=False)
    subjects.to_csv(os.path.join(out_dir, f'{prefix}_subject_predictions.csv'), index=False)
