"""
Step 3 - Multi-Instance Learning + Random Forest (MIL-RF), adapted from Wijaya et al. (2026) [8].

Original formulation in [8]: instance = one EEG channel within a trial (features over its
15 windows), bag = trial (8 instances), Random Forest (10 trees, max_depth = 16),
majority vote instance -> trial -> subject.

Adaptation in the shared PLV space (--mode pair, default): instance = one channel PAIR within
a trial, described by its 15-window PLV time course; bag = trial (28 instances).
--mode window (reference only): instance = one 1-s window (equivalent to a depth-limited RF).

Not reproduced: adaptive-DWT features and the post-hoc "inverted relabeling" of [8].

Usage:  python scripts/03_sota_mil_rf.py --dimension VV [--mode pair|window]
Output: results_mil_rf/
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import common as C

warnings.filterwarnings('ignore')
RF_PARAMS = dict(n_estimators=10, max_depth=16)   # as reported in [8]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dimension', choices=['VV', 'AR'], required=True)
parser.add_argument('--mode', choices=['pair', 'window'], default='pair')
args = parser.parse_args()

dim, mode = args.dimension, args.mode
out_dir = C.output_dir('mil_rf')
df, feature_cols, subjects = C.load_features(dim, 'plv')


def build_instances(part):
    X, rows = [], []
    for (subj, trial), g in part.groupby(['subject', 'trial'], sort=True):
        W = g[feature_cols].values                     # (windows, 28)
        label = int(g['y'].iloc[0])
        if mode == 'pair':
            if W.shape[0] != C.WINDOWS_PER_TRIAL:
                raise ValueError(f"Subject {subj}, trial {trial}: expected {C.WINDOWS_PER_TRIAL} windows, "
                                 f"found {W.shape[0]}.")
            for j, pair in enumerate(feature_cols):     # instance = PLV time course of one pair
                X.append(W[:, j])
                rows.append((subj, trial, pair, label))
        else:
            for w in range(W.shape[0]):                 # instance = one window
                X.append(W[w])
                rows.append((subj, trial, f'window_{w + 1}', label))
    return np.asarray(X), pd.DataFrame(rows, columns=['subject', 'trial', 'instance', 'y'])


print(f"MIL-RF [11] | {C.DIMENSION_NAMES[dim]} | instance = {mode} | bag = trial | RF {RF_PARAMS}")

all_units = []
for fold, test_subj in enumerate(subjects, start=1):
    C.set_seed()
    X_tr, meta_tr = build_instances(df[df['subject'] != test_subj])
    X_te, meta_te = build_instances(df[df['subject'] == test_subj])
    clf = RandomForestClassifier(**RF_PARAMS, random_state=C.SEED).fit(X_tr, meta_tr['y'].values)
    all_units.append(meta_te.assign(y_pred=clf.predict(X_te), fold=fold))
    print(f"  Fold {fold:>2}/{len(subjects)} | held-out subject {test_subj}")

units = pd.concat(all_units, ignore_index=True)
trials, subjects_df = C.aggregate(units)
summary = C.summarize(units, trials, subjects_df, unit_name='Instance')
C.print_subject_report('MIL-RF [11]', dim, subjects_df, summary)
print(f"  Tied trials (instance vote): {int(trials['tie'].sum())}")

prefix = f"{dim.lower()}_milrf" + ('' if mode == 'pair' else '_windowmode')
C.export_predictions(out_dir, prefix, units, trials, subjects_df, unit_name='instance')
pd.DataFrame([{'model': 'MIL-RF [11]', 'mode': mode, **summary}]).to_csv(
    os.path.join(out_dir, f'{prefix}_summary.csv'), index=False)
print(f"\nSaved to {out_dir}/")
