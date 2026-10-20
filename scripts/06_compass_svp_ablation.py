"""
Step 6 - Component-wise ablation of COMPASS-SVP (Table 2) and wPLI control analysis (Table 4).

Configurations (each with Euclidean and Cosine distance):
  Full System                         SVM sifting + K-Means (K = 2)
  Scenario A (w/o K-Means)            SVM sifting, every support vector is a prototype
  Scenario B (w/o SVM Sifting)        K-Means (K = 2) on all training windows
  Scenario C (Single Prototype, K=1)  SVM sifting + one prototype per class
  SVM (Baseline)                      the sifting SVM used directly as classifier
Voting ablation: every configuration reports window accuracy (no voting), hierarchical
subject accuracy (window -> trial -> subject) and flat subject accuracy (window -> subject).
Paired exact McNemar tests (Holm-corrected): Full System vs. each variant with the same metric.

Usage:  python scripts/06_compass_svp_ablation.py --dimension VV --feature plv
        python scripts/06_compass_svp_ablation.py --dimension VV --feature wpli
Output: results_ablation/
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

import common as C

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dimension', choices=['VV', 'AR'], required=True)
parser.add_argument('--feature', choices=['plv', 'wpli'], default='plv')
args = parser.parse_args()

dim, feature = args.dimension, args.feature
tag = f'{dim.lower()}_{feature}'
out_dir = C.output_dir('ablation')
df, feature_cols, subjects = C.load_features(dim, feature)

ABLATIONS = {   # name -> (use SVM sifting, prototype mode, K)
    'Full System':                        (True,  'kmeans', C.K_PROTOTYPES),
    'Scenario A (w/o K-Means)':           (True,  'raw',    None),
    'Scenario B (w/o SVM Sifting)':       (False, 'kmeans', C.K_PROTOTYPES),
    'Scenario C (Single Prototype, K=1)': (True,  'kmeans', 1),
}
METRICS = ['euclidean', 'cosine']
names = [f'{a} - {m.capitalize()}' for a in ABLATIONS for m in METRICS] + ['SVM (Baseline)']


def build_prototypes(X, y, mode, k):
    protos, labels = [], []
    for c in (0, 1):
        Xc = X[y == c]
        if mode == 'raw':
            protos.append(Xc)
            labels.append(np.full(len(Xc), c))
        else:
            km = KMeans(n_clusters=min(k, len(Xc)), n_init=C.KMEANS_N_INIT, random_state=C.SEED).fit(Xc)
            protos.append(km.cluster_centers_)
            labels.append(np.full(len(km.cluster_centers_), c))
    return np.vstack(protos), np.concatenate(labels)


print(f"COMPASS-SVP ablation | {C.DIMENSION_NAMES[dim]} | feature: {feature.upper()} | subjects: {len(subjects)}")

window_preds = {n: [] for n in names}
fold_log = []
for fold, test_subj in enumerate(subjects, start=1):
    C.set_seed()
    tr, te = df[df['subject'] != test_subj], df[df['subject'] == test_subj]
    X_tr, y_tr, X_te = tr[feature_cols].values, tr['y'].values, te[feature_cols].values
    base = te[['subject', 'trial', 'y']]

    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=C.SEED).fit(X_tr, y_tr)
    X_sv, y_sv = svm.support_vectors_, y_tr[svm.support_]
    fold_log.append({'fold': fold, 'test_subject': test_subj, 'n_train_windows': len(y_tr),
                     'n_support_vectors': len(y_sv), 'sv_fraction_%': round(len(y_sv) / len(y_tr) * 100, 2)})

    for abl, (use_sv, mode, k) in ABLATIONS.items():
        protos, labels = build_prototypes(*((X_sv, y_sv) if use_sv else (X_tr, y_tr)), mode, k)
        for metric in METRICS:
            pred = labels[np.argmin(cdist(X_te, protos, metric=metric), axis=1)]
            window_preds[f'{abl} - {metric.capitalize()}'].append(base.assign(y_pred=pred, fold=fold))
    window_preds['SVM (Baseline)'].append(base.assign(y_pred=svm.predict(X_te), fold=fold))
    print(f"  Fold {fold:>2}/{len(subjects)} | held-out subject {test_subj} | support vectors: {len(y_sv)}")

summary, subj_tables = {}, {}
for name in names:
    units = pd.concat(window_preds[name], ignore_index=True)
    trials, subj = C.aggregate(units)
    summary[name] = C.summarize(units, trials, subj)
    subj_tables[name] = subj
    C.export_predictions(out_dir, f'{tag}_{C.safe_name(name)}', units, trials, subj)

res = pd.DataFrame(summary).T
cols = ['Win Acc (%)', 'Trial Acc (%)', 'Subj Acc (%)', 'Subj Acc Flat Vote (%)', 'Subj Correct', '95% CI (%)',
        'Subj Macro Prec (%)', 'Subj Macro Rec (%)', 'Subj Macro F1 (%)', 'Subject Ties']
print("\n" + "=" * 140 + f"\n ABLATION RESULTS ({dim}, {feature.upper()})\n" + "=" * 140)
print(res[cols].to_string())
res.to_csv(os.path.join(out_dir, f'{tag}_ablation_summary.csv'))
pd.DataFrame(fold_log).to_csv(os.path.join(out_dir, f'{tag}_fold_log.csv'), index=False)

rows = []
for metric in METRICS:
    full = f'Full System - {metric.capitalize()}'
    block = []
    for other in [f'{a} - {metric.capitalize()}' for a in ABLATIONS if a != 'Full System'] + ['SVM (Baseline)']:
        a, b = subj_tables[full], subj_tables[other]
        ca, cb = a['correct'].values, b.set_index('subject').loc[a['subject'], 'correct'].values
        bb, cc = int((ca & ~cb).sum()), int((~ca & cb).sum())
        table = [[int((ca & cb).sum()), bb], [cc, int((~ca & ~cb).sum())]]
        block.append({'model_A': full, 'model_B': other, 'only_A_correct (b)': bb, 'only_B_correct (c)': cc,
                      'delta_acc_pp': round((ca.mean() - cb.mean()) * 100, 2),
                      'p_mcnemar_exact': round(mcnemar(table, exact=True).pvalue, 4)})
    for r, p in zip(block, multipletests([r['p_mcnemar_exact'] for r in block], method='holm')[1]):
        r['p_mcnemar_holm'] = round(p, 4)
    rows.extend(block)

stats = pd.DataFrame(rows)
print("\n" + "=" * 140 + "\n FULL SYSTEM vs. ABLATED VARIANTS (paired, subject level)\n" + "=" * 140)
print(stats.to_string(index=False))
stats.to_csv(os.path.join(out_dir, f'{tag}_ablation_statistics.csv'), index=False)
print(f"\nSaved to {out_dir}/")
