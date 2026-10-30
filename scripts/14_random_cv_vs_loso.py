"""
Step 14 - Random window-level cross-validation vs. LOSO (analysis plan v1, Analysis 3).

Quantifies how much accuracy rises when windows of the same subject are in both the training and the
test folds, as in the random-split evaluations of several published methods.
  Models   : COMPASS-SVP (Euclidean, Cosine), SVM, Random Forest, MLP, KNN; settings of step 05.
  Random CV: stratified 5-fold CV over windows, shuffled with seeds 0-9 (subject-wise z-scored PLV,
             as in the main benchmark).
  LOSO     : window accuracy of the main benchmark, read from results_compass/<dim>_benchmark_summary.csv
             (run step 05 first).
Only window accuracy is compared; subject-level accuracy is undefined under random CV because a
subject is split across folds.

Usage:  python scripts/14_random_cv_vs_loso.py --dimension VV
Output: results_cv_comparison/<dim>_random_cv_per_seed.csv
        results_cv_comparison/<dim>_random_cv_vs_loso.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import common as C

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dimension', choices=['VV', 'AR'], required=True)
parser.add_argument('--n-splits', type=int, default=5)
parser.add_argument('--seeds', type=int, default=10, help='shuffling seeds 0 .. seeds-1')
args = parser.parse_args()

dim, tag = args.dimension, args.dimension.lower()
out_dir = os.path.join(C.RESULTS_DIR, 'results_cv_comparison')
os.makedirs(out_dir, exist_ok=True)

loso_file = os.path.join(C.RESULTS_DIR, C.OUTPUT_DIRS['compass'], f'{tag}_benchmark_summary.csv')
if not os.path.isfile(loso_file):
    raise FileNotFoundError(f'{loso_file} not found; run scripts/05_compass_svp_benchmark.py --dimension {dim} first')
loso = pd.read_csv(loso_file, index_col=0)['Win Acc (%)']

df, cols, subjects = C.load_features(dim, 'plv')
X, y = df[cols].values, df['y'].values


def svm_factory():
    return SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=C.SEED)


BASELINES = {
    'SVM (Baseline)':           svm_factory,
    'Random Forest (Baseline)': lambda: RandomForestClassifier(n_estimators=100, random_state=C.SEED),
    'MLP (Baseline)':           lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=C.SEED),
    'KNN (Baseline)':           lambda: KNeighborsClassifier(n_neighbors=5),
}
COMPASS = {'COMPASS-SVP (Euclidean)': 'euclidean', 'COMPASS-SVP (Cosine)': 'cosine'}
names = list(COMPASS) + list(BASELINES)

print(f"Random CV vs. LOSO | {C.DIMENSION_NAMES[dim]} | {len(y)} windows, {len(subjects)} subjects | "
      f"{args.n_splits}-fold, seeds 0-{args.seeds - 1}")
rows = []
for seed in range(args.seeds):
    correct = dict.fromkeys(names, 0)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        C.set_seed()
        svm = svm_factory().fit(X[tr], y[tr])
        X_sv, y_sv = svm.support_vectors_, y[tr][svm.support_]
        protos, labels = [], []
        for c in (0, 1):
            km = KMeans(n_clusters=min(C.K_PROTOTYPES, int((y_sv == c).sum())), n_init=C.KMEANS_N_INIT,
                        random_state=C.SEED).fit(X_sv[y_sv == c])
            protos.append(km.cluster_centers_)
            labels.append(np.full(len(km.cluster_centers_), c))
        protos, labels = np.vstack(protos), np.concatenate(labels)
        for name, metric in COMPASS.items():
            correct[name] += int((labels[np.argmin(cdist(X[te], protos, metric=metric), axis=1)] == y[te]).sum())
        correct['SVM (Baseline)'] += int((svm.predict(X[te]) == y[te]).sum())
        for name, factory in BASELINES.items():
            if name != 'SVM (Baseline)':
                correct[name] += int((factory().fit(X[tr], y[tr]).predict(X[te]) == y[te]).sum())
    for name in names:
        rows.append({'seed': seed, 'model': name, 'random_cv_win_acc_%': round(correct[name] / len(y) * 100, 2)})
    print(f"  seed {seed}: " + ', '.join(f"{n.split(' (')[0]} {correct[n] / len(y) * 100:.1f}%" for n in names))

per_seed = pd.DataFrame(rows)
per_seed.to_csv(os.path.join(out_dir, f'{tag}_random_cv_per_seed.csv'), index=False)
agg = per_seed.groupby('model', sort=False)['random_cv_win_acc_%'].agg(['mean', 'std']).round(2)
table = pd.DataFrame({
    'dimension': dim,
    'loso_win_acc_%': [loso[n] for n in agg.index],
    'random_cv_win_acc_mean_%': agg['mean'].values,
    'random_cv_win_acc_sd_%': agg['std'].values,
}, index=agg.index)
table['difference_pp'] = (table['random_cv_win_acc_mean_%'] - table['loso_win_acc_%']).round(2)
table.index.name = 'model'
table.to_csv(os.path.join(out_dir, f'{tag}_random_cv_vs_loso.csv'))
print('\n' + table.to_string())
print(f"\nSaved to {out_dir}/")
