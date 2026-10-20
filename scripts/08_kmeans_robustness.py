"""
Step 8 - Robustness of COMPASS-SVP to the K-Means initialization (Section 3.6).

The SVM sifting stage is deterministic, but the K-Means macro-anchors depend on the random
initialization. The full LOSO evaluation of COMPASS-SVP is therefore repeated for 20 K-Means
random seeds, with a single initialization (n_init='auto') and with ten restarts (n_init=10,
the setting used in all other scripts), for both distance metrics and both dimensions.

Usage:  python scripts/08_kmeans_robustness.py                 # VV and AR, 20 seeds
        python scripts/08_kmeans_robustness.py --dimension VV --seeds 20
Output: results_compass/kmeans_initialization_robustness.csv         (one row per run)
        results_compass/kmeans_initialization_robustness_summary.csv (mean, SD, min, max)
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.svm import SVC

import common as C

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dimension', choices=['VV', 'AR'], nargs='+', default=['VV', 'AR'])
parser.add_argument('--seeds', type=int, default=20, help='number of K-Means seeds (0 .. seeds-1)')
args = parser.parse_args()

N_INIT_SETTINGS = ['auto', C.KMEANS_N_INIT]
METRICS = ['euclidean', 'cosine']
out_dir = C.output_dir('compass')


def robustness(dim, seeds):
    df, feats, subjects = C.load_features(dim, 'plv')

    # SVM sifting is deterministic: fit once per LOSO fold and reuse for every seed
    folds = []
    for s in subjects:
        tr, te = df[df.subject != s], df[df.subject == s]
        X, y = tr[feats].values, tr.y.values
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=C.SEED).fit(X, y)
        folds.append((te, svm.support_vectors_, y[svm.support_]))

    rows = []
    for n_init in N_INIT_SETTINGS:
        for metric in METRICS:
            for seed in seeds:
                units = []
                for te, Xs, ys in folds:
                    P, L = [], []
                    for c in (0, 1):
                        km = KMeans(n_clusters=C.K_PROTOTYPES, n_init=n_init, random_state=seed).fit(Xs[ys == c])
                        P.append(km.cluster_centers_)
                        L += [c] * C.K_PROTOTYPES
                    pred = np.array(L)[cdist(te[feats].values, np.vstack(P), metric=metric).argmin(1)]
                    units.append(te[['subject', 'trial', 'y']].assign(y_pred=pred))
                u = pd.concat(units)
                _, sj = C.aggregate(u)
                rows.append({'dimension': dim, 'n_init': str(n_init), 'metric': metric, 'seed': seed,
                             'win_acc': round((u.y == u.y_pred).mean() * 100, 2),
                             'subj_acc': round(sj.correct.mean() * 100, 2)})
        print(f"  {dim}: n_init={n_init} done")

    r = pd.DataFrame(rows)
    print(f"\n===== {dim}: subject accuracy over {len(seeds)} seeds =====")
    print(r.groupby(['n_init', 'metric'])['subj_acc'].agg(['mean', 'std', 'min', 'max']).round(2))
    print("\nHow often each accuracy occurred:")
    print(r.groupby(['n_init', 'metric'])['subj_acc'].value_counts().unstack(fill_value=0))
    return r


seeds = list(range(args.seeds))
results = pd.concat([robustness(dim, seeds) for dim in args.dimension], ignore_index=True)
results.to_csv(os.path.join(out_dir, 'kmeans_initialization_robustness.csv'), index=False)

summary = (results.groupby(['dimension', 'n_init', 'metric'])['subj_acc']
           .agg(mean='mean', sd='std', min='min', max='max', n_seeds='size').round(2).reset_index())
summary.to_csv(os.path.join(out_dir, 'kmeans_initialization_robustness_summary.csv'), index=False)
print(f"\nSaved to {out_dir}/kmeans_initialization_robustness*.csv")
