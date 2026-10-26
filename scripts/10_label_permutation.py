"""
Step 10 - Subject-level label-permutation test of COMPASS-SVP against chance (Section 3.6).

The class labels are permuted across subjects (the number of subjects per class is kept), and the
complete COMPASS-SVP pipeline (SVM sifting, K-Means condensation, nearest-prototype inference and
hierarchical voting) is retrained in every LOSO fold for each permutation. The p-value is the share
of permutations whose subject-level accuracy is at least the observed one (Ojala and Garriga, 2010).

All balanced label assignments are enumerated when --n-perm is 0 or not smaller than their number
(exact test; VV: 252 assignments). Otherwise --n-perm assignments are drawn without replacement
with a fixed seed and p = (count + 1) / (n_perm + 1).

Usage:  python scripts/10_label_permutation.py --dimension VV                # exact (252 assignments)
        python scripts/10_label_permutation.py --dimension AR --n-perm 300   # 300 random assignments
Output: results_compass/<dim>_label_permutation_summary.csv  (observed accuracy, null mean/95th pct., p)
        results_compass/<dim>_label_permutation_null.csv     (subject accuracy of every permutation)
Runtime (one CPU core): VV about 10 min (exact), AR about 1 h (300 permutations).
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.svm import SVC

import common as C

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dimension', choices=['VV', 'AR'], required=True)
parser.add_argument('--feature', choices=['plv', 'wpli'], default='plv')
parser.add_argument('--n-perm', type=int, default=0, help='0 = all balanced label assignments (exact)')
args = parser.parse_args()

METRICS = ['euclidean', 'cosine']
dim, tag = args.dimension, args.dimension.lower()
out_dir = C.output_dir('compass')

df, feats, subjects = C.load_features(dim, args.feature)
X, subj, trial = df[feats].values, df['subject'].values, df['trial'].values
idx = {s: np.where(subj == s)[0] for s in subjects}
true_labels = {s: int(df.loc[df['subject'] == s, 'y'].iloc[0]) for s in subjects}


def loso_accuracy(labels):
    """Subject-level accuracy of COMPASS-SVP (both metrics) for a subject -> class assignment."""
    y = np.array([labels[s] for s in subj])
    correct = dict.fromkeys(METRICS, 0)
    for s in subjects:
        C.set_seed()
        te = idx[s]
        tr = np.setdiff1d(np.arange(len(y)), te)
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=C.SEED).fit(X[tr], y[tr])
        X_sv, y_sv = svm.support_vectors_, y[tr][svm.support_]
        protos, proto_labels = [], []
        for c in (0, 1):
            km = KMeans(n_clusters=min(C.K_PROTOTYPES, int((y_sv == c).sum())), n_init=C.KMEANS_N_INIT,
                        random_state=C.SEED).fit(X_sv[y_sv == c])
            protos.append(km.cluster_centers_)
            proto_labels.append(np.full(len(km.cluster_centers_), c))
        protos, proto_labels = np.vstack(protos), np.concatenate(proto_labels)
        for metric in METRICS:
            pred = proto_labels[np.argmin(cdist(X[te], protos, metric=metric), axis=1)]
            trial_votes = pd.Series(pred).groupby(trial[te]).apply(C.majority_vote)
            correct[metric] += int(C.majority_vote(trial_votes.values) == labels[s])
    return {m: correct[m] / len(subjects) for m in METRICS}


observed = loso_accuracy(true_labels)
print(f"Label-permutation test | {C.DIMENSION_NAMES[dim]} | {args.feature.upper()} | observed: "
      + ', '.join(f"{m} {observed[m] * 100:.2f}%" for m in METRICS))

n_class1 = sum(true_labels.values())
assignments = list(itertools.combinations(range(len(subjects)), n_class1))
exact = args.n_perm <= 0 or args.n_perm >= len(assignments)
if not exact:
    rng = np.random.default_rng(C.SEED)
    assignments = [assignments[i] for i in rng.choice(len(assignments), size=args.n_perm, replace=False)]
print(f"{'Exact test' if exact else 'Random permutations'}: {len(assignments)} label assignments")

null_rows, null_acc = [], {m: [] for m in METRICS}
for k, comb in enumerate(assignments, start=1):
    labels = {s: int(i in comb) for i, s in enumerate(subjects)}
    acc = loso_accuracy(labels)
    for m in METRICS:
        null_acc[m].append(acc[m])
    null_rows.append({'permutation': k, 'class1_subjects': ' '.join(str(subjects[i]) for i in comb),
                      **{f'subj_acc_{m}_%': round(acc[m] * 100, 2) for m in METRICS}})
    if k % 25 == 0:
        print(f"  {k}/{len(assignments)}")
null = pd.DataFrame(null_rows)

summary = []
for m in METRICS:
    a = np.array(null_acc[m])
    count = int(np.sum(a >= observed[m] - 1e-9))   # unrounded accuracies; ties count as 'at least'
    p = count / len(a) if exact else (count + 1) / (len(a) + 1)
    summary.append({'dimension': dim, 'feature': args.feature, 'metric': m, 'test': 'exact' if exact else 'random',
                    'n_permutations': len(a), 'observed_subj_acc_%': round(observed[m] * 100, 2),
                    'null_mean_%': round(a.mean() * 100, 2), 'null_95th_percentile_%': round(np.percentile(a, 95) * 100, 2),
                    'n_null_at_least_observed': count, 'p_permutation': round(p, 4)})
summary = pd.DataFrame(summary)
print(summary.to_string(index=False))

summary.to_csv(os.path.join(out_dir, f'{tag}_label_permutation_summary.csv'), index=False)
null.to_csv(os.path.join(out_dir, f'{tag}_label_permutation_null.csv'), index=False)
print(f"\nSaved to {out_dir}/")
