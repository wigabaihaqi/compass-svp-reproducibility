"""
Step 1 - Within-fold WCSS profiles used to fix the number of prototypes K (Section 3.2, Fig. 3).

For every LOSO training fold (held-out subject excluded): fit the RBF-SVM filter on the
training subjects, keep the support vectors (alpha_i > 0), and compute the intra-class
K-Means WCSS for K = 1..6 per class. K was fixed at 2 for all folds based on these profiles.

Usage:  python scripts/01_wcss_profiles.py --dimension VV
Output: results_wcss/
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.svm import SVC

import common as C

MAX_K = 6

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dimension', choices=['VV', 'AR'], required=True)
parser.add_argument('--show', action='store_true', help='display the figure (default: save only)')
args = parser.parse_args()
if not args.show:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

dim = args.dimension
class_names = C.CLASS_NAMES[dim]
out_dir = C.output_dir('wcss')
df, feature_cols, subjects = C.load_features(dim, 'plv')
k_values = np.arange(1, MAX_K + 1)

print(f"Within-fold WCSS profiles | {C.DIMENSION_NAMES[dim]} | subjects: {len(subjects)} | K = 1..{MAX_K}")

rows = []
for fold, test_subj in enumerate(subjects, start=1):
    C.set_seed()
    tr = df[df['subject'] != test_subj]
    X, y = tr[feature_cols].values, tr['y'].values
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=C.SEED).fit(X, y)
    X_sv, y_sv = svm.support_vectors_, y[svm.support_]
    for c in (0, 1):
        Xc = X_sv[y_sv == c]
        for k in k_values:
            km = KMeans(n_clusters=int(k), n_init=C.KMEANS_N_INIT, random_state=C.SEED).fit(Xc)
            rows.append({'fold': fold, 'test_subject': test_subj, 'class': class_names[c], 'K': int(k),
                         'n_support_vectors': len(Xc), 'WCSS': km.inertia_})
    print(f"  Fold {fold:>2}/{len(subjects)} | held-out subject {test_subj} | "
          f"SV {class_names[0]} = {(y_sv == 0).sum()}, SV {class_names[1]} = {(y_sv == 1).sum()}")

wcss = pd.DataFrame(rows).sort_values(['fold', 'class', 'K'])
wcss['relative_reduction_from_prev_K_%'] = (
    wcss.groupby(['fold', 'class'])['WCSS'].pct_change().mul(-100).round(2))
summary = (wcss.groupby(['class', 'K'])
           .agg(WCSS_mean=('WCSS', 'mean'), WCSS_sd=('WCSS', 'std'),
                reduction_mean_pct=('relative_reduction_from_prev_K_%', 'mean'),
                reduction_sd_pct=('relative_reduction_from_prev_K_%', 'std'))
           .round(2).reset_index())

tag = dim.lower()
wcss.to_csv(os.path.join(out_dir, f'{tag}_wcss_per_fold.csv'), index=False)
summary.to_csv(os.path.join(out_dir, f'{tag}_wcss_summary.csv'), index=False)
print("\n" + summary.to_string(index=False))
print(f"\nK fixed at {C.K_PROTOTYPES} for all folds.")

# ---------- Figure (single panel, >= 10 pt at print size) ----------
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11, 'legend.fontsize': 10})
fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=300)
for c, color, marker in ((0, '#2166ac', 'o'), (1, '#b2182b', 's')):
    s = summary[summary['class'] == class_names[c]].sort_values('K')
    ax.plot(k_values, s['WCSS_mean'], marker=marker, markersize=5, linewidth=2.0, color=color,
            label=f'{class_names[c].capitalize()} boundary')
    ax.fill_between(k_values, s['WCSS_mean'] - s['WCSS_sd'], s['WCSS_mean'] + s['WCSS_sd'],
                    color=color, alpha=0.15)
ax.axvline(C.K_PROTOTYPES, color='#252525', linestyle=':', linewidth=1.2)
ax.set_xlabel('Number of prototypes ($K$)')
ax.set_ylabel('Intra-class WCSS')
ax.set_xticks(k_values)
ax.set_ylim(bottom=0)
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='best', frameon=True, facecolor='white', edgecolor='none', framealpha=0.85)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f'{tag}_wcss_convergence.png'), dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, f'{tag}_wcss_convergence.pdf'), bbox_inches='tight')
if args.show:
    plt.show()
print(f"Saved to {out_dir}/")
