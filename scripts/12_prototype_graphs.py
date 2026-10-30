"""
Step 12 - Connectivity graphs of the COMPASS-SVP prototypes and their stability across LOSO folds
(analysis plan v1, Analysis 1; descriptive, no inference).

Each prototype is a vector of 28 channel-pair values, so it is drawn as an 8-channel graph.
  1. Full-data prototypes: COMPASS-SVP fitted on all subjects of a dimension (2 prototypes per class).
  2. Fold prototypes: the prototypes of every LOSO training fold, matched to the full-data prototypes of
     the same class by the Hungarian algorithm on Pearson correlation.
  3. Stability: correlation of every fold prototype with its matched full-data prototype, and per edge
     the mean, SD and sign agreement across folds.
  4. Figure: one graph per prototype plus the difference graph (mean class-1 prototype minus mean
     class-0 prototype). Line width and colour show the edge value; a dashed line marks an edge whose
     sign disagrees with the full-data sign in more than 20% of the folds.
Values are within-subject z-scores (the features used by COMPASS-SVP).

Usage:  python scripts/12_prototype_graphs.py --dimension VV --feature plv
Output: results_prototypes/<dim>_<feature>_prototype_edges.csv      (per prototype and edge)
        results_prototypes/<dim>_<feature>_prototype_stability.csv  (per prototype: fold correlations)
        results_prototypes/<dim>_<feature>_prototype_graphs.png/.pdf
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.svm import SVC

import common as C

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dimension', choices=['VV', 'AR'], required=True)
parser.add_argument('--feature', choices=['plv', 'wpli'], default='plv')
parser.add_argument('--show', action='store_true', help='display the figure (default: save only)')
args = parser.parse_args()
if not args.show:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SIGN_FLAG = 0.80   # an edge is drawn dashed when its sign agrees in fewer than 80% of the folds
# approximate 2-D scalp positions (top view, nose up) of the 10-20 electrodes used
POS = {'Fp1': (-0.30, 0.88), 'Fp2': (0.30, 0.88), 'F3': (-0.42, 0.42), 'F4': (0.42, 0.42),
       'C3': (-0.55, 0.0), 'C4': (0.55, 0.0), 'O1': (-0.30, -0.88), 'O2': (0.30, -0.88)}

dim, feat, tag = args.dimension, args.feature, args.dimension.lower()
class_names = C.CLASS_NAMES[dim]
out_dir = os.path.join(C.RESULTS_DIR, 'results_prototypes')
os.makedirs(out_dir, exist_ok=True)
df, cols, subjects = C.load_features(dim, feat)
pairs = [c.split('-') for c in cols]
unknown = sorted({ch for p in pairs for ch in p} - set(POS))
if unknown:
    raise ValueError(f'No scalp position defined for channel(s) {unknown}')


def fit_prototypes(X, y):
    """COMPASS-SVP training stages 1-2; returns {class: array (K, n_features)}."""
    C.set_seed()
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=C.SEED).fit(X, y)
    X_sv, y_sv = svm.support_vectors_, y[svm.support_]
    out = {}
    for c in (0, 1):
        k = min(C.K_PROTOTYPES, int((y_sv == c).sum()))
        out[c] = KMeans(n_clusters=k, n_init=C.KMEANS_N_INIT, random_state=C.SEED).fit(X_sv[y_sv == c]).cluster_centers_
    return out


def match(ref, cand):
    """Reorder the rows of cand to the rows of ref (Hungarian algorithm on Pearson r)."""
    r = np.array([[np.corrcoef(a, b)[0, 1] for b in cand] for a in ref])
    rows, cols_ = linear_sum_assignment(-r)
    order = cols_[np.argsort(rows)]
    return cand[order], r[np.arange(len(ref)), order]


print(f"Prototype graphs | {C.DIMENSION_NAMES[dim]} | {feat.upper()} | subjects: {len(subjects)} | K = {C.K_PROTOTYPES}")
X_all, y_all = df[cols].values, df['y'].values
full = fit_prototypes(X_all, y_all)

fold_protos = {c: [] for c in (0, 1)}
fold_r = {c: [] for c in (0, 1)}
for fold, s in enumerate(subjects, start=1):
    tr = df['subject'] != s
    fp = fit_prototypes(X_all[tr], y_all[tr])
    for c in (0, 1):
        m, r = match(full[c], fp[c])
        fold_protos[c].append(m)
        fold_r[c].append(r)
    print(f"  Fold {fold:>2}/{len(subjects)} done")

# ---------- tables ----------
edge_rows, stab_rows, panels = [], [], []
for c in (0, 1):
    stack = np.stack(fold_protos[c])              # folds x K x edges
    rr = np.stack(fold_r[c])                      # folds x K
    for k in range(full[c].shape[0]):
        name = f'{class_names[c]} prototype {k + 1}'
        agree = (np.sign(stack[:, k, :]) == np.sign(full[c][k])).mean(axis=0)
        panels.append((name, full[c][k], agree))
        stab_rows.append({'dimension': dim, 'feature': feat, 'prototype': name, 'n_folds': len(rr),
                          'r_median': round(float(np.median(rr[:, k])), 3),
                          'r_min': round(float(rr[:, k].min()), 3), 'r_max': round(float(rr[:, k].max()), 3)})
        for e, col in enumerate(cols):
            edge_rows.append({'prototype': name, 'edge': col, 'full_data_value': round(float(full[c][k, e]), 4),
                              'fold_mean': round(float(stack[:, k, e].mean()), 4),
                              'fold_sd': round(float(stack[:, k, e].std(ddof=1)), 4),
                              'sign_agreement': round(float(agree[e]), 3)})

diff_full = full[1].mean(axis=0) - full[0].mean(axis=0)
diff_folds = np.stack(fold_protos[1]).mean(axis=1) - np.stack(fold_protos[0]).mean(axis=1)
diff_agree = (np.sign(diff_folds) == np.sign(diff_full)).mean(axis=0)
diff_name = f'{class_names[1]} minus {class_names[0]}'
panels.append((diff_name, diff_full, diff_agree))
for e, col in enumerate(cols):
    edge_rows.append({'prototype': diff_name, 'edge': col, 'full_data_value': round(float(diff_full[e]), 4),
                      'fold_mean': round(float(diff_folds[:, e].mean()), 4),
                      'fold_sd': round(float(diff_folds[:, e].std(ddof=1)), 4),
                      'sign_agreement': round(float(diff_agree[e]), 3)})

stab = pd.DataFrame(stab_rows)
edges = pd.DataFrame(edge_rows)
stab.to_csv(os.path.join(out_dir, f'{tag}_{feat}_prototype_stability.csv'), index=False)
edges.to_csv(os.path.join(out_dir, f'{tag}_{feat}_prototype_edges.csv'), index=False)
print('\n' + stab.to_string(index=False))
print(f"\nDifference graph: {int((diff_agree < SIGN_FLAG).sum())} of {len(cols)} edges change sign in more than "
      f"{(1 - SIGN_FLAG) * 100:.0f}% of the folds")

# ---------- figure ----------
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
vmax_p = max(np.abs(v).max() for _, v, _ in panels[:-1])
vmax_d = np.abs(diff_full).max()
fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.1), dpi=300)
for ax, (name, vals, agree) in zip(axes, panels):
    vmax = vmax_d if name == diff_name else vmax_p
    cmap = plt.get_cmap('RdBu_r')
    ax.add_patch(plt.Circle((0, 0), 1.05, fill=False, color='#888888', lw=0.8))
    ax.plot([-0.08, 0, 0.08], [1.03, 1.13, 1.03], color='#888888', lw=0.8)   # nose
    for (a, b), v, g in sorted(zip(pairs, vals, agree), key=lambda t: abs(t[1])):
        (x1, y1), (x2, y2) = POS[a], POS[b]
        ax.plot([x1, x2], [y1, y2], color=cmap(0.5 + 0.5 * v / vmax), lw=0.4 + 3.2 * abs(v) / vmax,
                ls='--' if g < SIGN_FLAG else '-', solid_capstyle='round', alpha=0.9)
    for ch, (x, y) in POS.items():
        ax.scatter(x, y, s=120, color='white', edgecolor='#333333', zorder=3, lw=0.8)
        ax.text(x, y, ch, ha='center', va='center', fontsize=6.5, zorder=4)
    ax.set_title(name, fontsize=9)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.25); ax.set_aspect('equal'); ax.axis('off')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label('z-score' if name != diff_name else 'difference (z)', fontsize=7)
fig.legend(handles=[Line2D([0], [0], color='#555555', ls='--', lw=1.2)],
           labels=[f'sign agrees in < {SIGN_FLAG * 100:.0f}% of LOSO folds'],
           loc='lower center', fontsize=7, frameon=False, ncol=1, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(out_dir, f'{tag}_{feat}_prototype_graphs.{ext}'), bbox_inches='tight')
if args.show:
    plt.show()
print(f"\nSaved to {out_dir}/")
