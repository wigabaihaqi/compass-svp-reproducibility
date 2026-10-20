"""
Step 7 - t-SNE visualization of the fitted COMPASS-SVP geometry for one LOSO training fold (Fig. 4; VV left, AR right).

All elements come from the REAL model fitted in the 28-D PLV feature space:
support vectors (alpha_i > 0), discarded samples (alpha_i = 0) and the K = 2 macro-anchors
per class. Training windows and macro-anchors are embedded together in one t-SNE run.
The f(x) = 0 and f(x) = +/-1 contours interpolate the real SVM decision values over the
embedding (t-SNE has no inverse mapping, so the contours are approximate).

Usage:  python scripts/07_compass_svp_geometry.py --dimension VV [--held-out-subject 10]
Output: results_compass-svp_geometry/
"""

import argparse
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC

import common as C

warnings.filterwarnings('ignore')
PERPLEXITY = {'VV': 30, 'AR': 50}       # Section 3.9

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dimension', choices=['VV', 'AR'], required=True)
parser.add_argument('--held-out-subject', default=None, help='fold to show (default: last subject)')
parser.add_argument('--width', type=float, default=6.7, help='printed figure width in inches')
parser.add_argument('--show', action='store_true')
args = parser.parse_args()
if not args.show:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

dim = args.dimension
names = [n.capitalize() for n in C.CLASS_NAMES[dim]]
out_dir = C.output_dir('geometry')
df, feature_cols, subjects = C.load_features(dim, 'plv')
test_subj = C.clean_subject_id(args.held_out_subject) if args.held_out_subject else subjects[-1]
if test_subj not in subjects:
    raise ValueError(f"Subject {test_subj} not in data; available: {subjects}")

C.set_seed()
tr = df[df['subject'] != test_subj]
X, y = tr[feature_cols].values, tr['y'].values

svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=C.SEED).fit(X, y)
is_sv = np.zeros(len(X), dtype=bool)
is_sv[svm.support_] = True
f_x = svm.decision_function(X)
protos, proto_lab = [], []
for c in (0, 1):
    km = KMeans(n_clusters=C.K_PROTOTYPES, n_init=C.KMEANS_N_INIT, random_state=C.SEED).fit(X[is_sv & (y == c)])
    protos.append(km.cluster_centers_)
    proto_lab.append(np.full(C.K_PROTOTYPES, c))
protos, proto_lab = np.vstack(protos), np.concatenate(proto_lab)

print(f"{C.DIMENSION_NAMES[dim]} | held-out subject {test_subj} | training windows {len(X)} | "
      f"support vectors {is_sv.sum()} ({is_sv.mean() * 100:.1f}%) | inside margin |f(x)|<1: "
      f"{(np.abs(f_x) < 1).mean() * 100:.1f}%")

emb_all = TSNE(n_components=2, perplexity=PERPLEXITY[dim], learning_rate='auto', init='pca',
               random_state=C.SEED).fit_transform(np.vstack([X, protos]))
emb, emb_p = emb_all[:len(X)], emb_all[len(X):]

# Interpolated contours of the real decision function (masked where there is no data)
pad = 0.05 * (emb.max(axis=0) - emb.min(axis=0))
(x0, y0), (x1, y1) = emb.min(axis=0) - pad, emb.max(axis=0) + pad
xx, yy = np.meshgrid(np.linspace(x0, x1, 400), np.linspace(y0, y1, 400))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = KNeighborsRegressor(n_neighbors=max(50, int(0.05 * len(emb)))).fit(emb, f_x).predict(grid)
tree = cKDTree(emb)
Z[tree.query(grid)[0] > 3 * np.percentile(tree.query(emb, k=2)[0][:, 1], 99)] = np.nan
Z = Z.reshape(xx.shape)

plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11, 'legend.fontsize': 10})
C0, C1 = '#2166ac', '#b2182b'
fig, ax = plt.subplots(figsize=(args.width, args.width * 0.95), dpi=300)
ax.scatter(*emb[~is_sv].T, c='#bdbdbd', s=6, alpha=0.25, edgecolors='none', zorder=1)
ax.scatter(*emb[is_sv & (y == 0)].T, c=C0, s=9, alpha=0.45, edgecolors='none', zorder=2)
ax.scatter(*emb[is_sv & (y == 1)].T, c=C1, s=9, alpha=0.45, edgecolors='none', zorder=2)
ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['#252525', '#000000', '#252525'],
           linestyles=['--', '-', '--'], linewidths=[1.0, 1.8, 1.0], zorder=4)
ax.scatter(*emb_p[proto_lab == 1].T, c=C1, marker='P', s=320, edgecolors='white', linewidth=1.5, zorder=10)
ax.scatter(*emb_p[proto_lab == 0].T, c=C0, marker='X', s=150, edgecolors='white', linewidth=1.2, zorder=11)
handles = [
    Line2D([], [], marker='o', ls='', color=C0, alpha=0.7, label=f'Support vectors: {names[0]}'),
    Line2D([], [], marker='o', ls='', color=C1, alpha=0.7, label=f'Support vectors: {names[1]}'),
    Line2D([], [], marker='o', ls='', color='#bdbdbd', label=r'Discarded samples ($\alpha_i = 0$)'),
    Line2D([], [], marker='X', ls='', color=C0, markersize=11, markeredgecolor='white',
           label=f'{names[0]} macro-anchors ($K=2$)'),
    Line2D([], [], marker='P', ls='', color=C1, markersize=11, markeredgecolor='white',
           label=f'{names[1]} macro-anchors ($K=2$)'),
    Line2D([], [], color='#000000', lw=1.8, label=r'$f(x)=0$ (interpolated)'),
    Line2D([], [], color='#252525', lw=1.0, ls='--', label=r'$f(x)=\pm 1$ (interpolated)'),
]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
ax.set_xlabel('t-SNE dimension 1')
ax.set_ylabel('t-SNE dimension 2')
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()

prefix = os.path.join(out_dir, f'{dim.lower()}_tsne_geometry')
plt.savefig(f'{prefix}.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{prefix}.pdf', bbox_inches='tight')
if args.show:
    plt.show()
pd.DataFrame({'subject': tr['subject'].values, 'trial': tr['trial'].values, 'class': y,
              'is_support_vector': is_sv, 'svm_decision_value': f_x,
              'tsne_1': emb[:, 0], 'tsne_2': emb[:, 1]}).to_csv(f'{prefix}_coordinates.csv', index=False)
pd.DataFrame({'class': proto_lab, 'tsne_1': emb_p[:, 0], 'tsne_2': emb_p[:, 1]}).to_csv(
    f'{prefix}_prototypes.csv', index=False)
print(f"Saved to {out_dir}/")
