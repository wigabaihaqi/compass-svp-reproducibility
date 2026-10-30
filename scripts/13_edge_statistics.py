"""
Step 13 - Subject-level tests of class differences per connectivity edge (analysis plan v1, Analysis 2).

Null hypothesis: no edge differs between the classes (Verbal vs. Visual; Active vs. Reflective).
  1. Per subject, the mean RAW connectivity (PLV or wPLI) of each of the 28 edges over all its windows.
     Raw values are used because within-subject z-scoring sets every subject's mean to zero.
  2. Per edge: difference of class means, Welch t, and Hedges' g with an approximate 95% CI.
  3. Exact subject-label permutation over all balanced assignments (VV 252, AR 48,620).
  4. Multiple-comparison correction within the family of 28 edges:
     primary   - permutation max-|t| (family-wise error rate, alpha = 0.05)
     supporting - Benjamini-Hochberg (q = 0.05) on the raw permutation p-values
  The sign of each mean difference is compared, descriptively, with the Analysis 1 difference graph
  (results_prototypes/) when that file exists.

Families: VV-PLV and AR-PLV (primary), VV-wPLI and AR-wPLI (volume-conduction control).

Usage:  python scripts/13_edge_statistics.py --dimension VV --feature plv
Output: results_edges/<dim>_<feature>_edge_statistics.csv   (all 28 edges)
        results_edges/<dim>_<feature>_edge_summary.csv      (one row: number of edges surviving)
        results_edges/<dim>_<feature>_significant_edges.png/.pdf  (only when an edge survives max-|t|)
"""

import argparse
import itertools
import os

import matplotlib
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import common as C

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dimension', choices=['VV', 'AR'], required=True)
parser.add_argument('--feature', choices=['plv', 'wpli'], default='plv')
args = parser.parse_args()
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ALPHA = 0.05
POS = {'Fp1': (-0.30, 0.88), 'Fp2': (0.30, 0.88), 'F3': (-0.42, 0.42), 'F4': (0.42, 0.42),
       'C3': (-0.55, 0.0), 'C4': (0.55, 0.0), 'O1': (-0.30, -0.88), 'O2': (0.30, -0.88)}

dim, feat, tag = args.dimension, args.feature, args.dimension.lower()
class_names = C.CLASS_NAMES[dim]
out_dir = os.path.join(C.RESULTS_DIR, 'results_edges')
os.makedirs(out_dir, exist_ok=True)

df, cols, subjects = C.load_features(dim, feat, zscore=False)
subj = df.groupby('subject', sort=False)[cols + ['y']].mean().loc[subjects]
X = subj[cols].values                       # subjects x edges (raw means)
y = subj['y'].round().astype(int).values
n1, n0 = int(y.sum()), int((1 - y).sum())
print(f"Edge statistics | {C.DIMENSION_NAMES[dim]} | {feat.upper()} (raw) | {class_names[0]} n = {n0}, "
      f"{class_names[1]} n = {n1} | edges: {len(cols)}")


def welch_t(A):
    """Welch t of every edge for every assignment; A = assignments x subjects (1 = class 1)."""
    A = A.astype(float)
    s1, q1 = A @ X, A @ X ** 2
    s0, q0 = X.sum(0) - s1, (X ** 2).sum(0) - q1
    m1, m0 = s1 / n1, s0 / n0
    v1, v0 = (q1 - s1 ** 2 / n1) / (n1 - 1), (q0 - s0 ** 2 / n0) / (n0 - 1)
    return (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)


combos = list(itertools.combinations(range(len(y)), n1))
assignments = np.zeros((len(combos), len(y)), dtype=np.int8)
for i, c in enumerate(combos):
    assignments[i, list(c)] = 1
print(f"Exact permutation test over {len(combos)} label assignments")

t_obs = welch_t(y[None, :])[0]
t_null = np.vstack([welch_t(assignments[i:i + 5000]) for i in range(0, len(assignments), 5000)])
abs_null = np.abs(t_null)
p_raw = (abs_null >= np.abs(t_obs) - 1e-12).mean(axis=0)
max_null = abs_null.max(axis=1)
p_fwer = (max_null[:, None] >= np.abs(t_obs)[None, :] - 1e-12).mean(axis=0)
q_fdr = multipletests(p_raw, alpha=ALPHA, method='fdr_bh')[1]

m1, m0 = X[y == 1].mean(0), X[y == 0].mean(0)
sd_pooled = np.sqrt(((n1 - 1) * X[y == 1].var(0, ddof=1) + (n0 - 1) * X[y == 0].var(0, ddof=1)) / (n1 + n0 - 2))
g = (1 - 3 / (4 * (n1 + n0) - 9)) * (m1 - m0) / sd_pooled
se_g = np.sqrt((n1 + n0) / (n1 * n0) + g ** 2 / (2 * (n1 + n0)))

res = pd.DataFrame({
    'edge': cols,
    f'mean_{class_names[0]}': m0.round(4), f'mean_{class_names[1]}': m1.round(4),
    'difference_class1_minus_class0': (m1 - m0).round(4),
    'welch_t': t_obs.round(3), 'hedges_g': g.round(3),
    'g_ci95_low': (g - 1.96 * se_g).round(3), 'g_ci95_high': (g + 1.96 * se_g).round(3),
    'p_permutation_raw': p_raw.round(4), 'p_fwer_maxT': p_fwer.round(4), 'q_fdr_bh': q_fdr.round(4),
})
res['significant_fwer'] = res['p_fwer_maxT'] < ALPHA
res['significant_fdr'] = res['q_fdr_bh'] < ALPHA

proto_file = os.path.join(C.RESULTS_DIR, 'results_prototypes', f'{tag}_{feat}_prototype_edges.csv')
agreement = None
if os.path.isfile(proto_file):
    pe = pd.read_csv(proto_file)
    diff = pe[pe['prototype'] == f'{class_names[1]} minus {class_names[0]}'].set_index('edge')['full_data_value']
    res['prototype_difference_sign'] = np.sign(res['edge'].map(diff)).astype(int)
    agree = np.sign(res['difference_class1_minus_class0']) == res['prototype_difference_sign']
    res['sign_agrees_with_prototypes'] = agree
    agreement = int(agree.sum())
else:
    print(f"[note] {proto_file} not found; run step 12 first to compare with the prototype difference graph")

res = res.sort_values('p_permutation_raw').reset_index(drop=True)
res.to_csv(os.path.join(out_dir, f'{tag}_{feat}_edge_statistics.csv'), index=False)
summary = pd.DataFrame([{
    'dimension': dim, 'feature': feat, 'n_class0': n0, 'n_class1': n1, 'n_permutations': len(combos),
    'edges': len(cols), 'min_p_raw': round(float(p_raw.min()), 4),
    'edges_p_raw_lt_0.05': int((p_raw < ALPHA).sum()),
    'edges_significant_fwer': int(res['significant_fwer'].sum()),
    'edges_significant_fdr': int(res['significant_fdr'].sum()),
    'max_abs_g': round(float(np.abs(g).max()), 3),
    'sign_agreement_with_prototypes': f'{agreement}/{len(cols)}' if agreement is not None else 'n/a',
}])
summary.to_csv(os.path.join(out_dir, f'{tag}_{feat}_edge_summary.csv'), index=False)
print('\n' + res.head(10).to_string(index=False))
print('\n' + summary.to_string(index=False))

sig = res[res['significant_fwer']]
if sig.empty:
    print(f"\nNo edge survives the max-|t| correction (FWER {ALPHA}); no figure is drawn.")
else:
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(3.2, 3.4), dpi=300)
    ax.add_patch(plt.Circle((0, 0), 1.05, fill=False, color='#888888', lw=0.8))
    vmax = np.abs(sig['hedges_g']).max()
    for _, r in sig.iterrows():
        a, b = r['edge'].split('-')
        (x1, y1), (x2, y2) = POS[a], POS[b]
        ax.plot([x1, x2], [y1, y2], color=plt.get_cmap('RdBu_r')(0.5 + 0.5 * r['hedges_g'] / vmax),
                lw=1 + 3 * abs(r['hedges_g']) / vmax)
    for ch, (x, y_) in POS.items():
        ax.scatter(x, y_, s=120, color='white', edgecolor='#333333', zorder=3, lw=0.8)
        ax.text(x, y_, ch, ha='center', va='center', fontsize=6.5, zorder=4)
    ax.set_title(f'{dim} {feat.upper()}: edges with FWER p < {ALPHA}\n(colour: Hedges g, '
                 f'{class_names[1]} minus {class_names[0]})', fontsize=8)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect('equal'); ax.axis('off')
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(out_dir, f'{tag}_{feat}_significant_edges.{ext}'), bbox_inches='tight')
print(f"\nSaved to {out_dir}/")
