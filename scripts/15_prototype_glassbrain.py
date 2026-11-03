"""Glass-brain rendering of the COMPASS-SVP prototype graphs.

Presentation only: reads the prototype edge values written by 12_prototype_graphs.py
(results_prototypes/<dim>_<feature>_prototype_edges.csv) and redraws them on a
top view of a glass brain. No model is fitted and no number changes.

Electrode positions are the mean cortical projections in MNI space of the
electrodes, as reported in Table 1 of Scrivener and Reader (2022), Brain and
Behavior 12, e2476, doi:10.1002/brb3.2476 (20 participants, open access). They
are an approximate projection of scalp electrodes; the connectivity is
sensor-level.

Usage: python scripts/15_prototype_glassbrain.py --dimension VV --feature plv
Requires nilearn (pip install nilearn).
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from nilearn import plotting  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--dimension', choices=['VV', 'AR'], required=True)
parser.add_argument('--feature', choices=['plv', 'wpli'], default='plv')
parser.add_argument('--results-dir', default=os.environ.get(
    'COMPASS_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
args = parser.parse_args()

SIGN_FLAG = 0.80   # same rule as script 12: dashed when the sign agrees in < 80% of folds
MNI = {'Fp1': (-24.54, 66.41, 11.97), 'Fp2': (25.25, 66.62, 12.19), 'F3': (-38.23, 34.47, 46.92),
       'F4': (38.59, 34.99, 48.23), 'C3': (-50.88, -21.18, 59.95), 'C4': (50.78, -23.18, 63.58),
       'O1': (-27.11, -99.78, 6.68), 'O2': (22.51, -100.07, 8.94)}   # cortex columns of Table 1

tag, feat = args.dimension.lower(), args.feature
in_dir = os.path.join(args.results_dir, 'results_prototypes')
df = pd.read_csv(os.path.join(in_dir, f'{tag}_{feat}_prototype_edges.csv'))
panels = list(dict.fromkeys(df['prototype']))
is_diff = [('minus' in p) for p in panels]
vmax_p = df[~df['prototype'].str.contains('minus')]['full_data_value'].abs().max()
vmax_d = df[df['prototype'].str.contains('minus')]['full_data_value'].abs().max()
coords = np.array(list(MNI.values()))
cmap = plt.get_cmap('RdBu_r')

plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 3.7), dpi=300)
for ax, name, diff in zip(axes, panels, is_diff):
    vmax = vmax_d if diff else vmax_p
    disp = plotting.plot_connectome(np.zeros((8, 8)), coords, display_mode='z', axes=ax,
                                    node_size=0, annotate=False, colorbar=False, black_bg=False)
    a = disp.axes['z'].ax
    rows = df[df['prototype'] == name].copy()
    rows = rows.reindex(rows['full_data_value'].abs().sort_values().index)
    for _, r in rows.iterrows():
        u, v = r['edge'].split('-')
        (x1, y1, _), (x2, y2, _) = MNI[u], MNI[v]
        val = r['full_data_value']
        a.plot([x1, x2], [y1, y2], color=cmap(0.5 + 0.5 * val / vmax), lw=0.4 + 3.2 * abs(val) / vmax,
               ls='--' if r['sign_agreement'] < SIGN_FLAG else '-', solid_capstyle='round',
               alpha=0.9, zorder=5)
    for ch, (x, y, _) in MNI.items():
        a.scatter(x, y, s=150, color='white', edgecolor='#333333', lw=0.8, zorder=6)
        a.text(x, y, ch, ha='center', va='center', fontsize=6, zorder=7)
    ax.set_title(name, fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label('difference (z)' if diff else 'z-score', fontsize=7)
fig.legend(handles=[Line2D([0], [0], color='#555555', ls='--', lw=1.2)],
           labels=[f'sign agrees in < {SIGN_FLAG * 100:.0f}% of LOSO folds'],
           loc='lower center', fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.03))
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(in_dir, f'{tag}_{feat}_prototype_glassbrain.{ext}'), bbox_inches='tight')
print(f'Saved {tag}_{feat}_prototype_glassbrain.png/.pdf to {in_dir}/')
