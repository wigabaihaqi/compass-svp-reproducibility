"""
Step 9 - Supplementary figure: accuracy at the window, trial and subject levels (hierarchical voting).
The figure is not part of the revised manuscript; the same values are reported in Table 1 and Section 3.10.

Reads the benchmark summaries written by step 05 (no model is retrained) and plots the
three decision levels for COMPASS-SVP: VV with Euclidean distance and AR with both metrics.

Usage:  python scripts/09_fusion_levels_figure.py
Output: results_compass/fig6_hierarchical_fusion_levels.png / .pdf / .csv
"""

import argparse
import os

import matplotlib
import pandas as pd

import common as C

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--show', action='store_true')
args = parser.parse_args()
if not args.show:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

out_dir = C.output_dir('compass')
SERIES = [  # (label, dimension, model row in <dim>_benchmark_summary.csv, colour, hatch)
    ('VV - Euclidean', 'vv', 'COMPASS-SVP (Euclidean)', '#2f6fb0', ''),
    ('AR - Cosine', 'ar', 'COMPASS-SVP (Cosine)', '#c8323a', ''),
    ('AR - Euclidean', 'ar', 'COMPASS-SVP (Euclidean)', '#e08a2c', '///'),
]
LEVELS = [('Window', 'Win Acc (%)'), ('Trial', 'Trial Acc (%)'), ('Subject', 'Subj Acc (%)')]

rows = []
for label, dim, model, _, _ in SERIES:
    s = pd.read_csv(os.path.join(out_dir, f'{dim}_benchmark_summary.csv'), index_col=0).loc[model]
    rows.append({'series': label, **{lvl: float(s[col]) for lvl, col in LEVELS}})
data = pd.DataFrame(rows).set_index('series')
data.to_csv(os.path.join(out_dir, 'fig6_hierarchical_fusion_levels.csv'))
print(data)

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 10, 'legend.fontsize': 8.5})
fig, ax = plt.subplots(figsize=(3.6, 2.9), dpi=300)
width, gap = 0.27, 0.015
handles = []
for i, (label, _, _, colour, hatch) in enumerate(SERIES):
    xs = [j + (i - 1) * (width + gap) for j in range(len(LEVELS))]
    vals = data.loc[label].values
    bars = ax.bar(xs, vals, width=width, color=colour, hatch=hatch, edgecolor='white', linewidth=0.6, zorder=3)
    handles.append(bars)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f'{v:.1f}', ha='center', va='bottom',
                fontsize=6.2, color='#222222')
chance = ax.axhline(50, color='#777777', lw=0.8, ls='--', zorder=2)
ax.set_xticks(range(len(LEVELS)))
ax.set_xticklabels([lvl for lvl, _ in LEVELS])
ax.set_xlabel('Decision level')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 100)
ax.set_xlim(-0.55, 2.55)
ax.grid(axis='y', color='#e5e5e5', lw=0.6, zorder=0)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.legend(handles + [chance], [label for label, *_ in SERIES] + ['Chance (50%)'], loc='upper center',
          bbox_to_anchor=(0.5, -0.24), ncol=2,
          frameon=False, handlelength=1.4, columnspacing=1.0)
plt.tight_layout()
prefix = os.path.join(out_dir, 'fig6_hierarchical_fusion_levels')
plt.savefig(f'{prefix}.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{prefix}.pdf', bbox_inches='tight')
if args.show:
    plt.show()
print(f"Saved to {prefix}.png/.pdf")
