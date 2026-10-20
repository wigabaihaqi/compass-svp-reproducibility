"""
Step 0b - Functional connectivity features (PLV and wPLI) for every 1-s window (Section 2.2, Table 4).

For each window of the filtered EEG produced by step 0a (8 channels x 250 samples):
  1. the instantaneous phase of every channel is obtained with the Hilbert transform;
  2. PLV  = | (1/M) * sum_n exp(j * (phi_x[n] - phi_y[n])) |                       (Eq. 1)
     wPLI = | sum_n Im(S_xy[n]) | / sum_n | Im(S_xy[n]) |,  S_xy = z_x * conj(z_y)   (Vinck et al., 2011)
  3. only the upper triangle of the 8 x 8 connectivity matrix is kept (i < j): no diagonal and
     no duplicated symmetric pair, i.e. 8 x 7 / 2 = 28 features named 'Fp1-Fp2', ..., 'O1-O2'.

All subjects of one dimension are written to a single file data/combined_fe_<feature>_<dim>.csv
(columns: subject, trial, window, label, 28 features). The subject ID and label are read from the
'subject' and 'class' columns of the window files.

Usage:  python scripts/00b_extract_connectivity.py --dimension VV --feature plv
        python scripts/00b_extract_connectivity.py --dimension VV --feature plv --verify path/to/old.csv
--verify compares the new features with an existing feature file, row by row (matched on trial and
window), and reports the largest absolute difference and the subject-ID correspondence.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.signal import hilbert

import common as C

CHANNEL_NAMES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'O1', 'O2']   # OpenBCI channels 0-7
SAMPLES_PER_WINDOW = 250
PAIRS = [(i, j) for i in range(len(CHANNEL_NAMES)) for j in range(i + 1, len(CHANNEL_NAMES))]
FEATURE_NAMES = [f'{CHANNEL_NAMES[i]}-{CHANNEL_NAMES[j]}' for i, j in PAIRS]
assert len(FEATURE_NAMES) == len(CHANNEL_NAMES) * (len(CHANNEL_NAMES) - 1) // 2 == 28

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dimension', choices=['VV', 'AR'], required=True)
parser.add_argument('--feature', choices=['plv', 'wpli'], required=True)
parser.add_argument('--verify', default=None, help='existing feature CSV to compare with')
parser.add_argument('--output', default=None, help='output path (default: data/combined_fe_<feature>_<dim>.csv)')
args = parser.parse_args()


def connectivity(window, feature):
    """window: array (8 channels, 250 samples) -> 28 upper-triangular connectivity values."""
    z = hilbert(np.asarray(window, dtype=float), axis=1)          # analytic signal per channel
    values = []
    if feature == 'plv':
        phase = np.angle(z)
        for i, j in PAIRS:
            values.append(np.abs(np.mean(np.exp(1j * (phase[i] - phase[j])))))
    else:
        for i, j in PAIRS:
            im = np.imag(z[i] * np.conj(z[j]))
            denom = np.sum(np.abs(im))
            values.append(np.abs(np.sum(im)) / denom if denom > 0 else 0.0)
    return np.asarray(values, dtype=np.float32)


dim, feature = args.dimension, args.feature
in_dir = os.path.join(C.DATA_DIR, C.RAW_EEG_DIRS[dim])
files = sorted(glob.glob(os.path.join(in_dir, '*.csv')))
if not files:
    raise FileNotFoundError(f"No window files in {in_dir}; run scripts/00a_segment_eeg.py first.")
dp_cols = [f'datapoint{i}' for i in range(1, SAMPLES_PER_WINDOW + 1)]

print(f"{feature.upper()} extraction | {C.DIMENSION_NAMES[dim]} | {len(files)} subject files")
rows, seen = [], set()
for path in files:
    df = pd.read_csv(path)
    subject = C.clean_subject_id(df['subject'].iloc[0])
    label = str(df['class'].iloc[0]).strip().lower()
    C.encode_labels([label], dim)                                  # raises on an unknown label
    if subject in seen:
        raise ValueError(f"Duplicate subject ID '{subject}' ({os.path.basename(path)})")
    seen.add(subject)

    n_skipped = 0
    for (trial, window), g in df.groupby(['nomor', 'window'], sort=True):
        if len(g) != len(CHANNEL_NAMES):
            n_skipped += 1
            continue
        g = g.sort_values('channel')                               # channel 0..7 -> Fp1..O2
        rows.append({'subject': subject, 'trial': trial, 'window': window, 'label': label,
                     **dict(zip(FEATURE_NAMES, connectivity(g[dp_cols].values, feature)))})
    n_windows = sum(1 for r in rows if r['subject'] == subject)
    print(f"  {os.path.basename(path):<40} subject {subject:>4} | {label:<10} | windows: {n_windows}"
          f"{f' | skipped (incomplete): {n_skipped}' if n_skipped else ''}")

out = pd.DataFrame(rows)
out_path = args.output or os.path.join(C.DATA_DIR, C.FEATURE_FILES[(dim, feature)])
out.to_csv(out_path, index=False)
print(f"\nSaved {len(out)} windows x {len(FEATURE_NAMES)} features -> {out_path}")

# ---------- optional verification against an existing feature file ----------
if args.verify:
    old = pd.read_csv(args.verify)
    key = ['trial', 'window']                                      # trial (= nomor) is unique across subjects
    m = out.merge(old, on=key, suffixes=('_new', '_old'))
    print(f"\nVerification against {args.verify}")
    print(f"  rows: new = {len(out)}, old = {len(old)}, matched on (trial, window) = {len(m)}")
    diff = np.abs(m[[f'{c}_new' for c in FEATURE_NAMES]].values - m[[f'{c}_old' for c in FEATURE_NAMES]].values)
    print(f"  max |difference| over 28 features: {diff.max():.3e}  (float32 precision ~ 1e-6)")
    print(f"  labels identical: {(m['label_new'] == m['label_old'].astype(str).str.strip().str.lower()).all()}")
    mapping = (m.groupby('subject_old')['subject_new'].agg(lambda s: sorted(set(s)))
               .reset_index().rename(columns={'subject_old': 'subject in old file',
                                              'subject_new': 'subject in new file'}))
    print("  subject-ID correspondence:\n" + mapping.to_string(index=False))
