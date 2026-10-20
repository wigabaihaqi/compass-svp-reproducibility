"""
Step 0a - EEG preprocessing and 1-s windowing (Section 2.1.3).

For every subject listed in data/subjects_manifest.csv:
  1. load the raw OpenBCI Cyton recording (8 channels, 250 Hz);
  2. apply a 4th-order zero-phase Butterworth band-pass filter (8-30 Hz, filtfilt) to the
     whole recording, so that the filter introduces no phase distortion for the PLV;
  3. starting at the recorded stimulus onset, cut 10 RAPM questions x 2 phases
     (question only / question + options) x 15 consecutive 1-s windows = 300 windows;
  4. keep 250 samples per window and channel (linear interpolation to 250 points when a
     window contains fewer samples; the number of such windows is reported).

Output: one CSV per subject in data/raw_eeg_vv/ or data/raw_eeg_ar/ with the columns
subject, class, nomor (trial), stimulant (question), session (phase), window, channel,
datapoint1..datapoint250. These files are the input of the LSTM-CNN (step 4) and of the
connectivity feature extraction.

Usage:  python scripts/00a_segment_eeg.py                     # all subjects in the manifest
        python scripts/00a_segment_eeg.py --subject subject10 # one subject
Raw recordings are read from COMPASS_RAW_SOURCE_DIR (default: data/raw_source).
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import common as C

# ---------- acquisition and protocol settings ----------
FS = 250.0
LOWCUT, HIGHCUT, FILTER_ORDER = 8.0, 30.0, 4
RAW_CHANNELS = [f'EXG Channel {i}' for i in range(8)]   # OpenBCI column names
TIMESTAMP_COLUMN = 'Timestamp (Formatted)'
N_QUESTIONS, N_PHASES, N_WINDOWS = 10, 2, 15
WINDOW_SECONDS = 1.0
SAMPLES_PER_WINDOW = 250
START_TIME_FORMAT = '%d-%m-%Y %H:%M:%S.%f'               # format used in the manifest

RAW_SOURCE_DIR = os.environ.get('COMPASS_RAW_SOURCE_DIR', os.path.join(C.DATA_DIR, 'raw_source'))
MANIFEST = os.path.join(C.DATA_DIR, 'subjects_manifest.csv')

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--subject', default=None, help='process only this subject_id from the manifest')
args = parser.parse_args()


def dimension_of(style):
    for dim, mapping in C.LABEL_TO_INDEX.items():
        if style in mapping:
            return dim
    raise ValueError(f"Unknown learning style '{style}'")


def segment_subject(row):
    style = str(row['learning_style']).strip().lower()
    dim = dimension_of(style)
    raw_path = os.path.join(RAW_SOURCE_DIR, row['raw_file'])
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Raw recording not found: {raw_path}")

    raw = pd.read_csv(raw_path, header=0)
    raw = raw.rename(columns=lambda x: x.strip() if isinstance(x, str) else x)
    raw[TIMESTAMP_COLUMN] = pd.to_datetime(raw[TIMESTAMP_COLUMN])

    # Zero-phase band-pass filter over the whole recording
    b, a = butter(FILTER_ORDER, [LOWCUT / (0.5 * FS), HIGHCUT / (0.5 * FS)], btype='band')
    signals = {str(i): filtfilt(b, a, raw[ch].values) for i, ch in enumerate(RAW_CHANNELS)}
    timestamps = raw[TIMESTAMP_COLUMN].values

    # Start time parsed explicitly as day-month-year (avoids month/day ambiguity)
    start = pd.to_datetime(row['start_time'], format=START_TIME_FORMAT)
    subject_id, subject_index = row['subject_id'], int(row['subject_index'])

    records, n_interpolated, min_samples = [], 0, SAMPLES_PER_WINDOW
    row_id = 1
    for question in range(1, N_QUESTIONS + 1):
        for phase in range(1, N_PHASES + 1):
            nomor = (subject_index - 1) * N_QUESTIONS * N_PHASES + (question - 1) * N_PHASES + phase
            for window in range(1, N_WINDOWS + 1):
                end = start + pd.Timedelta(seconds=WINDOW_SECONDS)
                mask = (timestamps >= np.datetime64(start)) & (timestamps <= np.datetime64(end))
                n = int(mask.sum())
                if n == 0:
                    raise ValueError(f"{subject_id}: empty window (question {question}, phase {phase}, "
                                     f"window {window}, start {start}). Check start_time in the manifest.")
                min_samples = min(min_samples, n)
                if n < SAMPLES_PER_WINDOW:
                    n_interpolated += 1
                for ch, sig in signals.items():
                    x = sig[mask][:SAMPLES_PER_WINDOW]
                    if len(x) < SAMPLES_PER_WINDOW:              # resample short windows to 250 points
                        x = np.interp(np.linspace(0, len(x) - 1, SAMPLES_PER_WINDOW), np.arange(len(x)), x)
                    records.append({
                        'trial': row_id,                         # running row counter (kept for compatibility)
                        'subject': subject_id,
                        'timestamp_start': start.strftime(START_TIME_FORMAT)[:-3],
                        'timestamp_end': end.strftime(START_TIME_FORMAT)[:-3],
                        'stimulant': question,
                        'session': phase,
                        'window': window,
                        'channel': ch,
                        'nomor': nomor,
                        **{f'datapoint{i + 1}': v for i, v in enumerate(x)},
                        'class': style,
                    })
                    row_id += 1
                # next window starts 1 ms after the end of the current one
                start = end + pd.Timedelta(milliseconds=1)

    out_dir = os.path.join(C.DATA_DIR, C.RAW_EEG_DIRS[dim])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{style}-{subject_id}-preprocessed.csv')
    pd.DataFrame(records).to_csv(out_path, index=False)
    return {'subject_id': subject_id, 'dimension': dim, 'class': style,
            'windows': N_QUESTIONS * N_PHASES * N_WINDOWS, 'windows_interpolated': n_interpolated,
            'min_samples_in_window': min_samples, 'output': os.path.relpath(out_path, C.REPO_DIR)}


if not os.path.isfile(MANIFEST):
    raise FileNotFoundError(f"Manifest not found: {MANIFEST} (see data/README.md)")
manifest = pd.read_csv(MANIFEST, dtype=str)
if args.subject:
    manifest = manifest[manifest['subject_id'] == args.subject]
    if manifest.empty:
        raise ValueError(f"Subject '{args.subject}' not in {MANIFEST}")

report = []
for _, r in manifest.iterrows():
    info = segment_subject(r)
    report.append(info)
    print(f"  {info['subject_id']:<12} {info['dimension']} {info['class']:<10} windows: {info['windows']} | "
          f"interpolated: {info['windows_interpolated']} | min samples: {info['min_samples_in_window']}")

qc_path = os.path.join(C.DATA_DIR, 'segmentation_report.csv')
pd.DataFrame(report).to_csv(qc_path, index=False)
print(f"\nDone. Quality report: {qc_path}")
