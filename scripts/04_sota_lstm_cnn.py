"""
Step 4 - Adapted LSTM-CNN (Jawed et al., 2024) [11] on band-pass-filtered EEG windows.

Input: one CSV per subject in data/raw_eeg_vv/ or data/raw_eeg_ar/ (output of step 00a) with the columns
subject, class, nomor (trial), window, channel, datapoint1..datapoint250 (8 channels x 250 samples
per 1-s window). Each channel is z-scored per subject (label-free, same principle as the
other models). Architecture: LSTM(400, return_sequences) -> Conv1D(256, k=3, ReLU) ->
Dropout(0.2) -> MaxPool(3) -> Flatten -> Dense(2, softmax); Adam (lr = 0.001), 15 epochs,
batch size 16. Hierarchical majority voting window -> trial -> subject.

Usage:  python scripts/04_sota_lstm_cnn.py --dimension VV [--epochs 15]
Output: results_lstmcnn/
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import common as C

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dimension', choices=['VV', 'AR'], required=True)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch-size', type=int, default=16)
args = parser.parse_args()

import tensorflow as tf  # noqa: E402  (imported after argparse so --help is fast)
from tensorflow.keras import Sequential  # noqa: E402
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402

TIME_STEPS, N_CHANNELS, LEARNING_RATE = 250, 8, 0.001
dim = args.dimension
out_dir = C.output_dir('lstmcnn')
raw_dir = os.path.join(C.DATA_DIR, C.RAW_EEG_DIRS[dim])


def set_all_seeds(seed):
    C.set_seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_subject(path):
    d = pd.read_csv(path).sort_values(['nomor', 'window', 'channel'])
    dp_cols = [f'datapoint{i}' for i in range(1, TIME_STEPS + 1)]
    subject = C.clean_subject_id(d['subject'].iloc[0])
    label = int(C.encode_labels([d['class'].iloc[0]], dim)[0])
    X, meta = [], []
    for (trial, window), g in d.groupby(['nomor', 'window']):
        if len(g) == N_CHANNELS:                       # keep complete windows only
            X.append(g[dp_cols].values.T)              # (250, 8)
            meta.append((subject, trial, window, label))
    if not X:
        raise ValueError(f"No complete 8-channel windows in {path}")
    X = np.asarray(X, dtype=np.float32)
    X = StandardScaler().fit_transform(X.reshape(-1, N_CHANNELS)).reshape(X.shape)   # subject-wise z-score
    return X, pd.DataFrame(meta, columns=['subject', 'trial', 'window', 'y'])


def build_model():
    model = Sequential([
        Input(shape=(TIME_STEPS, N_CHANNELS)),
        LSTM(400, return_sequences=True),
        Conv1D(256, kernel_size=3, activation='relu'),
        Dropout(0.2),
        MaxPooling1D(pool_size=3),
        Flatten(),
        Dense(2, activation='softmax'),
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if not os.path.isdir(raw_dir):
    raise FileNotFoundError(f"Raw EEG folder not found: {raw_dir}")
files = sorted(glob.glob(os.path.join(raw_dir, '*.csv')))
if not files:
    raise FileNotFoundError(f"No CSV files in {raw_dir}")

data = {}
for f in files:
    X, meta = load_subject(f)
    s = meta['subject'].iloc[0]
    if s in data:
        raise ValueError(f"Duplicate subject ID '{s}' (file {f})")
    data[s] = (X, meta)
    print(f"  Loaded subject {s:>4} | {C.CLASS_NAMES[dim][meta['y'].iloc[0]]:<10} | windows: {len(X)} | "
          f"trials: {meta['trial'].nunique()} | windows/trial: {len(X) / meta['trial'].nunique():.1f}")
subjects = sorted(data, key=C.subject_sort_key)

print(f"\nAdapted LSTM-CNN [14] | {C.DIMENSION_NAMES[dim]} | subjects: {len(subjects)} | epochs: {args.epochs}")
all_units = []
for fold, test_subj in enumerate(subjects, start=1):
    X_tr = np.concatenate([data[s][0] for s in subjects if s != test_subj])
    y_tr = np.concatenate([data[s][1]['y'].values for s in subjects if s != test_subj])
    X_te, meta_te = data[test_subj]

    tf.keras.backend.clear_session()
    set_all_seeds(C.SEED + fold)
    model = build_model()
    model.fit(X_tr, y_tr, epochs=args.epochs, batch_size=args.batch_size, verbose=0)
    prob = model.predict(X_te, verbose=0)
    all_units.append(meta_te.assign(y_pred=np.argmax(prob, axis=1), prob_class1=prob[:, 1], fold=fold))
    print(f"  Fold {fold:>2}/{len(subjects)} | held-out subject {test_subj} | "
          f"window acc: {(all_units[-1]['y'] == all_units[-1]['y_pred']).mean() * 100:.2f}%")

units = pd.concat(all_units, ignore_index=True)
trials, subjects_df = C.aggregate(units)
summary = C.summarize(units, trials, subjects_df)
C.print_subject_report('Adapted LSTM-CNN [14]', dim, subjects_df, summary)

prefix = f'lstmcnn_{dim.lower()}'
C.export_predictions(out_dir, prefix, units, trials, subjects_df)
pd.DataFrame([{'model': 'Adapted LSTM-CNN [14]', 'epochs': args.epochs, **summary}]).to_csv(
    os.path.join(out_dir, f'{prefix}_summary.csv'), index=False)
print(f"\nSaved to {out_dir}/")
