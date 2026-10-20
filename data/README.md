# Input data

All input files are placed in this single folder (override with the environment variable `COMPASS_DATA_DIR`).

```
data/
├── subjects_manifest_template.csv   # template for subjects_manifest.csv              (step 00a input)
├── raw_source/                      # raw OpenBCI recordings                           (step 00a input, not redistributed)
├── raw_eeg_vv/                      # filtered 1-s windows, one CSV per subject, VV     (00a output; 00b and 04 input, not redistributed)
├── raw_eeg_ar/                      # filtered 1-s windows, one CSV per subject, AR     (00a output; 00b and 04 input, not redistributed)
├── combined_fe_plv_vv.csv           # PLV features, Verbal-Visual                      (00b output; input of 01, 02, 03, 05, 06, 07)
├── combined_fe_plv_ar.csv           # PLV features, Active-Reflective                  (00b output; input of 01, 02, 03, 05, 06, 07)
├── combined_fe_wpli_vv.csv          # wPLI features, Verbal-Visual                     (00b output; input of 06 --feature wpli)
└── combined_fe_wpli_ar.csv          # wPLI features, Active-Reflective                 (00b output; input of 06 --feature wpli)
```

Only the four `combined_fe_*.csv` files and the manifest template are published. The raw recordings and
the segmented EEG windows are excluded by `.gitignore` and are available from the corresponding author upon
reasonable request, subject to the ethical approval of the study.

## 1. Subject manifest (`subjects_manifest.csv`, step 00a)

Copy `subjects_manifest_template.csv` to `subjects_manifest.csv` and add one row per subject.

| Column | Description |
|---|---|
| `dimension` | `VV` or `AR` (informative; the output folder is derived from `learning_style`) |
| `subject_index` | integer used to build the global trial index `nomor` |
| `subject_id` | subject code written to the output files (no participant names); must be unique |
| `learning_style` | `verbal`, `visual`, `active` / `aktif`, `reflective` / `reflektif` |
| `raw_file` | path of the OpenBCI CSV relative to `data/raw_source/` (or `COMPASS_RAW_SOURCE_DIR`) |
| `start_time` | onset of the first RAPM question, **day-month-year** `DD-MM-YYYY HH:MM:SS.mmm` |

Step 00a also writes `segmentation_report.csv` (number of windows with fewer than 250 samples per subject).

## 2. Filtered EEG windows (`raw_eeg_<vv|ar>/*.csv`)

One CSV per subject (file name `<learning_style>-<subject_id>-preprocessed.csv`): 8-30 Hz zero-phase
Butterworth band-pass filtered, 250 Hz, one row per channel per 1-s window (300 windows x 8 channels).

| Column | Description |
|---|---|
| `subject` | subject ID (used as the subject identifier by steps 00b and 04) |
| `class` | class label (same spellings as above) |
| `nomor` | global trial index (10 RAPM questions x 2 phases = 20 per subject) |
| `stimulant`, `session` | RAPM question (1-10) and presentation phase (1 = question only, 2 = with options) |
| `window` | window index within the trial (1-15) |
| `channel` | 0-7 = Fp1, Fp2, F3, F4, C3, C4, O1, O2 |
| `datapoint1` ... `datapoint250` | samples of the 1-s window |
| `trial`, `timestamp_start`, `timestamp_end` | running row counter and window time stamps (informative only) |

## 3. Connectivity features (`combined_fe_<plv|wpli>_<vv|ar>.csv`)

One row per 1-s window (300 windows per subject).

| Column | Description |
|---|---|
| `subject` | subject ID, taken from the `subject` column of the window files (`subject10` -> `10`) |
| `trial` | global trial index `nomor` (20 distinct values per subject) |
| `window` | window index within the trial (1-15) |
| `label` | `verbal` / `visual`, or `active` / `reflective` (`aktif` / `reflektif` also accepted) |
| `Fp1-Fp2`, ..., `O1-O2` | 28 upper-triangular channel-pair values (i < j; 8 x 7 / 2), no diagonal, no duplicated pairs |

Features are stored without normalization; every analysis script applies subject-wise z-scoring.
