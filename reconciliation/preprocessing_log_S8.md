# Preprocessing log: AR input file of subject 8

The window files in `data/raw_eeg_ar/` are not redistributed (see `data/README.md`). This log records what
was found in the original file and what was changed, so that the change can be audited.

## Files

| File | Rows | Trial indices (`nomor`) | SHA-256 |
|---|---|---|---|
| `active-subject8-preprocessed.csv`, original (first submission and commit c482849); kept locally as `active-subject8-preprocessed_ORIGINAL_7200rows.csv.bak` | 7,200 | 141–180 | `ae906b233a7ed630a5a45541b5c04c4175a38110299637abd4a572ab5de1672a` |
| `active-subject8-preprocessed.csv`, corrected (from commit 0975acf) | 2,400 | 141–160 | `f0b1436ab252d4a2c2bd00fe1566d9cc9525e21d9e1536123f32141b189ee376` |
| `active-subject9-preprocessed.csv`, unchanged (for reference) | 2,400 | 161–180 | `3460abbdd416c9c7f6076734600dc7eaf6b4fea379a611dab48ba8b122a14e2f` |

A correct subject file has 2,400 rows: 20 trials × 15 windows × 8 channels.

## What the original file contained

| Rows | Content |
|---|---|
| 1–2,400 | Subject 8 recording, trials 141–160 (recorded 02-07-2025, 09:59:10–10:04:10) |
| 2,401–4,800 | An exact copy of the subject 9 recording (trials 161–180, recorded 30-11-2024), labelled `subject8` |
| 4,801–7,200 | The subject 8 block of rows 1–2,400 repeated |

## Effect on the features

The feature extraction (`scripts/00b_extract_connectivity.py`; `FE-PLV-AR.ipynb` in the first submission) groups
rows by (trial, window) and only keeps groups of exactly 8 channels. As a result:

- Every subject 8 window appeared twice (16 rows), so all of them were dropped.
- The copied subject 9 windows (8 rows each) were kept.

The 300 feature windows labelled "subject 8" were therefore identical to those of subject 9. This is confirmed
in `s8_feature_check.csv`, which `scripts/11_reconciliation.py` produces from the feature files:

- **Original features:** subject 8 is identical to subject 9.
- **Corrected features:** subject 8 differs from subject 9.
- **All other 16 subjects:** identical in the original and corrected feature files.

In every LOSO fold, subject 8 or subject 9 was tested while its exact copy was in the training set. This
leakage explains why subjects 8 and 9 were both classified correctly in the first submission. For example, the
SVM baseline gave 0/20 Reflective trial votes for both.

## Correction

- **Change:** the subject 8 file was reduced to its single correct block (rows 1–2,400).
- **No other changes:** no other input file was modified, and no window was edited.
- **Rerun:** the features and all AR results were regenerated with the unchanged scripts (commit 0975acf).
- **Verification:** the corrected file is identical to rows 1–2,400 and to rows 4,801–7,200 of the original file.
