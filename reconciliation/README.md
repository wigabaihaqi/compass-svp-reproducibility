# Reconciliation of the first submission with the revised results

This folder traces every entry of Table 2 from the first submission to the revised manuscript. No model is
retrained. Everything is rebuilt from three stored sources by `python scripts/11_reconciliation.py`:

| State | Source | K-Means init | AR subject tie (10 vs 10 trials) | AR subject-8 input |
|---|---|---|---|---|
| **v1**, first submission | stored outputs of the original notebooks in `v1_notebooks/` | single (`n_init='auto'`) | Active | original (contains a copy of subject 9) |
| **A0** | commit `c482849` predictions, AR ties re-resolved as in v1 | 10 restarts | Active | original |
| **A** | commit `c482849` | 10 restarts | Class 1 (Reflective), as in VV | original |
| **B**, revised manuscript | current `results_compass/` | 10 restarts | Class 1 | corrected |

Each step changes one documented factor:

- **v1 → A0:** K-Means initialization. Software versions also differ, which affects only the stochastic RF
  and MLP baselines; the deterministic SVM and KNN baselines are identical at the window, trial and subject
  levels.
- **A0 → A:** the AR subject-level tie rule. The original AR notebook resolved 10-10 subject ties to Active,
  while the VV notebook and the revised code resolve them to Class 1.
- **A → B:** correction of the subject-8 AR input file (`preprocessing_log_S8.md`).
- **VV:** the VV data are identical in all states.

## Files

| File | Content |
|---|---|
| `v1_notebooks/` | The nine notebooks of the first submission, unchanged except that participant names in two LSTM-CNN output cells were replaced by P01, P02, …. `MANIFEST.csv` gives the SHA-256 of each original and archived file. |
| `table2_v1_check.csv` | Each entry of the first-manuscript Table 2 compared with the stored notebook output. Seven entries were transcription errors. |
| `stage_summary.csv` | Window, trial and subject accuracy, subject macro-F1 and subject confusion matrix for every model in every state. |
| `subject_diffs_vv.csv`, `subject_diffs_ar.csv` | Subject-by-subject predicted class, Class 1 trial votes (of 20) and correctness in every state, with the step in which the correctness changed. |
| `paired_transitions.csv` | For each step: paired 2×2 correct/incorrect counts, the subjects lost and gained, and exact McNemar p. |
| `s8_feature_check.csv` | Original vs. corrected subject-8 features compared with subject 9 and with all other subjects. |
| `preprocessing_log_S8.md` | Row counts, hashes and structure of the original and corrected subject-8 input file. |

Subject numbering is the same in all states. The SVM and KNN predictions, which do not depend on K-Means or on
random initialization, match subject by subject between v1 and A0.
