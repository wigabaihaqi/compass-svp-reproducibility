# Analysis plan v1: additional analyses for the new manuscript

- **Approved by the first author:** 30 October 2026, before any of the analyses below was run.
- **Locked:** by the git tag `analysis-plan-v1`. Any later change is recorded as a dated amendment at the end of
  this file, with its reason, and is reported in the manuscript.
- **Translation:** this file translates the plan approved in Indonesian; the content is unchanged.

## Aims

Three analyses are added and reported whatever their outcome, including null results:

1. Show what the COMPASS-SVP prototypes encode as connectivity graphs, and how stable they are across LOSO folds.
2. Test whether class differences in connectivity hold at the subject level.
3. Quantify how much random window-level cross-validation inflates accuracy compared with LOSO, for the same
   models and data.

## Common settings

- **Data:** RAPM PLV and wPLI features of this repository, with the corrected subject-8 AR file (see
  `reconciliation/`). There are 10 VV and 18 AR subjects, and no subject is excluded.
- **Models:** the main protocol is used throughout:
  - RBF-SVM: C = 1, gamma = 'scale', balanced class weights;
  - K-Means: K = 2, n_init = 10;
  - seed 42;
  - both distance metrics.

## Analysis 1: COMPASS-SVP prototype graphs (descriptive)

Each prototype is a vector of 28 channel-pair values, so it can be drawn directly as an 8-channel graph.

1. **Full-data prototypes:** fit COMPASS-SVP on all subjects of each dimension (PLV) and extract the 4 prototypes
   (2 per class).
2. **Fold prototypes:** extract the prototypes of every LOSO fold (VV 10 folds, AR 18 folds). Match each fold's
   prototypes to the full-data prototypes of the same class by the Hungarian algorithm on correlation.
3. **Stability:**
   - the correlation of each fold prototype with its matched full-data prototype (median and range);
   - the mean and SD of each edge across folds.
4. **Figures:**
   - one graph per prototype;
   - a difference graph (mean class-1 prototype minus mean class-0 prototype);
   - line width and colour show the edge value;
   - a dashed line marks an edge whose sign is inconsistent in more than 20% of folds.

**Reported:**
- **Figures and tables:** all graphs and the stability table, in units of within-subject z-score (stated in the
  captions).
- **wPLI:** the same graphs are drawn as a supplement, with no inference.
- **Interpretation:** limited to "the class-X prototype emphasises edge Y in the training data". No
  neurophysiological mechanism is claimed.

## Analysis 2: subject-level edge tests

**Null hypothesis:** no edge differs between the classes (Verbal vs. Visual; Active vs. Reflective).

1. **Subject values:** for each subject, the mean raw PLV of each of the 28 edges over its 300 windows. Raw values
   are used because within-subject z-scoring sets every subject's mean to zero.
2. **Edge statistics:**
   - difference of class means (Welch t);
   - Hedges' g with 95% CI.
3. **Exact subject-label permutation:**
   - VV: all 252 assignments of 5 vs. 5;
   - AR: all 48,620 assignments of 9 vs. 9.
4. **Multiple-comparison correction within each family of 28 edges:**
   - primary: permutation max-|t|, controlling the FWER at 0.05;
   - supporting: Benjamini-Hochberg at q = 0.05.

**Families:**

| Family | Role |
|---|---|
| VV, PLV | primary |
| AR, PLV | primary |
| VV, wPLI | volume-conduction control |
| AR, wPLI | volume-conduction control |

**Reported:**
- the full 28-edge table per family: difference, g with 95% CI, raw p, FWER p and FDR q;
- a graph of the edges that survive correction, or a statement that none does;
- descriptively, whether the direction agrees with the Analysis 1 difference graph.

**Notes:**
- Two participants contribute to both dimensions (VV1 = AR11, VV2 = AR12). This does not affect within-dimension
  tests and is stated in the Methods.
- With 10 and 18 subjects, power for small effects is low. A null result is not evidence of equal connectivity,
  and this is stated explicitly.

## Analysis 3: random cross-validation vs. LOSO

1. **Models:** COMPASS-SVP (Euclidean and Cosine), SVM, RF, MLP and KNN, with the settings of the main benchmark.
2. **Random CV:** stratified 5-fold CV at window level, shuffled with seed 42, so that windows of the same subject
   appear in both training and test folds.
3. **Comparison:** the LOSO window accuracy of the main benchmark (existing results).
4. **Repetitions:** 10 shuffling seeds (0–9), reported as mean and SD.

**Reported:**
- **Table:** window accuracy under random CV vs. LOSO per model, for VV and AR (PLV), with the difference.
- **Not computed:** subject-level accuracy under random CV, because a subject is split across folds.
- **Interpretation:** limited to how much the models exploit similarity between windows of the same subject. These
  figures are not used to claim any accuracy for COMPASS-SVP.

## Out of scope (not done in this manuscript)

- excluding subjects, changing frequency bands, or re-tuning model parameters;
- choosing a metric, tie rule or test family after seeing results;
- phase-specific (phase 1 vs. 2) or band-specific tests;
- HMT-S data.

## Implementation

| Script | Analysis | Main output |
|---|---|---|
| `scripts/12_prototype_graphs.py` | 1 | `results_prototypes/` |
| `scripts/13_edge_statistics.py` | 2 | `results_edges/` |
| `scripts/14_random_cv_vs_loso.py` | 3 | `results_cv_comparison/` |

## Amendments

None.
