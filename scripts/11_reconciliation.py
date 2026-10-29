"""
Step 11 - Reconciliation of the first submission with the revised results (response to Reviewer 3).

Traces every Table 2 result through the documented changes, without retraining any model:

  v1  first submission : stored outputs of the original notebooks (reconciliation/v1_notebooks/)
  A0  revised code, original AR input data, first-submission tie rule
      = commit c482849 predictions with tied AR subject votes (10 vs 10 trials) resolved to Active,
        as in the original AR notebook (VV notebook and revised code: ties -> Class 1)
  A   revised code, original AR input data (commit c482849)
  B   revised code, corrected AR input file of subject 8 (current results_compass/)

  v1 -> A0 : K-Means n_init 'auto' (one initialization) -> 10; software environment (only the
             stochastic baselines RF and MLP are affected; SVM and KNN are identical at window level)
  A0 -> A  : AR subject-level tie rule unified with VV (ties -> Class 1); VV unaffected
  A  -> B  : AR input file of subject 8 corrected; VV data identical in all states

Usage:  python scripts/11_reconciliation.py
Needs:  git (to read commit c482849) and the archived notebooks in reconciliation/v1_notebooks/
Output: reconciliation/
          table2_v1_check.csv          first-manuscript Table 2 vs. the stored notebook output
          stage_summary.csv            win/trial/subject metrics and subject confusion matrix per state
          subject_diffs_vv.csv / _ar.csv  subject-by-subject predictions and trial votes per state
          paired_transitions.csv       paired 2x2 (correct/incorrect) tables and exact McNemar p per change
          s8_feature_check.csv         subject-8 vs. subject-9 features before and after the correction
"""

import io
import json
import os
import re
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(REPO, 'reconciliation')
NB_DIR = os.path.join(REC, 'v1_notebooks')
COMMIT_A = 'c482849'

DIMS = {'VV': ('verbal', 'visual'), 'AR': ('active', 'reflective')}
# model key -> (name in the v1 notebook output per dimension, file stem in results_compass/)
MODELS = {
    'COMPASS-SVP (Euclidean)': ({'VV': 'COMPASS-SVM (Euclidean)', 'AR': 'COMPASS-SVP (Euclidean)'}, 'compass_svp_euclidean'),
    'COMPASS-SVP (Cosine)':    ({'VV': 'COMPASS-SVM (Cosine)', 'AR': 'COMPASS-SVP (Cosine)'}, 'compass_svp_cosine'),
    'SVM (Baseline)':          ({'VV': 'SVM (Baseline)', 'AR': 'SVM (Baseline)'}, 'svm_baseline'),
    'Random Forest (Baseline)': ({'VV': 'Random Forest (Baseline)', 'AR': 'Random Forest (Baseline)'}, 'random_forest_baseline'),
    'MLP (Baseline)':          ({'VV': 'MLP (Baseline)', 'AR': 'MLP (Baseline)'}, 'mlp_baseline'),
    'KNN (Baseline)':          ({'VV': 'KNN (Baseline)', 'AR': 'KNN (Baseline)'}, 'knn_baseline'),
}
# Table 2 of the first submission as printed in the manuscript (Win, Trial, Subj acc, Subj F1; %)
TABLE2_V1 = {
    ('VV', 'COMPASS-SVP (Euclidean)'): (68.80, 74.00, 80.00, 80.00),
    ('VV', 'COMPASS-SVP (Cosine)'):    (42.15, 38.00, 30.00, 28.50),
    ('VV', 'SVM (Baseline)'):          (55.47, 56.00, 70.00, 69.70),
    ('VV', 'MLP (Baseline)'):          (35.83, 36.00, 40.00, 37.50),
    ('VV', 'KNN (Baseline)'):          (33.80, 34.50, 40.00, 37.50),
    ('AR', 'COMPASS-SVP (Cosine)'):    (53.98, 55.56, 61.11, 60.99),
    ('AR', 'COMPASS-SVP (Euclidean)'): (51.20, 52.78, 55.56, 54.80),
    ('AR', 'KNN (Baseline)'):          (56.15, 51.11, 50.00, 49.85),
    ('AR', 'SVM (Baseline)'):          (54.59, 48.61, 44.44, 43.75),
    ('AR', 'MLP (Baseline)'):          (55.37, 47.50, 44.44, 43.75),
    ('AR', 'Random Forest (Baseline)'): (50.74, 46.11, 44.44, 43.75),
}
STAGES = ['v1', 'A0', 'A', 'B']
TRANSITIONS = [('v1->A0 (n_init; software)', 'v1', 'A0'), ('A0->A (AR subject tie rule)', 'A0', 'A'),
               ('A->B (AR subject-8 data)', 'A', 'B'), ('v1->B (total)', 'v1', 'B')]


# ---------------------------------------------------------------- readers
def notebook_text(dim):
    """Concatenated text output of the benchmark cell (first evaluation of all models) of a v1 notebook."""
    nb = json.load(open(os.path.join(NB_DIR, f'COMPASS_SVP_Evaluation_plv_{dim}.ipynb'), encoding='utf-8'))
    for cell in nb['cells']:
        text = ''.join(''.join(o.get('text', '')) for o in cell.get('outputs', []))
        if cell['cell_type'] == 'code' and 'FINAL BENCHMARK SUMMARY' in text:
            return text
    raise RuntimeError(f'benchmark cell not found in the {dim} notebook')


def v1_subjects(text, nb_name, dim):
    """Subject-level table printed by the v1 notebook for one model."""
    start = text.index(f'SUBJECT-LEVEL PERFORMANCE ({nb_name})')
    rows = []
    for line in text[start:].splitlines()[1:]:
        m = re.match(r'\s*(\d+)\s+(\w+)\s+(\w+)\s+(\d+)\s+(\d+)\s+(\w+)\s*$', line)
        if m:
            rows.append({'subject': int(m.group(1)), 'true_class': DIMS[dim].index(m.group(2).lower()),
                         'pred_subject': DIMS[dim].index(m.group(3).lower()), 'trials_class1': int(m.group(5))})
        elif rows:
            break
    return pd.DataFrame(rows)


def v1_summary(text, nb_name):
    """Win/trial/subject metrics from the final summary table of the v1 notebook."""
    block = text[text.rindex('FINAL BENCHMARK SUMMARY'):]
    for line in block.splitlines():
        if line.startswith(nb_name + ' '):
            v = [float(x) for x in line[len(nb_name):].split()]
            return {'win_acc_%': v[0], 'trial_acc_%': v[1], 'subj_acc_%': v[2], 'subj_macro_f1_%': v[5]}
    raise RuntimeError(f'{nb_name} not in summary table')


def git_csv(path):
    out = subprocess.check_output(['git', '-C', REPO, 'show', f'{COMMIT_A}:{path}'])
    return pd.read_csv(io.BytesIO(out))


def results_csv(stage, path):
    return git_csv(path) if stage == 'A' else pd.read_csv(os.path.join(REPO, path))


def repo_summary(stage, dim, model):
    s = results_csv(stage, f'results_compass/{dim.lower()}_benchmark_summary.csv').set_index('Unnamed: 0')
    r = s.loc[model]
    return {'win_acc_%': r['Win Acc (%)'], 'trial_acc_%': r['Trial Acc (%)'],
            'subj_acc_%': r['Subj Acc (%)'], 'subj_macro_f1_%': r['Subj Macro F1 (%)']}


# ---------------------------------------------------------------- helpers
def confusion(df):
    t, p = df['true_class'].values, df['pred_subject'].values
    return {'TN': int(((t == 0) & (p == 0)).sum()), 'FP': int(((t == 0) & (p == 1)).sum()),
            'FN': int(((t == 1) & (p == 0)).sum()), 'TP': int(((t == 1) & (p == 1)).sum())}


def mcnemar_exact(b, c):
    return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue


# ---------------------------------------------------------------- main
os.makedirs(REC, exist_ok=True)
check_rows, summary_rows, transition_rows = [], [], []

for dim, names in DIMS.items():
    text = notebook_text(dim)
    diff_rows = []
    for model, (nb_names, stem) in MODELS.items():
        preds = {'v1': v1_subjects(text, nb_names[dim], dim)}
        for st in ('A', 'B'):
            preds[st] = results_csv(st, f'results_compass/{dim.lower()}_{stem}_subject_predictions.csv')
        preds['A0'] = preds['A'].copy()
        if dim == 'AR':   # original AR notebook: active_trials >= reflective_trials -> Active
            preds['A0']['pred_subject'] = (preds['A0']['trials_class1'] * 2 > preds['A0']['n_trials']).astype(int)
        n = len(preds['B'])
        for st in STAGES:
            assert len(preds[st]) == n and (preds[st]['true_class'].values == preds['B']['true_class'].values).all(), \
                f'subject order/labels differ: {dim} {model} {st}'

        # metrics per state
        for st in STAGES:
            m = v1_summary(text, nb_names[dim]) if st == 'v1' else repo_summary('A' if st == 'A0' else st, dim, model)
            if st == 'A0':   # window/trial levels are those of A; subject level recomputed with the old tie rule
                t, p = preds[st]['true_class'], preds[st]['pred_subject']
                m.update({'subj_acc_%': round(accuracy_score(t, p) * 100, 2),
                          'subj_macro_f1_%': round(f1_score(t, p, average='macro', zero_division=0) * 100, 2)})
            correct = int((preds[st]['pred_subject'] == preds[st]['true_class']).sum())
            summary_rows.append({'dimension': dim, 'model': model, 'state': st, **m,
                                 'subj_correct': f'{correct}/{n}', **confusion(preds[st])})
        # first-manuscript table vs. notebook
        if (dim, model) in TABLE2_V1:
            nbv = v1_summary(text, nb_names[dim])
            for col, ms in zip(['win_acc_%', 'trial_acc_%', 'subj_acc_%', 'subj_macro_f1_%'], TABLE2_V1[(dim, model)]):
                check_rows.append({'dimension': dim, 'model': model, 'metric': col, 'manuscript_v1': ms,
                                   'notebook_v1': nbv[col],
                                   'status': 'match' if abs(ms - nbv[col]) < 0.005 else 'transcription error'})
        # subject-by-subject diffs
        ok = {st: (preds[st]['pred_subject'] == preds[st]['true_class']).values for st in STAGES}
        for i in range(n):
            changed = [lab.split(' ')[0] for lab, a, b in TRANSITIONS[:3] if ok[a][i] != ok[b][i]]
            row = {'model': model, 'subject': int(preds['B']['subject'][i]),
                   'true_class': names[int(preds['B']['true_class'][i])]}
            for st in STAGES:
                row[f'{st}_pred'] = names[int(preds[st]['pred_subject'][i])]
                row[f'{st}_trials_class1_of_20'] = int(preds[st]['trials_class1'][i])
                row[f'{st}_correct'] = bool(ok[st][i])
            row['correctness_changed_in'] = ' '.join(changed) or '-'
            diff_rows.append(row)
        # paired 2x2 tables
        for lab, a, b in TRANSITIONS:
            if dim == 'VV' and a in ('A0', 'A') and b in ('A', 'B'):
                assert (ok[a] == ok[b]).all(), f'VV predictions changed in {lab} although nothing changed for VV'
            bc, cb = int((ok[a] & ~ok[b]).sum()), int((~ok[a] & ok[b]).sum())
            transition_rows.append({'dimension': dim, 'model': model, 'change': lab,
                                    'correct_before': int(ok[a].sum()), 'correct_after': int(ok[b].sum()),
                                    'both_correct': int((ok[a] & ok[b]).sum()), 'only_before_correct': bc,
                                    'only_after_correct': cb, 'both_wrong': int((~ok[a] & ~ok[b]).sum()),
                                    'subjects_lost': ' '.join(str(s) for s in preds['B']['subject'][ok[a] & ~ok[b]]) or '-',
                                    'subjects_gained': ' '.join(str(s) for s in preds['B']['subject'][~ok[a] & ok[b]]) or '-',
                                    'p_mcnemar_exact': round(mcnemar_exact(bc, cb), 4)})
    pd.DataFrame(diff_rows).to_csv(os.path.join(REC, f'subject_diffs_{dim.lower()}.csv'), index=False)

pd.DataFrame(check_rows).to_csv(os.path.join(REC, 'table2_v1_check.csv'), index=False)
pd.DataFrame(summary_rows).to_csv(os.path.join(REC, 'stage_summary.csv'), index=False)
pd.DataFrame(transition_rows).to_csv(os.path.join(REC, 'paired_transitions.csv'), index=False)

# subject-8 feature check (AR PLV features before and after the correction)
feat_old = git_csv('data/combined_fe_plv_ar.csv')
feat_new = pd.read_csv(os.path.join(REPO, 'data', 'combined_fe_plv_ar.csv'))
cols = [c for c in feat_new.columns if '-' in c]
get = lambda d, s: d[d['subject'] == s].sort_values(['trial', 'window'])[cols].values
s8_rows = [
    {'comparison': 'original subject 8 vs subject 9', 'identical': bool(np.array_equal(get(feat_old, 8), get(feat_old, 9))),
     'trial_ids_subject8': f"{feat_old.loc[feat_old.subject == 8, 'trial'].min()}-{feat_old.loc[feat_old.subject == 8, 'trial'].max()}"},
    {'comparison': 'corrected subject 8 vs subject 9', 'identical': bool(np.array_equal(get(feat_new, 8), get(feat_new, 9))),
     'trial_ids_subject8': f"{feat_new.loc[feat_new.subject == 8, 'trial'].min()}-{feat_new.loc[feat_new.subject == 8, 'trial'].max()}"},
    {'comparison': 'all other 16 subjects, original vs corrected file',
     'identical': all(np.array_equal(get(feat_old, s), get(feat_new, s)) for s in range(1, 19) if s not in (8, 9)),
     'trial_ids_subject8': '-'},
]
pd.DataFrame(s8_rows).to_csv(os.path.join(REC, 's8_feature_check.csv'), index=False)

# console report
tr = pd.DataFrame(transition_rows)
print(tr[['dimension', 'model', 'change', 'correct_before', 'correct_after', 'subjects_lost', 'subjects_gained']]
      .to_string(index=False))
print('\nFirst-manuscript Table 2 entries that do not match the notebook output:')
ck = pd.DataFrame(check_rows)
print(ck[ck['status'] != 'match'].to_string(index=False))
print('\n' + pd.DataFrame(s8_rows).to_string(index=False))
print(f'\nSaved to {REC}/')
