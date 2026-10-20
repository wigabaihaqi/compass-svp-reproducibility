"""
Step 5 - COMPASS-SVP benchmark and subject-level statistics (Tables 1, 3 and Section 3.6).

Models trained here (subject-wise z-scored PLV features, strict LOSO):
  COMPASS-SVP (Euclidean / Cosine): RBF-SVM sifting (alpha_i > 0) -> intra-class K-Means (K = 2)
                                     -> nearest-prototype inference
  Baselines: SVM, Random Forest, MLP, KNN (default settings, Section 2.6)

Statistics (n = independent subjects): Clopper-Pearson exact 95% CI for every model and, for each
COMPASS-SVP variant vs. every other model, the paired 2x2 table, exact McNemar test, exact
(fully enumerated) paired permutation test, and Holm correction. Predictions from steps 2-4 are
loaded automatically from results_anova_ml/, results_mil_rf/ and results_lstmcnn/ when present.

Usage:  python scripts/05_compass_svp_benchmark.py --dimension VV
Output: results_compass/
"""

import argparse
import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

import common as C

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dimension', choices=['VV', 'AR'], required=True)
args = parser.parse_args()

dim = args.dimension
tag = dim.lower()
out_dir = C.output_dir('compass')
df, feature_cols, subjects = C.load_features(dim, 'plv')


def svm_factory():
    return SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=C.SEED)


BASELINES = {
    'SVM (Baseline)':           svm_factory,
    'Random Forest (Baseline)': lambda: RandomForestClassifier(n_estimators=100, random_state=C.SEED),
    'MLP (Baseline)':           lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=C.SEED),
    'KNN (Baseline)':           lambda: KNeighborsClassifier(n_neighbors=5),
}
COMPASS = {'COMPASS-SVP (Euclidean)': 'euclidean', 'COMPASS-SVP (Cosine)': 'cosine'}

EXTERNAL = {   # model name -> subject-level prediction file produced by steps 2-4
    'Adapted LSTM-CNN [14]':      (C.OUTPUT_DIRS['lstmcnn'],  f'lstmcnn_{tag}_subject_predictions.csv'),
    'Random Forest + ANOVA [15]': (C.OUTPUT_DIRS['anova_ml'], f'{tag}_random_forest_anova_subject_predictions.csv'),
    'KNN + ANOVA [15]':           (C.OUTPUT_DIRS['anova_ml'], f'{tag}_knn_anova_subject_predictions.csv'),
    'MLP + ANOVA [15]':           (C.OUTPUT_DIRS['anova_ml'], f'{tag}_mlp_anova_subject_predictions.csv'),
    'MIL-RF [11]':                (C.OUTPUT_DIRS['mil_rf'],   f'{tag}_milrf_subject_predictions.csv'),
}

print(f"COMPASS-SVP benchmark | {C.DIMENSION_NAMES[dim]} | subjects: {len(subjects)} | "
      f"features: {len(feature_cols)} | K = {C.K_PROTOTYPES}")

# ==========================================
# 1. STRICT LOSO
# ==========================================
names = list(COMPASS) + list(BASELINES)
window_preds = {n: [] for n in names}
fold_log = []
for fold, test_subj in enumerate(subjects, start=1):
    C.set_seed()
    tr, te = df[df['subject'] != test_subj], df[df['subject'] == test_subj]
    X_tr, y_tr, X_te = tr[feature_cols].values, tr['y'].values, te[feature_cols].values
    base = te[['subject', 'trial', 'y']]

    # COMPASS-SVP: stage 1 (sifting) + stage 2 (condensation) shared by both metrics
    svm = svm_factory().fit(X_tr, y_tr)
    X_sv, y_sv = svm.support_vectors_, y_tr[svm.support_]
    protos, proto_labels = [], []
    for c in (0, 1):
        km = KMeans(n_clusters=min(C.K_PROTOTYPES, int((y_sv == c).sum())), n_init=C.KMEANS_N_INIT,
                    random_state=C.SEED).fit(X_sv[y_sv == c])
        protos.append(km.cluster_centers_)
        proto_labels.append(np.full(len(km.cluster_centers_), c))
    protos, proto_labels = np.vstack(protos), np.concatenate(proto_labels)
    for name, metric in COMPASS.items():   # stage 3: nearest-prototype inference
        pred = proto_labels[np.argmin(cdist(X_te, protos, metric=metric), axis=1)]
        window_preds[name].append(base.assign(y_pred=pred, fold=fold))

    window_preds['SVM (Baseline)'].append(base.assign(y_pred=svm.predict(X_te), fold=fold))
    for name, factory in BASELINES.items():
        if name != 'SVM (Baseline)':
            window_preds[name].append(base.assign(y_pred=factory().fit(X_tr, y_tr).predict(X_te), fold=fold))

    fold_log.append({'fold': fold, 'test_subject': test_subj, 'n_train_windows': len(y_tr),
                     'n_support_vectors': len(y_sv), 'sv_fraction_%': round(len(y_sv) / len(y_tr) * 100, 2),
                     'n_sv_class0': int((y_sv == 0).sum()), 'n_sv_class1': int((y_sv == 1).sum())})
    print(f"  Fold {fold:>2}/{len(subjects)} | held-out subject {test_subj} | support vectors: {len(y_sv)} "
          f"({len(y_sv) / len(y_tr) * 100:.1f}% of training windows)")

# ==========================================
# 2. HIERARCHICAL VOTING, METRICS, EXPORT
# ==========================================
summary, subject_preds = {}, {}
for name in names:
    units = pd.concat(window_preds[name], ignore_index=True)
    trials, subj = C.aggregate(units)
    summary[name] = C.summarize(units, trials, subj)
    subject_preds[name] = subj[['subject', 'true_class', 'pred_subject']]
    C.print_subject_report(name, dim, subj, summary[name])
    C.export_predictions(out_dir, f'{tag}_{C.safe_name(name)}', units, trials, subj)

res = pd.DataFrame(summary).T
print("\n" + "=" * 120 + f"\n BENCHMARK SUMMARY ({dim})\n" + "=" * 120)
print(res.drop(columns=['TN', 'FP', 'FN', 'TP']).to_string())
res.to_csv(os.path.join(out_dir, f'{tag}_benchmark_summary.csv'))
pd.DataFrame(fold_log).to_csv(os.path.join(out_dir, f'{tag}_compass_fold_log.csv'), index=False)

# ==========================================
# 3. SUBJECT-LEVEL STATISTICS
# ==========================================
for ext_name, (folder, fname) in EXTERNAL.items():
    path = os.path.join(C.RESULTS_DIR, folder, fname)
    if not os.path.isfile(path):
        print(f"[SKIPPED] {ext_name}: {path} not found (run the corresponding step first)")
        continue
    ext = pd.read_csv(path)[['subject', 'true_class', 'pred_subject']]
    ext['subject'] = ext['subject'].apply(C.clean_subject_id)
    subject_preds[ext_name] = ext

ref_ids = set(subject_preds['COMPASS-SVP (Euclidean)']['subject'])
print("\nSubject ID check:")
for name, d in subject_preds.items():
    ok = set(d['subject']) == ref_ids and len(d) == len(ref_ids)
    if not ok:
        raise ValueError(f"Subject IDs of '{name}' do not match the PLV feature file.\n"
                         f"  only in PLV file: {sorted(ref_ids - set(d['subject']))}\n"
                         f"  only in {name}: {sorted(set(d['subject']) - ref_ids)}")
print(f"  all {len(subject_preds)} models share the same {len(ref_ids)} subjects [OK]")


def exact_permutation_p(ca, cb):
    d = ca.astype(int) - cb.astype(int)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    obs = abs(d.sum())
    return sum(abs(np.dot(s, np.abs(d))) >= obs for s in product([1, -1], repeat=len(d))) / 2 ** len(d)


rows = []
for i, comp in enumerate(COMPASS):
    block = []
    for other in subject_preds:
        if other == comp or other in list(COMPASS)[:i]:
            continue
        m = subject_preds[comp].merge(subject_preds[other], on='subject', suffixes=('_a', '_b'))
        if not (m['true_class_a'] == m['true_class_b']).all():
            raise ValueError(f"Ground-truth labels of '{other}' differ from the PLV feature file.")
        ca = (m['pred_subject_a'] == m['true_class_a']).values
        cb = (m['pred_subject_b'] == m['true_class_a']).values
        b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
        table = [[int((ca & cb).sum()), b], [c, int((~ca & ~cb).sum())]]
        block.append({'model_A': comp, 'model_B': other, 'n_subjects': len(m),
                      'both_correct': table[0][0], 'only_A_correct (b)': b, 'only_B_correct (c)': c,
                      'both_wrong': table[1][1], 'delta_acc_pp': round((ca.mean() - cb.mean()) * 100, 2),
                      'p_mcnemar_exact': round(mcnemar(table, exact=True).pvalue, 4),
                      'p_permutation_exact': round(exact_permutation_p(ca, cb), 4)})
    for r, p in zip(block, multipletests([r['p_mcnemar_exact'] for r in block], method='holm')[1]):
        r['p_mcnemar_holm'] = round(p, 4)
    rows.extend(block)

stats = pd.DataFrame(rows)
print("\n" + "=" * 120 + f"\n PAIRED SUBJECT-LEVEL TESTS ({dim})\n" + "=" * 120)
print(stats[['model_A', 'model_B', 'only_A_correct (b)', 'only_B_correct (c)', 'delta_acc_pp',
             'p_mcnemar_exact', 'p_permutation_exact', 'p_mcnemar_holm']].to_string(index=False))
stats.to_csv(os.path.join(out_dir, f'{tag}_subject_level_statistics.csv'), index=False)

ci_rows = []
for name, d in subject_preds.items():
    k = int((d['pred_subject'] == d['true_class']).sum())
    lo, hi = C.exact_ci(k, len(d))
    ci_rows.append({'model': name, 'correct': k, 'n': len(d), 'acc_%': round(k / len(d) * 100, 2),
                    'ci_low_%': round(lo, 1), 'ci_high_%': round(hi, 1)})
pd.DataFrame(ci_rows).to_csv(os.path.join(out_dir, f'{tag}_all_models_subject_accuracy_ci.csv'), index=False)
print(f"\nSaved to {out_dir}/")
