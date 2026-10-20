"""
Step 2 - ANOVA feature selection + classical ML, adapted from Yuvaraj et al. (2024) [12].

Adaptation: the same 28 subject-wise z-scored PLV features as COMPASS-SVP (the original
study used time-domain descriptors); ANOVA F-value selection of the top 30% features,
fitted on the training subjects of each LOSO fold only; RF / KNN / MLP classifiers;
hierarchical majority voting (window -> trial -> subject).

Usage:  python scripts/02_sota_anova_ml.py --dimension VV
Output: results_anova_ml/
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import common as C

warnings.filterwarnings('ignore')
ANOVA_PERCENTILE = 30

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dimension', choices=['VV', 'AR'], required=True)
args = parser.parse_args()

dim = args.dimension
out_dir = C.output_dir('anova_ml')
df, feature_cols, subjects = C.load_features(dim, 'plv')

MODELS = {   # name -> (file tag, factory)
    'Random Forest + ANOVA [15]': ('random_forest_anova',
                                   lambda: RandomForestClassifier(n_estimators=100, random_state=C.SEED)),
    'KNN + ANOVA [15]':           ('knn_anova', lambda: KNeighborsClassifier(n_neighbors=5)),
    'MLP + ANOVA [15]':           ('mlp_anova',
                                   lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=C.SEED)),
}

print(f"ANOVA (top {ANOVA_PERCENTILE}%) + RF/KNN/MLP [15] | {C.DIMENSION_NAMES[dim]} | subjects: {len(subjects)}")

window_preds = {name: [] for name in MODELS}
feature_log = []
for fold, test_subj in enumerate(subjects, start=1):
    C.set_seed()
    tr, te = df[df['subject'] != test_subj], df[df['subject'] == test_subj]
    selector = SelectPercentile(f_classif, percentile=ANOVA_PERCENTILE)
    X_tr = selector.fit_transform(tr[feature_cols].values, tr['y'].values)   # training subjects only
    X_te = selector.transform(te[feature_cols].values)
    feature_log.append({'fold': fold, 'test_subject': test_subj,
                        'selected_features': ';'.join(np.array(feature_cols)[selector.get_support()])})
    for name, (_, factory) in MODELS.items():
        clf = factory().fit(X_tr, tr['y'].values)
        window_preds[name].append(te[['subject', 'trial', 'y']].assign(y_pred=clf.predict(X_te), fold=fold))
    print(f"  Fold {fold:>2}/{len(subjects)} | held-out subject {test_subj}")

summary = {}
for name, (tag, _) in MODELS.items():
    units = pd.concat(window_preds[name], ignore_index=True)
    trials, subjects_df = C.aggregate(units)
    summary[name] = C.summarize(units, trials, subjects_df)
    C.print_subject_report(name, dim, subjects_df, summary[name])
    C.export_predictions(out_dir, f'{dim.lower()}_{tag}', units, trials, subjects_df)

res = pd.DataFrame(summary).T
print("\n" + res.to_string())
res.to_csv(os.path.join(out_dir, f'{dim.lower()}_anova_ml_summary.csv'))
pd.DataFrame(feature_log).to_csv(os.path.join(out_dir, f'{dim.lower()}_anova_selected_features_per_fold.csv'),
                                 index=False)
print(f"\nSaved to {out_dir}/")
