#!/usr/bin/env bash
# Amendment 1 (analysis_plan/analysis_plan_v1.md): recompute every subject-level result with the symmetric
# tie rule. Step 04 (LSTM-CNN) is not rerun: it has no tied subjects, so its results do not change.
# Steps 12-14 do not use subject-level votes. Logs are written to logs_amendment1/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs_amendment1
for DIM in VV AR; do
  python scripts/02_sota_anova_ml.py         --dimension "$DIM" | tee "logs_amendment1/02_${DIM}.txt"
  python scripts/03_sota_mil_rf.py           --dimension "$DIM" | tee "logs_amendment1/03_${DIM}.txt"
  python scripts/05_compass_svp_benchmark.py --dimension "$DIM" | tee "logs_amendment1/05_${DIM}.txt"
  python scripts/06_compass_svp_ablation.py  --dimension "$DIM" --feature plv  | tee "logs_amendment1/06_${DIM}_plv.txt"
  python scripts/06_compass_svp_ablation.py  --dimension "$DIM" --feature wpli | tee "logs_amendment1/06_${DIM}_wpli.txt"
done
python scripts/08_kmeans_robustness.py   | tee logs_amendment1/08.txt
python scripts/09_fusion_levels_figure.py | tee logs_amendment1/09.txt
python scripts/10_label_permutation.py --dimension VV             | tee logs_amendment1/10_VV.txt
python scripts/10_label_permutation.py --dimension AR --n-perm 300 | tee logs_amendment1/10_AR.txt
python scripts/11_reconciliation.py      | tee logs_amendment1/11.txt
python -c "import sklearn, numpy, pandas, scipy, sys; print(sys.version); print('sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'scipy', scipy.__version__)" > logs_amendment1/versions.txt
echo "Amendment 1 rerun finished."
