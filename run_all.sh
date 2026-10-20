#!/usr/bin/env bash
# Reproduces steps 01-09 for both dimensions from the released feature files in data/.
# Steps 02-04 run before step 05, which loads their subject-level predictions.
# Step 04 (LSTM-CNN) needs the filtered EEG windows in data/raw_eeg_*/ and is skipped when they are absent.
# Steps 00a-00b (raw recordings -> features) are run separately; see README.md.
set -euo pipefail
cd "$(dirname "$0")"

for DIM in VV AR; do
  dim=$(echo "$DIM" | tr '[:upper:]' '[:lower:]')
  echo "================ ${DIM} ================"
  python scripts/01_wcss_profiles.py         --dimension "$DIM"
  python scripts/02_sota_anova_ml.py         --dimension "$DIM"
  python scripts/03_sota_mil_rf.py           --dimension "$DIM"
  if ls "data/raw_eeg_${dim}"/*.csv >/dev/null 2>&1; then
    python scripts/04_sota_lstm_cnn.py       --dimension "$DIM"
  else
    echo "[skip] 04_sota_lstm_cnn.py: no files in data/raw_eeg_${dim}/ (using results_lstmcnn/ if present)"
  fi
  python scripts/05_compass_svp_benchmark.py --dimension "$DIM"
  python scripts/06_compass_svp_ablation.py  --dimension "$DIM" --feature plv
  python scripts/06_compass_svp_ablation.py  --dimension "$DIM" --feature wpli
  python scripts/07_compass_svp_geometry.py  --dimension "$DIM"
done
python scripts/08_kmeans_robustness.py
python scripts/09_fusion_levels_figure.py
echo "All results written to results_*/"
