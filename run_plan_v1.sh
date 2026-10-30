#!/usr/bin/env bash
# Runs the three analyses of analysis_plan/analysis_plan_v1.md (tag analysis-plan-v1) on the released features.
# Step 14 reads the LOSO window accuracies from results_compass/ (step 05), which are already in the repository.
# Console output of every step is saved in logs_plan_v1/.
set -euo pipefail
cd "$(dirname "$0")"
LOG=logs_plan_v1
mkdir -p "$LOG"

run () {   # run <log-name> <script> [args...]
  local name=$1; shift
  echo ">>> python $*"
  python "$@" 2>&1 | tee "$LOG/$name.txt"
}

for DIM in VV AR; do
  for FEAT in plv wpli; do
    run "12_${DIM}_${FEAT}" scripts/12_prototype_graphs.py --dimension "$DIM" --feature "$FEAT"
    run "13_${DIM}_${FEAT}" scripts/13_edge_statistics.py  --dimension "$DIM" --feature "$FEAT"
  done
done
for DIM in VV AR; do
  run "14_${DIM}" scripts/14_random_cv_vs_loso.py --dimension "$DIM"
done
echo "Done. Results in results_prototypes/, results_edges/, results_cv_comparison/; logs in $LOG/"
