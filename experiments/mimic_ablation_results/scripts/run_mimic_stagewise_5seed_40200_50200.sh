#!/bin/bash

set -e
set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_ROOT="experiments/mimic_ablation_results/runs/paper_evidence_supplements_5seed_${EXP_TIME}"
RUN_LOG="${RESULT_ROOT}/run_40200_50200.log"
STATUS_FILE="${RESULT_ROOT}/status.txt"
ROOT_POINTER="current_5seed_supp_root.txt"

mkdir -p "$RESULT_ROOT"
echo "$RESULT_ROOT" > "$ROOT_POINTER"

if [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$RUN_LOG") 2>&1
fi

write_status() {
  cat > "$STATUS_FILE" <<EOF
timestamp=$(date +"%F %T")
step=$1
message=$2
result_root=${RESULT_ROOT}
training_summary=${RESULT_ROOT}/new_multiseed_stage2/summary.csv
clinical_summary=${RESULT_ROOT}/multiseed_clinical_eval/summary.csv
EOF
}

echo "============================================================"
echo ">>> MIMIC 5-seed supplement: running new seeds 40200/50200"
echo ">>> result_root=${RESULT_ROOT}"
echo ">>> root_pointer=${ROOT_POINTER}"
echo "============================================================"

write_status "TRAIN" "running S0/S1/S2 for seeds 40200 and 50200"

env -u CMCRL_AUTO_LOGGED \
  CUDA=0 \
  RESULT_ROOT="${RESULT_ROOT}/new_multiseed_stage2" \
  SEEDS_STR="40200 50200" \
  EPOCHS=3 \
  RUN_S0=1 \
  RUN_S1=1 \
  RUN_S2=1 \
  bash experiments/mimic_ablation_results/scripts/run_multiseed_stage2_supplements.sh

write_status "EVAL" "running test report dump and clinical metrics"

env -u CMCRL_AUTO_LOGGED \
  CUDA=0 \
  RESULT_ROOT="${RESULT_ROOT}" \
  MULTISEED_STAGE2_ROOT="${RESULT_ROOT}/new_multiseed_stage2" \
  NEW_STAGE2_SEEDS_STR="40200 50200" \
  RUN_STRONG_BASELINE_MULTI_SEED=0 \
  RUN_NEW_STAGE2_MULTI_SEED=1 \
  RUN_MULTI_SEED_CLINICAL=1 \
  RUN_CCD_VAL_SELECTION=0 \
  UPDATE_SUMMARY=0 \
  bash experiments/mimic_ablation_results/scripts/run_paper_evidence_supplements.sh

write_status "DONE" "completed"

echo ""
echo "============================================================"
echo ">>> Finished MIMIC 40200/50200 stage-wise supplement"
echo ">>> result_root=${RESULT_ROOT}"
echo ">>> training_summary=${RESULT_ROOT}/new_multiseed_stage2/summary.csv"
echo ">>> clinical_summary=${RESULT_ROOT}/multiseed_clinical_eval/summary.csv"
echo "============================================================"
