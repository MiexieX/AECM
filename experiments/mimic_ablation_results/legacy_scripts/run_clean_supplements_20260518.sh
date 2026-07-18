#!/bin/bash

set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_ROOT="${RESULT_ROOT:-results/mimic_cxr/clean_supplements_20260518}"
BATCH_DIR="${RESULT_ROOT}/batch_${EXP_TIME}"
mkdir -p "$BATCH_DIR"

MASTER_LOG="${BATCH_DIR}/run.log"
STATUS_FILE="${BATCH_DIR}/status.txt"
SUMMARY_FILE="${BATCH_DIR}/summary.txt"

if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$MASTER_LOG") 2>&1
fi

SEEDS="${SEEDS:-10200}"
EPOCHS="${EPOCHS:-3}"
RUN_CLEAN_ABLATIONS="${RUN_CLEAN_ABLATIONS:-1}"
RUN_STAGE2_PROTOCOL="${RUN_STAGE2_PROTOCOL:-1}"
RUN_CLINICAL="${RUN_CLINICAL:-0}"
RUN_CLINICAL_CCD="${RUN_CLINICAL_CCD:-0}"
USE_S0_AS_STAGE1="${USE_S0_AS_STAGE1:-1}"

write_status() {
  cat > "$STATUS_FILE" <<EOF
timestamp=$(date +"%F %T")
current_step=$1
message=$2
seeds=${SEEDS}
epochs=${EPOCHS}
master_log=${MASTER_LOG}
summary_file=${SUMMARY_FILE}
EOF
}

echo "Clean supplement batch: ${BATCH_DIR}" > "$SUMMARY_FILE"
echo "Created: $(date)" >> "$SUMMARY_FILE"
echo "Seeds: ${SEEDS}" >> "$SUMMARY_FILE"
echo "Epochs: ${EPOCHS}" >> "$SUMMARY_FILE"
echo "Run clean ablations: ${RUN_CLEAN_ABLATIONS}" >> "$SUMMARY_FILE"
echo "Run Stage-II protocol: ${RUN_STAGE2_PROTOCOL}" >> "$SUMMARY_FILE"
echo "Run clinical: ${RUN_CLINICAL}" >> "$SUMMARY_FILE"
echo "Run clinical CCD: ${RUN_CLINICAL_CCD}" >> "$SUMMARY_FILE"
echo "Use S0 as Stage-I: ${USE_S0_AS_STAGE1}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "============================================================"
echo ">>> Starting clean supplement batch"
echo ">>> Batch dir: ${BATCH_DIR}"
echo ">>> Seeds: ${SEEDS}"
echo ">>> Epochs: ${EPOCHS}"
echo "============================================================"

write_status "INIT" "starting"

if [ "$RUN_CLEAN_ABLATIONS" = "1" ]; then
  write_status "CLEAN_ABLATIONS" "running"
  echo ""
  echo "============================================================"
  echo ">>> Running clean strict ablations: A, B, AB"
  echo "============================================================"
  env -u CMCRL_AUTO_LOGGED \
    SEEDS="$SEEDS" \
    EPOCHS="$EPOCHS" \
    RESULT_ROOT="results/mimic_cxr/clean_ablations_20260518" \
    bash experiments/mimic_ablation_results/legacy_scripts/clean_ablations_20260518/scripts/run_all_clean_ablations.sh
  exit_code=$?
  echo "[Clean ablations] exit_code=${exit_code}" >> "$SUMMARY_FILE"
  if [ $exit_code -ne 0 ]; then
    write_status "CLEAN_ABLATIONS" "failed"
    exit $exit_code
  fi
  write_status "CLEAN_ABLATIONS" "completed"
fi

if [ "$RUN_STAGE2_PROTOCOL" = "1" ]; then
  write_status "STAGE2_PROTOCOL" "running"
  echo ""
  echo "============================================================"
  echo ">>> Running clean Stage-II protocol: S0 -> S1/S2"
  echo "============================================================"
  env -u CMCRL_AUTO_LOGGED \
    SEEDS="$SEEDS" \
    EPOCHS="$EPOCHS" \
    RUN_S0=1 \
    RUN_S1=1 \
    RUN_S2=1 \
    USE_S0_AS_STAGE1="$USE_S0_AS_STAGE1" \
    RUN_CLINICAL="$RUN_CLINICAL" \
    RUN_CLINICAL_CCD="$RUN_CLINICAL_CCD" \
    RESULT_ROOT="results/mimic_cxr/stage2_protocol_20260518" \
    bash experiments/mimic_ablation_results/legacy_scripts/stage2_protocol_20260518/scripts/run_all_stage2_protocol.sh
  exit_code=$?
  echo "[Stage-II protocol] exit_code=${exit_code}" >> "$SUMMARY_FILE"
  if [ $exit_code -ne 0 ]; then
    write_status "STAGE2_PROTOCOL" "failed"
    exit $exit_code
  fi
  write_status "STAGE2_PROTOCOL" "completed"
fi

write_status "DONE" "all_completed"
echo ""
echo "============================================================"
echo ">>> Clean supplement batch finished at $(date)"
echo ">>> Summary file: ${SUMMARY_FILE}"
echo "============================================================"

