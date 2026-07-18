#!/bin/bash

set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_ROOT="${RESULT_ROOT:-results/mimic_cxr/clean_ablations_20260518}"
BATCH_ROOT="${RESULT_ROOT}/batches"
BATCH_NAME="${BATCH_NAME:-clean_ablation_batch_${EXP_TIME}}"
BATCH_DIR="${BATCH_ROOT}/${BATCH_NAME}"
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
RUN_C1="${RUN_C1:-1}"
RUN_C2="${RUN_C2:-1}"
RUN_C3="${RUN_C3:-1}"
BASELINE_CKPT="${BASELINE_CKPT:-/home/sjx/work/project/CMCRL-main/results/mimic_cxr/train/baseline_potential2_20260424_113736/model_best.pth}"
SCRIPT_DIR="/home/sjx/work/project/CMCRL-main/experiments/mimic_ablation_results/legacy_scripts/clean_ablations_20260518/scripts"

echo "Clean ablation batch: ${BATCH_NAME}" > "$SUMMARY_FILE"
echo "Created: $(date)" >> "$SUMMARY_FILE"
echo "Seeds: ${SEEDS}" >> "$SUMMARY_FILE"
echo "Epochs: ${EPOCHS}" >> "$SUMMARY_FILE"
echo "Baseline checkpoint: ${BASELINE_CKPT}" >> "$SUMMARY_FILE"
echo "Run C1: ${RUN_C1}" >> "$SUMMARY_FILE"
echo "Run C2: ${RUN_C2}" >> "$SUMMARY_FILE"
echo "Run C3: ${RUN_C3}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

write_status() {
  cat > "$STATUS_FILE" <<EOF
batch_name=${BATCH_NAME}
timestamp=$(date +"%F %T")
current_step=$1
message=$2
seed=${3:-}
epochs=${EPOCHS}
baseline_ckpt=${BASELINE_CKPT}
master_log=${MASTER_LOG}
summary_file=${SUMMARY_FILE}
EOF
}

latest_run_dir() {
  local group_dir="$1"
  ls -td "${group_dir}"/* 2>/dev/null | head -1 || true
}

append_result_summary() {
  local label="$1"
  local run_dir="$2"
  local status="$3"
  local exit_code="${4:-0}"
  {
    echo "[${label}]"
    echo "status=${status}"
    echo "exit_code=${exit_code}"
    echo "run_dir=${run_dir}"
    if [ -f "${run_dir}/mimic_cxr.csv" ]; then
      echo "csv=${run_dir}/mimic_cxr.csv"
      tail -n 2 "${run_dir}/mimic_cxr.csv"
    fi
    echo ""
  } >> "$SUMMARY_FILE"
}

run_step() {
  local step_key="$1"
  local step_name="$2"
  local script_path="$3"
  local seed="$4"
  local group_dir="$5"

  write_status "$step_key" "running" "$seed"
  echo ""
  echo "============================================================"
  echo ">>> Running ${step_key}: ${step_name}"
  echo ">>> Seed: ${seed}"
  echo ">>> Script: ${script_path}"
  echo ">>> Started at: $(date)"
  echo "============================================================"

  local before_latest
  before_latest=$(latest_run_dir "$group_dir")

  env -u CMCRL_AUTO_LOGGED \
    SEED="$seed" \
    EPOCHS="$EPOCHS" \
    BASELINE_CKPT="$BASELINE_CKPT" \
    bash "$script_path"
  local exit_code=$?

  local after_latest
  after_latest=$(latest_run_dir "$group_dir")
  if [ -z "$after_latest" ] || [ "$after_latest" = "$before_latest" ]; then
    after_latest="$before_latest"
  fi

  if [ $exit_code -eq 0 ]; then
    echo ">>> ${step_key} completed successfully at $(date)"
    write_status "$step_key" "completed" "$seed"
    append_result_summary "${step_key} ${step_name} seed=${seed}" "$after_latest" "completed" "$exit_code"
  else
    echo ">>> ${step_key} failed with exit code ${exit_code} at $(date)"
    write_status "$step_key" "failed" "$seed"
    append_result_summary "${step_key} ${step_name} seed=${seed}" "$after_latest" "failed" "$exit_code"
    return $exit_code
  fi
}

echo "============================================================"
echo ">>> Starting clean ablation batch: ${BATCH_NAME}"
echo ">>> Batch dir: ${BATCH_DIR}"
echo ">>> Master log: ${MASTER_LOG}"
echo ">>> Seeds: ${SEEDS}"
echo ">>> Epochs: ${EPOCHS}"
echo ">>> Baseline checkpoint: ${BASELINE_CKPT}"
echo ">>> Planned rows: clean A, clean B, clean AB"
echo "============================================================"

write_status "INIT" "starting" ""

C1_GROUP="${RESULT_ROOT}/C1_baseline_plus_A_clean"
C2_GROUP="${RESULT_ROOT}/C2_baseline_plus_B_clean"
C3_GROUP="${RESULT_ROOT}/C3_baseline_plus_AB_clean"

for seed in $SEEDS; do
  if [ "$RUN_C1" = "1" ]; then
    run_step "C1" "clean baseline + A" "${SCRIPT_DIR}/run_c1_clean_baseline_plus_a.sh" "$seed" "$C1_GROUP" || exit 1
  fi
  if [ "$RUN_C2" = "1" ]; then
    run_step "C2" "clean baseline + B" "${SCRIPT_DIR}/run_c2_clean_baseline_plus_b.sh" "$seed" "$C2_GROUP" || exit 1
  fi
  if [ "$RUN_C3" = "1" ]; then
    run_step "C3" "clean baseline + AB" "${SCRIPT_DIR}/run_c3_clean_baseline_plus_ab.sh" "$seed" "$C3_GROUP" || exit 1
  fi
done

write_status "DONE" "all_completed" ""
echo ""
echo "============================================================"
echo ">>> All requested clean ablation steps finished at $(date)"
echo ">>> Summary file: ${SUMMARY_FILE}"
echo "============================================================"

