#!/bin/bash

set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_ROOT="${RESULT_ROOT:-results/mimic_cxr/clean_ablations_6epoch_20260522}"
BATCH_ROOT="${RESULT_ROOT}/batches"
BATCH_NAME="${BATCH_NAME:-clean_6epoch_batch_${EXP_TIME}}"
BATCH_DIR="${BATCH_ROOT}/${BATCH_NAME}"
mkdir -p "$BATCH_DIR"

MASTER_LOG="${BATCH_DIR}/run.log"
STATUS_FILE="${BATCH_DIR}/status.txt"
SUMMARY_FILE="${BATCH_DIR}/summary.txt"

if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$MASTER_LOG") 2>&1
fi

SEED="${SEED:-10200}"
EPOCHS="${EPOCHS:-6}"
RUN_A="${RUN_A:-1}"
RUN_B="${RUN_B:-1}"
RUN_AB="${RUN_AB:-1}"
RUN_BC="${RUN_BC:-1}"
RUN_ABC="${RUN_ABC:-1}"
BASELINE_CKPT="${BASELINE_CKPT:-/home/sjx/work/project/CMCRL-main/results/mimic_cxr/train/baseline_potential2_20260424_113736/model_best.pth}"

CLEAN_3E_SCRIPT_DIR="/home/sjx/work/project/CMCRL-main/experiments/mimic_ablation_results/legacy_scripts/clean_ablations_20260518/scripts"
SCRIPT_DIR="/home/sjx/work/project/CMCRL-main/experiments/mimic_ablation_results/legacy_scripts/clean_ablations_6epoch_20260522/scripts"

echo "Clean 6-epoch ablation batch: ${BATCH_NAME}" > "$SUMMARY_FILE"
echo "Created: $(date)" >> "$SUMMARY_FILE"
echo "Seed: ${SEED}" >> "$SUMMARY_FILE"
echo "Epochs: ${EPOCHS}" >> "$SUMMARY_FILE"
echo "Baseline checkpoint: ${BASELINE_CKPT}" >> "$SUMMARY_FILE"
echo "Run A: ${RUN_A}" >> "$SUMMARY_FILE"
echo "Run B: ${RUN_B}" >> "$SUMMARY_FILE"
echo "Run AB: ${RUN_AB}" >> "$SUMMARY_FILE"
echo "Run BC: ${RUN_BC}" >> "$SUMMARY_FILE"
echo "Run ABC: ${RUN_ABC}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

write_status() {
  cat > "$STATUS_FILE" <<EOF
batch_name=${BATCH_NAME}
timestamp=$(date +"%F %T")
current_step=$1
message=$2
seed=${SEED}
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
  local group_dir="$4"

  write_status "$step_key" "running"
  echo ""
  echo "============================================================"
  echo ">>> Running ${step_key}: ${step_name}"
  echo ">>> Script: ${script_path}"
  echo ">>> Seed: ${SEED}, epochs=${EPOCHS}"
  echo ">>> Started at: $(date)"
  echo "============================================================"

  local before_latest
  before_latest=$(latest_run_dir "$group_dir")

  env -u CMCRL_AUTO_LOGGED \
    SEED="$SEED" \
    EPOCHS="$EPOCHS" \
    BASELINE_CKPT="$BASELINE_CKPT" \
    GROUP_DIR="$group_dir" \
    bash "$script_path"
  local exit_code=$?

  local after_latest
  after_latest=$(latest_run_dir "$group_dir")
  if [ -z "$after_latest" ] || [ "$after_latest" = "$before_latest" ]; then
    after_latest="$before_latest"
  fi

  if [ $exit_code -eq 0 ]; then
    echo ">>> ${step_key} completed successfully at $(date)"
    write_status "$step_key" "completed"
    append_result_summary "${step_key} ${step_name}" "$after_latest" "completed" "$exit_code"
  else
    echo ">>> ${step_key} failed with exit code ${exit_code} at $(date)"
    write_status "$step_key" "failed"
    append_result_summary "${step_key} ${step_name}" "$after_latest" "failed" "$exit_code"
    return $exit_code
  fi
}

echo "============================================================"
echo ">>> Starting clean 6-epoch ablation batch: ${BATCH_NAME}"
echo ">>> Batch dir: ${BATCH_DIR}"
echo ">>> Seed: ${SEED} | Epochs: ${EPOCHS}"
echo ">>> Matrix: A, B, AB, BC, ABC"
echo "============================================================"

write_status "INIT" "starting"

if [ "$RUN_A" = "1" ]; then
  run_step "A" "clean baseline + A, 6 epoch" \
    "${CLEAN_3E_SCRIPT_DIR}/run_c1_clean_baseline_plus_a.sh" \
    "${RESULT_ROOT}/A_clean_6epoch" || exit 1
fi

if [ "$RUN_B" = "1" ]; then
  run_step "B" "clean baseline + B, 6 epoch" \
    "${CLEAN_3E_SCRIPT_DIR}/run_c2_clean_baseline_plus_b.sh" \
    "${RESULT_ROOT}/B_clean_6epoch" || exit 1
fi

if [ "$RUN_AB" = "1" ]; then
  run_step "AB" "clean baseline + AB, 6 epoch" \
    "${CLEAN_3E_SCRIPT_DIR}/run_c3_clean_baseline_plus_ab.sh" \
    "${RESULT_ROOT}/AB_clean_6epoch" || exit 1
fi

if [ "$RUN_BC" = "1" ]; then
  run_step "BC" "clean baseline + BC, 6 epoch" \
    "${SCRIPT_DIR}/run_c4_clean_baseline_plus_bc_6epoch.sh" \
    "${RESULT_ROOT}/BC_clean_6epoch" || exit 1
fi

if [ "$RUN_ABC" = "1" ]; then
  run_step "ABC" "clean baseline + ABC, 6 epoch" \
    "${SCRIPT_DIR}/run_c5_clean_baseline_plus_abc_6epoch.sh" \
    "${RESULT_ROOT}/ABC_clean_6epoch" || exit 1
fi

write_status "DONE" "all_completed"
echo ""
echo "============================================================"
echo ">>> All requested clean 6-epoch ablation steps finished at $(date)"
echo ">>> Summary file: ${SUMMARY_FILE}"
echo "============================================================"

