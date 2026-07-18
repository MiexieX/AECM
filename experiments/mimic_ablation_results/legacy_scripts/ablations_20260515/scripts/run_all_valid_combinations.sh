#!/bin/bash

set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
BATCH_ROOT="results/mimic_cxr/ablations_20260515/batches"
BATCH_NAME="ablation_batch_${EXP_TIME}"
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
EPOCHS="${EPOCHS:-3}"
RUN_E1="${RUN_E1:-1}"
RUN_E2="${RUN_E2:-1}"
RUN_E3="${RUN_E3:-1}"
RUN_E4="${RUN_E4:-1}"
RUN_E5="${RUN_E5:-1}"

SCRIPT_DIR="/home/sjx/work/project/CMCRL-main/experiments/mimic_ablation_results/legacy_scripts/ablations_20260515/scripts"

echo "Ablation batch: ${BATCH_NAME}" > "$SUMMARY_FILE"
echo "Created: $(date)" >> "$SUMMARY_FILE"
echo "Seed: ${SEED}" >> "$SUMMARY_FILE"
echo "Epochs: ${EPOCHS}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

write_status() {
  cat > "$STATUS_FILE" <<EOF
batch_name=${BATCH_NAME}
timestamp=$(date +"%F %T")
current_step=$1
message=$2
seed=${SEED}
epochs=${EPOCHS}
master_log=${MASTER_LOG}
summary_file=${SUMMARY_FILE}
EOF
}

run_step() {
  local step_key="$1"
  local step_name="$2"
  local script_path="$3"

  write_status "$step_key" "running"
  echo ""
  echo "============================================================"
  echo ">>> Running ${step_key}: ${step_name}"
  echo ">>> Script: ${script_path}"
  echo ">>> Started at: $(date)"
  echo "============================================================"

  local before_latest
  before_latest=$(ls -td results/mimic_cxr/ablations_20260515/E*/* 2>/dev/null | head -1 || true)

  env -u CMCRL_AUTO_LOGGED SEED="$SEED" EPOCHS="$EPOCHS" bash "$script_path"
  local exit_code=$?

  local after_latest
  after_latest=$(ls -td results/mimic_cxr/ablations_20260515/E*/* 2>/dev/null | head -1 || true)

  if [ $exit_code -eq 0 ]; then
    echo ">>> ${step_key} completed successfully at $(date)"
    write_status "$step_key" "completed"
    {
      echo "[${step_key}] ${step_name}"
      echo "status=completed"
      if [ -n "$after_latest" ] && [ "$after_latest" != "$before_latest" ]; then
        echo "run_dir=${after_latest}"
      fi
      echo ""
    } >> "$SUMMARY_FILE"
  else
    echo ">>> ${step_key} failed with exit code ${exit_code} at $(date)"
    write_status "$step_key" "failed"
    {
      echo "[${step_key}] ${step_name}"
      echo "status=failed"
      echo "exit_code=${exit_code}"
      if [ -n "$after_latest" ] && [ "$after_latest" != "$before_latest" ]; then
        echo "run_dir=${after_latest}"
      fi
      echo ""
    } >> "$SUMMARY_FILE"
    return $exit_code
  fi
}

echo "============================================================"
echo ">>> Starting ablation batch: ${BATCH_NAME}"
echo ">>> Batch dir: ${BATCH_DIR}"
echo ">>> Master log: ${MASTER_LOG}"
echo ">>> Status file: ${STATUS_FILE}"
echo ">>> Seed: ${SEED} | Epochs: ${EPOCHS}"
echo ">>> Planned matrix: baseline+A, baseline+B, baseline+AB, baseline+BC, baseline+ABC"
echo "============================================================"

write_status "INIT" "starting"

if [ "$RUN_E1" = "1" ]; then
  run_step "E1" "baseline + A" "${SCRIPT_DIR}/run_e1_baseline_plus_a.sh" || exit 1
fi

if [ "$RUN_E2" = "1" ]; then
  run_step "E2" "baseline + B" "${SCRIPT_DIR}/run_e2_baseline_plus_b.sh" || exit 1
fi

if [ "$RUN_E3" = "1" ]; then
  run_step "E3" "baseline + AB" "${SCRIPT_DIR}/run_e3_baseline_plus_ab.sh" || exit 1
fi

if [ "$RUN_E4" = "1" ]; then
  run_step "E4" "baseline + BC" "${SCRIPT_DIR}/run_e4_baseline_plus_bc.sh" || exit 1
fi

if [ "$RUN_E5" = "1" ]; then
  run_step "E5" "baseline + ABC" "${SCRIPT_DIR}/run_e5_baseline_plus_abc.sh" || exit 1
fi

write_status "DONE" "all_completed"
echo ""
echo "============================================================"
echo ">>> All requested ablation steps finished at $(date)"
echo ">>> Summary file: ${SUMMARY_FILE}"
echo "============================================================"
