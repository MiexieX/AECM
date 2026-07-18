#!/bin/bash

set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_ROOT="${RESULT_ROOT:-results/mimic_cxr/stage2_protocol_20260518}"
BATCH_ROOT="${RESULT_ROOT}/batches"
BATCH_NAME="${BATCH_NAME:-stage2_protocol_batch_${EXP_TIME}}"
BATCH_DIR="${BATCH_ROOT}/${BATCH_NAME}"
mkdir -p "$BATCH_DIR"

MASTER_LOG="${BATCH_DIR}/run.log"
STATUS_FILE="${BATCH_DIR}/status.txt"
SUMMARY_FILE="${BATCH_DIR}/summary.txt"

if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$MASTER_LOG") 2>&1
fi

SEEDS="${SEEDS:-10200 20200 30200}"
EPOCHS="${EPOCHS:-3}"
RUN_S0="${RUN_S0:-0}"
RUN_S1="${RUN_S1:-1}"
RUN_S2="${RUN_S2:-1}"
RUN_CLINICAL="${RUN_CLINICAL:-0}"
RUN_CLINICAL_CCD="${RUN_CLINICAL_CCD:-0}"
USE_S0_AS_STAGE1="${USE_S0_AS_STAGE1:-0}"
STAGE1_SELECT="${STAGE1_SELECT:-model_best_val.pth}"
STAGE1_CKPT="${STAGE1_CKPT:-/home/sjx/work/project/CMCRL-main/results/mimic_cxr/train/distill_prompt_best_f1/model_best.pth}"
BASELINE_CKPT="${BASELINE_CKPT:-/home/sjx/work/project/CMCRL-main/results/mimic_cxr/train/baseline_potential2_20260424_113736/model_best.pth}"
SCRIPT_DIR="/home/sjx/work/project/CMCRL-main/experiments/mimic_ablation_results/legacy_scripts/stage2_protocol_20260518/scripts"

echo "Stage-II protocol batch: ${BATCH_NAME}" > "$SUMMARY_FILE"
echo "Created: $(date)" >> "$SUMMARY_FILE"
echo "Seeds: ${SEEDS}" >> "$SUMMARY_FILE"
echo "Epochs: ${EPOCHS}" >> "$SUMMARY_FILE"
echo "Stage-I checkpoint: ${STAGE1_CKPT}" >> "$SUMMARY_FILE"
echo "Baseline checkpoint: ${BASELINE_CKPT}" >> "$SUMMARY_FILE"
echo "Run S0: ${RUN_S0}" >> "$SUMMARY_FILE"
echo "Run S1: ${RUN_S1}" >> "$SUMMARY_FILE"
echo "Run S2: ${RUN_S2}" >> "$SUMMARY_FILE"
echo "Run clinical: ${RUN_CLINICAL}" >> "$SUMMARY_FILE"
echo "Run clinical CCD: ${RUN_CLINICAL_CCD}" >> "$SUMMARY_FILE"
echo "Use S0 as Stage-I: ${USE_S0_AS_STAGE1}" >> "$SUMMARY_FILE"
echo "Stage-I select: ${STAGE1_SELECT}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

write_status() {
  cat > "$STATUS_FILE" <<EOF
batch_name=${BATCH_NAME}
timestamp=$(date +"%F %T")
current_step=$1
message=$2
seed=${3:-}
epochs=${EPOCHS}
stage1_ckpt=${STAGE1_CKPT}
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

run_train_step() {
  local step_key="$1"
  local step_name="$2"
  local script_path="$3"
  local seed="$4"
  local group_dir="$5"
  local stage1_ckpt_override="${6:-$STAGE1_CKPT}"

  write_status "$step_key" "running" "$seed"
  echo ""
  echo "============================================================"
  echo ">>> Running ${step_key}: ${step_name}"
  echo ">>> Seed: ${seed}"
  echo ">>> Stage-I override: ${stage1_ckpt_override}"
  echo ">>> Script: ${script_path}"
  echo ">>> Started at: $(date)"
  echo "============================================================"

  local before_latest
  before_latest=$(latest_run_dir "$group_dir")

  env -u CMCRL_AUTO_LOGGED \
    SEED="$seed" \
    EPOCHS="$EPOCHS" \
    STAGE1_CKPT="$stage1_ckpt_override" \
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

run_clinical_eval() {
  local label="$1"
  local seed="$2"
  local run_dir="$3"
  local use_ccli="$4"
  local ccd_gamma="$5"
  local model_name="$6"

  local ckpt="${run_dir}/model_best_val.pth"
  if [ ! -f "$ckpt" ]; then
    ckpt="${run_dir}/model_best.pth"
  fi
  if [ ! -f "$ckpt" ]; then
    echo ">>> Clinical eval skipped for ${label}: no checkpoint found in ${run_dir}"
    return 0
  fi

  local eval_name="${label}_seed${seed}"
  if [ -n "$ccd_gamma" ]; then
    eval_name="${eval_name}_ccd${ccd_gamma}"
  fi
  local eval_dir="${RESULT_ROOT}/clinical/${eval_name}"

  write_status "CLINICAL_${label}" "running" "$seed"
  echo ""
  echo "------------------------------------------------------------"
  echo ">>> Clinical eval: ${label}, seed=${seed}, ckpt=${ckpt}"
  echo ">>> model=${model_name}"
  echo ">>> use_ccli=${use_ccli}, ccd_gamma=${ccd_gamma:-none}"
  echo "------------------------------------------------------------"

  if [ -n "$ccd_gamma" ]; then
    env -u CMCRL_AUTO_LOGGED \
      BEST_MODEL="$ckpt" \
      TASK_NAME="$eval_name" \
      SAVE_DIR="$eval_dir" \
      SEED="$seed" \
      MODEL_NAME="$model_name" \
      USE_COUNTERFACTUAL_INTERVENTION="$use_ccli" \
      CCD_GAMMA="$ccd_gamma" \
      bash "${SCRIPT_DIR}/eval_checkpoint_clinical.sh"
  else
    env -u CMCRL_AUTO_LOGGED \
      BEST_MODEL="$ckpt" \
      TASK_NAME="$eval_name" \
      SAVE_DIR="$eval_dir" \
      SEED="$seed" \
      MODEL_NAME="$model_name" \
      USE_COUNTERFACTUAL_INTERVENTION="$use_ccli" \
      bash "${SCRIPT_DIR}/eval_checkpoint_clinical.sh"
  fi
  local exit_code=$?

  {
    echo "[Clinical ${label} seed=${seed}]"
    echo "status=$([ $exit_code -eq 0 ] && echo completed || echo failed)"
    echo "exit_code=${exit_code}"
    echo "eval_dir=${eval_dir}"
    if [ -f "${eval_dir}/mimic_cxr_clinical.csv" ]; then
      echo "clinical_csv=${eval_dir}/mimic_cxr_clinical.csv"
      tail -n 1 "${eval_dir}/mimic_cxr_clinical.csv"
    fi
    echo ""
  } >> "$SUMMARY_FILE"

  if [ $exit_code -ne 0 ]; then
    write_status "CLINICAL_${label}" "failed" "$seed"
    return $exit_code
  fi
  write_status "CLINICAL_${label}" "completed" "$seed"
}

echo "============================================================"
echo ">>> Starting Stage-II protocol batch: ${BATCH_NAME}"
echo ">>> Batch dir: ${BATCH_DIR}"
echo ">>> Master log: ${MASTER_LOG}"
echo ">>> Seeds: ${SEEDS}"
echo ">>> Epochs: ${EPOCHS}"
echo ">>> Stage-I checkpoint: ${STAGE1_CKPT}"
echo ">>> Baseline checkpoint: ${BASELINE_CKPT}"
echo ">>> Planned steps: optional S0 clean AB, S1 Stage-I continue no CCLI, S2 Stage-I + CCLI"
echo ">>> USE_S0_AS_STAGE1=${USE_S0_AS_STAGE1}, STAGE1_SELECT=${STAGE1_SELECT}"
echo "============================================================"

write_status "INIT" "starting" ""

S0_GROUP="${RESULT_ROOT}/S0_clean_stage1_AB"
S1_GROUP="${RESULT_ROOT}/S1_stage1_continue_no_CCLI"
S2_GROUP="${RESULT_ROOT}/S2_stage1_plus_CCLI"

for seed in $SEEDS; do
  STAGE1_FOR_SEED="$STAGE1_CKPT"

  if [ "$RUN_S0" = "1" ]; then
    run_train_step \
      "S0" \
      "Clean Stage-I AB from baseline" \
      "${SCRIPT_DIR}/run_s0_clean_stage1_ab.sh" \
      "$seed" \
      "$S0_GROUP" || exit 1
    S0_RUN_DIR=$(latest_run_dir "$S0_GROUP")
    if [ "$USE_S0_AS_STAGE1" = "1" ]; then
      candidate_ckpt="${S0_RUN_DIR}/${STAGE1_SELECT}"
      if [ ! -f "$candidate_ckpt" ]; then
        candidate_ckpt="${S0_RUN_DIR}/model_best_val.pth"
      fi
      if [ ! -f "$candidate_ckpt" ]; then
        candidate_ckpt="${S0_RUN_DIR}/model_best.pth"
      fi
      if [ ! -f "$candidate_ckpt" ]; then
        echo ">>> Cannot use S0 as Stage-I: no checkpoint found in ${S0_RUN_DIR}"
        exit 1
      fi
      STAGE1_FOR_SEED="$candidate_ckpt"
      echo ">>> Using clean S0 checkpoint as Stage-I for seed ${seed}: ${STAGE1_FOR_SEED}"
      {
        echo "[S0 selected Stage-I seed=${seed}]"
        echo "stage1_ckpt=${STAGE1_FOR_SEED}"
        echo ""
      } >> "$SUMMARY_FILE"
    fi
  else
    S0_RUN_DIR=""
  fi

  if [ "$RUN_S1" = "1" ]; then
    run_train_step \
      "S1" \
      "Stage-I continue without CCLI" \
      "${SCRIPT_DIR}/run_s1_stage1_continue_no_ccli.sh" \
      "$seed" \
      "$S1_GROUP" \
      "$STAGE1_FOR_SEED" || exit 1
    S1_RUN_DIR=$(latest_run_dir "$S1_GROUP")
  else
    S1_RUN_DIR=""
  fi

  if [ "$RUN_S2" = "1" ]; then
    run_train_step \
      "S2" \
      "Stage-I plus CCLI" \
      "${SCRIPT_DIR}/run_s2_stage1_plus_ccli.sh" \
      "$seed" \
      "$S2_GROUP" \
      "$STAGE1_FOR_SEED" || exit 1
    S2_RUN_DIR=$(latest_run_dir "$S2_GROUP")
  else
    S2_RUN_DIR=""
  fi

  if [ "$RUN_CLINICAL" = "1" ]; then
    if [ -n "${S0_RUN_DIR:-}" ]; then
      run_clinical_eval "s0_clean_ab" "$seed" "$S0_RUN_DIR" "0" "" "vlci_ccli_clean" || exit 1
    fi
    if [ -n "${S1_RUN_DIR:-}" ]; then
      run_clinical_eval "s1_no_ccli" "$seed" "$S1_RUN_DIR" "0" "" "vlci_ccli_clean" || exit 1
    fi
    if [ -n "${S2_RUN_DIR:-}" ]; then
      run_clinical_eval "s2_ccli" "$seed" "$S2_RUN_DIR" "1" "" "vlci_ccli_clean" || exit 1
      if [ "$RUN_CLINICAL_CCD" = "1" ]; then
        run_clinical_eval "s2_ccli" "$seed" "$S2_RUN_DIR" "1" "0.1" "vlci_ccli_clean" || exit 1
      fi
    fi
  fi
done

write_status "DONE" "all_completed" ""
echo ""
echo "============================================================"
echo ">>> All requested Stage-II protocol steps finished at $(date)"
echo ">>> Summary file: ${SUMMARY_FILE}"
echo "============================================================"
