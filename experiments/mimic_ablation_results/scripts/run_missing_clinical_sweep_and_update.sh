#!/bin/bash

set -e
set -u
set -o pipefail

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
SUMMARY_DIR="${SUMMARY_DIR:-experiments/mimic_ablation_results/summaries/experiment_summary_20260527}"
ORIGINAL_SUMMARY="${ORIGINAL_SUMMARY:-${SUMMARY_DIR}/all_ablation_summary.csv}"
ORIGINAL_MD="${ORIGINAL_MD:-${SUMMARY_DIR}/all_ablation_summary.md}"
RESULT_ROOT="${RESULT_ROOT:-experiments/mimic_ablation_results/runs/missing_clinical_sweep_${EXP_TIME}}"
BASELINE_ROOT="${BASELINE_ROOT:-${RESULT_ROOT}/baseline_potential2_full_eval}"
BACKUP_DIR="${BACKUP_DIR:-${SUMMARY_DIR}/backups/${EXP_TIME}}"
LOG_FILE="${RESULT_ROOT}/run.log"
STATUS_FILE="${RESULT_ROOT}/status.txt"

MISSING_EVAL_SCRIPT="${MISSING_EVAL_SCRIPT:-experiments/mimic_ablation_results/scripts/run_missing_metric_evals.sh}"
APPLY_SCRIPT="${APPLY_SCRIPT:-experiments/mimic_ablation_results/scripts/apply_missing_clinical_summary.py}"

MODEL_NAME="${MODEL_NAME:-vlci}"
CONFIG_PATH="${CONFIG_PATH:-config/mimic_cxr/vlci.json}"
CUDA="${CUDA:-0}"
SEED="${SEED:-10200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BEAM_SIZE="${BEAM_SIZE:-3}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-80}"
FORCE="${FORCE:-0}"
UPDATE_CANONICAL="${UPDATE_CANONICAL:-1}"

BASELINE_CKPT="${BASELINE_CKPT:-results/mimic_cxr/train/baseline_potential2_20260424_113736/model_best.pth}"
SUPPLEMENTED_SUMMARY="${RESULT_ROOT}/all_ablation_summary_with_missing_clinical_20260608.csv"
BASELINE_CLINICAL="${BASELINE_ROOT}/test_metrics/mimic_cxr_clinical.csv"
APPLY_AUDIT="${RESULT_ROOT}/canonical_update_audit.csv"

mkdir -p "$RESULT_ROOT" "$BASELINE_ROOT"

if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

write_status() {
  cat > "$STATUS_FILE" <<EOF
timestamp=$(date +"%F %T")
current_step=$1
message=$2
result_root=${RESULT_ROOT}
summary_dir=${SUMMARY_DIR}
original_summary=${ORIGINAL_SUMMARY}
original_md=${ORIGINAL_MD}
backup_dir=${BACKUP_DIR}
log_file=${LOG_FILE}
model_name=${MODEL_NAME}
cuda=${CUDA}
seed=${SEED}
force=${FORCE}
update_canonical=${UPDATE_CANONICAL}
EOF
}

echo "============================================================"
echo ">>> MIMIC missing clinical sweep + table update"
echo ">>> started_at=$(date)"
echo ">>> result_root=${RESULT_ROOT}"
echo ">>> log_file=${LOG_FILE}"
echo ">>> canonical_csv=${ORIGINAL_SUMMARY}"
echo ">>> canonical_md=${ORIGINAL_MD}"
echo ">>> update_canonical=${UPDATE_CANONICAL}"
echo "============================================================"

write_status "MISSING_EVAL" "running missing metric evaluations"

RESULT_ROOT="$RESULT_ROOT" \
BASELINE_ROOT="$BASELINE_ROOT" \
MODEL_NAME="$MODEL_NAME" \
CONFIG_PATH="$CONFIG_PATH" \
CUDA="$CUDA" \
SEED="$SEED" \
BATCH_SIZE="$BATCH_SIZE" \
BEAM_SIZE="$BEAM_SIZE" \
MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH" \
FORCE="$FORCE" \
BASELINE_CKPT="$BASELINE_CKPT" \
ORIGINAL_SUMMARY="$ORIGINAL_SUMMARY" \
DISABLE_AUTO_LOG=1 \
bash "$MISSING_EVAL_SCRIPT"

if [ ! -s "$SUPPLEMENTED_SUMMARY" ]; then
  echo "Missing supplemented summary: ${SUPPLEMENTED_SUMMARY}"
  exit 1
fi

if [ "$UPDATE_CANONICAL" = "1" ]; then
  write_status "APPLY_UPDATE" "rewriting canonical summary tables"
  python "$APPLY_SCRIPT" \
    --supplemented-summary "$SUPPLEMENTED_SUMMARY" \
    --target-summary "$ORIGINAL_SUMMARY" \
    --target-md "$ORIGINAL_MD" \
    --backup-dir "$BACKUP_DIR" \
    --audit-out "$APPLY_AUDIT" \
    --baseline-clinical "$BASELINE_CLINICAL" \
    --baseline-row-id "B_potential2_0424" \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --baseline-result-dir "$BASELINE_ROOT" \
    --model-name "$MODEL_NAME" \
    --seed "$SEED"
else
  echo ">>> UPDATE_CANONICAL=0, leaving canonical summary unchanged."
fi

write_status "DONE" "all_completed"

echo "============================================================"
echo ">>> Finished at $(date)"
echo ">>> log_file=${LOG_FILE}"
echo ">>> eval_summary=${RESULT_ROOT}/summary.csv"
echo ">>> supplemented_summary=${SUPPLEMENTED_SUMMARY}"
echo ">>> baseline_clinical=${BASELINE_CLINICAL}"
echo ">>> canonical_update_audit=${APPLY_AUDIT}"
echo ">>> backup_dir=${BACKUP_DIR}"
echo "============================================================"
