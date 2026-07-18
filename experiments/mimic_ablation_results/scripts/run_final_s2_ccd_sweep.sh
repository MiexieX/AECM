#!/bin/bash

set -e
set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")

BEST_MODEL="${BEST_MODEL:-/home/sjx/work/project/CMCRL-main/experiments/mimic_ablation_results/archive/ABLATION_RESULTS_20260522/stage2_protocol_20260518/S2_stage1_plus_CCLI/s2_stage1_plus_ccli_seed10200_20260521_152617/model_best_val.pth}"
RESULT_ROOT="${RESULT_ROOT:-experiments/mimic_ablation_results/runs/final_s2_ccd_sweep_${EXP_TIME}}"
SUMMARY_FILE="${RESULT_ROOT}/summary.csv"
LOG_FILE="${RESULT_ROOT}/run.log"

GAMMAS_STR="${GAMMAS_STR:-0.05 0.10 0.15 0.20 0.25 0.30}"
USE_CONFIDENCE_GATE="${USE_CONFIDENCE_GATE:-1}"
CCLI_ANOMALY_THRESHOLD="${CCLI_ANOMALY_THRESHOLD:-0.35}"
CCLI_ANOMALY_TEMPERATURE="${CCLI_ANOMALY_TEMPERATURE:-0.10}"
SEED="${SEED:-10200}"
CUDA="${CUDA:-0}"
SPLIT="${SPLIT:-test}"

mkdir -p "$RESULT_ROOT"

if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo ">>> Final S2 clean CCD sweep"
echo ">>> checkpoint=${BEST_MODEL}"
echo ">>> result_root=${RESULT_ROOT}"
echo ">>> gammas=${GAMMAS_STR}"
echo ">>> model=vlci_ccli_clean"
echo ">>> split=${SPLIT}, seed=${SEED}, cuda=${CUDA}"
echo ">>> confidence_gate=${USE_CONFIDENCE_GATE}"
echo ">>> anomaly_threshold=${CCLI_ANOMALY_THRESHOLD}"
echo ">>> anomaly_temperature=${CCLI_ANOMALY_TEMPERATURE}"

echo "gamma,task_name,split,result_dir,clinical_csv" > "$SUMMARY_FILE"

for GAMMA in ${GAMMAS_STR}; do
  GAMMA_TAG=${GAMMA/./p}
  TASK_NAME="s2_clean_ccd_g${GAMMA_TAG}"
  SAVE_DIR="${RESULT_ROOT}/${TASK_NAME}"
  REPORT_CSV="${SAVE_DIR}/mimic_cxr_reports.csv"

  echo ""
  echo "============================================================"
  echo ">>> Running S2 clean CCD gamma=${GAMMA}"
  echo ">>> save_dir=${SAVE_DIR}"
  echo "============================================================"

  python eval_dump_reports.py \
    --c config/mimic_cxr/vlci.json \
    --model vlci_ccli_clean \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --load_model_path "$BEST_MODEL" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    --batch_size 16 \
    --beam_size 3 \
    --sample_method "beam_search" \
    --max_seq_length 80 \
    --suppress_UNK 1 \
    --split "$SPLIT" \
    --use_counterfactual_intervention 1 \
    --ccd_gamma "$GAMMA" \
    --ccd_use_confidence_gate "$USE_CONFIDENCE_GATE" \
    --ccli_anomaly_threshold "$CCLI_ANOMALY_THRESHOLD" \
    --ccli_anomaly_temperature "$CCLI_ANOMALY_TEMPERATURE"

  if [ "$SPLIT" = "both" ] || [ "$SPLIT" = "val" ]; then
    python eval_clinical.py \
      --c config/mimic_cxr/vlci.json \
      --task_name "${TASK_NAME}_val" \
      --result_dir "${SAVE_DIR}/val_metrics" \
      --report_csv "$REPORT_CSV" \
      --report_split "val" \
      --cuda "$CUDA" \
      --seed "$SEED"
    echo "${GAMMA},${TASK_NAME},val,${SAVE_DIR},${SAVE_DIR}/val_metrics/mimic_cxr_clinical.csv" >> "$SUMMARY_FILE"
  fi

  if [ "$SPLIT" = "both" ] || [ "$SPLIT" = "test" ]; then
    python eval_clinical.py \
      --c config/mimic_cxr/vlci.json \
      --task_name "${TASK_NAME}_test" \
      --result_dir "${SAVE_DIR}/test_metrics" \
      --report_csv "$REPORT_CSV" \
      --report_split "test" \
      --cuda "$CUDA" \
      --seed "$SEED"
    echo "${GAMMA},${TASK_NAME},test,${SAVE_DIR},${SAVE_DIR}/test_metrics/mimic_cxr_clinical.csv" >> "$SUMMARY_FILE"
  fi
done

echo ""
echo "============================================================"
echo ">>> Final S2 clean CCD sweep finished at $(date)"
echo ">>> Summary file: ${SUMMARY_FILE}"
echo "============================================================"
