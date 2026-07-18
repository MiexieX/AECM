#!/bin/bash

set -e

BEST_MODEL="${BEST_MODEL:-/home/sjx/work/project/CMCRL-main/results/mimic_cxr/train/ccli_stage2_bleu_0.1214/model_best.pth}"
GAMMAS=("0.1" "0.2" "0.3" "0.5" "0.7")
USE_CONFIDENCE_GATE=1
CCLI_ANOMALY_THRESHOLD=0.35
CCLI_ANOMALY_TEMPERATURE=0.10
SWEEP_TIME=$(date +"%Y%m%d_%H%M%S")
SWEEP_DIR="${SWEEP_DIR:-results/mimic_cxr/test/ccd_sweep_${SWEEP_TIME}}"
mkdir -p "$SWEEP_DIR"
LOG_FILE="${SWEEP_DIR}/run.log"
if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo ">>> CCD sweep using checkpoint: ${BEST_MODEL}"
echo ">>> sweep_dir=${SWEEP_DIR}"
echo ">>> log_file=${LOG_FILE}"
echo ">>> gammas: ${GAMMAS[*]}"
echo ">>> confidence_gate=${USE_CONFIDENCE_GATE}, anomaly_tau=${CCLI_ANOMALY_THRESHOLD}, anomaly_temp=${CCLI_ANOMALY_TEMPERATURE}"

for GAMMA in "${GAMMAS[@]}"; do
  GAMMA_TAG=${GAMMA/./p}
  TASK_NAME="ccd_g${GAMMA_TAG}"
  SAVE_DIR="results/mimic_cxr/test/${TASK_NAME}"
  REPORT_CSV="${SAVE_DIR}/mimic_cxr_reports.csv"

  echo "============================================================"
  echo ">>> Running CCD gamma=${GAMMA}"
  echo ">>> save_dir=${SAVE_DIR}"
  echo "============================================================"

  python eval_dump_reports.py \
    --c config/mimic_cxr/vlci.json \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --load_model_path "$BEST_MODEL" \
    --cuda 0 \
    --seed 10200 \
    --batch_size 16 \
    --beam_size 3 \
    --sample_method "beam_search" \
    --max_seq_length 80 \
    --suppress_UNK 1 \
    --split "both" \
    --use_counterfactual_intervention 1 \
    --ccd_gamma "$GAMMA" \
    --ccd_use_confidence_gate "$USE_CONFIDENCE_GATE" \
    --ccli_anomaly_threshold "$CCLI_ANOMALY_THRESHOLD" \
    --ccli_anomaly_temperature "$CCLI_ANOMALY_TEMPERATURE"

  python eval_clinical.py \
    --c config/mimic_cxr/vlci.json \
    --task_name "${TASK_NAME}_val" \
    --result_dir "${SAVE_DIR}/val_metrics" \
    --report_csv "$REPORT_CSV" \
    --report_split "val" \
    --cuda 0 \
    --seed 10200

  python eval_clinical.py \
    --c config/mimic_cxr/vlci.json \
    --task_name "${TASK_NAME}_test" \
    --result_dir "${SAVE_DIR}/test_metrics" \
    --report_csv "$REPORT_CSV" \
    --report_split "test" \
    --cuda 0 \
    --seed 10200
done

echo ">>> CCD sweep finished."
