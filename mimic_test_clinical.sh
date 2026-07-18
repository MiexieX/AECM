#!/bin/bash
# Clinical-balanced evaluation entrypoint.
# Default decoding uses CCD gamma=0.1 with confidence gate enabled.
#
# Example:
#   BEST_MODEL=/abs/path/to/model_best.pth bash mimic_test_clinical.sh

BEST_MODEL="${BEST_MODEL:-/home/sjx/work/project/CMCRL-main/results/mimic_cxr/train/ccli_stage2_bleu_0.1214/model_best.pth}"
TASK_NAME="${TASK_NAME:-view_clinical}"
SAVE_DIR="${SAVE_DIR:-results/mimic_cxr/test/${TASK_NAME}}"
REPORT_CSV="${SAVE_DIR}/mimic_cxr_reports.csv"
mkdir -p "$SAVE_DIR"
LOG_FILE="${SAVE_DIR}/run.log"
if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

CCD_GAMMA="${CCD_GAMMA:-0.1}"
CCD_USE_CONFIDENCE_GATE="${CCD_USE_CONFIDENCE_GATE:-1}"
CCLI_ANOMALY_THRESHOLD="${CCLI_ANOMALY_THRESHOLD:-0.35}"
CCLI_ANOMALY_TEMPERATURE="${CCLI_ANOMALY_TEMPERATURE:-0.10}"

echo ">>> 🧪 正在启动 MIMIC-CXR Clinical-balanced 测试任务，加载权重: ${BEST_MODEL}"
echo ">>> save_dir=${SAVE_DIR}"
echo ">>> log_file=${LOG_FILE}"
echo ">>> ccd_gamma=${CCD_GAMMA}, confidence_gate=${CCD_USE_CONFIDENCE_GATE}, anomaly_tau=${CCLI_ANOMALY_THRESHOLD}, anomaly_temp=${CCLI_ANOMALY_TEMPERATURE}"

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
    --split "test" \
    --use_counterfactual_intervention 1 \
    --ccd_gamma "$CCD_GAMMA" \
    --ccd_use_confidence_gate "$CCD_USE_CONFIDENCE_GATE" \
    --ccli_anomaly_threshold "$CCLI_ANOMALY_THRESHOLD" \
    --ccli_anomaly_temperature "$CCLI_ANOMALY_TEMPERATURE"

python eval_clinical.py \
    --c config/mimic_cxr/vlci.json \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --report_csv "$REPORT_CSV" \
    --cuda 0 \
    --seed 10200
