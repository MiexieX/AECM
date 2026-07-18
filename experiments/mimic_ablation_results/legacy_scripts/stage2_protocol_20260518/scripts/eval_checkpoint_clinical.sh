#!/bin/bash

BEST_MODEL="${BEST_MODEL:?BEST_MODEL is required}"
TASK_NAME="${TASK_NAME:-clinical_eval_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-results/mimic_cxr/stage2_protocol_20260518/clinical/${TASK_NAME}}"
REPORT_CSV="${SAVE_DIR}/mimic_cxr_reports.csv"
SEED="${SEED:-10200}"
USE_COUNTERFACTUAL_INTERVENTION="${USE_COUNTERFACTUAL_INTERVENTION:-0}"
MODEL_NAME="${MODEL_NAME:-vlci}"
CCD_GAMMA="${CCD_GAMMA:-}"
CCD_USE_CONFIDENCE_GATE="${CCD_USE_CONFIDENCE_GATE:-1}"
CCLI_ANOMALY_THRESHOLD="${CCLI_ANOMALY_THRESHOLD:-0.35}"
CCLI_ANOMALY_TEMPERATURE="${CCLI_ANOMALY_TEMPERATURE:-0.10}"

mkdir -p "$SAVE_DIR"
LOG_FILE="${SAVE_DIR}/run.log"
if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo ">>> Clinical/offline evaluation"
echo ">>> best_model=${BEST_MODEL}"
echo ">>> save_dir=${SAVE_DIR}"
echo ">>> seed=${SEED}"
echo ">>> model=${MODEL_NAME}"
echo ">>> use_counterfactual_intervention=${USE_COUNTERFACTUAL_INTERVENTION}"
echo ">>> ccd_gamma=${CCD_GAMMA:-none}"

dump_args=(
  --c config/mimic_cxr/vlci.json
  --task_name "$TASK_NAME"
  --result_dir "$SAVE_DIR"
  --model "$MODEL_NAME"
  --load_model_path "$BEST_MODEL"
  --cuda 0
  --seed "$SEED"
  --batch_size 16
  --beam_size 3
  --sample_method "beam_search"
  --max_seq_length 80
  --suppress_UNK 1
  --split "test"
  --use_counterfactual_intervention "$USE_COUNTERFACTUAL_INTERVENTION"
  --ccli_anomaly_threshold "$CCLI_ANOMALY_THRESHOLD"
  --ccli_anomaly_temperature "$CCLI_ANOMALY_TEMPERATURE"
)

if [ -n "$CCD_GAMMA" ]; then
  dump_args+=(--ccd_gamma "$CCD_GAMMA")
  dump_args+=(--ccd_use_confidence_gate "$CCD_USE_CONFIDENCE_GATE")
fi

python eval_dump_reports.py "${dump_args[@]}"

python eval_clinical.py \
    --c config/mimic_cxr/vlci.json \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --report_csv "$REPORT_CSV" \
    --cuda 0 \
    --seed "$SEED"
