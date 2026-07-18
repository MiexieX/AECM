#!/bin/bash

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
GROUP_DIR="${GROUP_DIR:-results/mimic_cxr/stage2_protocol_20260518/S1_stage1_continue_no_CCLI}"
TASK_NAME="${TASK_NAME:-s1_stage1_continue_no_ccli_seed${SEED:-10200}_${EXP_TIME}}"
SAVE_DIR="${SAVE_DIR:-${GROUP_DIR}/${TASK_NAME}}"
mkdir -p "$SAVE_DIR"
LOG_FILE="${SAVE_DIR}/run.log"
if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

SEED="${SEED:-10200}"
EPOCHS="${EPOCHS:-3}"
STAGE1_CKPT="${STAGE1_CKPT:-/home/sjx/work/project/CMCRL-main/results/mimic_cxr/train/distill_prompt_best_f1/model_best.pth}"

PATHOLOGY_LOSS_WEIGHT="${PATHOLOGY_LOSS_WEIGHT:-0.001}"
TEMPLATE_LOSS_WEIGHT="${TEMPLATE_LOSS_WEIGHT:-0.001}"
DISTILL_LOSS_WEIGHT="${DISTILL_LOSS_WEIGHT:-0.30}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
PSEUDO_LABEL_PATH="${PSEUDO_LABEL_PATH:-data/mimic_cxr_pseudo_labels.json}"
TEMPLATE_LABEL_PATH="${TEMPLATE_LABEL_PATH:-data/mimic_cxr_template_labels.json}"
USE_CONFIDENCE_GATED_PATHOLOGY_PROMPT="${USE_CONFIDENCE_GATED_PATHOLOGY_PROMPT:-1}"
PATHOLOGY_GATE_THRESHOLD="${PATHOLOGY_GATE_THRESHOLD:-0.50}"
PATHOLOGY_GATE_TEMPERATURE="${PATHOLOGY_GATE_TEMPERATURE:-0.12}"
USE_NORMAL_AWARE_SUPPRESSION="${USE_NORMAL_AWARE_SUPPRESSION:-1}"
NORMAL_SUPPRESSION_THRESHOLD="${NORMAL_SUPPRESSION_THRESHOLD:-0.45}"
NORMAL_SUPPRESSION_TEMPERATURE="${NORMAL_SUPPRESSION_TEMPERATURE:-0.10}"
LOG_CLINICAL_METRICS_DURING_TRAIN="${LOG_CLINICAL_METRICS_DURING_TRAIN:-0}"

echo ">>> S1 Stage-I continue WITHOUT CCLI: ${TASK_NAME}"
echo ">>> save_dir=${SAVE_DIR}"
echo ">>> log_file=${LOG_FILE}"
echo ">>> seed=${SEED}, epochs=${EPOCHS}"
echo ">>> Stage-I checkpoint=${STAGE1_CKPT}"
echo ">>> model=vlci_ccli_clean"
echo ">>> A=on, B=on, CCLI=off"
echo ">>> purpose=matched Stage-II control for continued training without counterfactual intervention"

python main.py --c config/mimic_cxr/vlci.json \
    --task finetune \
    --model vlci_ccli_clean \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --cuda 0 \
    --seed "$SEED" \
    --load_model_path "$STAGE1_CKPT" \
    --epochs "$EPOCHS" \
    --batch_size 16 \
    --val_intervals 4 \
    --lr_en 1e-6 \
    --lr_de 1e-6 \
    --weight_decay 5e-5 \
    --num_workers 12 \
    --pin_memory 1 \
    --prefetch_factor 2 \
    --en_num_layers 6 \
    --de_num_layers 6 \
    --max_seq_length 80 \
    --threshold 10 \
    --use_anatomical_adapter 1 \
    --use_memory_prompt 1 \
    --use_counterfactual_intervention 0 \
    --train_only_innovations 1 \
    --pathology_loss_weight "$PATHOLOGY_LOSS_WEIGHT" \
    --template_loss_weight "$TEMPLATE_LOSS_WEIGHT" \
    --distill_loss_weight "$DISTILL_LOSS_WEIGHT" \
    --distill_temperature "$DISTILL_TEMPERATURE" \
    --teacher_disable_memory_prompt 1 \
    --teacher_disable_counterfactual 1 \
    --pseudo_label_path "$PSEUDO_LABEL_PATH" \
    --template_label_path "$TEMPLATE_LABEL_PATH" \
    --use_confidence_gated_pathology_prompt "$USE_CONFIDENCE_GATED_PATHOLOGY_PROMPT" \
    --pathology_gate_threshold "$PATHOLOGY_GATE_THRESHOLD" \
    --pathology_gate_temperature "$PATHOLOGY_GATE_TEMPERATURE" \
    --use_normal_aware_suppression "$USE_NORMAL_AWARE_SUPPRESSION" \
    --normal_suppression_threshold "$NORMAL_SUPPRESSION_THRESHOLD" \
    --normal_suppression_temperature "$NORMAL_SUPPRESSION_TEMPERATURE" \
    --log_clinical_metrics_during_train "$LOG_CLINICAL_METRICS_DURING_TRAIN"
