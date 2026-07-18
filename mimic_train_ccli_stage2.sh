#!/bin/bash

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
TASK_NAME="ccli_stage2_${EXP_TIME}"
SAVE_DIR="results/mimic_cxr/train/${TASK_NAME}"
mkdir -p "$SAVE_DIR"
LOG_FILE="${SAVE_DIR}/run.log"
if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

# Stage-II CCLI training. Start from the Stage-I gated prompt checkpoint and
# enable counterfactual causal logit intervention for continued finetuning.
BASELINE_CKPT="/home/sjx/work/project/CMCRL-main/results/mimic_cxr/train/distill_prompt_best_f1/model_best.pth"

# Ordinary distillation setting A:
# - freeze backbone, train only innovation modules
# - keep phrase memory off
# - use very small clinical/template supervision
# - use stronger KL to prevent BLEU collapse
PATHOLOGY_LOSS_WEIGHT=0.001
TEMPLATE_LOSS_WEIGHT=0.001
DISTILL_LOSS_WEIGHT=0.30
DISTILL_TEMPERATURE=2.0
PSEUDO_LABEL_PATH="data/mimic_cxr_pseudo_labels.json"
TEMPLATE_LABEL_PATH="data/mimic_cxr_template_labels.json"
USE_CONFIDENCE_GATED_PATHOLOGY_PROMPT=1
PATHOLOGY_GATE_THRESHOLD=0.50
PATHOLOGY_GATE_TEMPERATURE=0.12
USE_NORMAL_AWARE_SUPPRESSION=1
NORMAL_SUPPRESSION_THRESHOLD=0.45
NORMAL_SUPPRESSION_TEMPERATURE=0.10
USE_COUNTERFACTUAL_INTERVENTION=1
CCLI_ANOMALY_THRESHOLD=0.35
CCLI_ANOMALY_TEMPERATURE=0.10
CCLI_MAX_ALPHA=0.25
CCLI_HIDDEN_DIM=16
LOG_CLINICAL_METRICS_DURING_TRAIN=0

echo ">>> Stage-II CCLI + 蒸馏保护训练: ${TASK_NAME}"
echo ">>> 结果目录: ${SAVE_DIR}"
echo ">>> 日志文件: ${LOG_FILE}"
echo ">>> Stage-I 初始化权重: ${BASELINE_CKPT}"
echo ">>> train_only_innovations=1, memory_prompt=on, phrase_memory=off"
echo ">>> pathology=${PATHOLOGY_LOSS_WEIGHT}, template=${TEMPLATE_LOSS_WEIGHT}, distill=${DISTILL_LOSS_WEIGHT}"
echo ">>> confidence_gate=${USE_CONFIDENCE_GATED_PATHOLOGY_PROMPT}, gate_tau=${PATHOLOGY_GATE_THRESHOLD}, gate_temp=${PATHOLOGY_GATE_TEMPERATURE}"
echo ">>> normal_suppress=${USE_NORMAL_AWARE_SUPPRESSION}, normal_tau=${NORMAL_SUPPRESSION_THRESHOLD}, normal_temp=${NORMAL_SUPPRESSION_TEMPERATURE}"
echo ">>> ccli=${USE_COUNTERFACTUAL_INTERVENTION}, ccli_tau=${CCLI_ANOMALY_THRESHOLD}, ccli_temp=${CCLI_ANOMALY_TEMPERATURE}, ccli_max_alpha=${CCLI_MAX_ALPHA}"
echo ">>> log_clinical_metrics_during_train=${LOG_CLINICAL_METRICS_DURING_TRAIN}"
echo ">>> dataloader: num_workers=12, pin_memory=1, prefetch_factor=2"

python main.py --c config/mimic_cxr/vlci.json \
    --task finetune \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --cuda 0 \
    --seed 10200 \
    --load_model_path "$BASELINE_CKPT" \
    --epochs 3 \
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
    --use_phrase_memory 0 \
    --train_only_innovations 1 \
    --pathology_loss_weight "$PATHOLOGY_LOSS_WEIGHT" \
    --template_loss_weight "$TEMPLATE_LOSS_WEIGHT" \
    --distill_loss_weight "$DISTILL_LOSS_WEIGHT" \
    --distill_temperature "$DISTILL_TEMPERATURE" \
    --teacher_disable_memory_prompt 1 \
    --pseudo_label_path "$PSEUDO_LABEL_PATH" \
    --template_label_path "$TEMPLATE_LABEL_PATH" \
    --use_confidence_gated_pathology_prompt "$USE_CONFIDENCE_GATED_PATHOLOGY_PROMPT" \
    --pathology_gate_threshold "$PATHOLOGY_GATE_THRESHOLD" \
    --pathology_gate_temperature "$PATHOLOGY_GATE_TEMPERATURE" \
    --use_normal_aware_suppression "$USE_NORMAL_AWARE_SUPPRESSION" \
    --normal_suppression_threshold "$NORMAL_SUPPRESSION_THRESHOLD" \
    --normal_suppression_temperature "$NORMAL_SUPPRESSION_TEMPERATURE" \
    --use_counterfactual_intervention "$USE_COUNTERFACTUAL_INTERVENTION" \
    --ccli_anomaly_threshold "$CCLI_ANOMALY_THRESHOLD" \
    --ccli_anomaly_temperature "$CCLI_ANOMALY_TEMPERATURE" \
    --ccli_max_alpha "$CCLI_MAX_ALPHA" \
    --ccli_hidden_dim "$CCLI_HIDDEN_DIM" \
    --log_clinical_metrics_during_train "$LOG_CLINICAL_METRICS_DURING_TRAIN" \
    --teacher_disable_counterfactual 1
