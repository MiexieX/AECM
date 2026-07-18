#!/bin/bash

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
GROUP_DIR="${GROUP_DIR:-results/mimic_cxr/clean_ablations_20260518/C1_baseline_plus_A_clean}"
TASK_NAME="${TASK_NAME:-c1_clean_baseline_plus_a_seed${SEED:-10200}_${EXP_TIME}}"
SAVE_DIR="${SAVE_DIR:-${GROUP_DIR}/${TASK_NAME}}"
mkdir -p "$SAVE_DIR"
LOG_FILE="${SAVE_DIR}/run.log"
if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

SEED="${SEED:-10200}"
EPOCHS="${EPOCHS:-3}"
BASELINE_CKPT="${BASELINE_CKPT:-/home/sjx/work/project/CMCRL-main/results/mimic_cxr/train/baseline_potential2_20260424_113736/model_best.pth}"
DISTILL_LOSS_WEIGHT="${DISTILL_LOSS_WEIGHT:-0.30}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"

echo ">>> C1 clean baseline + A: ${TASK_NAME}"
echo ">>> save_dir=${SAVE_DIR}"
echo ">>> seed=${SEED}, epochs=${EPOCHS}"
echo ">>> baseline_ckpt=${BASELINE_CKPT}"
echo ">>> model=vlci_ccli_clean"
echo ">>> A=on, B=off, C=off"

python main.py --c config/mimic_cxr/vlci.json \
    --task finetune \
    --model vlci_ccli_clean \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --cuda 0 \
    --seed "$SEED" \
    --load_model_path "$BASELINE_CKPT" \
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
    --use_memory_prompt 0 \
    --use_counterfactual_intervention 0 \
    --train_only_innovations 1 \
    --pathology_loss_weight 0.0 \
    --template_loss_weight 0.0 \
    --distill_loss_weight "$DISTILL_LOSS_WEIGHT" \
    --distill_temperature "$DISTILL_TEMPERATURE" \
    --teacher_disable_memory_prompt 1 \
    --teacher_disable_counterfactual 1

