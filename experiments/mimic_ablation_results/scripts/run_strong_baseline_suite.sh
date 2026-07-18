#!/bin/bash

set -e
set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_ROOT="${RESULT_ROOT:-experiments/mimic_ablation_results/runs/strong_baseline_${EXP_TIME}}"
LOG_FILE="${RESULT_ROOT}/run.log"
STATUS_FILE="${RESULT_ROOT}/status.txt"
SUMMARY_FILE="${RESULT_ROOT}/summary.csv"

SEED="${SEED:-10200}"
CUDA="${CUDA:-0}"
EPOCHS="${EPOCHS:-6}"
MODEL_NAME="${MODEL_NAME:-vlci_original_clean}"
AUTHOR_CKPT="${AUTHOR_CKPT:-/home/sjx/work/project/CMCRL-main/pretrained/mimic_cxr_model_best.pth}"

RUN_AUTHOR_EVAL="${RUN_AUTHOR_EVAL:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TRAINED_EVAL="${RUN_TRAINED_EVAL:-1}"

mkdir -p "$RESULT_ROOT"

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
summary_file=${SUMMARY_FILE}
seed=${SEED}
cuda=${CUDA}
epochs=${EPOCHS}
model_name=${MODEL_NAME}
author_ckpt=${AUTHOR_CKPT}
EOF
}

echo "row_id,kind,model_name,checkpoint,result_dir,report_csv,clinical_csv,notes" > "$SUMMARY_FILE"

append_summary() {
  echo "$1,$2,$3,$4,$5,$6,$7,$8" >> "$SUMMARY_FILE"
}

eval_checkpoint_test_only() {
  local row_id="$1"
  local checkpoint="$2"
  local notes="$3"
  local out_dir="${RESULT_ROOT}/${row_id}"
  local report_csv="${out_dir}/mimic_cxr_reports.csv"
  local clinical_csv="${out_dir}/test_metrics/mimic_cxr_clinical.csv"

  python eval_dump_reports.py \
    --c config/mimic_cxr/vlci.json \
    --model "$MODEL_NAME" \
    --task_name "$row_id" \
    --result_dir "$out_dir" \
    --load_model_path "$checkpoint" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    --batch_size 16 \
    --beam_size 3 \
    --sample_method "beam_search" \
    --max_seq_length 80 \
    --suppress_UNK 1 \
    --split "test"

  python eval_clinical.py \
    --c config/mimic_cxr/vlci.json \
    --task_name "${row_id}_test" \
    --result_dir "${out_dir}/test_metrics" \
    --report_csv "$report_csv" \
    --report_split "test" \
    --cuda "$CUDA" \
    --seed "$SEED"

  append_summary "$row_id" "test_eval" "$MODEL_NAME" "$checkpoint" "$out_dir" "$report_csv" "$clinical_csv" "$notes"
}

train_strong_baseline() {
  local row_id="B0_original_clean_${EPOCHS}epoch_seed${SEED}"
  local out_dir="${RESULT_ROOT}/${row_id}"
  mkdir -p "$out_dir"

  python main.py --c config/mimic_cxr/vlci.json \
    --task finetune \
    --model "$MODEL_NAME" \
    --task_name "$row_id" \
    --result_dir "$out_dir" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    --load_model_path "$AUTHOR_CKPT" \
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
    --pathology_loss_weight 0.0 \
    --template_loss_weight 0.0 \
    --distill_loss_weight 0.0

  append_summary "$row_id" "train" "$MODEL_NAME" "$AUTHOR_CKPT" "$out_dir" "${out_dir}/mimic_cxr.csv" "" "strong original baseline continued from author checkpoint"
}

echo "============================================================"
echo ">>> Strong baseline suite started at $(date)"
echo ">>> result_root=${RESULT_ROOT}"
echo ">>> model=${MODEL_NAME}"
echo ">>> author_ckpt=${AUTHOR_CKPT}"
echo "============================================================"

write_status "INIT" "starting"

if [ "$RUN_AUTHOR_EVAL" = "1" ]; then
  write_status "AUTHOR_EVAL" "running"
  eval_checkpoint_test_only "B_author_0113_test" "$AUTHOR_CKPT" "author provided checkpoint test-only eval"
fi

TRAINED_CKPT=""
if [ "$RUN_TRAIN" = "1" ]; then
  write_status "TRAIN_STRONG_BASELINE" "running"
  train_strong_baseline
  TRAINED_CKPT="${RESULT_ROOT}/B0_original_clean_${EPOCHS}epoch_seed${SEED}/model_best.pth"
fi

if [ "$RUN_TRAINED_EVAL" = "1" ]; then
  if [ -z "$TRAINED_CKPT" ]; then
    TRAINED_CKPT="${TRAINED_CKPT:-${RESULT_ROOT}/B0_original_clean_${EPOCHS}epoch_seed${SEED}/model_best.pth}"
  fi
  if [ -f "$TRAINED_CKPT" ]; then
    write_status "TRAINED_EVAL" "running"
    eval_checkpoint_test_only "B0_original_clean_${EPOCHS}epoch_test" "$TRAINED_CKPT" "trained strong original baseline test-only eval"
  else
    echo ">>> Trained checkpoint not found; skip eval: ${TRAINED_CKPT}"
  fi
fi

write_status "DONE" "all_completed"

echo "============================================================"
echo ">>> Strong baseline suite finished at $(date)"
echo ">>> summary=${SUMMARY_FILE}"
echo "============================================================"

