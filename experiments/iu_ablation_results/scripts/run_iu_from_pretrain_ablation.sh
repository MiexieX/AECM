#!/bin/bash

set -e
set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_ROOT="${RESULT_ROOT:-experiments/iu_ablation_results/runs/iu_from_pretrain_${EXP_TIME}}"
LOG_FILE="${RESULT_ROOT}/run.log"
STATUS_FILE="${RESULT_ROOT}/status.txt"
SUMMARY_FILE="${RESULT_ROOT}/summary.csv"
SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR:-experiments/iu_ablation_results/summaries}"

CONFIG="${CONFIG:-config/iu_xray/vlci.json}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-/home/sjx/work/project/CMCRL-main/pretrained/nano_current_checkpoint.pth}"
P0_MODEL="${P0_MODEL:-vlci}"
P1_MODEL="${P1_MODEL:-vlci}"
P2_MODEL="${P2_MODEL:-vlci}"
CUDA="${CUDA:-0}"
SEED="${SEED:-10200}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BEAM_SIZE="${BEAM_SIZE:-3}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-60}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CCD_GAMMAS_STR="${CCD_GAMMAS_STR:-0.05 0.10 0.15 0.20 0.25 0.30}"

RUN_P0="${RUN_P0:-1}"
RUN_P1="${RUN_P1:-1}"
RUN_P2="${RUN_P2:-1}"
RUN_CCD="${RUN_CCD:-1}"
UPDATE_SUMMARY="${UPDATE_SUMMARY:-1}"

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
config=${CONFIG}
pretrain_ckpt=${PRETRAIN_CKPT}
p0_model=${P0_MODEL}
p1_model=${P1_MODEL}
p2_model=${P2_MODEL}
cuda=${CUDA}
seed=${SEED}
epochs=${EPOCHS}
lr=${LR}
num_workers=${NUM_WORKERS}
ccd_gammas=${CCD_GAMMAS_STR}
EOF
}

echo "row_id,kind,model_name,checkpoint,result_dir,report_csv,clinical_csv,epochs,seed,selected_gamma,notes" > "$SUMMARY_FILE"

append_summary() {
  echo "$1,$2,$3,$4,$5,$6,$7,$8,$9,${10},${11}" >> "$SUMMARY_FILE"
}

find_best_val_ckpt() {
  local run_dir="$1"
  if [ -f "${run_dir}/model_best_val.pth" ]; then
    echo "${run_dir}/model_best_val.pth"
  elif [ -f "${run_dir}/model_best.pth" ]; then
    echo "${run_dir}/model_best.pth"
  else
    echo ""
  fi
}

eval_checkpoint_test_only() {
  local row_id="$1"
  local model_name="$2"
  local checkpoint="$3"
  local out_dir="$4"
  local notes="$5"
  local selected_gamma="${6:-}"
  local use_ccli="${7:-0}"
  local ccd_gamma="${8:-0}"
  local use_anatomy="${9:-0}"
  local use_memory="${10:-0}"

  local report_csv="${out_dir}/iu_xray_reports.csv"
  local metrics_dir="${out_dir}/test_metrics"
  local clinical_csv="${metrics_dir}/iu_xray_clinical.csv"

  mkdir -p "$out_dir" "$metrics_dir"

  local extra_args=()
  if [ "$model_name" = "vlci" ] || [ "$model_name" = "vlci_ccli_clean" ]; then
    extra_args+=(
      --use_anatomical_adapter "$use_anatomy"
      --use_memory_prompt "$use_memory"
      --use_counterfactual_intervention "$use_ccli"
    )
  fi
  if [ -n "$ccd_gamma" ] && [ "$ccd_gamma" != "0" ]; then
    extra_args=(
      --use_anatomical_adapter 1
      --use_memory_prompt 1
      --use_counterfactual_intervention 1
      --ccd_gamma "$ccd_gamma"
      --ccd_use_confidence_gate 1
    )
  fi

  python eval_dump_reports.py \
    --c "$CONFIG" \
    --model "$model_name" \
    --task_name "$row_id" \
    --result_dir "$out_dir" \
    --load_model_path "$checkpoint" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --beam_size "$BEAM_SIZE" \
    --sample_method "beam_search" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --suppress_UNK 0 \
    --split "test" \
    "${extra_args[@]}"

  python eval_clinical.py \
    --c "$CONFIG" \
    --task_name "${row_id}_test" \
    --result_dir "$metrics_dir" \
    --report_csv "$report_csv" \
    --report_split "test" \
    --cuda "$CUDA" \
    --seed "$SEED"

  append_summary "$row_id" "test_eval" "$model_name" "$checkpoint" "$out_dir" "$report_csv" "$clinical_csv" "$EPOCHS" "$SEED" "$selected_gamma" "$notes"
}

train_p0_original() {
  local row_id="IU_P0_original_from_pretrain"
  local out_dir="${RESULT_ROOT}/${row_id}"
  mkdir -p "$out_dir"

  python main.py --c "$CONFIG" \
    --task finetune \
    --model "$P0_MODEL" \
    --task_name "$row_id" \
    --result_dir "$out_dir" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    --load_model_path "$PRETRAIN_CKPT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr_en "$LR" \
    --lr_de "$LR" \
    --weight_decay 5e-5 \
    --drop_prob_lm 0.5 \
    --num_workers "$NUM_WORKERS" \
    --use_anatomical_adapter 0 \
    --use_memory_prompt 0 \
    --use_counterfactual_intervention 0

  local ckpt
  ckpt=$(find_best_val_ckpt "$out_dir")
  if [ -z "$ckpt" ]; then
    echo "Missing checkpoint for ${row_id}: ${out_dir}"
    exit 1
  fi
  eval_checkpoint_test_only "$row_id" "$P0_MODEL" "$ckpt" "${out_dir}/test_eval_best_val" "vlci with A/B/C disabled, fine-tuned from nano pretrain" "" "0" "0" "0" "0"
}

train_p1_ab() {
  local row_id="IU_P1_AB_from_pretrain"
  local out_dir="${RESULT_ROOT}/${row_id}"
  mkdir -p "$out_dir"

  python main.py --c "$CONFIG" \
    --task finetune \
    --model "$P1_MODEL" \
    --task_name "$row_id" \
    --result_dir "$out_dir" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    --load_model_path "$PRETRAIN_CKPT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr_en "$LR" \
    --lr_de "$LR" \
    --weight_decay 5e-5 \
    --drop_prob_lm 0.5 \
    --num_workers "$NUM_WORKERS" \
    --use_anatomical_adapter 1 \
    --use_memory_prompt 1 \
    --use_counterfactual_intervention 0

  local ckpt
  ckpt=$(find_best_val_ckpt "$out_dir")
  if [ -z "$ckpt" ]; then
    echo "Missing checkpoint for ${row_id}: ${out_dir}"
    exit 1
  fi
  eval_checkpoint_test_only "$row_id" "$P1_MODEL" "$ckpt" "${out_dir}/test_eval_best_val" "vlci A+B fine-tuned from nano pretrain" "" "0" "0" "1" "1"
}

train_p2_abc() {
  local p1_ckpt="$1"
  local row_id="IU_P2_ABC_from_pretrain"
  local out_dir="${RESULT_ROOT}/${row_id}"
  mkdir -p "$out_dir"

  python main.py --c "$CONFIG" \
    --task finetune \
    --model "$P2_MODEL" \
    --task_name "$row_id" \
    --result_dir "$out_dir" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    --load_model_path "$p1_ckpt" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr_en "$LR" \
    --lr_de "$LR" \
    --weight_decay 5e-5 \
    --drop_prob_lm 0.5 \
    --num_workers "$NUM_WORKERS" \
    --use_anatomical_adapter 1 \
    --use_memory_prompt 1 \
    --use_counterfactual_intervention 1

  local ckpt
  ckpt=$(find_best_val_ckpt "$out_dir")
  if [ -z "$ckpt" ]; then
    echo "Missing checkpoint for ${row_id}: ${out_dir}"
    exit 1
  fi
  eval_checkpoint_test_only "$row_id" "$P2_MODEL" "$ckpt" "${out_dir}/test_eval_best_val" "vlci A+B+C fine-tuned from P1 best-val" "" "1" "0" "1" "1"
}

run_ccd_selected() {
  local checkpoint="$1"
  local block_root="${RESULT_ROOT}/IU_P2_CCD_selected"
  local val_summary="${block_root}/val_summary.csv"
  local selected_file="${block_root}/selected_gamma.txt"
  mkdir -p "$block_root"
  echo "gamma,result_dir,report_csv,clinical_csv,val_CE_F1,val_BLEU_4,val_CIDEr" > "$val_summary"

  for gamma in ${CCD_GAMMAS_STR}; do
    local gamma_tag="${gamma/./p}"
    local row_id="IU_P2_CCD_val_g${gamma_tag}"
    local out_dir="${block_root}/val_sweep/${row_id}"
    local report_csv="${out_dir}/iu_xray_reports.csv"
    local metrics_dir="${out_dir}/val_metrics"
    local clinical_csv="${metrics_dir}/iu_xray_clinical.csv"
    mkdir -p "$out_dir" "$metrics_dir"

    python eval_dump_reports.py \
      --c "$CONFIG" \
      --model "$P2_MODEL" \
      --task_name "$row_id" \
      --result_dir "$out_dir" \
      --load_model_path "$checkpoint" \
      --cuda "$CUDA" \
      --seed "$SEED" \
      --batch_size "$BATCH_SIZE" \
      --beam_size "$BEAM_SIZE" \
      --sample_method "beam_search" \
      --max_seq_length "$MAX_SEQ_LENGTH" \
      --suppress_UNK 0 \
      --split "val" \
      --use_anatomical_adapter 1 \
      --use_memory_prompt 1 \
      --use_counterfactual_intervention 1 \
      --ccd_gamma "$gamma" \
      --ccd_use_confidence_gate 1

    python eval_clinical.py \
      --c "$CONFIG" \
      --task_name "${row_id}_val" \
      --result_dir "$metrics_dir" \
      --report_csv "$report_csv" \
      --report_split "val" \
      --cuda "$CUDA" \
      --seed "$SEED"

    python - "$gamma" "$out_dir" "$report_csv" "$clinical_csv" "$val_summary" <<'PY'
import csv
import sys
from pathlib import Path

gamma, out_dir, report_csv, clinical_csv, summary = sys.argv[1:]
with Path(clinical_csv).open("r", newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))
with Path(summary).open("a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow([
        gamma,
        out_dir,
        report_csv,
        clinical_csv,
        row.get("test_CE_F1", ""),
        row.get("test_BLEU_4", ""),
        row.get("test_CIDEr", ""),
    ])
PY
  done

  python - "$val_summary" "$selected_file" <<'PY'
import csv
import sys
from pathlib import Path

summary, selected_file = map(Path, sys.argv[1:])
best = None
with summary.open("r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        candidate = (
            float(row["val_CE_F1"]),
            float(row["val_BLEU_4"]),
            float(row["val_CIDEr"]),
            row["gamma"],
        )
        if best is None or candidate > best:
            best = candidate
if best is None:
    raise SystemExit("No valid CCD validation results.")
selected_file.write_text(best[3], encoding="utf-8")
print(f"selected_gamma={best[3]}")
PY

  local selected_gamma
  selected_gamma=$(cat "$selected_file")
  eval_checkpoint_test_only "IU_P2_CCD_selected" "$P2_MODEL" "$checkpoint" "${block_root}/test_selected_gamma" "CCD selected by validation CE_F1 from P2" "$selected_gamma" "1" "$selected_gamma"
}

echo "============================================================"
echo ">>> IU-Xray from-pretrain ablation suite started at $(date)"
echo ">>> result_root=${RESULT_ROOT}"
echo "============================================================"

write_status "INIT" "starting"

if [ "$RUN_P0" = "1" ]; then
  write_status "P0_ORIGINAL_FROM_PRETRAIN" "running"
  train_p0_original
fi

P1_CKPT=""
if [ "$RUN_P1" = "1" ]; then
  write_status "P1_AB_FROM_PRETRAIN" "running"
  train_p1_ab
  P1_CKPT=$(find_best_val_ckpt "${RESULT_ROOT}/IU_P1_AB_from_pretrain")
fi

P2_CKPT=""
if [ "$RUN_P2" = "1" ]; then
  write_status "P2_ABC_FROM_PRETRAIN" "running"
  if [ -z "$P1_CKPT" ]; then
    P1_CKPT="${P1_CKPT_OVERRIDE:-}"
  fi
  if [ -z "$P1_CKPT" ]; then
    echo "RUN_P2=1 requires P1 checkpoint. Run P1 first or set P1_CKPT_OVERRIDE."
    exit 1
  fi
  train_p2_abc "$P1_CKPT"
  P2_CKPT=$(find_best_val_ckpt "${RESULT_ROOT}/IU_P2_ABC_from_pretrain")
fi

if [ "$RUN_CCD" = "1" ]; then
  write_status "P2_CCD_SELECTED" "running"
  if [ -z "$P2_CKPT" ]; then
    P2_CKPT="${P2_CKPT_OVERRIDE:-}"
  fi
  if [ -z "$P2_CKPT" ]; then
    echo "RUN_CCD=1 requires P2 checkpoint. Run P2 first or set P2_CKPT_OVERRIDE."
    exit 1
  fi
  run_ccd_selected "$P2_CKPT"
fi

if [ "$UPDATE_SUMMARY" = "1" ]; then
  write_status "UPDATE_SUMMARY" "running"
  python experiments/iu_ablation_results/scripts/update_iu_from_pretrain_summary.py \
    --run-root "$RESULT_ROOT" \
    --out-dir "$SUMMARY_OUT_DIR" \
    --seed "$SEED"
fi

write_status "DONE" "all_completed"

echo ""
echo "============================================================"
echo ">>> IU-Xray from-pretrain ablation suite finished at $(date)"
echo ">>> summary=${SUMMARY_FILE}"
echo ">>> summary_out_dir=${SUMMARY_OUT_DIR}"
echo "============================================================"
