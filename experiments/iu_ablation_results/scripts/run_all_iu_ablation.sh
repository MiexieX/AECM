#!/bin/bash

set -e
set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_ROOT="${RESULT_ROOT:-experiments/iu_ablation_results/runs/iu_ablation_${EXP_TIME}}"
LOG_FILE="${RESULT_ROOT}/run.log"
STATUS_FILE="${RESULT_ROOT}/status.txt"
SUMMARY_FILE="${RESULT_ROOT}/summary.csv"
SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR:-experiments/iu_ablation_results/summaries}"

CONFIG="${CONFIG:-config/iu_xray/vlci.json}"
AUTHOR_CKPT="${AUTHOR_CKPT:-/home/sjx/work/project/CMCRL-main/pretrained/iu_xray_model_best.pth}"
CUDA="${CUDA:-0}"
SEED="${SEED:-10200}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-6}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BEAM_SIZE="${BEAM_SIZE:-3}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-60}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CCD_GAMMAS_STR="${CCD_GAMMAS_STR:-0.05 0.10 0.15 0.20 0.25 0.30}"

RUN_B0="${RUN_B0:-1}"
RUN_B1="${RUN_B1:-1}"
RUN_S1="${RUN_S1:-1}"
RUN_S2="${RUN_S2:-1}"
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
author_ckpt=${AUTHOR_CKPT}
cuda=${CUDA}
seed=${SEED}
epochs=${EPOCHS}
lr=${LR}
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

  local report_csv="${out_dir}/iu_xray_reports.csv"
  local metrics_dir="${out_dir}/test_metrics"
  local clinical_csv="${metrics_dir}/iu_xray_clinical.csv"

  mkdir -p "$out_dir" "$metrics_dir"

  local extra_args=()
  if [ "$model_name" = "vlci_ccli_clean" ]; then
    extra_args+=(--use_counterfactual_intervention "$use_ccli")
  fi
  if [ -n "$ccd_gamma" ] && [ "$ccd_gamma" != "0" ]; then
    extra_args=(--use_counterfactual_intervention 1 --ccd_gamma "$ccd_gamma" --ccd_use_confidence_gate 1)
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

  append_summary "$row_id" "test_eval" "$model_name" "$checkpoint" "$out_dir" "$report_csv" "$clinical_csv" "" "$SEED" "$selected_gamma" "$notes"
}

train_original_continue() {
  local row_id="IU_B1_original_continue"
  local out_dir="${RESULT_ROOT}/${row_id}"
  mkdir -p "$out_dir"

  python main.py --c "$CONFIG" \
    --task finetune \
    --model vlci_original \
    --task_name "$row_id" \
    --result_dir "$out_dir" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    --load_model_path "$AUTHOR_CKPT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr_en "$LR" \
    --lr_de "$LR" \
    --weight_decay 5e-5 \
    --drop_prob_lm 0.5 \
    --num_workers "$NUM_WORKERS"

  local ckpt
  ckpt=$(find_best_val_ckpt "$out_dir")
  if [ -z "$ckpt" ]; then
    echo "Missing checkpoint for ${row_id}: ${out_dir}"
    exit 1
  fi
  eval_checkpoint_test_only "$row_id" "vlci_original" "$ckpt" "${out_dir}/test_eval_best_val" "original VLCI continuation baseline"
}

train_s1_ab() {
  local row_id="IU_S1_AB_continue"
  local out_dir="${RESULT_ROOT}/${row_id}"
  mkdir -p "$out_dir"

  python main.py --c "$CONFIG" \
    --task finetune \
    --model vlci_ccli_clean \
    --task_name "$row_id" \
    --result_dir "$out_dir" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    --load_model_path "$AUTHOR_CKPT" \
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
  eval_checkpoint_test_only "$row_id" "vlci_ccli_clean" "$ckpt" "${out_dir}/test_eval_best_val" "A+B continuation without CCLI"
}

train_s2_abc() {
  local stage1_ckpt="$1"
  local row_id="IU_S2_ABC_continue"
  local out_dir="${RESULT_ROOT}/${row_id}"
  mkdir -p "$out_dir"

  python main.py --c "$CONFIG" \
    --task finetune \
    --model vlci_ccli_clean \
    --task_name "$row_id" \
    --result_dir "$out_dir" \
    --cuda "$CUDA" \
    --seed "$SEED" \
    --load_model_path "$stage1_ckpt" \
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
  eval_checkpoint_test_only "$row_id" "vlci_ccli_clean" "$ckpt" "${out_dir}/test_eval_best_val" "A+B+C continuation with CCLI" "" "1"
}

run_ccd_selected() {
  local checkpoint="$1"
  local block_root="${RESULT_ROOT}/IU_CCD_selected"
  local val_summary="${block_root}/val_summary.csv"
  local selected_file="${block_root}/selected_gamma.txt"
  mkdir -p "$block_root"
  echo "gamma,result_dir,report_csv,clinical_csv,val_CE_F1,val_BLEU_4,val_CIDEr" > "$val_summary"

  for gamma in ${CCD_GAMMAS_STR}; do
    local gamma_tag="${gamma/./p}"
    local row_id="IU_CCD_val_g${gamma_tag}"
    local out_dir="${block_root}/val_sweep/${row_id}"
    local report_csv="${out_dir}/iu_xray_reports.csv"
    local metrics_dir="${out_dir}/val_metrics"
    local clinical_csv="${metrics_dir}/iu_xray_clinical.csv"
    mkdir -p "$out_dir" "$metrics_dir"

    python eval_dump_reports.py \
      --c "$CONFIG" \
      --model vlci_ccli_clean \
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
    writer = csv.writer(f)
    writer.writerow([
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
        score = float(row["val_CE_F1"])
        bleu4 = float(row["val_BLEU_4"])
        cider = float(row["val_CIDEr"])
        candidate = (score, bleu4, cider, row["gamma"])
        if best is None or candidate > best:
            best = candidate
if best is None:
    raise SystemExit("No valid CCD validation results.")
selected_file.write_text(best[3], encoding="utf-8")
print(f"selected_gamma={best[3]}")
PY

  local selected_gamma
  selected_gamma=$(cat "$selected_file")
  eval_checkpoint_test_only "IU_CCD_selected" "vlci_ccli_clean" "$checkpoint" "${block_root}/test_selected_gamma" "CCD selected by validation CE_F1" "$selected_gamma" "1" "$selected_gamma"
}

echo "============================================================"
echo ">>> IU-Xray ablation suite started at $(date)"
echo ">>> result_root=${RESULT_ROOT}"
echo "============================================================"

write_status "INIT" "starting"

if [ "$RUN_B0" = "1" ]; then
  write_status "B0_AUTHOR_TEST" "running"
  eval_checkpoint_test_only "IU_B0_author_original_test" "vlci_original" "$AUTHOR_CKPT" "${RESULT_ROOT}/IU_B0_author_original_test" "author original IU checkpoint reproduction"
fi

if [ "$RUN_B1" = "1" ]; then
  write_status "B1_ORIGINAL_CONTINUE" "running"
  train_original_continue
fi

S1_CKPT=""
if [ "$RUN_S1" = "1" ]; then
  write_status "S1_AB_CONTINUE" "running"
  train_s1_ab
  S1_CKPT=$(find_best_val_ckpt "${RESULT_ROOT}/IU_S1_AB_continue")
fi

S2_CKPT=""
if [ "$RUN_S2" = "1" ]; then
  write_status "S2_ABC_CONTINUE" "running"
  if [ -z "$S1_CKPT" ]; then
    S1_CKPT="${S1_CKPT_OVERRIDE:-}"
  fi
  if [ -z "$S1_CKPT" ]; then
    echo "RUN_S2=1 requires S1 checkpoint. Run S1 first or set S1_CKPT_OVERRIDE."
    exit 1
  fi
  train_s2_abc "$S1_CKPT"
  S2_CKPT=$(find_best_val_ckpt "${RESULT_ROOT}/IU_S2_ABC_continue")
fi

if [ "$RUN_CCD" = "1" ]; then
  write_status "CCD_SELECTED" "running"
  if [ -z "$S2_CKPT" ]; then
    S2_CKPT="${S2_CKPT_OVERRIDE:-}"
  fi
  if [ -z "$S2_CKPT" ]; then
    echo "RUN_CCD=1 requires S2 checkpoint. Run S2 first or set S2_CKPT_OVERRIDE."
    exit 1
  fi
  run_ccd_selected "$S2_CKPT"
fi

if [ "$UPDATE_SUMMARY" = "1" ]; then
  write_status "UPDATE_SUMMARY" "running"
  python experiments/iu_ablation_results/scripts/update_iu_ablation_summary.py \
    --run-root "$RESULT_ROOT" \
    --out-dir "$SUMMARY_OUT_DIR" \
    --seed "$SEED"
fi

write_status "DONE" "all_completed"

echo ""
echo "============================================================"
echo ">>> IU-Xray ablation suite finished at $(date)"
echo ">>> summary=${SUMMARY_FILE}"
echo ">>> summary_out_dir=${SUMMARY_OUT_DIR}"
echo "============================================================"
