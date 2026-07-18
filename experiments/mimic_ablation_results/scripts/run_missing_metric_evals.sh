#!/bin/bash

set -e
set -u
set -o pipefail

RESULT_ROOT="${RESULT_ROOT:-experiments/mimic_ablation_results/runs/missing_metric_evals_20260608}"
BASELINE_ROOT="${BASELINE_ROOT:-results/mimic_cxr/test/baseline_potential2_20260424_113736_full_eval}"
SUMMARY_FILE="${RESULT_ROOT}/summary.csv"
STATUS_FILE="${RESULT_ROOT}/status.txt"
LOG_FILE="${RESULT_ROOT}/run.log"

MODEL_NAME="${MODEL_NAME:-vlci}"
CONFIG_PATH="${CONFIG_PATH:-config/mimic_cxr/vlci.json}"
CUDA="${CUDA:-0}"
SEED="${SEED:-10200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BEAM_SIZE="${BEAM_SIZE:-3}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-80}"
FORCE="${FORCE:-0}"

BASELINE_CKPT="${BASELINE_CKPT:-results/mimic_cxr/train/baseline_potential2_20260424_113736/model_best.pth}"
ORIGINAL_SUMMARY="${ORIGINAL_SUMMARY:-experiments/mimic_ablation_results/summaries/experiment_summary_20260527/all_ablation_summary.csv}"

mkdir -p "$RESULT_ROOT"
mkdir -p "$BASELINE_ROOT"

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
baseline_root=${BASELINE_ROOT}
summary_file=${SUMMARY_FILE}
model_name=${MODEL_NAME}
cuda=${CUDA}
seed=${SEED}
force=${FORCE}
EOF
}

write_summary_header() {
  echo "row_id,kind,modules,A,B,C,model_name,checkpoint,result_dir,report_csv,clinical_csv,status,notes" > "$SUMMARY_FILE"
}

append_summary() {
  echo "$1,$2,$3,$4,$5,$6,$7,$8,$9,${10},${11},${12},${13}" >> "$SUMMARY_FILE"
}

find_checkpoint() {
  local run_dir="$1"
  if [ -f "${run_dir}/model_best_val.pth" ]; then
    echo "${run_dir}/model_best_val.pth"
  elif [ -f "${run_dir}/model_best.pth" ]; then
    echo "${run_dir}/model_best.pth"
  else
    echo ""
  fi
}

run_eval_from_checkpoint() {
  local row_id="$1"
  local kind="$2"
  local modules="$3"
  local use_a="$4"
  local use_b="$5"
  local use_c="$6"
  local checkpoint="$7"
  local out_dir="$8"
  local notes="$9"

  local report_csv="${out_dir}/mimic_cxr_reports.csv"
  local clinical_dir="${out_dir}/test_metrics"
  local clinical_csv="${clinical_dir}/mimic_cxr_clinical.csv"

  if [ -z "$checkpoint" ] || [ ! -f "$checkpoint" ]; then
    append_summary "$row_id" "$kind" "$modules" "$use_a" "$use_b" "$use_c" "$MODEL_NAME" "$checkpoint" "$out_dir" "$report_csv" "$clinical_csv" "missing_checkpoint" "$notes"
    echo ">>> Missing checkpoint for ${row_id}: ${checkpoint}"
    return 1
  fi

  mkdir -p "$out_dir"
  mkdir -p "$clinical_dir"

  if [ "$FORCE" = "1" ] || [ ! -s "$report_csv" ]; then
    echo ""
    echo "============================================================"
    echo ">>> Dump reports: ${row_id}"
    echo ">>> checkpoint=${checkpoint}"
    echo ">>> modules=${modules} A=${use_a} B=${use_b} C=${use_c}"
    echo "============================================================"

    python eval_dump_reports.py \
      --c "$CONFIG_PATH" \
      --model "$MODEL_NAME" \
      --task_name "$row_id" \
      --result_dir "$out_dir" \
      --load_model_path "$checkpoint" \
      --cuda "$CUDA" \
      --seed "$SEED" \
      --batch_size "$BATCH_SIZE" \
      --beam_size "$BEAM_SIZE" \
      --sample_method "beam_search" \
      --max_seq_length "$MAX_SEQ_LENGTH" \
      --suppress_UNK 1 \
      --split "test" \
      --use_anatomical_adapter "$use_a" \
      --use_memory_prompt "$use_b" \
      --use_counterfactual_intervention "$use_c"
  else
    echo ">>> Reusing existing report CSV for ${row_id}: ${report_csv}"
  fi

  if [ "$FORCE" = "1" ] || [ ! -s "$clinical_csv" ]; then
    echo ""
    echo "================================================------------"
    echo ">>> Compute clinical metrics: ${row_id}"
    echo "================================================------------"

    python eval_clinical.py \
      --c "$CONFIG_PATH" \
      --task_name "${row_id}_test" \
      --result_dir "$clinical_dir" \
      --report_csv "$report_csv" \
      --report_split "test" \
      --cuda "$CUDA" \
      --seed "$SEED"
  else
    echo ">>> Reusing existing clinical CSV for ${row_id}: ${clinical_csv}"
  fi

  append_summary "$row_id" "$kind" "$modules" "$use_a" "$use_b" "$use_c" "$MODEL_NAME" "$checkpoint" "$out_dir" "$report_csv" "$clinical_csv" "completed" "$notes"
}

run_ablation_row() {
  local row_id="$1"
  local modules="$2"
  local use_a="$3"
  local use_b="$4"
  local use_c="$5"
  local run_dir="$6"

  local checkpoint
  checkpoint=$(find_checkpoint "$run_dir")
  local out_dir="${RESULT_ROOT}/${row_id}"
  run_eval_from_checkpoint "$row_id" "ablation" "$modules" "$use_a" "$use_b" "$use_c" "$checkpoint" "$out_dir" "missing_clinical_metric_fill"
}

echo "============================================================"
echo ">>> Missing metric evaluations started at $(date)"
echo ">>> result_root=${RESULT_ROOT}"
echo ">>> baseline_root=${BASELINE_ROOT}"
echo ">>> model=${MODEL_NAME} from models/vlci.py"
echo "============================================================"

write_status "INIT" "starting"
write_summary_header

write_status "BASELINE" "running baseline_potential2 full eval"
run_eval_from_checkpoint \
  "baseline_potential2_20260424_113736" \
  "baseline" \
  "baseline" \
  "0" "0" "0" \
  "$BASELINE_CKPT" \
  "$BASELINE_ROOT" \
  "baseline_potential2_full_test_eval"

write_status "ABLATIONS" "running missing clinical metric rows"
run_ablation_row "C1" "A" "1" "0" "0" "experiments/mimic_ablation_results/archive/ABLATION_RESULTS_20260522/clean_ablations_20260518/C1_baseline_plus_A_clean/c1_clean_baseline_plus_a_seed10200_20260519_154209"
run_ablation_row "C2" "B" "0" "1" "0" "experiments/mimic_ablation_results/archive/ABLATION_RESULTS_20260522/clean_ablations_20260518/C2_baseline_plus_B_clean/c2_clean_baseline_plus_b_seed10200_20260520_011351"
run_ablation_row "C3" "AB" "1" "1" "0" "experiments/mimic_ablation_results/archive/ABLATION_RESULTS_20260522/clean_ablations_20260518/C3_baseline_plus_AB_clean/c3_clean_baseline_plus_ab_seed10200_20260520_105102"
run_ablation_row "E4" "BC" "0" "1" "1" "experiments/mimic_ablation_results/archive/ABLATION_RESULTS_20260522/ablations_20260515/E4_baseline_plus_BC/e4_baseline_plus_bc_20260516_150355"
run_ablation_row "E5" "ABC" "1" "1" "1" "experiments/mimic_ablation_results/archive/ABLATION_RESULTS_20260522/ablations_20260515/E5_baseline_plus_ABC/e5_baseline_plus_abc_20260517_035206"
run_ablation_row "A6" "A" "1" "0" "0" "experiments/mimic_ablation_results/runs/clean_ablations_6epoch_20260522/A_clean_6epoch/c1_clean_baseline_plus_a_seed10200_20260522_104252"
run_ablation_row "B6" "B" "0" "1" "0" "experiments/mimic_ablation_results/runs/clean_ablations_6epoch_20260522/B_clean_6epoch/c2_clean_baseline_plus_b_seed10200_20260523_054423"
run_ablation_row "AB6" "AB" "1" "1" "0" "experiments/mimic_ablation_results/runs/clean_ablations_6epoch_20260522/AB_clean_6epoch/c3_clean_baseline_plus_ab_seed10200_20260524_004355"
run_ablation_row "BC6" "BC" "0" "1" "1" "experiments/mimic_ablation_results/runs/clean_ablations_6epoch_20260522/BC_clean_6epoch/c4_clean_baseline_plus_bc_seed10200_20260524_194312"
run_ablation_row "ABC6" "ABC" "1" "1" "1" "experiments/mimic_ablation_results/runs/clean_ablations_6epoch_20260522/ABC_clean_6epoch/c5_clean_baseline_plus_abc_seed10200_20260525_211943"

write_status "SUMMARY_UPDATE" "merging clinical metrics into summary copy"
python experiments/mimic_ablation_results/scripts/update_missing_clinical_metrics.py \
  --original-summary "$ORIGINAL_SUMMARY" \
  --eval-summary "$SUMMARY_FILE" \
  --output-dir "$RESULT_ROOT" \
  --baseline-clinical "${BASELINE_ROOT}/test_metrics/mimic_cxr_clinical.csv"

write_status "DONE" "all_completed"

echo "============================================================"
echo ">>> Missing metric evaluations finished at $(date)"
echo ">>> eval_summary=${SUMMARY_FILE}"
echo ">>> supplemented_summary=${RESULT_ROOT}/all_ablation_summary_with_missing_clinical_20260608.csv"
echo "============================================================"
