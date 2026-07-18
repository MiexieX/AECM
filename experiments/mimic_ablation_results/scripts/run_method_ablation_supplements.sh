#!/bin/bash

set -e
set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_ROOT="${RESULT_ROOT:-experiments/mimic_ablation_results/runs/method_ablation_supplements_${EXP_TIME}}"
LOG_FILE="${RESULT_ROOT}/run.log"
STATUS_FILE="${RESULT_ROOT}/status.txt"
SUMMARY_FILE="${RESULT_ROOT}/summary.csv"

SEED="${SEED:-10200}"
CUDA="${CUDA:-0}"

S0_CKPT="${S0_CKPT:-/home/sjx/work/project/CMCRL-main/experiments/mimic_ablation_results/archive/ABLATION_RESULTS_20260522/stage2_protocol_20260518/S0_clean_stage1_AB/s0_clean_stage1_ab_seed10200_20260520_202611/model_best_val.pth}"
S1_CKPT="${S1_CKPT:-/home/sjx/work/project/CMCRL-main/experiments/mimic_ablation_results/archive/ABLATION_RESULTS_20260522/stage2_protocol_20260518/S1_stage1_continue_no_CCLI/s1_stage1_continue_no_ccli_seed10200_20260521_055531/model_best_val.pth}"
S2_CKPT="${S2_CKPT:-/home/sjx/work/project/CMCRL-main/experiments/mimic_ablation_results/archive/ABLATION_RESULTS_20260522/stage2_protocol_20260518/S2_stage1_plus_CCLI/s2_stage1_plus_ccli_seed10200_20260521_152617/model_best_val.pth}"

RUN_S0_S1_S2_TEST_EVAL="${RUN_S0_S1_S2_TEST_EVAL:-1}"
RUN_CCD_SWEEP="${RUN_CCD_SWEEP:-1}"
CCD_GAMMAS_STR="${CCD_GAMMAS_STR:-0.05 0.10 0.15 0.20 0.25 0.30}"

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
EOF
}

echo "row_id,kind,model_name,checkpoint,result_dir,report_csv,clinical_csv,notes" > "$SUMMARY_FILE"

append_summary() {
  echo "$1,$2,$3,$4,$5,$6,$7,$8" >> "$SUMMARY_FILE"
}

eval_clean_checkpoint_test_only() {
  local row_id="$1"
  local checkpoint="$2"
  local notes="$3"
  local out_dir="${RESULT_ROOT}/${row_id}"
  local report_csv="${out_dir}/mimic_cxr_reports.csv"
  local clinical_csv="${out_dir}/test_metrics/mimic_cxr_clinical.csv"

  python eval_dump_reports.py \
    --c config/mimic_cxr/vlci.json \
    --model vlci_ccli_clean \
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

  append_summary "$row_id" "test_eval" "vlci_ccli_clean" "$checkpoint" "$out_dir" "$report_csv" "$clinical_csv" "$notes"
}

run_ccd_sweep() {
  local ccd_root="${RESULT_ROOT}/S2_CCD_test_sweep"
  mkdir -p "$ccd_root"

  DISABLE_AUTO_LOG=1 \
  RESULT_ROOT="$ccd_root" \
  BEST_MODEL="$S2_CKPT" \
  GAMMAS_STR="$CCD_GAMMAS_STR" \
  SPLIT="test" \
  SEED="$SEED" \
  CUDA="$CUDA" \
  bash experiments/mimic_ablation_results/scripts/run_final_s2_ccd_sweep.sh

  append_summary "S2_CCD_test_sweep" "ccd_sweep" "vlci_ccli_clean" "$S2_CKPT" "$ccd_root" "" "${ccd_root}/summary.csv" "test-only CCD decoding supplement"
}

echo "============================================================"
echo ">>> Method ablation supplements started at $(date)"
echo ">>> result_root=${RESULT_ROOT}"
echo "============================================================"

write_status "INIT" "starting"

if [ "$RUN_S0_S1_S2_TEST_EVAL" = "1" ]; then
  write_status "S0_TEST_EVAL" "running"
  eval_clean_checkpoint_test_only "S0_clean_AB_test_eval" "$S0_CKPT" "Stage-I AB model_best_val test-only eval"
  write_status "S1_TEST_EVAL" "running"
  eval_clean_checkpoint_test_only "S1_no_CCLI_test_eval" "$S1_CKPT" "Stage-II no-CCLI model_best_val test-only eval"
  write_status "S2_TEST_EVAL" "running"
  eval_clean_checkpoint_test_only "S2_CCLI_test_eval" "$S2_CKPT" "Stage-II CCLI model_best_val test-only eval"
fi

if [ "$RUN_CCD_SWEEP" = "1" ]; then
  write_status "CCD_SWEEP" "running"
  run_ccd_sweep
fi

write_status "DONE" "all_completed"

echo "============================================================"
echo ">>> Method ablation supplements finished at $(date)"
echo ">>> summary=${SUMMARY_FILE}"
echo "============================================================"

