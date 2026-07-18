#!/bin/bash

set -e
set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_ROOT="${RESULT_ROOT:-experiments/mimic_ablation_results/runs/all_supplementary_${EXP_TIME}}"
LOG_FILE="${RESULT_ROOT}/run.log"
STATUS_FILE="${RESULT_ROOT}/status.txt"
SUMMARY_FILE="${RESULT_ROOT}/summary.txt"

RUN_STRONG_BASELINE="${RUN_STRONG_BASELINE:-1}"
RUN_METHOD_SUPPLEMENTS="${RUN_METHOD_SUPPLEMENTS:-1}"
RUN_MULTI_SEED="${RUN_MULTI_SEED:-1}"

SEEDS_STR="${SEEDS_STR:-20200 30200}"
EPOCHS_BASELINE="${EPOCHS_BASELINE:-6}"
EPOCHS_MULTI_SEED="${EPOCHS_MULTI_SEED:-3}"

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
run_strong_baseline=${RUN_STRONG_BASELINE}
run_method_supplements=${RUN_METHOD_SUPPLEMENTS}
run_multi_seed=${RUN_MULTI_SEED}
seeds=${SEEDS_STR}
EOF
}

append_summary() {
  {
    echo "[$1]"
    echo "status=$2"
    echo "result_dir=$3"
    echo "summary=$4"
    echo ""
  } >> "$SUMMARY_FILE"
}

run_block() {
  local key="$1"
  local script_path="$2"
  local block_root="$3"
  shift 3

  mkdir -p "$block_root"
  write_status "$key" "running"

  echo ""
  echo "============================================================"
  echo ">>> Running block: ${key}"
  echo ">>> script=${script_path}"
  echo ">>> block_root=${block_root}"
  echo "============================================================"

  env -u CMCRL_AUTO_LOGGED \
    RESULT_ROOT="$block_root" \
    "$@" \
    bash "$script_path"

  local block_summary="${block_root}/summary.csv"
  if [ ! -f "$block_summary" ]; then
    block_summary="${block_root}/summary.txt"
  fi
  append_summary "$key" "completed" "$block_root" "$block_summary"
}

echo "Supplementary experiment master run" > "$SUMMARY_FILE"
echo "created=$(date)" >> "$SUMMARY_FILE"
echo "result_root=${RESULT_ROOT}" >> "$SUMMARY_FILE"
echo "seeds=${SEEDS_STR}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "============================================================"
echo ">>> All supplementary experiments started at $(date)"
echo ">>> result_root=${RESULT_ROOT}"
echo "============================================================"

write_status "INIT" "starting"

if [ "$RUN_STRONG_BASELINE" = "1" ]; then
  run_block \
    "strong_baseline" \
    "experiments/mimic_ablation_results/scripts/run_strong_baseline_suite.sh" \
    "${RESULT_ROOT}/strong_baseline" \
    EPOCHS="$EPOCHS_BASELINE"
fi

if [ "$RUN_METHOD_SUPPLEMENTS" = "1" ]; then
  run_block \
    "method_ablation_supplements" \
    "experiments/mimic_ablation_results/scripts/run_method_ablation_supplements.sh" \
    "${RESULT_ROOT}/method_ablation_supplements"
fi

if [ "$RUN_MULTI_SEED" = "1" ]; then
  run_block \
    "multiseed_stage2" \
    "experiments/mimic_ablation_results/scripts/run_multiseed_stage2_supplements.sh" \
    "${RESULT_ROOT}/multiseed_stage2" \
    SEEDS_STR="$SEEDS_STR" \
    EPOCHS="$EPOCHS_MULTI_SEED"
fi

write_status "DONE" "all_completed"

echo ""
echo "============================================================"
echo ">>> All supplementary experiments finished at $(date)"
echo ">>> summary=${SUMMARY_FILE}"
echo "============================================================"

