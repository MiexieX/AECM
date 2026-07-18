#!/bin/bash

set -e
set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_ROOT="${RESULT_ROOT:-experiments/mimic_ablation_results/runs/multiseed_stage2_${EXP_TIME}}"
LOG_FILE="${RESULT_ROOT}/run.log"
STATUS_FILE="${RESULT_ROOT}/status.txt"
SUMMARY_FILE="${RESULT_ROOT}/summary.csv"

SEEDS_STR="${SEEDS_STR:-20200 30200}"
EPOCHS="${EPOCHS:-3}"
BASELINE_CKPT="${BASELINE_CKPT:-/home/sjx/work/project/CMCRL-main/results/mimic_cxr/train/baseline_potential2_20260424_113736/model_best.pth}"

RUN_S0="${RUN_S0:-1}"
RUN_S1="${RUN_S1:-1}"
RUN_S2="${RUN_S2:-1}"

S0_SCRIPT="${S0_SCRIPT:-experiments/mimic_ablation_results/legacy_scripts/stage2_protocol_20260518/scripts/run_s0_clean_stage1_ab.sh}"
S1_SCRIPT="${S1_SCRIPT:-experiments/mimic_ablation_results/legacy_scripts/stage2_protocol_20260518/scripts/run_s1_stage1_continue_no_ccli.sh}"
S2_SCRIPT="${S2_SCRIPT:-experiments/mimic_ablation_results/legacy_scripts/stage2_protocol_20260518/scripts/run_s2_stage1_plus_ccli.sh}"

mkdir -p "$RESULT_ROOT"

LAST_RUN_DIR=""

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
seeds=${SEEDS_STR}
epochs=${EPOCHS}
baseline_ckpt=${BASELINE_CKPT}
EOF
}

latest_run_dir() {
  local group_dir="$1"
  ls -td "${group_dir}"/* 2>/dev/null | head -1 || true
}

append_summary() {
  local seed="$1"
  local row_id="$2"
  local run_dir="$3"
  local init_ckpt="$4"
  local notes="$5"
  local csv_path=""
  local best_val_ckpt=""
  local best_test_ckpt=""

  if [ -f "${run_dir}/mimic_cxr.csv" ]; then
    csv_path="${run_dir}/mimic_cxr.csv"
  fi
  if [ -f "${run_dir}/model_best_val.pth" ]; then
    best_val_ckpt="${run_dir}/model_best_val.pth"
  fi
  if [ -f "${run_dir}/model_best.pth" ]; then
    best_test_ckpt="${run_dir}/model_best.pth"
  fi

  echo "${seed},${row_id},${run_dir},${init_ckpt},${csv_path},${best_val_ckpt},${best_test_ckpt},${notes}" >> "$SUMMARY_FILE"
}

run_stage_step() {
  local seed="$1"
  local row_id="$2"
  local script_path="$3"
  local group_dir="$4"
  local init_key="$5"
  local init_ckpt="$6"

  mkdir -p "$group_dir"
  local before_latest
  before_latest=$(latest_run_dir "$group_dir")

  echo ""
  echo "============================================================"
  echo ">>> Running ${row_id} seed=${seed}"
  echo ">>> script=${script_path}"
  echo ">>> group_dir=${group_dir}"
  echo ">>> ${init_key}=${init_ckpt}"
  echo "============================================================"

  env -u CMCRL_AUTO_LOGGED \
    SEED="$seed" \
    EPOCHS="$EPOCHS" \
    GROUP_DIR="$group_dir" \
    "$init_key"="$init_ckpt" \
    bash "$script_path"

  local after_latest
  after_latest=$(latest_run_dir "$group_dir")
  if [ -z "$after_latest" ] || [ "$after_latest" = "$before_latest" ]; then
    after_latest="$before_latest"
  fi

  append_summary "$seed" "$row_id" "$after_latest" "$init_ckpt" "multi-seed stage-wise supplement"
  LAST_RUN_DIR="$after_latest"
}

echo "seed,row_id,run_dir,init_ckpt,csv_path,model_best_val,model_best,notes" > "$SUMMARY_FILE"

echo "============================================================"
echo ">>> Multi-seed Stage-II supplements started at $(date)"
echo ">>> result_root=${RESULT_ROOT}"
echo ">>> seeds=${SEEDS_STR}"
echo ">>> epochs=${EPOCHS}"
echo "============================================================"

write_status "INIT" "starting"

for SEED_VALUE in ${SEEDS_STR}; do
  SEED_ROOT="${RESULT_ROOT}/seed_${SEED_VALUE}"
  mkdir -p "$SEED_ROOT"

  S0_GROUP="${SEED_ROOT}/S0_clean_stage1_AB"
  S1_GROUP="${SEED_ROOT}/S1_stage1_continue_no_CCLI"
  S2_GROUP="${SEED_ROOT}/S2_stage1_plus_CCLI"

  STAGE1_CKPT_FOR_SEED=""

  if [ "$RUN_S0" = "1" ]; then
    write_status "S0_seed_${SEED_VALUE}" "running"
    run_stage_step "$SEED_VALUE" "S0" "$S0_SCRIPT" "$S0_GROUP" "BASELINE_CKPT" "$BASELINE_CKPT"
    S0_RUN_DIR="$LAST_RUN_DIR"
    if [ -f "${S0_RUN_DIR}/model_best_val.pth" ]; then
      STAGE1_CKPT_FOR_SEED="${S0_RUN_DIR}/model_best_val.pth"
    elif [ -f "${S0_RUN_DIR}/model_best.pth" ]; then
      STAGE1_CKPT_FOR_SEED="${S0_RUN_DIR}/model_best.pth"
    else
      echo ">>> Could not find S0 checkpoint for seed=${SEED_VALUE}: ${S0_RUN_DIR}"
      exit 1
    fi
  else
    STAGE1_CKPT_FOR_SEED="${STAGE1_CKPT:-}"
    if [ -z "$STAGE1_CKPT_FOR_SEED" ]; then
      echo ">>> RUN_S0=0 requires STAGE1_CKPT to be set."
      exit 1
    fi
  fi

  if [ "$RUN_S1" = "1" ]; then
    write_status "S1_seed_${SEED_VALUE}" "running"
    run_stage_step "$SEED_VALUE" "S1" "$S1_SCRIPT" "$S1_GROUP" "STAGE1_CKPT" "$STAGE1_CKPT_FOR_SEED"
  fi

  if [ "$RUN_S2" = "1" ]; then
    write_status "S2_seed_${SEED_VALUE}" "running"
    run_stage_step "$SEED_VALUE" "S2" "$S2_SCRIPT" "$S2_GROUP" "STAGE1_CKPT" "$STAGE1_CKPT_FOR_SEED"
  fi
done

write_status "DONE" "all_completed"

echo ""
echo "============================================================"
echo ">>> Multi-seed Stage-II supplements finished at $(date)"
echo ">>> summary=${SUMMARY_FILE}"
echo "============================================================"

