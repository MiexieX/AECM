#!/bin/bash

set -e
set -u

EXP_TIME=$(date +"%Y%m%d_%H%M%S")

# Focused sweep around the strongest previous IU recipe:
# lr=1e-5, seed=30200, epoch=10, BLEU_4=0.185443.
# Goal: find a best-val checkpoint that exceeds the paper IU BLEU_4=0.189340.
export SWEEP_ROOT="${SWEEP_ROOT:-experiments/iu_ablation_results/runs/iu_focused_20epoch_sweep_${EXP_TIME}}"
export SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR:-experiments/iu_ablation_results/summaries/iu_focused_20epoch_sweep_${EXP_TIME}}"

export EPOCHS_LIST="${EPOCHS_LIST:-20}"
export LR_LIST="${LR_LIST:-1e-5 1.2e-5 1.5e-5 2e-5}"
export SEED_LIST="${SEED_LIST:-30200 40200 50200 60200 70200}"
export BATCH_SIZE_LIST="${BATCH_SIZE_LIST:-16}"

export RUN_P2_ON_EACH="${RUN_P2_ON_EACH:-0}"
export RUN_CCD_ON_EACH="${RUN_CCD_ON_EACH:-0}"
export TARGET_BLEU4="${TARGET_BLEU4:-0.189340}"

echo "============================================================"
echo ">>> IU focused 20-epoch sweep"
echo ">>> target BLEU_4: ${TARGET_BLEU4}"
echo ">>> sweep_root: ${SWEEP_ROOT}"
echo ">>> summary_out_dir: ${SUMMARY_OUT_DIR}"
echo ">>> epochs: ${EPOCHS_LIST}"
echo ">>> lrs: ${LR_LIST}"
echo ">>> seeds: ${SEED_LIST}"
echo ">>> batch sizes: ${BATCH_SIZE_LIST}"
echo ">>> RUN_P2_ON_EACH=${RUN_P2_ON_EACH}, RUN_CCD_ON_EACH=${RUN_CCD_ON_EACH}"
echo "============================================================"

bash experiments/iu_ablation_results/scripts/run_iu_recipe_sweep.sh
