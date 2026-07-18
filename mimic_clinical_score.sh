#!/bin/bash
REPORT_CSV="/home/sjx/work/project/CMCRL-main/results/mimic_cxr/train/baseline_potential2_20260424_113736/model_best.pth"
TASK_NAME="baseline"
SAVE_DIR="results/mimic_cxr/clinical/${TASK_NAME}"

echo ">>> 🧪 正在基于已导出的报告 CSV 计算临床指标: ${REPORT_CSV}"

python eval_clinical.py \
    --c config/mimic_cxr/vlci.json \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --report_csv "$REPORT_CSV" \
    --cuda 0 \
    --seed 10200
