#!/bin/bash

REPORT_CSV="/home/sjx/work/project/CMCRL-main/results/iu_xray/test/test_best/iu_xray_reports.csv"
TASK_NAME="test_best"
SAVE_DIR="results/iu_xray/clinical/${TASK_NAME}"

echo ">>> 🧪 正在基于已导出的 IU 报告 CSV 计算临床指标: ${REPORT_CSV}"

python eval_clinical.py \
    --c config/iu_xray/vlci.json \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --report_csv "$REPORT_CSV" \
    --cuda 0 \
    --seed 10200
