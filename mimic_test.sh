#!/bin/bash
# BLEU-first evaluation entrypoint.
# You can override the checkpoint externally:
#   BEST_MODEL=/abs/path/to/model_best.pth bash mimic_test.sh
BEST_MODEL="${BEST_MODEL:-/home/sjx/work/project/CMCRL-main/results/mimic_cxr/train/baseline_potential2_20260424_113736/model_best.pth}"

# Result naming can also be overridden if needed.
TASK_NAME="${TASK_NAME:-baseline}"
SAVE_DIR="${SAVE_DIR:-results/mimic_cxr/test/${TASK_NAME}}"
REPORT_CSV="${SAVE_DIR}/mimic_cxr_reports.csv"
mkdir -p "$SAVE_DIR"
LOG_FILE="${SAVE_DIR}/run.log"
if [ "${DISABLE_AUTO_LOG:-0}" != "1" ] && [ -z "${CMCRL_AUTO_LOGGED:-}" ]; then
  export CMCRL_AUTO_LOGGED=1
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo ">>> 🧪 正在启动 MIMIC-CXR BLEU-first 测试任务，加载权重: ${BEST_MODEL}"
echo ">>> save_dir=${SAVE_DIR}"
echo ">>> log_file=${LOG_FILE}"

# 1. 导出 test split 的生成报告
python eval_dump_reports.py \
    --c config/mimic_cxr/vlci.json \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --load_model_path "$BEST_MODEL" \
    --cuda 0 \
    --seed 10200 \
    --batch_size 16 \
    --beam_size 3 \
    --sample_method "beam_search" \
    --max_seq_length 80 \
    --suppress_UNK 1 \
    --split "test" 

# 2. 离线评分
python eval_clinical.py \
    --c config/mimic_cxr/vlci.json \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --report_csv "$REPORT_CSV" \
    --cuda 0 \
    --seed 10200
