#!/bin/bash


# 2. 指定你想测试的权重路径
# 这里填入你之前ls看到的 iu_xray_model_best.pth 的绝对路径
BEST_MODEL="/home/sjx/work/project/CMCRL-main/pretrained/iu_xray_model_best.pth"

# 3. 设置测试任务名称
TASK_NAME="test_best"
SAVE_DIR="results/iu_xray/test/${TASK_NAME}"
REPORT_CSV="${SAVE_DIR}/iu_xray_reports.csv"

echo ">>> 🧪 正在启动测试任务，加载权重: ${BEST_MODEL}"

# 4. 先导出原始报告和模型生成报告
python eval_dump_reports.py \
    --c config/iu_xray/vlci.json \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --load_model_path "$BEST_MODEL" \
    --cuda 0 \
    --seed 10200 \
    --batch_size 16 \
    --beam_size 3 \
    --sample_method "beam_search" \
    --max_seq_length 60

# 5. 再基于导出的报告 CSV 进行离线评估
python eval_clinical.py \
    --c config/iu_xray/vlci.json \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --report_csv "$REPORT_CSV" \
    --cuda 0 \
    --seed 10200
