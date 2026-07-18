#!/bin/bash

# 2. 设置实验名称（加入时间戳防止覆盖）
EXP_TIME=$(date +"%Y%m%d_%H%M%S")
TASK_NAME="new1_${EXP_TIME}"
SAVE_DIR="results/iu_xray/train/${TASK_NAME}"

echo ">>> 🚀 正在启动训练任务: ${TASK_NAME}"
echo ">>> 📁 结果将保存在: ${SAVE_DIR}"

# 3. 运行程序（在此手动修改任何你想改的超参数）
python main.py --c config/iu_xray/vlci.json \
    --task finetune \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --cuda 0 \
    --seed 10200 \
    --load_model_path "/home/sjx/work/project/CMCRL-main/pretrained/iu_xray_model_best.pth" \
    --epochs 10 \
    --batch_size 16 \
    --lr_en 1e-5 \
    --lr_de 1e-5 \
    --weight_decay 5e-5 \
    --drop_prob_lm 0.5 \
    --num_workers 2
