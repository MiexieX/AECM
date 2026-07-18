#!/bin/bash

# 2. 设置实验名称
EXP_TIME=$(date +"%Y%m%d_%H%M%S")
TASK_NAME="new3_${EXP_TIME}"
SAVE_DIR="results/mimic_cxr/train/${TASK_NAME}"

# Optional auxiliary losses for the paper branch.
# Keep them at 0.0 for a pure LM-loss memory-prompt run.
# Suggested first BCE try: PATHOLOGY_LOSS_WEIGHT=0.03
# Distillation still needs Trainer-side teacher wiring before enabling.
PATHOLOGY_LOSS_WEIGHT=0.0
DISTILL_LOSS_WEIGHT=0.0
PSEUDO_LABEL_PATH="data/mimic_cxr_pseudo_labels.json"

echo ">>> 🚀 正在启动 MIMIC-CXR 训练任务: ${TASK_NAME}"
echo ">>> 📁 结果将保存在: ${SAVE_DIR}"

# 3. 运行程序（动态覆盖 MIMIC-CXR 的特定超参数）
# 注意：如果要基于预训练模型微调，请把 load_model_path 的双引号内换成预训练权重路径
python main.py --c config/mimic_cxr/vlci.json \
    --task finetune \
    --task_name "$TASK_NAME" \
    --result_dir "$SAVE_DIR" \
    --cuda 0 \
    --seed 10200 \
    --load_model_path "/home/sjx/work/project/CMCRL-main/pretrained/mimic_cxr_model_best.pth" \
    --epochs 6 \
    --batch_size 16 \
    --val_intervals 4 \
    --lr_en 1e-6 \
    --lr_de 1e-6 \
    --weight_decay 5e-5 \
	    --num_workers 12 \
	    --en_num_layers 6 \
	    --de_num_layers 6 \
	    --max_seq_length 80 \
	    --threshold 10 \
	    --pathology_loss_weight "$PATHOLOGY_LOSS_WEIGHT" \
	    --distill_loss_weight "$DISTILL_LOSS_WEIGHT" \
	    --pseudo_label_path "$PSEUDO_LABEL_PATH"
