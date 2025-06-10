#!/bin/bash

# =============================================================================
# VMamba-B LaSBiRD Dataset Evaluation Script for Single A800 80GB GPU
# =============================================================================

# 设置基本参数
MODEL_NAME="vmamba_base_224_lasbird"
CONFIG_FILE="classification/configs/vssm/vmambav2_base_224_lasbird.yaml"
OUTPUT_DIR="output/${MODEL_NAME}"
BATCH_SIZE=512  # 评估时可以使用更大的batch size
CHECKPOINT_PATH=""  # 需要用户指定检查点路径

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 参数检查
if [ $# -eq 0 ]; then
    echo "用法: $0 <checkpoint_path>"
    echo "示例: $0 output/vmamba_base_224_lasbird/ckpt_epoch_100.pth"
    exit 1
fi

CHECKPOINT_PATH=$1

# 检查必要文件是否存在
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "错误: 配置文件 ${CONFIG_FILE} 不存在!"
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "错误: 检查点文件 ${CHECKPOINT_PATH} 不存在!"
    exit 1
fi

echo "开始评估 VMamba-B 模型在 LaSBiRD 数据集上..."
echo "配置文件: ${CONFIG_FILE}"
echo "检查点: ${CHECKPOINT_PATH}"
echo "批处理大小: ${BATCH_SIZE}"
echo "使用GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=================================================================="

# 评估命令
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=1 \
    --master_addr="127.0.0.1" \
    --master_port=29502 \
    classification/main.py \
    --cfg ${CONFIG_FILE} \
    --batch-size ${BATCH_SIZE} \
    --eval \
    --resume ${CHECKPOINT_PATH} \
    --output ${OUTPUT_DIR}_eval \
    --tag "eval_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee logs/eval_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log

# 检查评估是否成功完成
if [ $? -eq 0 ]; then
    echo "=================================================================="
    echo "评估完成! 评估日志保存在 logs/ 目录中"
    echo "=================================================================="
else
    echo "=================================================================="
    echo "评估失败! 请检查错误日志。"
    echo "=================================================================="
    exit 1
fi 