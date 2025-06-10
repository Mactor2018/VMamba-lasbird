#!/bin/bash

# =============================================================================
# VMamba-B LaSBiRD Dataset Training Script for Single A800 80GB GPU
# =============================================================================



# 设置基本参数
MODEL_NAME="vmamba_base_224_lasbird"
CONFIG_FILE="classification/configs/vssm/vmambav2_base_224_lasbird.yaml"
OUTPUT_DIR="output/${MODEL_NAME}"
BATCH_SIZE=256  # A800 80GB可以支持更大的batch size
NUM_WORKERS=16  # 根据CPU核心数调整
TAG_PREFIX="train_A800_single"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export HF_ENDPOINT=https://hf-mirror.com

# 检查必要文件是否存在
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "错误: 配置文件 ${CONFIG_FILE} 不存在!"
    exit 1
fi


echo "开始训练 VMamba-B 模型在 LaSBiRD 数据集上..."
echo "配置文件: ${CONFIG_FILE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "批处理大小: ${BATCH_SIZE}"
echo "使用GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=================================================================="

# 训练命令 - 单GPU训练
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=1 \
    --master_addr="127.0.0.1" \
    --master_port=29501 \
    classification/main.py \
    --cfg ${CONFIG_FILE} \
    --batch-size ${BATCH_SIZE} \
    --output ${OUTPUT_DIR} \
    --tag "${TAG_PREFIX}_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee logs/train_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log

# 检查训练是否成功完成
if [ $? -eq 0 ]; then
    echo "=================================================================="
    echo "训练完成! 日志和模型保存在:"
    echo "- 模型权重: ${OUTPUT_DIR}/"
    echo "- 训练日志: logs/"
    echo "=================================================================="
else
    echo "=================================================================="
    echo "训练失败! 请检查错误日志。"
    echo "=================================================================="
    exit 1
fi 