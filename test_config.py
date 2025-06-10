#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from config import get_config
import argparse

def test_config():
    parser = argparse.ArgumentParser('配置测试脚本')
    parser.add_argument('--cfg', type=str, default='configs/vssm/vmambav2_base_224_lasbird.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    try:
        config = get_config(args)
        print("✅ 配置加载成功!")
        print(f"数据集: {config.DATA.DATASET}")
        print(f"训练CSV: {config.DATA.TRAIN_CSV}")
        print(f"测试CSV: {config.DATA.TEST_CSV}")
        print(f"验证集比例: {config.DATA.VAL_SUBSET_RATIO}")
        print(f"批处理大小: {config.DATA.BATCH_SIZE}")
        print(f"模型类型: {config.MODEL.TYPE}")
        print(f"模型名称: {config.MODEL.NAME}")
        print(f"AMP启用: {config.AMP_ENABLE}")
        print(f"随机种子: {config.SEED}")
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

if __name__ == '__main__':
    success = test_config()
    sys.exit(0 if success else 1) 