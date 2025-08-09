"""
XLPro Model: CNN-BiLSTM Hybrid Model for Automatic Piano Fingering Generation

基于论文:
"CNN-BiLSTM Hybrid Model with Physical Constraints for Automatic Piano Fingering Generation"
by Shingyui He, Tianze Zhang

主要特性:
- CNN-BiLSTM混合架构
- 物理约束集成
- 注意力机制
- 前向规划
"""

__version__ = "0.1.0"
__author__ = "Based on work by Shingyui He, Tianze Zhang"

# 导入核心组件
from .data import PianoFingeringDataset, FeatureExtractor, PhysicalConstraints
from .models import CNNBiLSTMModel

__all__ = [
    "PianoFingeringDataset",
    "FeatureExtractor", 
    "CNNBiLSTMModel",
    "PhysicalConstraints"
] 