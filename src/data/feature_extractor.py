"""
特征提取器

实现论文Section 3.2中的基础特征提取
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from .physical_constraints import PhysicalConstraints


class FeatureExtractor:
    """
    基础特征提取器
    
    提取论文中定义的基础音乐特征和物理约束特征
    """
    
    def __init__(self, use_physical_constraints: bool = True):
        """
        初始化特征提取器
        
        Args:
            use_physical_constraints: 是否使用物理约束特征
        """
        self.use_physical_constraints = use_physical_constraints
        
        if use_physical_constraints:
            self.physical_constraints = PhysicalConstraints()
    
    def extract_basic_features(self, notes: List[Dict]) -> torch.Tensor:
        """
        提取基础特征
        
        Args:
            notes: 音符序列
            
        Returns:
            基础特征tensor [seq_len, feature_dim]
        """
        features = []
        
        for note in notes:
            feature_vector = [
                note['midi_number'] / 127.0,     # 归一化MIDI号
                note['duration'],                # 时长
                float(note['is_black_key']),     # 黑键标识
                note['onset_velocity'] / 127.0,  # 归一化力度
                note['channel']                  # 声道
            ]
            features.append(feature_vector)
        
        return torch.FloatTensor(features)
    
    def extract_physical_features(self, notes: List[Dict]) -> torch.Tensor:
        """
        提取物理约束特征
        
        Args:
            notes: 音符序列
            
        Returns:
            物理约束特征tensor [seq_len, constraint_dim]
        """
        if not self.use_physical_constraints:
            return torch.zeros(len(notes), 5)
        
        return self.physical_constraints.compute_sequence_constraints(notes)
    
    def extract_all_features(self, notes: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取所有特征
        
        Args:
            notes: 音符序列
            
        Returns:
            (基础特征, 物理约束特征)
        """
        basic_features = self.extract_basic_features(notes)
        physical_features = self.extract_physical_features(notes)
        
        return basic_features, physical_features 