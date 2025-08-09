"""
前向规划算法

实现论文Section 3.5中的前向规划机制
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


class ForwardPlanner:
    """
    前向规划器
    
    实现论文中的两阶段规划算法
    """
    
    def __init__(
        self,
        planning_depth: int = 3,
        top_k_candidates: int = 3,
        physical_weight: float = 0.3
    ):
        """
        初始化前向规划器
        
        Args:
            planning_depth: 规划深度 dmax
            top_k_candidates: Top-k候选数
            physical_weight: 物理约束权重
        """
        self.planning_depth = planning_depth
        self.top_k_candidates = top_k_candidates
        self.physical_weight = physical_weight
    
    def plan_sequence(
        self,
        model,
        input_features: torch.Tensor,
        physical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        对整个序列进行前向规划
        
        Args:
            model: 训练好的模型
            input_features: 输入特征
            physical_features: 物理约束特征
            
        Returns:
            优化后的指法序列
        """
        batch_size, seq_len, _ = input_features.shape
        
        # 获取初始预测
        with torch.no_grad():
            outputs = model(input_features, physical_features)
            initial_probs = F.softmax(outputs['logits'], dim=-1)
        
        # 对每个序列进行规划
        planned_sequences = []
        
        for b in range(batch_size):
            sequence_probs = initial_probs[b]  # [seq_len, num_classes]
            sequence_physical = physical_features[b] if physical_features is not None else None
            
            planned_seq = self._plan_single_sequence(sequence_probs, sequence_physical)
            planned_sequences.append(planned_seq)
        
        return torch.stack(planned_sequences)
    
    def _plan_single_sequence(
        self,
        probs: torch.Tensor,
        physical_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        对单个序列进行规划
        
        Args:
            probs: 预测概率 [seq_len, num_classes]
            physical_features: 物理约束特征 [seq_len, constraint_dim]
            
        Returns:
            规划后的指法序列 [seq_len]
        """
        seq_len, num_classes = probs.shape
        
        # 简化版本：贪心选择
        # 在实际实现中，这里应该使用动态规划或搜索算法
        
        best_sequence = []
        
        for t in range(seq_len):
            step_probs = probs[t]
            
            # 获取top-k候选
            top_k_probs, top_k_indices = torch.topk(step_probs, self.top_k_candidates)
            
            # 评估每个候选
            best_score = float('-inf')
            best_finger = top_k_indices[0].item()
            
            for i, finger_idx in enumerate(top_k_indices):
                score = top_k_probs[i].item()
                
                # 添加物理约束评分
                if physical_features is not None and t < len(physical_features):
                    physical_score = self._evaluate_physical_constraints(
                        finger_idx.item(), physical_features[t]
                    )
                    score += self.physical_weight * physical_score
                
                # 添加序列一致性评分
                if len(best_sequence) > 0:
                    consistency_score = self._evaluate_consistency(
                        best_sequence[-1], finger_idx.item()
                    )
                    score += 0.1 * consistency_score
                
                if score > best_score:
                    best_score = score
                    best_finger = finger_idx.item()
            
            best_sequence.append(best_finger)
        
        return torch.tensor(best_sequence)
    
    def _evaluate_physical_constraints(
        self,
        finger: int,
        physical_features: torch.Tensor
    ) -> float:
        """
        评估物理约束得分
        
        Args:
            finger: 指法
            physical_features: 物理约束特征
            
        Returns:
            物理约束得分
        """
        # 简化版本：基于物理特征的启发式评分
        if len(physical_features) >= 5:
            stretching = physical_features[0].item()
            crossing = physical_features[1].item()
            natural_violation = physical_features[3].item()
            strength_violation = physical_features[4].item()
            
            # 惩罚高违反程度
            penalty = stretching + crossing + natural_violation + strength_violation
            return -penalty
        
        return 0.0
    
    def _evaluate_consistency(self, prev_finger: int, curr_finger: int) -> float:
        """
        评估序列一致性得分
        
        Args:
            prev_finger: 前一个指法
            curr_finger: 当前指法
            
        Returns:
            一致性得分
        """
        # 简化版本：惩罚大幅跳跃
        if prev_finger == 5:  # 忽略padding
            return 0.0
        
        # 转换为实际指法 (-5到5)
        prev_actual = prev_finger - 5
        curr_actual = curr_finger - 5
        
        # 如果是同一只手，惩罚大跳跃
        if (prev_actual > 0) == (curr_actual > 0):  # 同一只手
            jump = abs(abs(prev_actual) - abs(curr_actual))
            return -0.1 * jump
        
        return 0.0


class BeamSearchPlanner(ForwardPlanner):
    """
    使用束搜索的前向规划器
    
    更高级的搜索算法实现
    """
    
    def __init__(
        self,
        beam_width: int = 5,
        **kwargs
    ):
        """
        初始化束搜索规划器
        
        Args:
            beam_width: 束宽度
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.beam_width = beam_width
    
    def _plan_single_sequence(
        self,
        probs: torch.Tensor,
        physical_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        使用束搜索进行序列规划
        
        Args:
            probs: 预测概率
            physical_features: 物理约束特征
            
        Returns:
            规划后的指法序列
        """
        seq_len, num_classes = probs.shape
        
        # 初始化束
        # 每个beam: (sequence, score)
        beams = [([], 0.0)]
        
        for t in range(seq_len):
            new_beams = []
            
            for sequence, score in beams:
                step_probs = probs[t]
                
                # 获取top-k候选
                top_k_probs, top_k_indices = torch.topk(
                    step_probs, min(self.top_k_candidates, num_classes)
                )
                
                for i, finger_idx in enumerate(top_k_indices):
                    new_sequence = sequence + [finger_idx.item()]
                    new_score = score + torch.log(top_k_probs[i]).item()
                    
                    # 添加物理约束和一致性评分
                    if physical_features is not None and t < len(physical_features):
                        physical_score = self._evaluate_physical_constraints(
                            finger_idx.item(), physical_features[t]
                        )
                        new_score += self.physical_weight * physical_score
                    
                    if len(sequence) > 0:
                        consistency_score = self._evaluate_consistency(
                            sequence[-1], finger_idx.item()
                        )
                        new_score += 0.1 * consistency_score
                    
                    new_beams.append((new_sequence, new_score))
            
            # 保留最好的beam_width个候选
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:self.beam_width]
        
        # 返回最佳序列
        best_sequence = beams[0][0]
        return torch.tensor(best_sequence) 