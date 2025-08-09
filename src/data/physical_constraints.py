"""
物理约束计算模块

实现论文Section 3.3中的生物力学指法约束:
- 拉伸率 (Stretching Rate)
- 交叉距离 (Crossing Distance) 
- 手部位置 (Hand Position)
- 自然违反 (Natural Violation)
- 力度违反 (Strength Violation)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger


class PhysicalConstraints:
    """
    物理约束计算器
    
    计算论文中定义的5种生物力学约束特征
    """
    
    def __init__(
        self,
        velocity_threshold: float = 80.0,
        finger_weights: Optional[Dict[int, float]] = None
    ):
        """
        初始化物理约束计算器
        
        Args:
            velocity_threshold: 力度阈值 τ
            finger_weights: 手指特定权重 ω_f
        """
        self.velocity_threshold = velocity_threshold
        
        # 默认手指权重 (4指和5指较弱)
        self.finger_weights = finger_weights or {
            1: 1.0,   # 拇指
            2: 1.0,   # 食指  
            3: 1.0,   # 中指
            4: 1.2,   # 无名指 (较弱)
            5: 1.5,   # 小指 (最弱)
            -1: 1.0,  # 左手拇指
            -2: 1.0,  # 左手食指
            -3: 1.0,  # 左手中指
            -4: 1.2,  # 左手无名指
            -5: 1.5   # 左手小指
        }
    
    def compute_all_constraints(
        self,
        notes: List[Dict],
        current_idx: int
    ) -> Dict[str, float]:
        """
        计算当前音符的所有物理约束特征
        
        Args:
            notes: 音符序列
            current_idx: 当前音符索引
            
        Returns:
            包含所有约束特征的字典
        """
        current_note = notes[current_idx]
        
        constraints = {
            'stretching_rate': 0.0,
            'crossing_distance': 0.0,
            'hand_position': self._compute_hand_position(notes, current_idx),
            'natural_violation': self._compute_natural_violation(current_note),
            'strength_violation': self._compute_strength_violation(current_note)
        }
        
        # 计算与前一个音符的约束 (如果存在)
        if current_idx > 0:
            prev_note = notes[current_idx - 1]
            constraints['stretching_rate'] = self._compute_stretching_rate(
                prev_note, current_note
            )
            constraints['crossing_distance'] = self._compute_crossing_distance(
                prev_note, current_note
            )
        
        return constraints
    
    def _compute_stretching_rate(
        self,
        note_i: Dict,
        note_j: Dict
    ) -> float:
        """
        计算拉伸率 φ_stretch
        
        φ_stretch(i,j) = |mi - mj| / |fi - fj|
        
        量化连续手指之间所需的扩展
        """
        # 只计算同一只手的拉伸
        if note_i['channel'] != note_j['channel']:
            return 0.0
        
        finger_i = note_i.get('finger', 0)
        finger_j = note_j.get('finger', 0)
        
        # 跳过无效指法
        if finger_i == 0 or finger_j == 0:
            return 0.0
        
        midi_i = note_i['midi_number']
        midi_j = note_j['midi_number']
        
        finger_diff = abs(finger_i - finger_j)
        if finger_diff == 0:
            return 0.0
        
        # 计算拉伸率
        pitch_diff = abs(midi_i - midi_j)
        stretching_rate = pitch_diff / finger_diff
        
        # 归一化 (正常手指跨度约为7个半音)
        return min(stretching_rate / 7.0, 2.0)
    
    def _compute_crossing_distance(
        self,
        note_i: Dict,
        note_j: Dict
    ) -> float:
        """
        计算交叉距离 φ_cross
        
        φ_cross(i,j) = |mi - mj| if crossing occurs, else 0
        
        衡量手指交叉的物理难度
        """
        # 只计算同一只手的交叉
        if note_i['channel'] != note_j['channel']:
            return 0.0
        
        finger_i = note_i.get('finger', 0)
        finger_j = note_j.get('finger', 0)
        
        if finger_i == 0 or finger_j == 0:
            return 0.0
        
        midi_i = note_i['midi_number']
        midi_j = note_j['midi_number']
        
        # 检查是否发生交叉
        # 交叉条件: (fi < fj ∧ mi > mj) ∨ (fi > fj ∧ mi < mj)
        crossing_occurs = (
            (finger_i < finger_j and midi_i > midi_j) or
            (finger_i > finger_j and midi_i < midi_j)
        )
        
        if crossing_occurs:
            crossing_distance = abs(midi_i - midi_j)
            return min(crossing_distance / 12.0, 2.0)  # 归一化
        
        return 0.0
    
    def _compute_hand_position(
        self,
        notes: List[Dict],
        current_idx: int,
        window_size: int = 5
    ) -> float:
        """
        计算手部位置 φ_pos
        
        φ_pos(t) = (1/|Nt|) * Σ mi
        
        追踪当前位置和手部移动
        """
        current_note = notes[current_idx]
        channel = current_note['channel']
        
        # 获取时间窗口内同一只手的音符
        start_idx = max(0, current_idx - window_size)
        end_idx = min(len(notes), current_idx + window_size + 1)
        
        same_hand_notes = [
            note for note in notes[start_idx:end_idx]
            if note['channel'] == channel
        ]
        
        if not same_hand_notes:
            return current_note['midi_number'] / 127.0
        
        # 计算平均位置
        avg_position = sum(note['midi_number'] for note in same_hand_notes) / len(same_hand_notes)
        
        return avg_position / 127.0  # 归一化
    
    def _compute_natural_violation(self, note: Dict) -> float:
        """
        计算自然违反 φ_nat
        
        φ_nat(i) = 1 if violations occur, else 0
        
        识别违反自然手型的指法
        """
        finger = note.get('finger', 0)
        is_black_key = note.get('is_black_key', False)
        midi_number = note['midi_number']
        
        violations = 0.0
        
        # 违反规则1: 拇指按黑键 (不推荐)
        if abs(finger) == 1 and is_black_key:
            violations += 0.3
        
        # 违反规则2: 小指按极端位置
        if abs(finger) == 5:
            if midi_number < 36 or midi_number > 96:  # 极低或极高
                violations += 0.4
        
        # 违反规则3: 手指跨度过大的位置
        # 右手
        if finger > 0:
            if finger == 5 and midi_number > 84:  # 右手小指在高音区
                violations += 0.3
            elif finger == 1 and midi_number < 48:  # 右手拇指在低音区
                violations += 0.2
        # 左手  
        else:
            if finger == -5 and midi_number < 48:  # 左手小指在低音区
                violations += 0.3
            elif finger == -1 and midi_number > 72:  # 左手拇指在高音区
                violations += 0.2
        
        return min(violations, 1.0)
    
    def _compute_strength_violation(self, note: Dict) -> float:
        """
        计算力度违反 φ_str
        
        φ_str(i) = von,i * ω_f if (fi ∈ {4,5} ∧ von,i > τ), else 0
        
        惩罚弱手指的过度用力
        """
        finger = note.get('finger', 0)
        velocity = note.get('onset_velocity', 64)
        
        # 只对4指和5指(无名指和小指)应用此约束
        if abs(finger) in {4, 5}:
            if velocity > self.velocity_threshold:
                finger_weight = self.finger_weights.get(finger, 1.0)
                violation = (velocity / 127.0) * finger_weight
                return min(violation, 1.0)
        
        return 0.0
    
    def compute_sequence_constraints(
        self,
        notes: List[Dict]
    ) -> torch.Tensor:
        """
        计算整个序列的物理约束特征
        
        Args:
            notes: 音符序列
            
        Returns:
            形状为 (sequence_length, num_constraints) 的tensor
        """
        constraint_features = []
        
        for i in range(len(notes)):
            constraints = self.compute_all_constraints(notes, i)
            
            # 转换为特征向量
            feature_vector = [
                constraints['stretching_rate'],
                constraints['crossing_distance'], 
                constraints['hand_position'],
                constraints['natural_violation'],
                constraints['strength_violation']
            ]
            
            constraint_features.append(feature_vector)
        
        return torch.FloatTensor(constraint_features)
    
    def compute_constraint_loss(
        self,
        predicted_fingers: torch.Tensor,
        notes: List[Dict],
        weight: float = 0.3
    ) -> torch.Tensor:
        """
        计算物理约束损失
        
        L_phys(y, ŷ, Φ) = L_CE(y, ŷ) · E[ω(Φ)]
        
        Args:
            predicted_fingers: 预测的指法 [batch_size, seq_len, num_classes]
            notes: 原始音符数据
            weight: 物理约束权重
            
        Returns:
            物理约束损失项
        """
        batch_size, seq_len = predicted_fingers.shape[:2]
        device = predicted_fingers.device
        
        total_penalty = 0.0
        
        for batch_idx in range(batch_size):
            batch_notes = notes[batch_idx] if isinstance(notes[0], list) else notes
            
            for seq_idx in range(seq_len):
                if seq_idx < len(batch_notes):
                    note = batch_notes[seq_idx].copy()
                    
                    # 获取预测的指法
                    pred_finger_prob = predicted_fingers[batch_idx, seq_idx]
                    pred_finger = torch.argmax(pred_finger_prob).item()
                    
                    # 转换指法编号 (0-10 -> -5到5)
                    if pred_finger <= 5:
                        note['finger'] = pred_finger - 5  # -5 到 0
                    else:
                        note['finger'] = pred_finger - 5  # 1 到 5
                    
                    # 计算物理约束
                    constraints = self.compute_all_constraints(batch_notes, seq_idx)
                    
                    # 计算惩罚
                    penalty = sum(constraints.values()) / len(constraints)
                    total_penalty += penalty
        
        avg_penalty = total_penalty / (batch_size * seq_len)
        return torch.tensor(avg_penalty * weight, device=device, requires_grad=True)


def create_physical_features(notes_batch: List[List[Dict]]) -> torch.Tensor:
    """
    为批量数据创建物理约束特征
    
    Args:
        notes_batch: 批量音符数据
        
    Returns:
        物理约束特征tensor [batch_size, seq_len, constraint_features]
    """
    constraint_calculator = PhysicalConstraints()
    batch_features = []
    
    for notes in notes_batch:
        sequence_features = constraint_calculator.compute_sequence_constraints(notes)
        batch_features.append(sequence_features)
    
    # 确保所有序列长度一致
    max_len = max(len(seq) for seq in batch_features)
    
    padded_features = []
    for features in batch_features:
        if len(features) < max_len:
            # 填充到最大长度
            padding = torch.zeros(max_len - len(features), features.shape[1])
            features = torch.cat([features, padding], dim=0)
        padded_features.append(features)
    
    return torch.stack(padded_features) 