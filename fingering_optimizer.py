#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
import pandas as pd

class FingeringOptimizer:
    """
    钢琴指法优化器
    用于后处理模型预测的指法，根据钢琴指法规则优化连续指法的合理性
    """
    
    def __init__(self):
        # 基本钢琴指法规则
        # 1. 避免同一个手指连续弹奏多个不同音高的音符，特别是3个以上
        # 2. 拇指和小指转指时需要特别注意
        # 3. 相邻半音应使用相邻手指
        # 4. 相邻音符间音程较大时，指法应有较大变化
        # 5. 考虑音符的音高和持续时间
        
        # 定义指法转换成本矩阵 (右手)
        # 行: 当前指法, 列: 下一个指法
        # 值越大表示转换代价越高
        self.right_transition_cost = np.array([
            # 1   2   3   4   5 (下一个指法)
            [0.0, 0.1, 0.3, 0.5, 0.8],  # 1 (当前指法)
            [0.1, 0.0, 0.1, 0.3, 0.5],  # 2
            [0.3, 0.1, 0.0, 0.1, 0.3],  # 3
            [0.5, 0.3, 0.1, 0.0, 0.1],  # 4
            [0.8, 0.5, 0.3, 0.1, 0.0],  # 5
        ])
        
        # 左手指法成本矩阵（负数指法对应）
        self.left_transition_cost = np.array([
            # -5  -4  -3  -2  -1 (下一个指法)
            [0.0, 0.1, 0.3, 0.5, 0.8],  # -5 (当前指法)
            [0.1, 0.0, 0.1, 0.3, 0.5],  # -4
            [0.3, 0.1, 0.0, 0.1, 0.3],  # -3
            [0.5, 0.3, 0.1, 0.0, 0.1],  # -2
            [0.8, 0.5, 0.3, 0.1, 0.0],  # -1
        ])
        
        # 音程与指法差异之间的期望关系
        # 一般来说，音程越大，指法差异也应越大
        self.pitch_interval_to_fingering_diff = {
            # 音程: (最小指法差异, 最优指法差异)
            0: (0, 0),     # 相同音高可用相同指法
            1: (0, 1),     # 半音最好用相邻指法
            2: (1, 1),     # 全音最好用相邻指法
            3: (1, 2),     # 小三度最好用间隔1个的指法
            4: (1, 2),     # 大三度最好用间隔1个的指法
            5: (2, 2),     # 纯四度最好用间隔2个的指法
            6: (2, 3),     # 增四度最好用间隔2-3个的指法
            7: (2, 3),     # 纯五度最好用间隔2-3个的指法
            8: (3, 4),     # 小六度最好用间隔3-4个的指法
            9: (3, 4),     # 大六度最好用间隔3-4个的指法
            10: (3, 4),    # 小七度最好用间隔3-4个的指法
            11: (3, 4),    # 大七度最好用间隔3-4个的指法
            12: (4, 4),    # 八度最好用大间隔的指法
        }
        
        # 连续使用同一指法的最大次数
        self.max_consecutive_count = 2
        
        # 和弦内不同音符的指法偏好
        # 通常低音用低指法数字，高音用高指法数字
        self.chord_fingering_preference = {
            'right': {  # 右手
                'low': [1, 2],    # 低音偏好拇指、食指
                'middle': [2, 3], # 中音偏好食指、中指
                'high': [4, 5]    # 高音偏好无名指、小指
            },
            'left': {   # 左手
                'low': [-5, -4],  # 低音偏好小指、无名指
                'middle': [-3, -2], # 中音偏好中指、食指
                'high': [-1]      # 高音偏好拇指
            }
        }

    def _get_finger_idx(self, finger):
        """将指法映射到数组索引"""
        if finger > 0:  # 右手
            return finger - 1
        else:  # 左手
            return abs(finger) - 1
    
    def _get_transition_cost(self, current_finger, next_finger):
        """计算指法转换的成本"""
        if current_finger > 0 and next_finger > 0:  # 右手
            return self.right_transition_cost[self._get_finger_idx(current_finger)][self._get_finger_idx(next_finger)]
        elif current_finger < 0 and next_finger < 0:  # 左手
            return self.left_transition_cost[self._get_finger_idx(current_finger)][self._get_finger_idx(next_finger)]
        else:
            return 0.0  # 不同手之间没有转换成本
    
    def _get_pitch_interval(self, current_pitch, next_pitch):
        """计算音高间隔（半音数）"""
        return abs(next_pitch - current_pitch)
    
    def _get_ideal_finger_diff(self, pitch_interval):
        """根据音高间隔获取理想的指法差异"""
        # 对于超过八度的间隔，使用八度的规则
        interval = min(pitch_interval, 12)
        _, optimal_diff = self.pitch_interval_to_fingering_diff.get(interval, (1, 2))
        return optimal_diff
    
    def _get_consecutive_cost(self, finger, count):
        """计算连续使用同一指法的成本"""
        if count <= self.max_consecutive_count:
            return 0.0
        else:
            # 连续次数越多，成本越高
            return 0.5 * (count - self.max_consecutive_count)
    
    def _get_finger_diff_cost(self, finger_diff, ideal_diff):
        """计算指法差异与理想差异之间的成本"""
        return 0.3 * abs(finger_diff - ideal_diff)
    
    def optimize_fingering(self, fingerings, midi_numbers, hand_info='right'):
        """
        优化指法序列
        
        参数:
        - fingerings: 指法序列，如 [1, 5, 5, 5, 4, 2, 1]
        - midi_numbers: 对应的MIDI音高，如 [60, 62, 64, 65, 67, 69, 71]
        - hand_info: 手部信息，'right' 或 'left'
        
        返回:
        - 优化后的指法序列
        """
        # 如果只有一个或零个音符，无需优化
        if len(fingerings) <= 1:
            return fingerings
        
        # 复制输入数据以避免修改原始数据
        fingerings = fingerings.copy()
        
        # 修正不合理的连续相同指法
        modified = True
        while modified:
            modified = False
            
            # 1. 找出所有连续相同指法的段落
            segments = []
            current_finger = fingerings[0]
            current_start = 0
            current_count = 1
            
            for i in range(1, len(fingerings)):
                if fingerings[i] == current_finger:
                    current_count += 1
                else:
                    if current_count > self.max_consecutive_count:
                        segments.append((current_start, current_start + current_count - 1, current_finger))
                    current_finger = fingerings[i]
                    current_start = i
                    current_count = 1
            
            # 处理最后一段
            if current_count > self.max_consecutive_count:
                segments.append((current_start, current_start + current_count - 1, current_finger))
            
            # 2. 优化每个连续段
            for start, end, finger in segments:
                # 处理长度超过阈值的连续相同指法
                if end - start + 1 > self.max_consecutive_count:
                    # 分析这段音符的音高变化
                    pitch_changes = []
                    for i in range(start, end):
                        pitch_changes.append(abs(midi_numbers[i+1] - midi_numbers[i]))
                    
                    # 确定要更改的位置
                    positions_to_change = []
                    
                    # 对于长连续段，我们在音高变化较大的位置更改指法
                    if len(pitch_changes) >= 3:
                        # 找出音高变化最大的几个位置
                        sorted_positions = sorted(range(len(pitch_changes)), 
                                                 key=lambda i: pitch_changes[i], 
                                                 reverse=True)
                        
                        # 选择音高变化最大的位置进行指法更改
                        num_changes = (end - start) // 2
                        positions_to_change = [start + pos + 1 for pos in sorted_positions[:num_changes]]
                    else:
                        # 对于较短的连续段，我们简单地更改中间位置
                        mid = (start + end) // 2
                        positions_to_change = [mid]
                    
                    # 更改选定位置的指法
                    for pos in positions_to_change:
                        if pos > 0 and pos < len(fingerings) - 1:
                            prev_midi = midi_numbers[pos-1]
                            curr_midi = midi_numbers[pos]
                            next_midi = midi_numbers[pos+1]
                            
                            # 计算与前后音符的音高差异
                            prev_interval = abs(curr_midi - prev_midi)
                            next_interval = abs(next_midi - curr_midi)
                            
                            # 根据音高差异确定适当的指法
                            if finger in [1, 5, -1, -5]:  # 对于拇指和小指，需要特殊处理
                                # 对于右手
                                if finger > 0:
                                    if finger == 1:
                                        fingerings[pos] = 2  # 拇指通常转到食指
                                    else:  # finger == 5
                                        fingerings[pos] = 4  # 小指通常转到无名指
                                # 对于左手
                                else:
                                    if finger == -1:
                                        fingerings[pos] = -2  # 拇指通常转到食指
                                    else:  # finger == -5
                                        fingerings[pos] = -4  # 小指通常转到无名指
                            else:
                                # 根据音程大小选择相邻或间隔指法
                                interval = max(prev_interval, next_interval)
                                if interval <= 2:  # 小间隔，使用相邻指法
                                    if finger > 0:  # 右手
                                        fingerings[pos] = max(1, min(5, finger + (-1 if finger > 3 else 1)))
                                    else:  # 左手
                                        fingerings[pos] = max(-5, min(-1, finger + (1 if finger > -3 else -1)))
                                else:  # 大间隔，使用间隔指法
                                    if finger > 0:  # 右手
                                        fingerings[pos] = max(1, min(5, finger + (-2 if finger > 3 else 2)))
                                    else:  # 左手
                                        fingerings[pos] = max(-5, min(-1, finger + (2 if finger > -3 else -2)))
                    
                    modified = True
            
            # 如果没有修改，退出循环
            if not modified:
                break
        
        return fingerings

    def optimize_fingerings_with_info(self, df):
        """
        优化DataFrame中的指法，使用更多的上下文信息
        
        参数:
        - df: 包含指法和MIDI音符信息的DataFrame
              必须包含列: 'finger_number', 'midi_number', 'hand'
        
        返回:
        - 包含优化后指法的DataFrame
        """
        # 复制输入DataFrame以避免修改原始数据
        df = df.copy()
        
        # 按手部分组处理
        for hand in df['hand'].unique():
            hand_df = df[df['hand'] == hand]
            indices = hand_df.index
            
            # 获取该手部的指法和MIDI音高
            fingerings = hand_df['finger_number'].values
            midi_numbers = hand_df['midi_number'].values
            
            # 优化指法
            optimized_fingerings = self.optimize_fingering(
                fingerings, 
                midi_numbers, 
                hand_info=hand
            )
            
            # 更新DataFrame
            for i, idx in enumerate(indices):
                df.at[idx, 'finger_number'] = optimized_fingerings[i]
        
        return df

    def postprocess_predicted_fingerings(self, predicted_fingerings, df):
        """
        后处理预测的指法，修正不合理的连续指法
        
        参数:
        - predicted_fingerings: 模型预测的指法序列
        - df: 包含音符信息的DataFrame，必须包含'midi_number'和'hand'列
        
        返回:
        - 优化后的指法序列
        """
        # 创建包含指法信息的临时DataFrame
        temp_df = df.copy()
        temp_df['finger_number'] = predicted_fingerings
        
        # 优化指法
        optimized_df = self.optimize_fingerings_with_info(temp_df)
        
        # 返回优化后的指法序列
        return optimized_df['finger_number'].tolist()


def main():
    # 测试代码
    optimizer = FingeringOptimizer()
    
    # 测试用例1: 连续使用同一指法的情况
    fingerings1 = [1, 5, 5, 5, 5, 5, 1]
    midi_numbers1 = [60, 62, 64, 65, 67, 69, 71]
    print("原始指法:", fingerings1)
    print("优化指法:", optimizer.optimize_fingering(fingerings1, midi_numbers1))
    
    # 测试用例2: 左手指法
    fingerings2 = [-5, -5, -5, -5, -1]
    midi_numbers2 = [48, 50, 52, 53, 55]
    print("\n原始指法:", fingerings2)
    print("优化指法:", optimizer.optimize_fingering(fingerings2, midi_numbers2, hand_info='left'))
    
    # 测试DataFrame优化
    print("\n测试DataFrame优化:")
    data = {
        'finger_number': [1, 5, 5, 5, 5, 5, 1, -5, -5, -5, -5, -1],
        'midi_number': [60, 62, 64, 65, 67, 69, 71, 48, 50, 52, 53, 55],
        'hand': ['right', 'right', 'right', 'right', 'right', 'right', 'right', 
                 'left', 'left', 'left', 'left', 'left']
    }
    df = pd.DataFrame(data)
    optimized_df = optimizer.optimize_fingerings_with_info(df)
    print("原始DataFrame:\n", df[['finger_number', 'midi_number', 'hand']])
    print("优化后DataFrame:\n", optimized_df[['finger_number', 'midi_number', 'hand']])


if __name__ == "__main__":
    main() 