#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
import pandas as pd
from data_utils import white_key_distance, is_black_key

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

        # 定义最大合理的手指跨度（白键数量）
        self.max_stretch = {
            'thumb_index': 5,    # 拇指到食指
            'thumb_middle': 6,   # 拇指到中指
            'thumb_ring': 7,     # 拇指到无名指
            'thumb_pinky': 8,    # 拇指到小指
            'index_middle': 4,   # 食指到中指
            'index_ring': 5,     # 食指到无名指
            'index_pinky': 6,    # 食指到小指
            'middle_ring': 3,    # 中指到无名指
            'middle_pinky': 5,   # 中指到小指
            'ring_pinky': 3,     # 无名指到小指
        }
        
        # 定义不合理的连续指法
        self.bad_finger_sequences = [
            [1, 1], [2, 2], [3, 3], [4, 4], [5, 5],  # 同一个手指连续使用
            [-1, -1], [-2, -2], [-3, -3], [-4, -4], [-5, -5],  # 左手同一个手指连续使用
            [3, 4, 3], [4, 3, 4],  # 3-4-3/4-3-4交替
            [-3, -4, -3], [-4, -3, -4],  # 左手3-4-3/4-3-4交替
            [2, 3, 2], [3, 2, 3],  # 2-3-2/3-2-3交替
            [-2, -3, -2], [-3, -2, -3],  # 左手2-3-2/3-2-3交替
        ]

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

    def postprocess_predicted_fingerings(self, fingerings, df):
        """
        后处理预测的指法，修正不合理的连续指法
        
        参数:
        - predicted_fingerings: 模型预测的指法序列
        - df: 包含音符信息的DataFrame，必须包含'midi_number'和'hand'列
        
        返回:
        - 优化后的指法序列
        """
        # 将指法和音符MIDI编号转换为numpy数组
        fingerings_array = np.array(fingerings)
        midi_numbers = df['midi_number'].values
        
        # 1. 修复不合理的连续手指使用
        optimized = self._optimize_finger_sequence(fingerings_array, midi_numbers, window=2)
        optimized = self._optimize_finger_sequence(optimized, midi_numbers, window=3)
        
        # 2. 平滑指法过渡
        optimized = self._smooth_finger_transitions(optimized, midi_numbers)
        
        # 3. 优化黑键指法
        optimized = self._handle_black_keys(optimized, midi_numbers)
        
        # 4. 检查伸展违反
        for i in range(1, len(optimized)):
            if fingerings_array[i] * fingerings_array[i-1] > 0:  # 同一只手
                if self._stretch_violation(
                    optimized[i-1], optimized[i], 
                    midi_numbers[i-1], midi_numbers[i]
                ):
                    # 检测到伸展违反，尝试找到更好的指法
                    # 简单策略：考虑拇指交叉或改变一个指法
                    if midi_numbers[i] > midi_numbers[i-1] and optimized[i-1] >= 3:
                        # 向上移动，考虑使用拇指
                        optimized[i] = 1 if optimized[i] > 0 else -1
                    elif midi_numbers[i] < midi_numbers[i-1] and optimized[i-1] <= 3:
                        # 向下移动，考虑使用小指
                        optimized[i] = 5 if optimized[i] > 0 else -5
        
        # 将numpy数组转换回list
        return optimized.tolist()

    def _stretch_violation(self, finger1, finger2, note1_midi, note2_midi):
        """检查两个手指之间的伸展是否违反人体工程学限制"""
        # 转换为右手指法并获取手指索引
        f1, f2 = abs(finger1), abs(finger2)
        
        # 确保手指按升序排列
        if f1 > f2:
            f1, f2 = f2, f1
            note1_midi, note2_midi = note2_midi, note1_midi
        
        # 获取手指之间的键盘距离
        distance = white_key_distance(note1_midi, note2_midi)
        
        # 获取最大允许的伸展
        stretch_key = None
        if f1 == 1 and f2 == 2:
            stretch_key = 'thumb_index'
        elif f1 == 1 and f2 == 3:
            stretch_key = 'thumb_middle'
        elif f1 == 1 and f2 == 4:
            stretch_key = 'thumb_ring'
        elif f1 == 1 and f2 == 5:
            stretch_key = 'thumb_pinky'
        elif f1 == 2 and f2 == 3:
            stretch_key = 'index_middle'
        elif f1 == 2 and f2 == 4:
            stretch_key = 'index_ring'
        elif f1 == 2 and f2 == 5:
            stretch_key = 'index_pinky'
        elif f1 == 3 and f2 == 4:
            stretch_key = 'middle_ring'
        elif f1 == 3 and f2 == 5:
            stretch_key = 'middle_pinky'
        elif f1 == 4 and f2 == 5:
            stretch_key = 'ring_pinky'
        
        if stretch_key and distance > self.max_stretch[stretch_key]:
            return True  # 违反伸展限制
        
        return False
    
    def _find_thumb_crossing(self, fingerings, midi_numbers):
        """识别拇指交叉点"""
        crossings = []
        
        for i in range(1, len(fingerings)):
            # 右手拇指交叉
            if (fingerings[i-1] in [2, 3, 4, 5] and fingerings[i] == 1 and 
                midi_numbers[i] > midi_numbers[i-1]):
                crossings.append(i)
            # 左手拇指交叉（反向）
            elif (fingerings[i-1] in [-2, -3, -4, -5] and fingerings[i] == -1 and 
                  midi_numbers[i] < midi_numbers[i-1]):
                crossings.append(i)
        
        return crossings
    
    def _optimize_finger_sequence(self, fingerings, midi_numbers, window=3):
        """优化指法序列"""
        optimized = fingerings.copy()
        
        # 逐位置检查
        for i in range(len(fingerings) - window + 1):
            seq = fingerings[i:i+window]
            
            # 检查是否为不良序列
            for bad_seq in self.bad_finger_sequences:
                if len(bad_seq) == window and all(abs(s1) == abs(s2) for s1, s2 in zip(seq, bad_seq)):
                    # 获取这个窗口的音符
                    notes_midi = midi_numbers[i:i+window]
                    
                    # 如果是连续同指
                    if window == 2 and abs(seq[0]) == abs(seq[1]):
                        # 检查音高差异
                        pitch_diff = abs(notes_midi[1] - notes_midi[0])
                        # 如果音高差异大，改变第二个指法
                        if pitch_diff > 3:
                            # 向上走，增加指法数字；向下走，减少指法数字
                            if notes_midi[1] > notes_midi[0]:
                                new_finger = seq[0] + 1 if seq[0] > 0 else seq[0] - 1
                                # 确保新指法在1-5/-1--5范围内
                                if 1 <= abs(new_finger) <= 5:
                                    optimized[i+1] = new_finger
                            else:
                                new_finger = seq[0] - 1 if seq[0] > 0 else seq[0] + 1
                                # 确保新指法在1-5/-1--5范围内
                                if 1 <= abs(new_finger) <= 5:
                                    optimized[i+1] = new_finger
                    
                    # 如果是3-4-3或4-3-4模式
                    elif window == 3 and ((abs(seq[0]) == 3 and abs(seq[1]) == 4 and abs(seq[2]) == 3) or
                                          (abs(seq[0]) == 4 and abs(seq[1]) == 3 and abs(seq[2]) == 4)):
                        # 检查音高模式
                        if (notes_midi[0] < notes_midi[1] and notes_midi[1] > notes_midi[2]) or \
                           (notes_midi[0] > notes_midi[1] and notes_midi[1] < notes_midi[2]):
                            # 音高先升后降或先降后升，考虑改变中间指法
                            if seq[1] > 0:  # 右手
                                optimized[i+1] = 2 if abs(seq[1]) == 3 else 3
                            else:  # 左手
                                optimized[i+1] = -2 if abs(seq[1]) == 3 else -3
                    
                    break
        
        return optimized
    
    def _smooth_finger_transitions(self, fingerings, midi_numbers):
        """平滑指法过渡"""
        optimized = fingerings.copy()
        
        for i in range(1, len(fingerings)-1):
            # 检查相邻指法是否存在不自然的跳跃
            # 例如，1-5-2或5-1-4这样的序列
            if (abs(fingerings[i-1]) == 1 and abs(fingerings[i]) == 5) or \
               (abs(fingerings[i-1]) == 5 and abs(fingerings[i]) == 1):
                # 检查跳跃后的指法是否与方向一致
                if i+1 < len(fingerings):
                    pitch_direction = midi_numbers[i] - midi_numbers[i-1]
                    next_pitch_direction = midi_numbers[i+1] - midi_numbers[i]
                    
                    # 如果方向一致，可能需要调整跳跃后的指法
                    if (pitch_direction > 0 and next_pitch_direction > 0) or \
                       (pitch_direction < 0 and next_pitch_direction < 0):
                        # 根据实际情况调整指法
                        if abs(fingerings[i]) == 5 and 1 <= abs(fingerings[i+1]) <= 3:
                            # 将5改为4，避免大跳跃后立即使用小指法
                            optimized[i] = 4 if fingerings[i] > 0 else -4
                        elif abs(fingerings[i]) == 1 and 3 <= abs(fingerings[i+1]) <= 5:
                            # 将1改为2，避免大跳跃后立即使用拇指
                            optimized[i] = 2 if fingerings[i] > 0 else -2
        
        return optimized
    
    def _handle_black_keys(self, fingerings, midi_numbers):
        """优化黑键指法"""
        optimized = fingerings.copy()
        
        for i in range(len(fingerings)):
            # 检查是否为黑键
            if is_black_key(midi_numbers[i]):
                # 避免在黑键上使用拇指和小指（除非必要）
                if abs(fingerings[i]) == 1:  # 拇指
                    # 查看周围上下文
                    if i > 0 and i < len(fingerings)-1:
                        # 如果拇指不是用于转向，考虑更改
                        if not ((midi_numbers[i-1] > midi_numbers[i] > midi_numbers[i+1]) or
                                (midi_numbers[i-1] < midi_numbers[i] < midi_numbers[i+1])):
                            # 改为食指或中指
                            optimized[i] = 2 if fingerings[i] > 0 else -2
                elif abs(fingerings[i]) == 5:  # 小指
                    # 小指在黑键上不灵活，考虑更改为无名指
                    optimized[i] = 4 if fingerings[i] > 0 else -4
        
        return optimized


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