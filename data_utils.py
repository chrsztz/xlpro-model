# data_utils.py
import re
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from gensim.models import Word2Vec

# 导入RL相关物理约束计算函数
import math
from collections import defaultdict

# Constants for hand feature matrix
# These represent the maximum stretchable distances between fingers for both hands
# Values represent white key distances
RIGHT_HAND_MATRIX = np.array([
    [0, 5, 6, 7, 8],    # Thumb (1) to fingers 1-5
    [-1, 0, 4, 5, 6],   # Index (2) to fingers 1-5
    [-1, -1, 0, 3, 5],  # Middle (3) to fingers 1-5
    [-1, -1, -1, 0, 3], # Ring (4) to fingers 1-5
    [-1, -1, -1, -1, 0] # Pinky (5) to fingers 1-5
])

LEFT_HAND_MATRIX = np.array([
    [0, 5, 6, 7, 8],    # Thumb (1) to fingers 1-5
    [-1, 0, 4, 5, 6],   # Index (2) to fingers 1-5
    [-1, -1, 0, 3, 5],  # Middle (3) to fingers 1-5
    [-1, -1, -1, 0, 3], # Ring (4) to fingers 1-5
    [-1, -1, -1, -1, 0] # Pinky (5) to fingers 1-5
])

# 升级版映射字典
ENHARMONIC_MAPPING = {
    # 双降音符
    'Cbb': 'Bb',
    'Dbb': 'C',
    'Ebb': 'D',
    'Fbb': 'Eb',
    'Gbb': 'F',
    'Abb': 'G',
    'Bbb': 'A',

    # 单降音符
    'Cb': 'B',
    'Db': 'C#',
    'Eb': 'D#',
    'Fb': 'E',
    'Gb': 'F#',
    'Ab': 'G#',
    'Bb': 'A#',

    # 纯音符
    'C': 'C',
    'D': 'D',
    'E': 'E',
    'F': 'F',
    'G': 'G',
    'A': 'A',
    'B': 'B',

    # 单升音符
    'C#': 'C#',
    'D#': 'D#',
    'E#': 'F',
    'F#': 'F#',
    'G#': 'G#',
    'A#': 'A#',
    'B#': 'C',

    # 双升音符
    'Cx': 'D',
    'Dx': 'E',
    'Ex': 'F#',
    'Fx': 'G',
    'Gx': 'A',
    'Ax': 'B',
    'Bx': 'C#',
}
TEMPO_MAPPING = {
    'Larghissimo': 24,
    'Grave': 40,
    'Largo': 40,
    'Larghetto': 50,
    'Adagio': 66,
    'Adagietto': 68,
    'Andante': 76,
    'Andantino': 80,
    'Moderato': 108,
    'Allegretto': 112,
    'Allegro': 120,
    'Vivace': 156,
    'Presto': 168,
    'Prestissimo': 200
}

# 创建反向映射，用于还原
REVERSE_ENHARMONIC_MAPPING = {}
for key, value in ENHARMONIC_MAPPING.items():
    if value not in REVERSE_ENHARMONIC_MAPPING:
        REVERSE_ENHARMONIC_MAPPING[value] = []
    REVERSE_ENHARMONIC_MAPPING[value].append(key)

class CustomFingeringEncoder:
    """自定义指法编码器，处理钢琴指法的编码和解码"""

    def __init__(self):
        self.mapping = {
            -5: 0, -4: 1, -3: 2, -2: 3, -1: 4,  # 左手
            1: 5, 2: 6, 3: 7, 4: 8, 5: 9  # 右手
        }
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}

    def fit(self, y):
        # 验证数据中的指法是否都在映射范围内
        invalid_fingers = set(y) - set(self.mapping.keys())
        if invalid_fingers:
            raise ValueError(f"Found invalid fingerings: {invalid_fingers}")
        return self

    def transform(self, y):
        return np.array([self.mapping.get(val, -1) for val in y])

    def inverse_transform(self, y):
        return np.array([self.inverse_mapping.get(val, 0) for val in y])

    def __getstate__(self):
        return {'mapping': self.mapping, 'inverse_mapping': self.inverse_mapping}

    def __setstate__(self, state):
        self.mapping = state['mapping']
        self.inverse_mapping = state['inverse_mapping']

# 物理约束相关函数
def get_hand_feature_matrix(hand='right'):
    """获取指定手的特征矩阵"""
    if hand.lower() == 'right':
        return RIGHT_HAND_MATRIX
    else:
        return LEFT_HAND_MATRIX

def get_white_keys():
    """获取钢琴所有白键的MIDI音符编号"""
    # MIDI note numbers for white keys (A0 to C8)
    return [21, 23, 24, 26, 28, 29, 31, 33, 35, 36, 38, 40, 41, 43, 45, 47, 48, 50, 52, 
            53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 
            86, 88, 89, 91, 93, 95, 96, 98, 100, 101, 103, 105, 107, 108]

def distance_from_a0(note_midi):
    """计算从A0（最左边的键）到指定音符的白键距离"""
    white_keys = get_white_keys()
    
    if note_midi in white_keys:
        return white_keys.index(note_midi)
    else:
        # 如果是黑键，取最近的白键并减去0.5
        for i, wk in enumerate(white_keys):
            if wk > note_midi:
                return i - 0.5
                
        # 如果音符高于所有白键
        return len(white_keys) - 0.5

def white_key_distance(note1_midi, note2_midi):
    """计算两个音符之间的白键距离"""
    return abs(distance_from_a0(note1_midi) - distance_from_a0(note2_midi))

def calculate_stretching_rate(finger1, finger2, note1_midi, note2_midi, hand='right'):
    """
    计算两个手指之间的伸展率
    
    Args:
        finger1 (int): 第一个手指 (1=拇指, 2=食指, ..., 5=小指)
        finger2 (int): 第二个手指
        note1_midi (int): 第一个音符的MIDI编号
        note2_midi (int): 第二个音符的MIDI编号
        hand (str): 'right' 或 'left'
        
    Returns:
        float: 伸展率 (0 到 1, 1 表示最大伸展)
    """
    # 调整手指索引（从1开始到从0开始）
    f1, f2 = finger1 - 1, finger2 - 1
    
    # 确保手指按升序排列
    if f1 > f2:
        f1, f2 = f2, f1
        note1_midi, note2_midi = note2_midi, note1_midi
    
    # 获取手特征矩阵
    hand_matrix = get_hand_feature_matrix(hand)
    
    # 计算手指之间的自然距离
    natural_distance = abs(f2 - f1)
    
    # 从矩阵计算最大伸展距离
    max_stretch_distance = hand_matrix[f1, f2]
    
    # 计算音符之间的实际距离
    actual_distance = white_key_distance(note1_midi, note2_midi)
    
    # 根据手性排序音符
    if hand.lower() == 'left':
        if note1_midi < note2_midi:
            actual_distance *= -1  # 左手反转
    else:  # 右手
        if note1_midi > note2_midi:
            actual_distance *= -1  # 右手反转
    
    # 如果是伸展
    if actual_distance > natural_distance:
        # 计算伸展率
        if max_stretch_distance > natural_distance:
            return (actual_distance - natural_distance) / (max_stretch_distance - natural_distance)
        else:
            return 1.0  # 最大伸展
    
    # 如果是收缩
    elif actual_distance < natural_distance:
        # 计算收缩率
        return (natural_distance - actual_distance) / natural_distance
    
    # 无伸展或收缩
    return 0.0

def calculate_hand_position(fingering, notes_midi, hand='right'):
    """
    根据指法和音符计算手位置
    
    Args:
        fingering (list): 手指编号列表
        notes_midi (list): MIDI音符编号列表
        hand (str): 'right' 或 'left'
        
    Returns:
        float: 手的位置
    """
    h = 1 if hand.lower() == 'right' else -1
    
    if len(fingering) == 1:
        # 单音符
        f = fingering[0]
        n = notes_midi[0]
        return distance_from_a0(n) + h * (3 - f)
    else:
        # 多音符（和弦）
        # 获取最低和最高的音符及其手指
        min_idx = notes_midi.index(min(notes_midi))
        max_idx = notes_midi.index(max(notes_midi))
        
        fl, nl = fingering[min_idx], notes_midi[min_idx]
        fh, nh = fingering[max_idx], notes_midi[max_idx]
        
        pos_l = distance_from_a0(nl) + h * (3 - fl)
        pos_h = distance_from_a0(nh) + h * (3 - fh)
        
        return (pos_l + pos_h) / 2

def calculate_hand_movement(prev_position, next_position):
    """计算手位置之间的移动距离"""
    return abs(prev_position - next_position)

def calculate_cross_fingering_distance(thumb_note_midi, cross_finger, cross_note_midi):
    """
    计算交叉指法距离
    
    Args:
        thumb_note_midi (int): 拇指的MIDI音符编号
        cross_finger (int): 交叉于拇指之上/之下的手指 (2, 3, 或 4)
        cross_note_midi (int): 交叉手指的MIDI音符编号
        
    Returns:
        float: 交叉指法距离
    """
    return white_key_distance(thumb_note_midi, cross_note_midi) + (cross_finger - 1)

def count_fingering_mismatches(prev_fingering, prev_notes_midi, next_fingering, next_notes_midi):
    """
    计算转换之间的指法不匹配数量
    """
    mismatches = 0
    
    # 创建音符到手指的映射
    prev_map = {n: f for n, f in zip(prev_notes_midi, prev_fingering)}
    next_map = {n: f for n, f in zip(next_notes_midi, next_fingering)}
    
    # 检查具有不同手指的公共音符
    for note in set(prev_notes_midi) & set(next_notes_midi):
        if prev_map[note] != next_map[note]:
            mismatches += 1
    
    return mismatches

def count_inverse_fingerings(prev_fingering, prev_notes_midi, next_fingering, next_notes_midi):
    """
    计算转换之间的逆向指法数量
    """
    inverse_count = 0
    
    # 对于右手：正常情况是较高的音符用较大的手指编号
    # 对于左手：正常情况是较高的音符用较小的手指编号
    
    # 从之前的音符创建（音符，手指）对
    prev_pairs = list(zip(prev_notes_midi, prev_fingering))
    
    # 从下一个音符创建（音符，手指）对
    next_pairs = list(zip(next_notes_midi, next_fingering))
    
    # 对所有可能的两对音符进行检查
    for (n1, f1) in prev_pairs:
        for (n2, f2) in next_pairs:
            # 检查是否为逆向指法
            if (n1 < n2 and f1 > f2) or (n1 > n2 and f1 < f2):
                inverse_count += 1
    
    return inverse_count

# 原有函数

def normalize_spelled_pitch(spelled_pitch):
    """
    将同音异名的音符标准化为统一的表示（例如，将所有降音符转换为升音符）。
    """
    # 分离音名和八度
    pitch_name = ''.join([c for c in spelled_pitch if c.isalpha() or c in ['#', 'b']])
    octave = ''.join([c for c in spelled_pitch if c.isdigit()])
    normalized_pitch = ENHARMONIC_MAPPING.get(pitch_name, pitch_name)  # 默认不变
    return f"{normalized_pitch}{octave}"

def denormalize_spelled_pitch(normalized_pitch):
    """
    根据需要将标准化后的音符还原为原始的同音异名形式。
    """
    # 简单选择映射中的第一个同音异名，如果需要特定的还原逻辑，可以进一步扩展
    pitch_name = ''.join([c for c in normalized_pitch if c.isalpha() or c in ['#', 'b']])
    octave = ''.join([c for c in normalized_pitch if c.isdigit()])
    original_pitch = REVERSE_ENHARMONIC_MAPPING.get(pitch_name, [pitch_name])[0]  # 默认选第一个
    return f"{original_pitch}{octave}"

def get_midi_number(spelled_pitch):
    """
    将标准化后的 spelled_pitch 转换为 MIDI 编号。
    """
    # 解析音名和八度
    match = re.match(r'^([A-Ga-g][#b]?)(\d+)$', spelled_pitch)
    if not match:
        return 60  # 默认C4
    pitch, octave = match.groups()
    octave = int(octave)
    note_to_semitone = {'C': 0, 'C#': 1, 'Db': 1,
                        'D': 2, 'D#': 3, 'Eb': 3,
                        'E': 4, 'Fb': 4,
                        'F': 5, 'F#': 6, 'Gb': 6,
                        'G': 7, 'G#': 8, 'Ab': 8,
                        'A': 9, 'A#': 10, 'Bb': 10,
                        'B': 11, 'Cb': 11}
    semitone = note_to_semitone.get(pitch, 0)
    midi_number = 12 * (octave + 1) + semitone
    return midi_number

def is_black_key(midi_number):
    """
    判断MIDI编号对应的音符是否为黑键。
    """
    black_keys = {1, 3, 6, 8, 10}  # C#:1, D#:3, F#:6, G#:8, A#:10
    return 1 if (midi_number % 12) in black_keys else 0


def calculate_speed_features(df, window=1.0):
    df = df.copy()

    # 计算真实时值
    df['real_duration'] = df['offset_time'] - df['onset_time']

    # 计算稠密度
    def calculate_density(df, window=1.0):
        density = []
        for idx, row in df.iterrows():
            start = row['onset_time']
            end = start + window
            count = df[(df['onset_time'] > start) & (df['onset_time'] <= end)].shape[0]
            density.append(count)
        return density

    df['note_density'] = calculate_density(df, window)
    return df

def calculate_midi_diff(df):
    """
    计算 MIDI 差值和处理后的 MIDI 差值。
    """
    df = df.copy()
    df['prev_midi_number'] = df['midi_number'].shift(1).fillna(60)  # 默认C4
    df['midi_diff'] = df['midi_number'] - df['prev_midi_number']
    # 添加和弦标识符
    if 'chord' not in df.columns:
        df['chord'] = 0  # 默认不是和弦，可以根据实际数据调整

    def process_midi_diff(row):
        if row['chord']:
            # 根据论文中的公式 (3)，假设 k=0
            return 200 * 0 - row['midi_diff']  # 需要根据实际情况调整
        else:
            if row['midi_diff'] < 100:
                return -row['midi_diff']
            else:
                return row['midi_diff']

    df['midi_diff_processed'] = df.apply(process_midi_diff, axis=1)
    return df

# 计算物理约束特征函数
def calculate_physical_constraint_features(df, window_size=3):
    """计算物理约束特征
    
    基于PIGdata格式，提取更全面的物理约束特征，专注于：
    1. 空间特征：音程距离、键盘物理距离、黑白键信息
    2. 时间特征：音符间隔时间、持续时间、重叠程度 
    3. 手部特征：交替模式、手部位置、跨度要求
    4. 指法约束：指法组合自然度、跨度限制、拇指交叉
    
    Args:
        df: 包含音符和指法数据的DataFrame
        window_size: 上下文窗口大小

    Returns:
        添加了物理约束特征的DataFrame
    """
    df = df.copy()
    
    # 解码真实指法 (-5到-1为左手，1到5为右手)
    if 'fingering_encoded' in df.columns:
        df['fingering_decoded'] = df['fingering_encoded'].apply(
            lambda x: x - 5 if x >= 5 else -(x + 1)
        )
    
    # 判断手性
    if 'hand_encoded' in df.columns:
        df['hand'] = df['hand_encoded'].apply(
            lambda x: 'right' if x == 1 else 'left'
        )
    else:
        df['hand'] = 'right'  # 默认右手
    
    feature_rows = []
    
    for i in range(len(df)):
        current_features = {}
        
        # 当前音符的基本信息
        current_midi = df.iloc[i]['midi_number']
        current_is_black = is_black_key(current_midi)
        
        # 添加当前音符的黑白键特征
        current_features['curr_black_key'] = float(current_is_black)
        
        # 如果有时间信息，提取时间特征
        time_features = {}
        if 'onset_time' in df.columns and 'offset_time' in df.columns:
            curr_onset = df.iloc[i]['onset_time']
            curr_offset = df.iloc[i]['offset_time']
            curr_duration = curr_offset - curr_onset
            
            time_features['note_duration'] = curr_duration
            
            # 检查前一个音符的时间关系
            if i > 0:
                prev_onset = df.iloc[i-1]['onset_time']
                prev_offset = df.iloc[i-1]['offset_time']
                
                # 计算音符间隔时间(IOI)
                time_features['ioi'] = curr_onset - prev_onset
                
                # 计算与前一个音符的重叠程度
                overlap = max(0, prev_offset - curr_onset) / curr_duration if curr_duration > 0 else 0
                time_features['overlap'] = overlap
        
        # 添加时间特征
        current_features.update(time_features)
        
        # 对于每个音符，计算与前后音符的物理约束特征
        for offset in range(-window_size, window_size + 1):
            if offset == 0:
                continue  # 跳过当前音符
            
            j = i + offset
            if 0 <= j < len(df):
                # 获取比较音符的信息
                comp_midi = df.iloc[j]['midi_number']
                comp_is_black = is_black_key(comp_midi)
                
                prefix = 'prev' if offset < 0 else 'next'
                abs_offset = abs(offset)
                
                # 计算音程距离（半音数）
                pitch_interval = abs(current_midi - comp_midi)
                current_features[f'{prefix}{abs_offset}_pitch_interval'] = pitch_interval
                
                # 计算键盘实际物理距离
                physical_distance = white_key_distance(current_midi, comp_midi)
                current_features[f'{prefix}{abs_offset}_physical_distance'] = physical_distance
                
                # 计算黑白键组合特征
                # 0=白到白, 1=白到黑, 2=黑到白, 3=黑到黑
                key_transition = 0
                if current_is_black and comp_is_black:
                    key_transition = 3
                elif current_is_black and not comp_is_black:
                    key_transition = 2
                elif not current_is_black and comp_is_black:
                    key_transition = 1
                current_features[f'{prefix}{abs_offset}_key_transition'] = key_transition
                
                if 'fingering_decoded' in df.columns:
                    curr_finger = df.iloc[i]['fingering_decoded']
                    comp_finger = df.iloc[j]['fingering_decoded']
                    
                    # 确保指法编号是正数 (1-5)，用于物理计算
                    curr_finger_abs = abs(curr_finger)
                    comp_finger_abs = abs(comp_finger)
                    
                    # 提取手部信息
                    curr_hand = 'right' if curr_finger > 0 else 'left'
                    comp_hand = 'right' if comp_finger > 0 else 'left'
                    
                    # 手部交替特征
                    current_features[f'{prefix}{abs_offset}_hand_switch'] = float(curr_hand != comp_hand)
                    
                    # 只有同一只手才计算指法伸展
                    if curr_finger * comp_finger > 0:  # 同一只手
                        # 计算伸展率
                        try:
                            stretch_rate = calculate_stretching_rate(
                                curr_finger_abs, comp_finger_abs, 
                                current_midi, comp_midi, 
                                curr_hand
                            )
                            current_features[f'{prefix}{abs_offset}_stretch_rate'] = stretch_rate
                        except:
                            current_features[f'{prefix}{abs_offset}_stretch_rate'] = 0.0
                        
                        # 计算自然指法顺序违反
                        # 如果手指间距小于音符间距，通常不自然
                        finger_distance = abs(curr_finger_abs - comp_finger_abs)
                        if pitch_interval > 5 and finger_distance < 2:
                            current_features[f'{prefix}{abs_offset}_natural_violation'] = 1.0
                        else:
                            current_features[f'{prefix}{abs_offset}_natural_violation'] = 0.0
                        
                        # 手指强度/大小特征
                        # 拇指(1)和小指(5)的强度比食指(2)、中指(3)和无名指(4)要弱
                        if curr_finger_abs in [1, 5] and pitch_interval > 7:
                            current_features[f'{prefix}{abs_offset}_finger_strength_violation'] = 1.0
                        else:
                            current_features[f'{prefix}{abs_offset}_finger_strength_violation'] = 0.0
                    else:
                        # 不同手，所有伸展特征为0
                        current_features[f'{prefix}{abs_offset}_stretch_rate'] = 0.0
                        current_features[f'{prefix}{abs_offset}_natural_violation'] = 0.0
                        current_features[f'{prefix}{abs_offset}_finger_strength_violation'] = 0.0
                    
                    # 计算拇指交叉特征
                    if (curr_finger_abs == 1 and comp_finger_abs in [2, 3, 4]) or \
                       (comp_finger_abs == 1 and curr_finger_abs in [2, 3, 4]):
                        thumb_note = current_midi if curr_finger_abs == 1 else comp_midi
                        cross_finger = comp_finger_abs if curr_finger_abs == 1 else curr_finger_abs
                        cross_note = comp_midi if curr_finger_abs == 1 else current_midi
                        
                        try:
                            cross_dist = calculate_cross_fingering_distance(
                                thumb_note, cross_finger, cross_note
                            )
                            current_features[f'{prefix}{abs_offset}_cross_dist'] = cross_dist
                            
                            # 拇指在黑键上交叉的难度评估
                            thumb_is_black = is_black_key(thumb_note)
                            current_features[f'{prefix}{abs_offset}_thumb_black_cross'] = float(thumb_is_black)
                        except:
                            current_features[f'{prefix}{abs_offset}_cross_dist'] = 0.0
                            current_features[f'{prefix}{abs_offset}_thumb_black_cross'] = 0.0
                    else:
                        current_features[f'{prefix}{abs_offset}_cross_dist'] = 0.0
                        current_features[f'{prefix}{abs_offset}_thumb_black_cross'] = 0.0
        
        feature_rows.append(current_features)
    
    # 将特征添加到DataFrame
    for feature_name in feature_rows[0].keys():
        df[feature_name] = [row.get(feature_name, 0.0) for row in feature_rows]
    
    return df

def create_word_column(df, feature_columns):
    """
    创建 'word' 列，将多个特征组合成一个字符串，用于Word2Vec训练。
    """
    df['word'] = df[feature_columns].astype(str).agg(' '.join, axis=1)
    return df

def train_word2vec(sentences, window=2, vector_size=128, min_count=1, workers=4):
    """
    训练 Word2Vec-CBOW 模型，并返回训练好的模型。
    """
    model = Word2Vec(sentences, window=window, vector_size=vector_size, min_count=min_count, workers=workers, sg=0)
    return model


# data_utils.py

def get_fused_features(df, word2vec_model, tokenized_sentences):
    """获取融合特征，确保维度为128（因为基础特征是8维，总共需要136维）"""

    def get_vector(tokens):
        vectors = []
        for token in tokens:
            if token in word2vec_model.wv:
                vectors.append(word2vec_model.wv[token])
            else:
                vectors.append(np.zeros(word2vec_model.vector_size))
        if vectors:
            base_vector = np.mean(vectors, axis=0)
            # 确保维度为128
            if len(base_vector) > 128:
                return base_vector[:128]
            elif len(base_vector) < 128:
                padding = np.zeros(128 - len(base_vector))
                return np.concatenate([base_vector, padding])
            return base_vector
        return np.zeros(128)

    # 使用list comprehension处理tokenized_sentences
    df['fused_feature'] = [get_vector(tokens) for tokens in tokenized_sentences]

    # 验证维度
    sample_dim = len(df['fused_feature'].iloc[0])
    print(f"Fused feature dimension before scaling: {sample_dim}")
    assert sample_dim == 128, f"Expected 128 features for fusion, got {sample_dim}"

    return df


def combine_features(df, feature_columns):
    """Combine original features with word2vec and CRF features."""
    combined_features = []
    
    for i, row in df.iterrows():
        # 获取原始特征
        orig_features = [float(row[col]) for col in feature_columns]
        
        # 获取融合特征
        fused_feature = row['fused_feature'] if 'fused_feature' in row else []
        
        # 获取CRF特征
        crf_feature = row['crf_feature'] if 'crf_feature' in row else []
        
        # 结合所有特征
        combined = np.concatenate([orig_features, fused_feature, crf_feature])
        combined_features.append(combined)
    
    df['combined_features'] = combined_features
    return df

def save_pickle(obj, filename):
    """
    保存对象为pickle文件。
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    """
    从pickle文件加载对象。
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
