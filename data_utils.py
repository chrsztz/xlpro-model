# data_utils.py
import re
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from gensim.models import Word2Vec

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
    """确保与训练集一致的密度计算方式"""
    # 先获取训练集的密度范围
    train_df = pd.read_pickle('df.pkl')
    max_density = train_df['note_density'].max()
    min_density = train_df['note_density'].min()

    # 计算密度并归一化到训练集范围
    def calculate_density(start_time, window):
        end_time = start_time + window
        count = df[(df['onset_time'] >= start_time) &
                   (df['onset_time'] < end_time)].shape[0]
        # 归一化到训练集范围
        normalized_count = int((count - min_density) / (max_density - min_density) * max_density)
        return normalized_count

    df['note_density'] = df['onset_time'].apply(lambda x: calculate_density(x, window))
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
    """将融合特征与原始特征组合，确保总维度为136"""

    def combine(row):
        # 基础特征
        base_features = row[feature_columns].values  # 8维
        # 融合特征应该是128维 (136-8=128)
        fused_features = row['fused_feature_scaled']
        return np.concatenate([base_features, fused_features])

    df['combined_features'] = df.apply(combine, axis=1)

    # 验证维度
    sample_dim = len(df['combined_features'].iloc[0])
    print(f"Combined feature dimension: {sample_dim}")
    assert sample_dim == 136, f"Expected 136 features, got {sample_dim}"

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
