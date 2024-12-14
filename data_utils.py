# data_utils.py

import re
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from gensim.models import Word2Vec


def get_midi_number(spelled_pitch):
    """
    将拼写音高（如 C4, D#5）转换为 MIDI 编号。
    A4 ≈ 440Hz 对应 MIDI 69。
    """
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


def create_word_column(df, feature_columns):
    """
    创建 'word' 列，将多个特征组合成一个字符串，用于Word2Vec训练。
    """
    df['word'] = df.apply(lambda row: '_'.join(map(str, row[feature_columns].values)), axis=1)
    return df


def train_word2vec(sentences, window=2, vector_size=128, min_count=1, workers=4):
    """
    训练 Word2Vec-CBOW 模型，并返回训练好的模型。
    """
    model = Word2Vec(sentences, window=window, vector_size=vector_size, min_count=min_count, workers=workers, sg=0)
    return model


def get_fused_features(df, word2vec_model):
    """
    使用 Word2Vec 模型将每个音符的特征向量转化为融合特征向量。
    """
    df['fused_feature'] = df['word'].apply(
        lambda x: word2vec_model.wv[x] if x in word2vec_model.wv else np.zeros(word2vec_model.vector_size))
    return df


def combine_features(df, feature_columns):
    """
    将融合特征与原始特征组合。
    """

    def combine(row):
        return np.concatenate((row[feature_columns].values, row['fused_feature_scaled']))

    df['combined_features'] = df.apply(combine, axis=1)
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
