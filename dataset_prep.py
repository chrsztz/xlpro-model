import os
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from data_utils import (
    get_midi_number,
    save_pickle,
    normalize_spelled_pitch,
    ENHARMONIC_MAPPING,
    REVERSE_ENHARMONIC_MAPPING,
    CustomFingeringEncoder
)

def parse_fingering_file(file_path):
    """解析单个指法文件，返回结构化数据"""
    piece_id = os.path.splitext(os.path.basename(file_path))[0]
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            note_id = parts[0]
            onset_time = float(parts[1])
            offset_time = float(parts[2])
            spelled_pitch = parts[3]
            onset_velocity = float(parts[4])
            offset_velocity = float(parts[5])
            channel = int(parts[6])
            finger_number = parts[7]

            # 处理指法中的复杂情况（如 "1_2"）
            if '_' in finger_number:
                finger_number = finger_number.split('_')[0]
            try:
                finger_number = int(finger_number)
            except ValueError:
                finger_number = 0

            # 提取音高和八度
            pitch_name = ''.join([c for c in spelled_pitch if c.isalpha() or c in ['#', 'b']])
            octave = ''.join([c for c in spelled_pitch if c.isdigit()])
            octave = int(octave) if octave else 4

            # 判断左右手
            hand = 'right' if channel == 0 else 'left'
            duration = round(offset_time - onset_time, 2)
            normalized_spelled_pitch = normalize_spelled_pitch(spelled_pitch)

            data.append({
                'piece_id': piece_id,
                'note_id': note_id,
                'onset_time': onset_time,
                'offset_time': offset_time,
                'spelled_pitch': spelled_pitch,
                'normalized_spelled_pitch': normalized_spelled_pitch,
                'pitch_name': pitch_name,
                'octave': octave,
                'duration': duration,
                'hand': hand,
                'finger_number': finger_number
            })
    return data

def load_pig_dataset(fingering_dir):
    """多线程加载所有指法文件"""
    files = [os.path.join(fingering_dir, f) for f in os.listdir(fingering_dir) if f.endswith('.txt')]
    with Pool(processes=4) as pool:
        all_data = pool.map(parse_fingering_file, files)
    all_data = [item for sublist in all_data for item in sublist]
    return pd.DataFrame(all_data)

def main():
    # 文件路径
    fingering_folder = 'ThumbSet/FingeringFiles'
    df = load_pig_dataset(fingering_folder)

    # 数据清洗
    df['finger_number'] = df['finger_number'].fillna(0).astype(int)

    # 浮点数精度控制，减少存储空间
    float_columns = ['onset_time', 'offset_time', 'duration']
    df[float_columns] = df[float_columns].round(2)

    # 计算MIDI音高
    df['midi_number'] = df['normalized_spelled_pitch'].apply(get_midi_number)

    # 特征编码
    le_pitch = LabelEncoder()
    le_duration = LabelEncoder()
    le_hand = LabelEncoder()
    fingering_encoder = CustomFingeringEncoder()

    df['pitch_encoded'] = le_pitch.fit_transform(df['normalized_spelled_pitch'])
    df['duration_encoded'] = le_duration.fit_transform(df['duration'].astype(str))
    df['hand_encoded'] = le_hand.fit_transform(df['hand'])
    fingering_encoder.fit(df['finger_number'])
    df['fingering_encoded'] = fingering_encoder.transform(df['finger_number'])

    # 选择关键特征，减少冗余
    X = df[['pitch_encoded', 'duration_encoded', 'hand_encoded', 'midi_number']].values
    y = df['fingering_encoded'].values

    # 划分训练集和验证集，并记录索引
    train_idx, val_idx = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y
    )

    # 在 df 中添加 'train' 列，1 表示训练集，0 表示验证集
    df['train'] = 0
    df.loc[train_idx, 'train'] = 1

    # 根据索引提取训练集和验证集
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    # 生成序列
    sequence_length = 10

    def create_sequences(X, y, seq_length):
        X_seq = []
        y_seq = []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)

    # 保存数据
    np.save('X_train.npy', X_train_seq)
    np.save('X_val.npy', X_val_seq)
    np.save('y_train.npy', y_train_seq)
    np.save('y_val.npy', y_val_seq)

    # 保存编码器和数据
    save_pickle(le_pitch, 'le_pitch.pkl')
    save_pickle(le_duration, 'le_duration.pkl')
    save_pickle(le_hand, 'le_hand.pkl')
    save_pickle(fingering_encoder, 'le_fingering.pkl')
    save_pickle(df, 'df.pkl')  # 现在 df 包含 'train' 列
    save_pickle(ENHARMONIC_MAPPING, 'enharmonic_mapping.pkl')
    save_pickle(REVERSE_ENHARMONIC_MAPPING, 'reverse_enharmonic_mapping.pkl')

if __name__ == "__main__":
    main()