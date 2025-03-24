# dataset_prep.py

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from data_utils import (
    get_midi_number,
    save_pickle,
    normalize_spelled_pitch,        # 导入标准化函数
    ENHARMONIC_MAPPING,             # 导入映射字典
    REVERSE_ENHARMONIC_MAPPING,     # 导入反向映射字典
    CustomFingeringEncoder
)


def mark_chords(df):
    """
    标记 'chord' 列为 1 如果同一曲目 (piece_id) 同一手部 (hand) 内有多个音符的时间区间重叠。
    否则，标记为 0。
    """
    df['chord'] = 0  # 初始化为 0

    # 按 'piece_id' 和 'hand' 分组
    grouped = df.groupby(['piece_id', 'hand'])

    def mark_chords_group(group):
        """
        对每个分组（同一曲目同一手部）标记和弦。
        """
        group = group.sort_values('onset_time').reset_index(drop=True)
        n = len(group)
        chord_indices = set()

        for i in range(n):
            current_onset = group.loc[i, 'onset_time']
            current_offset = group.loc[i, 'offset_time']

            for j in range(i + 1, n):
                next_onset = group.loc[j, 'onset_time']
                next_offset = group.loc[j, 'offset_time']

                if next_onset < current_offset:  # 重叠
                    chord_indices.add(i)
                    chord_indices.add(j)
                else:
                    break  # 因为已排序，后续不会再重叠

        # 将重叠的音符标记为和弦
        if chord_indices:
            group.loc[list(chord_indices), 'chord'] = 1

        return group

    # 应用到每个分组
    df = grouped.apply(mark_chords_group).reset_index(drop=True)

    return df

def parse_fingering_file(file_path):
    """
    解析单个 fingering 文件，返回包含所有音符信息的列表。
    """
    piece_id = os.path.splitext(os.path.basename(file_path))[0]  # 使用文件名（不含扩展名）作为 piece_id
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # 去除首尾空白字符并按空格分割
            parts = line.strip().split()
            if len(parts) < 8:
                continue  # 跳过格式不完整的行
            note_id = parts[0]
            onset_time = float(parts[1])
            offset_time = float(parts[2])
            spelled_pitch = parts[3]
            onset_velocity = float(parts[4])
            offset_velocity = float(parts[5])
            channel = int(parts[6])
            finger_number = parts[7]

            # 处理指法替换（例如 '3_1'）
            if '_' in finger_number:
                finger_number = finger_number.split('_')[0]  # 仅取主要指法

            # 转换指法为整数
            try:
                finger_number = int(finger_number)
            except ValueError:
                finger_number = None  # 处理无法转换的指法

            # 解析音高和八度
            pitch_name = ''.join([c for c in spelled_pitch if c.isalpha() or c in ['#', 'b']])
            octave = ''.join([c for c in spelled_pitch if c.isdigit()])
            octave = int(octave) if octave else 4  # 默认八度为4

            # 确定手部
            hand = 'right' if channel == 0 else 'left'

            # 计算音符持续时间，并统一小数位数
            duration = round(offset_time - onset_time, 2)

            # 标准化 spelled_pitch
            normalized_spelled_pitch = normalize_spelled_pitch(spelled_pitch)

            # 统一小数位数的处理
            onset_time = round(onset_time, 3)
            offset_time = round(offset_time, 3)

            data.append({
                'piece_id': piece_id,  # 添加 piece_id
                'note_id': note_id,
                'onset_time': onset_time,
                'offset_time': offset_time,
                'spelled_pitch': spelled_pitch,
                'normalized_spelled_pitch': normalized_spelled_pitch,  # 添加标准化后的音符
                'pitch_name': pitch_name,
                'octave': octave,
                'duration': duration,
                'hand': hand,
                'finger_number': finger_number
            })
    return data

def load_pig_dataset(fingering_dir):
    """
    加载 PIG Dataset 中所有 fingering 文件，返回一个包含所有音符数据的 DataFrame。
    """
    all_data = []
    for file_name in os.listdir(fingering_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(fingering_dir, file_name)
            piece_data = parse_fingering_file(file_path)
            all_data.extend(piece_data)
    df = pd.DataFrame(all_data)
    return df

def plot_duration_distribution(df):
    """
    绘制音符持续时间的分布图。
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['duration'], bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Note Durations')
    plt.xlabel('Duration (s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('duration_distribution.png')
    plt.show()

def main():
    # 设置 fingering 文件夹路径
    fingering_folder = 'PIGdata/FingeringFiles'  # 请根据实际路径调整

    # 加载数据
    df = load_pig_dataset(fingering_folder)

    # 查看数据
    print("初始数据样例：")
    print(df.head())

    # 删除指法缺失的音符
    df = df.dropna(subset=['finger_number'])

    # 确保指法为整数类型
    df['finger_number'] = df['finger_number'].astype(int)

    # 统一小数位数
    float_columns = ['onset_time', 'offset_time', 'duration']
    df[float_columns] = df[float_columns].round(2)

    # 计算 MIDI 编号，使用标准化后的音符
    df['midi_number'] = df['normalized_spelled_pitch'].apply(get_midi_number)

    # 初始化 LabelEncoder
    le_pitch = LabelEncoder()
    le_duration = LabelEncoder()
    le_hand = LabelEncoder()

    # 对类别特征进行标签编码，使用标准化后的音符
    df['pitch_encoded'] = le_pitch.fit_transform(df['normalized_spelled_pitch'])
    df['duration_encoded'] = le_duration.fit_transform(df['duration'].astype(str))
    df['hand_encoded'] = le_hand.fit_transform(df['hand'])

    fingering_encoder = CustomFingeringEncoder()
    print("Original finger numbers:", np.unique(df['finger_number']))

    # 确保所有指法都是合法的
    fingering_encoder.fit(df['finger_number'])
    df['fingering_encoded'] = fingering_encoder.transform(df['finger_number'])

    print("Encoded finger numbers:", np.unique(df['fingering_encoded']))
    print("\nFingering mapping:")
    for orig, encoded in fingering_encoder.mapping.items():
        print(f"{orig} -> {encoded}")

    df = mark_chords(df)

    # 特征和标签
    X = df[['pitch_encoded', 'duration_encoded', 'hand_encoded']].values
    y = df['fingering_encoded'].values

    print(f"特征形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    # 保存 LabelEncoder 和映射
    save_pickle(le_pitch, 'le_pitch.pkl')
    save_pickle(le_duration, 'le_duration.pkl')
    save_pickle(le_hand, 'le_hand.pkl')
    save_pickle(fingering_encoder, 'le_fingering.pkl')
    save_pickle(df, "df.pkl")
    # 保存标准化映射
    save_pickle(ENHARMONIC_MAPPING, 'enharmonic_mapping.pkl')
    save_pickle(REVERSE_ENHARMONIC_MAPPING, 'reverse_enharmonic_mapping.pkl')

    # 绘制 duration 分布图
    plot_duration_distribution(df)

    # 创建序列
    sequence_length = 10  # 使用前10个音符预测第11个音符

    def create_sequences(X, y, seq_length):
        X_seq = []
        y_seq = []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(X, y, sequence_length)

    print(f"序列特征形状: {X_seq.shape}")  # (样本数, sequence_length, 特征数量)
    print(f"序列标签形状: {y_seq.shape}")  # (样本数,)

    # 查看不同时长的分布
    duration_counts = df['duration'].value_counts().sort_index()
    print("\n不同时长的分布：")
    print(duration_counts)

    # 将序列数据保存为 npy 文件
    np.save('X_train.npy', X_seq)
    np.save('X_val.npy', X_seq)  # 这里需要根据实际划分修改
    np.save('y_train.npy', y_seq)
    np.save('y_val.npy', y_seq)  # 这里需要根据实际划分修改

    print("序列数据已保存。")

if __name__ == "__main__":
    main()