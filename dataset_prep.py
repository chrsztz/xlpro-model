import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def parse_fingering_file(file_path):
    """
    解析单个 fingering 文件，返回包含所有音符信息的列表。
    """
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

            # 计算音符持续时间
            duration = offset_time - onset_time

            data.append({
                'note_id': note_id,
                'onset_time': onset_time,
                'offset_time': offset_time,
                'spelled_pitch': spelled_pitch,
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

def get_midi_number(spelled_pitch):
    """
    将拼写音高（如 C4, D#5）转换为 MIDI 编号。
    A4 ≈ 440Hz 对应 MIDI 69。
    """
    import re
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

# 设置 fingering 文件夹路径
fingering_folder = 'PIGdata/FingeringFiles'  # 请根据实际路径调整

# 加载数据
df = load_pig_dataset(fingering_folder)

# 查看数据
print(df.head())
# 删除指法缺失的音符
df = df.dropna(subset=['finger_number'])

# 确保指法为整数类型
df['finger_number'] = df['finger_number'].astype(int)

# 计算 MIDI 编号
df['midi_number'] = df['spelled_pitch'].apply(get_midi_number)

# 初始化 LabelEncoder
le_pitch = LabelEncoder()
le_duration = LabelEncoder()
le_hand = LabelEncoder()
le_fingering = LabelEncoder()

# 对类别特征进行标签编码
df['pitch_encoded'] = le_pitch.fit_transform(df['spelled_pitch'])
df['duration_encoded'] = le_duration.fit_transform(df['duration'].astype(str))
df['hand_encoded'] = le_hand.fit_transform(df['hand'])

# 对目标标签进行标签编码
df['fingering_encoded'] = le_fingering.fit_transform(df['finger_number'])

# 特征和标签
X = df[['pitch_encoded', 'duration_encoded', 'hand_encoded']].values
y = df['fingering_encoded'].values

print(f"特征形状: {X.shape}")
print(f"标签形状: {y.shape}")
# 保存 LabelEncoder
with open('le_pitch.pkl', 'wb') as f:
    pickle.dump(le_pitch, f)

with open('le_duration.pkl', 'wb') as f:
    pickle.dump(le_duration, f)

with open('le_hand.pkl', 'wb') as f:
    pickle.dump(le_hand, f)

with open('le_fingering.pkl', 'wb') as f:
    pickle.dump(le_fingering, f)

sequence_length = 10  # 使用前10个音符预测第11个音符

def create_sequences(X, y, seq_length):
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, sequence_length)

print(f"序列特征形状: {X_seq.shape}")  # (样本数, sequence_length, 特征数量)
print(f"序列标签形状: {y_seq.shape}")  # (样本数,)
# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"验证集样本数: {X_val.shape[0]}")
# 将序列数据保存为 npy 文件
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('df.npy', df)