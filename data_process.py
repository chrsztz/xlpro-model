import dataset_prep
import numpy as np
from gensim.models import Word2Vec
import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from dataset_prep import le_fingering


def calculate_midi_diff(df):
    df = df.copy()
    df['prev_midi_number'] = df['midi_number'].shift(1).fillna(60)  # 默认C4
    df['midi_diff'] = df['midi_number'] - df['prev_midi_number']

    # 假设有一个 'is_chord' 列标识当前音符是否为和弦的一部分
    # 如果没有，需要根据您的数据进行相应处理
    # 这里我们简单假设 'chord' 列已经存在并表示是否为和弦
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

df = np.load('df.npy')
df = calculate_midi_diff(df)


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


df = calculate_speed_features(df)
# 添加黑键标识符
def is_black_key(midi_number):
    # MIDI 音符编号对应的黑键
    black_keys = {1, 3, 6, 8, 10}  # C#:1, D#:3, F#:6, G#:8, A#:10
    return 1 if (midi_number % 12) in black_keys else 0

df['black_key'] = df['midi_number'].apply(is_black_key)

# 添加和弦标识符
# 这里假设您有一个方法或列来标识是否为和弦音符
# 例如，根据 'note_id' 来确定和弦，或者使用其他方式
# 这里简单假设 'is_chord' 列已经存在
if 'is_chord' not in df.columns:
    df['is_chord'] = 0  # 默认不是和弦，可以根据实际数据调整

df['chord'] = df['is_chord']  # 0 或 1

def train_word2vec(df, window=2, vector_size=128, min_count=1, workers=4):
    """
    训练 Word2Vec-CBOW 模型，并返回训练好的模型。
    """
    # 将每首乐谱视为一个句子，每个音符的特征向量视为一个词
    # 这里我们需要将特征向量转换为字符串，因为 Word2Vec 需要字符串输入
    # 另一种方法是使用 Embedding 层代替 Word2Vec

    # 为每个音符生成一个唯一的字符串表示
    df['word'] = df.apply(lambda row: '_'.join(map(str, row[['pitch_encoded', 'duration_encoded', 'hand_encoded',
                                                             'midi_diff_processed', 'real_duration',
                                                             'note_density', 'black_key', 'chord']].values)), axis=1)

    # 按乐谱分组（假设 'note_id' 可用于分组，具体根据数据调整）
    # 如果没有明确的分组信息，可以按文件名或其他标识进行分组
    # 这里假设每个 'note_id' 是唯一的，且按时间顺序排列

    # 创建句子列表
    sentences = []
    current_sentence = []
    prev_onset = -1
    threshold = 5.0  # 根据需要调整，确定句子分割点

    for idx, row in df.iterrows():
        if row['onset_time'] - prev_onset > threshold:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        current_sentence.append(row['word'])
        prev_onset = row['onset_time']

    if current_sentence:
        sentences.append(current_sentence)

    # 训练 Word2Vec-CBOW 模型
    model = Word2Vec(sentences, window=window, vector_size=vector_size, min_count=min_count, workers=workers, sg=0)
    return model


# 训练 Word2Vec-CBOW 模型
word2vec_model = train_word2vec(df, window=2, vector_size=128, min_count=1, workers=4)

# 保存模型
word2vec_model.save("word2vec_cbow.model")

# 加载模型（如有需要）
word2vec_model = Word2Vec.load("word2vec_cbow.model")
def get_fused_features(df, word2vec_model):
    """
    使用 Word2Vec 模型将每个音符的特征向量转化为融合特征向量。
    """
    df = df.copy()
    df['fused_feature'] = df['word'].apply(lambda x: word2vec_model.wv[x] if x in word2vec_model.wv else np.zeros(word2vec_model.vector_size))
    return df

df = get_fused_features(df, word2vec_model)

# 将融合特征向量转化为多维特征
fused_features = np.vstack(df['fused_feature'].values)

# 标准化融合特征
scaler_fused = StandardScaler()
fused_features_scaled = scaler_fused.fit_transform(fused_features)

# 将融合特征添加到原始特征中
df['fused_feature_scaled'] = list(fused_features_scaled)

# 更新特征集
feature_columns = ['pitch_encoded', 'duration_encoded', 'hand_encoded',
                   'midi_diff_processed', 'real_duration',
                   'note_density', 'black_key', 'chord']

# 我们将融合特征作为新的特征，替换或附加到现有特征中
# 这里选择替换音高编码等原始特征，使用融合特征
# 根据论文，需要保留原始特征以确保特征不丢失
# 这里我们将融合特征附加到原始特征

def combine_features(row):
    return np.concatenate((row[feature_columns].values, row['fused_feature_scaled']))

df['combined_features'] = df.apply(combine_features, axis=1)

X = np.stack(df['combined_features'].values)
y = df['fingering_encoded'].values

print(f"新特征形状: {X.shape}")
print(f"标签形状: {y.shape}")

# 标准化数值特征（包括融合特征）
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 重新创建序列
sequence_length = 10  # 使用前10个音符预测第11个音符

def create_sequences(X, y, seq_length):
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, sequence_length)

print(f"序列特征形状（包含融合特征）: {X_seq.shape}")  # (样本数, sequence_length, 特征数量)
print(f"序列标签形状: {y_seq.shape}")  # (样本数,)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"验证集样本数: {X_val.shape[0]}")

# 使用 SMOTE 进行过采样（针对序列数据，需谨慎使用）
# SMOTE 主要适用于非序列数据，以下为一种处理方法
# 您也可以选择仅使用类别权重而不使用 SMOTE
smote = SMOTE(random_state=42)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_val_reshaped = X_val.reshape(X_val.shape[0], -1)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train)
X_val_resampled, y_val_resampled = smote.fit_resample(X_val_reshaped, y_val)

# 将数据重新转换为序列格式
X_train_resampled = X_train_resampled.reshape(-1, sequence_length, X_seq.shape[2])
X_val_resampled = X_val_resampled.reshape(-1, sequence_length, X_seq.shape[2])

print(f"过采样后训练集序列形状: {X_train_resampled.shape}, 标签形状: {y_train_resampled.shape}")
print(f"过采样后验证集序列形状: {X_val_resampled.shape}, 标签形状: {y_val_resampled.shape}")


def augment_mirror_symmetry(X, y, le_fingering, le_hand):
    """
    利用左右手镜像对称进行数据增强。
    """
    X_aug = []
    y_aug = []

    for i in range(len(X)):
        # 假设 'hand_encoded' 是特征中的一个维度，且为特定索引
        hand_index = 2  # 根据实际特征顺序调整
        if X[i, -1, hand_index] == le_hand.transform(['left'])[0]:
            # 将左手数据转换为右手数据
            X_mirror = X[i].copy()
            X_mirror[:, hand_index] = le_hand.transform(['right'])[0]

            # 翻转指法（具体翻转规则需根据手指编号定义）
            # 假设有 5 个手指，翻转规则如：1↔5, 2↔4, 3不变
            finger_flip = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
            y_mirror = finger_flip.get(y[i], y[i])

            X_aug.append(X_mirror)
            y_aug.append(y_mirror)

    if X_aug:
        X_aug = np.array(X_aug)
        y_aug = np.array(y_aug)
        return np.concatenate((X, X_aug), axis=0), np.concatenate((y, y_aug), axis=0)
    else:
        return X, y


# # 加载过采样后的数据
# X_train = np.load('X_train_resampled.npy')
# y_train = np.load('y_train_resampled.npy')
# X_val = np.load('X_val_resampled.npy')
# y_val = np.load('y_val_resampled.npy')

with open('le_pitch.pkl', 'rb') as f:
    le_pitch = pickle.load(f)
with open('le_duration.pkl', 'rb') as f:
    le_duration = pickle.load(f)
with open('le_hand.pkl', 'rb') as f:
    le_hand = pickle.load(f)
with open('le_fingering.pkl', 'rb') as f:
    le_fingering = pickle.load(f)

# 执行数据增强
X_train_aug, y_train_aug = augment_mirror_symmetry(X_train_resampled, y_train_resampled, le_fingering, le_hand)
X_val_aug, y_val_aug = augment_mirror_symmetry(X_val_resampled, y_val_resampled, le_fingering, le_hand)

print(f"增强后训练集序列形状: {X_train_aug.shape}, 标签形状: {y_train_aug.shape}")
print(f"增强后验证集序列形状: {X_val_aug.shape}, 标签形状: {y_val_aug.shape}")

# 更新数据加载器
import torch
from torch.utils.data import Dataset, DataLoader


class FingeringDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # 输入特征
        self.y = torch.tensor(y, dtype=torch.long)  # 指法标签

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


batch_size = 64

train_dataset = FingeringDataset(X_train_aug, y_train_aug)
val_dataset = FingeringDataset(X_val_aug, y_val_aug)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 将序列数据保存为 npy 文件
np.save('X_train_aug.npy', X_train_aug)
np.save('X_val_aug.npy', X_val_aug)
np.save('y_train_aug.npy', y_train_aug)
np.save('y_val_aug.npy', y_val_aug)
