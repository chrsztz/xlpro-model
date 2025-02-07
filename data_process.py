# data_process.py

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from data_utils import (
    get_midi_number,
    is_black_key,
    calculate_speed_features,
    calculate_midi_diff,
    create_word_column,
    train_word2vec,
    get_fused_features,
    combine_features,
    save_pickle,
    load_pickle,
    process_fused_features
)
from gensim.models import Word2Vec


def main():
    # 加载序列数据
    try:
        X_train = np.load('X_train.npy')
        X_val = np.load('X_val.npy')
        y_train = np.load('y_train.npy')
        y_val = np.load('y_val.npy')
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("请确保已运行 'dataset_prep.py' 并生成所需的 .npy 文件。")
        return

    # 加载 LabelEncoders
    try:
        le_pitch = load_pickle('le_pitch.pkl')
        le_duration = load_pickle('le_duration.pkl')
        le_hand = load_pickle('le_hand.pkl')
        le_fingering = load_pickle('le_fingering.pkl')
    except FileNotFoundError as e:
        print(f"Error loading LabelEncoders: {e}")
        print("请确保已运行 'dataset_prep.py' 并生成相关的 .pkl 文件。")
        return

    # 加载 DataFrame
    try:
        df = pd.read_pickle('df.pkl')
        print(type(df),len(df))
    except FileNotFoundError:
        print("Error: 'df.pkl' not found. 请在 'dataset_prep.py' 中添加保存 DataFrame 的代码。")
        return

    # 计算额外特征
    df = calculate_midi_diff(df)
    print("done")
    df = calculate_speed_features(df)
    print("done")

    # 添加黑键标识符
    df['black_key'] = df['midi_number'].apply(is_black_key)
    if 'is_chord' not in df.columns:
        df['is_chord'] = 0
    df['chord'] = df['is_chord']  # 0 或 1

    # 特征提取
    feature_columns = ['pitch_encoded', 'duration_encoded', 'hand_encoded',
                       'midi_diff_processed', 'real_duration',
                       'note_density', 'black_key', 'chord']
    df = create_word_column(df, feature_columns)

    print("部分 'word' 列样例：")
    print(df['word'].head())
    save_pickle(df, "df.pkl")
    # **修改部分开始**
    # 将 'word' 列拆分为单词列表
    tokenized_sentences = df['word'].apply(lambda x: x.split()).tolist()
    # 训练 Word2Vec-CBOW 模型
    word2vec_model = train_word2vec(tokenized_sentences, window=2, vector_size=128, min_count=1, workers=4)
    # **修改部分结束**

    # 保存模型
    word2vec_model.save("word2vec_cbow.model")

    print("Word2Vec 模型已训练并保存。")

    # 获取融合特征
    df = get_fused_features(df, word2vec_model, tokenized_sentences)

    # 标准化融合特征
    scaler_fused, fused_features_scaled = process_fused_features(df, scaler_fused=None, batch_size=10000)
    df['fused_feature_scaled'] = list(fused_features_scaled)
    print("done")

    # 更新特征集
    feature_columns_extended = ['pitch_encoded', 'duration_encoded', 'hand_encoded',
                                'midi_diff_processed', 'real_duration',
                                'note_density', 'black_key', 'chord']

    # 将融合特征附加到原始特征
    df = combine_features(df, feature_columns_extended)

    print("部分 'combined_features' 样例：")
    print(df['combined_features'].head())

    # 提取特征和标签
    X = np.stack(df['combined_features'].values)
    y = df['fingering_encoded'].values

    print(f"新特征形状: {X.shape}")
    print(f"标签形状: {y.shape}")

    # 标准化数值特征（包括融合特征）
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 保存标准化器
    save_pickle(scaler, 'scaler.pkl')

    # 重新创建序列
    sequence_length = 10  # 使用前10个音符预测第11个音符

    def create_sequences_full(X, y, seq_length):
        X_seq = []
        y_seq = []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences_full(X, y, sequence_length)

    print(f"序列特征形状（包含融合特征）: {X_seq.shape}")  # (样本数, sequence_length, 特征数量)
    print(f"序列标签形状: {y_seq.shape}")  # (样本数,)

    # 划分训练集和验证集
    X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    print(f"训练集样本数: {X_train_seq.shape[0]}")
    print(f"验证集样本数: {X_val_seq.shape[0]}")

    # 使用 SMOTE 进行过采样（针对序列数据，需谨慎使用）
    # 注意：SMOTE 主要适用于非序列数据，以下为一种处理方法
    # 您也可以选择仅使用类别权重而不使用 SMOTE
    smote = SMOTE(random_state=42)
    X_train_reshaped = X_train_seq.reshape(X_train_seq.shape[0], -1)
    X_val_reshaped = X_val_seq.reshape(X_val_seq.shape[0], -1)

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train_seq)
    X_val_resampled, y_val_resampled = smote.fit_resample(X_val_reshaped, y_val_seq)

    # 将数据重新转换为序列格式
    X_train_resampled = X_train_resampled.reshape(-1, sequence_length, X_seq.shape[2])
    X_val_resampled = X_val_resampled.reshape(-1, sequence_length, X_seq.shape[2])

    print(f"过采样后训练集序列形状: {X_train_resampled.shape}, 标签形状: {y_train_resampled.shape}")
    print(f"过采样后验证集序列形状: {X_val_resampled.shape}, 标签形状: {y_val_resampled.shape}")

    # 数据增强：镜像对称
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
                y_mirror = np.array([finger_flip.get(f, f) for f in y[i]])

                X_aug.append(X_mirror)
                y_aug.append(y_mirror)

        if X_aug:
            X_aug = np.array(X_aug)
            y_aug = np.array(y_aug)
            return np.concatenate((X, X_aug), axis=0), np.concatenate((y, y_aug), axis=0)
        else:
            return X, y

    # 执行数据增强
    X_train_aug, y_train_aug = augment_mirror_symmetry(X_train_resampled, y_train_resampled, le_fingering, le_hand)
    X_val_aug, y_val_aug = augment_mirror_symmetry(X_val_resampled, y_val_resampled, le_fingering, le_hand)

    print(f"增强后训练集序列形状: {X_train_aug.shape}, 标签形状: {y_train_aug.shape}")
    print(f"增强后验证集序列形状: {X_val_aug.shape}, 标签形状: {y_val_aug.shape}")

    # 将增强后的数据保存为 npy 文件
    np.save('X_train_aug.npy', X_train_aug)
    np.save('X_val_aug.npy', X_val_aug)
    np.save('y_train_aug.npy', y_train_aug)
    np.save('y_val_aug.npy', y_val_aug)

    print("增强后的数据已保存。")


if __name__ == "__main__":
    main()
