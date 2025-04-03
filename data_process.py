# data_process.py

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn_crfsuite  # Add CRF import
from sklearn_crfsuite import metrics
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
    calculate_physical_constraint_features,
    calculate_stretching_rate,
    calculate_hand_position,
    calculate_hand_movement,
    calculate_cross_fingering_distance,
    count_fingering_mismatches,
    count_inverse_fingerings,
    white_key_distance
)
from gensim.models import Word2Vec


# Function to convert sequences to CRF features
def sequence_to_crf_features(sequence, window_size=2):
    """
    Convert a sequence of notes to CRF features with context window.

    Args:
        sequence: A sequence of feature dictionaries
        window_size: Number of notes to consider before and after
        
    Returns:
        List of dictionaries with features for CRF
    """
    features = []
    seq_len = len(sequence)
    
    for i in range(seq_len):
        # Basic features for current note
        note_features = {
            'pitch': sequence[i]['pitch_encoded'],
            'duration': sequence[i]['duration_encoded'],
            'hand': sequence[i]['hand_encoded'],
            'black_key': sequence[i]['black_key'],
            'chord': sequence[i]['chord'],
            'midi_diff': sequence[i]['midi_diff_processed'],
            'density': sequence[i]['note_density'],
        }
        
        # 添加物理约束特征（如果存在）
        for key, value in sequence[i].items():
            if any(substr in key for substr in ['stretch_rate', 'cross_dist', 'white_key_dist']):
                note_features[key] = value
        
        # Add context features (previous and next notes)
        for offset in range(-window_size, window_size + 1):
            if offset == 0:  # Skip current note (already added)
                continue
                
            idx = i + offset
            # Handle boundary conditions
            if 0 <= idx < seq_len:
                prefix = 'prev' if offset < 0 else 'next'
                abs_offset = abs(offset)
                note_features.update({
                    f'{prefix}{abs_offset}_pitch': sequence[idx]['pitch_encoded'],
                    f'{prefix}{abs_offset}_duration': sequence[idx]['duration_encoded'],
                    f'{prefix}{abs_offset}_hand': sequence[idx]['hand_encoded'],
                    f'{prefix}{abs_offset}_black_key': sequence[idx]['black_key'],
                    f'{prefix}{abs_offset}_chord': sequence[idx]['chord'],
                })
            else:
                # For out of bounds, use special indicators
                prefix = 'prev' if offset < 0 else 'next'
                abs_offset = abs(offset)
                note_features.update({
                    f'{prefix}{abs_offset}_pitch': -1,
                    f'{prefix}{abs_offset}_duration': -1,
                    f'{prefix}{abs_offset}_hand': -1,
                    f'{prefix}{abs_offset}_black_key': -1,
                    f'{prefix}{abs_offset}_chord': -1,
                })
        
        features.append(note_features)
    
    return features


# Function to train CRF model and extract CRF features
def extract_crf_features(df, sequence_length=10):
    """
    Train a CRF model and extract CRF features
    
    Args:
        df: DataFrame with note data
        sequence_length: Length of sequences to consider
        
    Returns:
        DataFrame with added CRF features
    """
    print("Training CRF model and extracting CRF features...")
    
    # Convert DataFrame rows to dictionaries for CRF feature extraction
    df_dict = df.to_dict('records')
    
    # Extract CRF features for each sequence
    crf_features = []
    for i in range(0, len(df_dict), sequence_length):
        if i + sequence_length <= len(df_dict):
            seq = df_dict[i:i+sequence_length]
            crf_features.extend(sequence_to_crf_features(seq))
            
    # Ensure the feature list is the same length as the DataFrame
    if len(crf_features) < len(df):
        # Pad with empty features for any remaining rows
        remainder = len(df) - len(crf_features)
        empty_features = [{'crf_placeholder': 0} for _ in range(remainder)]
        crf_features.extend(empty_features)
    elif len(crf_features) > len(df):
        # Trim excess features
        crf_features = crf_features[:len(df)]
    
    # Train CRF model on sequences and extract probabilities
    X_crf = []
    y_crf = []
    
    for i in range(0, len(df) - sequence_length):
        X_crf.append(crf_features[i:i+sequence_length])
        y_crf.append([str(y) for y in df['fingering_encoded'].values[i:i+sequence_length]])
    
    # Train CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    if len(X_crf) > 0:
        print(f"Training CRF model with {len(X_crf)} sequences...")
        crf.fit(X_crf, y_crf)
        
        # Extract probability features for each position
        crf_probs = []
        for seq_features in X_crf:
            seq_probs = crf.predict_marginals_single(seq_features)
            crf_probs.extend(seq_probs)
        
        # Flatten CRF probabilities to a fixed-size vector for each note
        max_classes = max(len(probs) for probs in crf_probs) if crf_probs else 0
        crf_vectors = []
        
        for probs in crf_probs:
            vector = []
            for i in range(max_classes):
                class_key = str(i)
                vector.append(probs.get(class_key, 0.0))
            crf_vectors.append(vector)
        
        # Pad with zeros for notes without CRF features
        zero_vector = [0.0] * max_classes
        while len(crf_vectors) < len(df):
            crf_vectors.append(zero_vector)
        
        # Add CRF vectors to DataFrame
        df['crf_feature'] = crf_vectors[:len(df)]
    else:
        print("WARNING: Not enough data to train CRF model")
        # Add empty CRF features
        df['crf_feature'] = [[0.0]] * len(df)
    
    return df


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
        print(type(df))
    except FileNotFoundError:
        print("Error: 'df.pkl' not found. 请在 'dataset_prep.py' 中添加保存 DataFrame 的代码。")
        return

    # 计算额外特征
    df = calculate_midi_diff(df)
    df = calculate_speed_features(df)

    # 添加黑键标识符
    df['black_key'] = df['midi_number'].apply(is_black_key)
    if 'is_chord' not in df.columns:
        df['is_chord'] = 0
    df['chord'] = df['is_chord']  # 0 或 1

    # 计算物理约束特征
    print("计算物理约束特征...")
    df = calculate_physical_constraint_features(df, window_size=3)
    
    # 特征提取 - 使用优化后的物理特征集
    # 基础特征
    base_features = [
        'pitch_encoded', 'duration_encoded', 'hand_encoded',
        'midi_diff_processed', 'real_duration',
        'note_density', 'black_key', 'chord'
    ]
    
    # 获取新的物理约束特征列，按类别分组
    spatial_features = [col for col in df.columns if any(substr in col for substr in 
                        ['physical_distance', 'pitch_interval', 'key_transition', 'curr_black_key'])]
    
    temporal_features = [col for col in df.columns if any(substr in col for substr in 
                         ['note_duration', 'ioi', 'overlap'])]
    
    hand_features = [col for col in df.columns if any(substr in col for substr in 
                      ['hand_switch', 'stretch_rate'])]
    
    fingering_features = [col for col in df.columns if any(substr in col for substr in 
                          ['natural_violation', 'finger_strength_violation', 'cross_dist', 'thumb_black_cross'])]
    
    # 所有物理特征
    physical_features = spatial_features + temporal_features + hand_features + fingering_features
    
    # 最终使用的特征列表
    feature_columns = base_features + physical_features
    
    print(f"使用的特征集:")
    print(f"- 基础特征: {len(base_features)} 个")
    print(f"- 空间特征: {len(spatial_features)} 个")
    print(f"- 时间特征: {len(temporal_features)} 个")
    print(f"- 手部特征: {len(hand_features)} 个")
    print(f"- 指法特征: {len(fingering_features)} 个")
    print(f"- 总特征数: {len(feature_columns)} 个")
    
    # 保存当前处理的DataFrame
    save_pickle(df, "df_features.pkl")
    
    # 决定是否使用word2vec和CRF
    use_word2vec = False  # 设置为False禁用word2vec特征
    use_crf = False       # 设置为False禁用CRF特征
    
    # 条件性Word2Vec处理
    if use_word2vec:
        print("使用Word2Vec进行特征增强...")
        df = create_word_column(df, feature_columns)
        print("部分 'word' 列样例：")
        print(df['word'].head())
        
        # 训练 Word2Vec-CBOW 模型
        tokenized_sentences = df['word'].apply(lambda x: x.split()).tolist()
        word2vec_model = train_word2vec(tokenized_sentences, window=2, vector_size=64, min_count=1, workers=4)
        
        # 保存模型
        word2vec_model.save("word2vec_cbow.model")
        print("Word2Vec 模型已训练并保存。")
        
        # 获取融合特征
        df = get_fused_features(df, word2vec_model, tokenized_sentences)
        
        # 将融合特征向量转化为多维特征
        fused_features = np.vstack(df['fused_feature'].values)
        
        # 标准化融合特征
        scaler_fused = StandardScaler()
        fused_features_scaled = scaler_fused.fit_transform(fused_features)
        
        # 将融合特征添加到原始特征中
        df['fused_feature_scaled'] = list(fused_features_scaled)
    else:
        print("跳过Word2Vec特征提取。")
        df['fused_feature'] = [np.zeros(64) for _ in range(len(df))]
    
    # 条件性CRF处理
    if use_crf:
        print("使用CRF进行特征增强...")
        # 应用CRF特征提取
        df = extract_crf_features(df, sequence_length=10)
    else:
        print("跳过CRF特征提取。")
        df['crf_feature'] = [[0.0] for _ in range(len(df))]
    
    # 在没有Word2Vec和CRF特征的情况下，直接使用物理特征
    # 构建特征矩阵
    X_features = []
    for i, row in df.iterrows():
        # 提取所有数值特征
        feature_values = np.array([float(row[col]) for col in feature_columns])
        
        # 如果使用了Word2Vec或CRF，添加这些特征
        if use_word2vec:
            feature_values = np.concatenate([feature_values, row['fused_feature']])
        
        if use_crf:
            feature_values = np.concatenate([feature_values, row['crf_feature']])
        
        X_features.append(feature_values)
    
    # 转换为numpy数组
    X = np.array(X_features)
    y = df['fingering_encoded'].values
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签形状: {y.shape}")

    # 处理无限值和NaN
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    print("已将无限值和NaN替换为有限值")
    
    # 标准化数值特征
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

    print(f"序列特征形状（包含融合特征和物理约束）: {X_seq.shape}")  # (样本数, sequence_length, 特征数量)
    print(f"序列标签形状: {y_seq.shape}")  # (样本数,)

    # 划分训练集和验证集
    X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    print(f"训练集样本数: {X_train_seq.shape[0]}")
    print(f"验证集样本数: {X_val_seq.shape[0]}")

    # 移除SMOTE过采样代码
    # 直接使用原始的训练集和验证集
    X_train_resampled = X_train_seq
    y_train_resampled = y_train_seq
    X_val_resampled = X_val_seq
    y_val_resampled = y_val_seq

    print(f"处理后训练集序列形状: {X_train_resampled.shape}, 标签形状: {y_train_resampled.shape}")
    print(f"处理后验证集序列形状: {X_val_resampled.shape}, 标签形状: {y_val_resampled.shape}")

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
                y_mirror = finger_flip.get(y[i], y[i])

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
