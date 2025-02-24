import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
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
from joblib import Parallel, delayed

def process_batch(df_batch, word2vec_model, feature_columns, tokenized_sentences):
    """分批处理特征工程"""
    df_batch = create_word_column(df_batch, feature_columns)
    df_batch = get_fused_features(df_batch, word2vec_model, tokenized_sentences)
    return df_batch

def sequence_generator(X, y, seq_length, batch_size):
    """生成器，按需生成序列"""
    for i in range(0, len(X) - seq_length, batch_size):
        end = min(i + batch_size, len(X) - seq_length)
        X_batch = [X[j:j + seq_length] for j in range(i, end)]
        y_batch = [y[j + seq_length] for j in range(i, end)]
        yield np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.int64)

def main():
    # 加载初始数据
    try:
        X_train = np.load('X_train.npy', mmap_mode='r')
        X_val = np.load('X_val.npy', mmap_mode='r')
        y_train = np.load('y_train.npy', mmap_mode='r')
        y_val = np.load('y_val.npy', mmap_mode='r')
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        return

    try:
        df = pd.read_pickle('df.pkl')
    except FileNotFoundError as e:
        print(f"Error loading DataFrame: {e}")
        return

    if 'train' not in df.columns:
        print("Error: 'train' column not found in df.")
        return

    # 划分训练集和验证集
    df_train = df[df['train'] == 1]
    df_val = df[df['train'] == 0]
    print("Number of training samples: {}".format(len(df_train)))

    # 优化数据类型以减少内存占用
    df_train = df_train.astype({
        'pitch_encoded': 'int16',
        'duration_encoded': 'int16',
        'hand_encoded': 'int8',
        'midi_number': 'float32',
        'midi_diff_processed': 'float32',
        'real_duration': 'float32',
        'note_density': 'float32',
        'black_key': 'int8',
        'chord': 'int8'
    })
    df_val = df_val.astype({
        'pitch_encoded': 'int16',
        'duration_encoded': 'int16',
        'hand_encoded': 'int8',
        'midi_number': 'float32',
        'midi_diff_processed': 'float32',
        'real_duration': 'float32',
        'note_density': 'float32',
        'black_key': 'int8',
        'chord': 'int8'
    })

    # 计算音乐特征
    df_train = calculate_midi_diff(df_train)
    df_train = calculate_speed_features(df_train)
    df_train['black_key'] = df_train['midi_number'].apply(is_black_key)
    if 'is_chord' not in df_train.columns:
        df_train['is_chord'] = 0
    df_train['chord'] = df_train['is_chord']

    df_val = calculate_midi_diff(df_val)
    df_val = calculate_speed_features(df_val)
    df_val['black_key'] = df_val['midi_number'].apply(is_black_key)
    if 'is_chord' not in df_val.columns:
        df_val['is_chord'] = 0
    df_val['chord'] = df_val['is_chord']
    print("已完成音乐特征计算")

    # 训练 Word2Vec 模型
    feature_columns = ['pitch_encoded', 'duration_encoded', 'hand_encoded',
                       'midi_diff_processed', 'real_duration',
                       'note_density', 'black_key', 'chord']
    tokenized_sentences_train = df_train['word'].apply(lambda x: x.split()).tolist() if 'word' in df_train.columns else None
    if tokenized_sentences_train is None:
        df_train = create_word_column(df_train, feature_columns)
        tokenized_sentences_train = df_train['word'].apply(lambda x: x.split()).tolist()
    word2vec_model = train_word2vec(
        tokenized_sentences_train, window=5, vector_size=64, min_count=5, workers=4
    )
    word2vec_model.save("word2vec_cbow.model")
    print("word2vec model saved")

    # 分批并行处理特征工程
    batch_size = 50000
    df_train_batches = [df_train.iloc[i:i + batch_size] for i in range(0, len(df_train), batch_size)]
    df_val_batches = [df_val.iloc[i:i + batch_size] for i in range(0, len(df_val), batch_size)]

    df_train_processed = Parallel(n_jobs=-1)(
        delayed(process_batch)(batch, word2vec_model, feature_columns, tokenized_sentences_train)
        for batch in df_train_batches
    )
    df_train = pd.concat(df_train_processed)

    tokenized_sentences_val = df_val['word'].apply(lambda x: x.split()).tolist() if 'word' in df_val.columns else None
    if tokenized_sentences_val is None:
        tokenized_sentences_val = [batch['word'].apply(lambda x: x.split()).tolist() for batch in df_val_batches]
        tokenized_sentences_val = [item for sublist in tokenized_sentences_val for item in sublist]
    df_val_processed = [process_batch(batch, word2vec_model, feature_columns, tokenized_sentences_val)
                        for batch in df_val_batches]
    df_val = pd.concat(df_val_processed)

    scaler_fused, fused_features_scaled_train = process_fused_features(df_train, scaler_fused=None, batch_size=10000)
    df_train['fused_feature_scaled'] = list(fused_features_scaled_train)
    _, fused_features_scaled_val = process_fused_features(df_val, scaler_fused=scaler_fused, batch_size=10000)
    df_val['fused_feature_scaled'] = list(fused_features_scaled_val)
    print("特征工程完成")

    # 组合特征
    feature_columns_extended = ['pitch_encoded', 'duration_encoded', 'hand_encoded',
                                'midi_diff_processed', 'real_duration',
                                'note_density', 'black_key', 'chord']
    df_train = combine_features(df_train, feature_columns_extended)
    df_val = combine_features(df_val, feature_columns_extended)

    # 转换为数组
    X_train = np.stack(df_train['combined_features'].values)
    y_train = df_train['fingering_encoded'].values
    X_val = np.stack(df_val['combined_features'].values)
    y_val = df_val['fingering_encoded'].values

    # GPU加速标准化
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device='cuda')
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device='cuda')
    mean = torch.mean(X_train_tensor, dim=0)
    std = torch.std(X_train_tensor, dim=0)
    X_train_scaled = (X_train_tensor - mean) / std
    X_val_scaled = (X_val_tensor - mean) / std
    X_train = X_train_scaled.cpu().numpy()
    X_val = X_val_scaled.cpu().numpy()
    print("已完成标准化")

    # 使用生成器生成序列
    sequence_length = 10
    batch_size_seq = 10000

    with open('X_train_seq.npy', 'wb') as f_x, open('y_train_seq.npy', 'wb') as f_y:
        for X_batch, y_batch in sequence_generator(X_train, y_train, sequence_length, batch_size_seq):
            np.save(f_x, X_batch)
            np.save(f_y, y_batch)

    with open('X_val_seq.npy', 'wb') as f_x, open('y_val_seq.npy', 'wb') as f_y:
        for X_batch, y_batch in sequence_generator(X_val, y_val, sequence_length, batch_size_seq):
            np.save(f_x, X_batch)
            np.save(f_y, y_batch)

    print("序列生成并保存完成")

if __name__ == "__main__":
    main()