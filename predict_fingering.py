# predict_fingering.py

from music21 import converter
from musicxml_utils import extract_all_features, apply_predicted_fingering
import torch
import numpy as np
from models import BiLSTMWithAttention
from data_utils import (
    load_pickle,
    normalize_spelled_pitch,
    calculate_midi_diff,
    create_word_column,
    get_fused_features,
    combine_features,
    is_black_key,
    get_midi_number
)
from gensim.models import Word2Vec
import pandas as pd

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

def validate_features(df_train, df_pred, feature_name):
    """验证特征分布"""
    print(f"\nValidating {feature_name}:")
    print("Training data statistics:")
    print(df_train[feature_name].describe())
    print("\nPrediction data statistics:")
    print(df_pred[feature_name].describe())


def preprocess_score_data(score_path):
    """预处理乐谱数据，确保与训练数据格式完全一致"""
    # 1. 加载乐谱和初始特征
    score = converter.parse(score_path)
    df = extract_all_features(score)
    print(f"Initial DataFrame shape: {df.shape}")

    # 2. 加载所有预处理器
    try:
        le_pitch = load_pickle('le_pitch.pkl')
        le_duration = load_pickle('le_duration.pkl')
        le_hand = load_pickle('le_hand.pkl')
        le_fingering = load_pickle('le_fingering.pkl')
        word2vec_model = Word2Vec.load("word2vec_cbow.model")
        scaler = load_pickle('scaler.pkl')
    except FileNotFoundError as e:
        print(f"Error loading preprocessors: {e}")
        raise

    # 3. 基础特征处理
    # 标准化音高
    df['normalized_spelled_pitch'] = df['note'].apply(normalize_spelled_pitch)

    # 计算MIDI编号并添加black_key特征
    df['midi_number'] = df['normalized_spelled_pitch'].apply(get_midi_number)
    df['black_key'] = df['midi_number'].apply(is_black_key)

    # 计算MIDI差异和速度特征
    df = calculate_midi_diff(df)
    print("After midi_diff:", df.columns.tolist())
    df = calculate_speed_features(df, window=1.0)
    print("After speed features:", df.columns.tolist())

    # 4. 特征编码
    try:
        df['pitch_encoded'] = le_pitch.transform(df['normalized_spelled_pitch'])
        df['duration_encoded'] = le_duration.transform(df['duration'].astype(str))
        df['hand_encoded'] = le_hand.transform(df['hand'])
    except ValueError as e:
        print(f"Error in feature encoding: {e}")
        raise

    # 5. 创建完整的特征列表
    feature_columns = [
        'pitch_encoded', 'duration_encoded', 'hand_encoded',
        'midi_diff_processed', 'real_duration', 'note_density',
        'black_key', 'chord'
    ]
    # 加载训练数据样本进行分布对比
    train_df = pd.read_pickle('df.pkl')

    # 对比并打印特征分布
    for col in feature_columns:
        print(f"\nDistribution for {col}:")
        print("Training data:")
        print(train_df[col].value_counts(normalize=True))
        print("\nPrediction data:")
        print(df[col].value_counts(normalize=True))

    # 6. 创建word列并进行Word2Vec处理
    df = create_word_column(df, feature_columns)
    tokenized_sentences = df['word'].apply(lambda x: x.split())

    # 7. 获取融合特征（128维）
    df = get_fused_features(df, word2vec_model, tokenized_sentences)

    # 8. 组合基础特征和融合特征
    X = []
    for idx, row in df.iterrows():
        # 获取基础特征（8维）
        base_features = row[feature_columns].values
        # 获取融合特征（128维）
        fused_features = row['fused_feature']
        # 组合特征（136维）
        combined_features = np.concatenate([base_features, fused_features])
        X.append(combined_features)

    X = np.array(X)
    print(f"Combined features shape before scaling: {X.shape}")
    assert X.shape[1] == 136, f"Expected 136 features, got {X.shape[1]}"

    # 9. 对136维特征进行标准化
    X_scaled = scaler.transform(X)
    print(f"Scaled features shape: {X_scaled.shape}")

    # 10. 创建序列
    sequence_length = 10
    X_seq = []
    for i in range(len(X_scaled) - sequence_length + 1):
        X_seq.append(X_scaled[i:i + sequence_length])

    X_seq = np.array(X_seq, dtype=np.float32)
    print(f"Final sequence shape: {X_seq.shape}")

    return score, X_seq, df


def predict_fingering(score_path, model_path='fingering_bilstm_model.pth'):
    """预测指法并应用到乐谱"""
    # 预处理数据
    score, X_seq, df = preprocess_score_data(score_path)

    # 加载模型
    model = BiLSTMWithAttention(
        input_size=136,
        hidden_size=512,
        num_layers=3,
        num_classes=10,  # 确认是10类（0-9）
        dropout=0.5,
        bidirectional=True
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    model.eval()

    # 加载自定义指法编码器
    fingering_encoder = load_pickle('le_fingering.pkl')

    # 预测指法
    predicted_fingerings = []
    with torch.no_grad():
        for x in X_seq:
            x = torch.tensor(x).unsqueeze(0)
            output = model(x)
            predicted = torch.argmax(output, dim=1).item()
            # 直接使用inverse_transform，不需要额外处理
            actual_fingering = fingering_encoder.inverse_transform([predicted])[0]
            predicted_fingerings.append(actual_fingering)

    # 添加验证
    print("Prediction statistics:")
    unique_fingers, counts = np.unique(predicted_fingerings, return_counts=True)
    for finger, count in zip(unique_fingers, counts):
        print(f"Fingering {finger}: {count} times ({count / len(predicted_fingerings) * 100:.2f}%)")

    # 应用指法到乐谱
    modified_score = apply_predicted_fingering(score, predicted_fingerings)

    return modified_score



def main():
    # 输入和输出文件路径
    input_path = './output/001_Bach_Invention_No1_C.mxl'
    output_path = './output_with_fingering.mxl'

    # 预测指法
    modified_score = predict_fingering(input_path)

    # 保存结果
    modified_score.write('mxl', fp=output_path)
    print(f"Fingering prediction completed. Modified score saved to {output_path}")
    pass


if __name__ == "__main__":
    main()