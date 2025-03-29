 # crf_feature_extractor.py

import numpy as np
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics

def sequence_to_crf_features(sequence, window_size=2):
    """
    将音符序列转换为CRF特征，包含上下文窗口
    
    Args:
        sequence: 特征字典序列
        window_size: 考虑前后多少个音符
        
    Returns:
        用于CRF的特征字典列表
    """
    features = []
    seq_len = len(sequence)
    
    for i in range(seq_len):
        # 当前音符的基本特征
        note_features = {
            'pitch': sequence[i]['pitch_encoded'],
            'duration': sequence[i]['duration_encoded'],
            'hand': sequence[i]['hand_encoded'],
            'black_key': sequence[i]['black_key'],
            'chord': sequence[i]['chord'],
            'midi_diff': sequence[i]['midi_diff_processed'],
            'density': sequence[i]['note_density'],
        }
        
        # 添加上下文特征(前后音符)
        for offset in range(-window_size, window_size + 1):
            if offset == 0:  # 跳过当前音符(已添加)
                continue
                
            idx = i + offset
            # 处理边界条件
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
                # 对于超出范围的，使用特殊指示符
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

def extract_crf_features(df, crf_model_path='crf_model.pkl', sequence_length=10):
    """
    提取CRF特征用于预测
    
    Args:
        df: 包含音符数据的DataFrame
        crf_model_path: 保存的CRF模型路径
        sequence_length: 考虑的序列长度
        
    Returns:
        包含CRF特征的DataFrame
    """
    from data_utils import load_pickle
    
    print("提取CRF特征...")
    
    # 将DataFrame行转换为字典供CRF特征提取
    df_dict = df.to_dict('records')
    
    # 为每个序列提取CRF特征
    crf_features = []
    for i in range(0, len(df_dict), sequence_length//2):  # 使用半重叠窗口
        if i + sequence_length <= len(df_dict):
            seq = df_dict[i:i+sequence_length]
            crf_features.extend(sequence_to_crf_features(seq))
    
    # 确保特征列表与DataFrame长度相同
    if len(crf_features) < len(df):
        # 不足的部分用空特征填充
        remainder = len(df) - len(crf_features)
        empty_features = [{'crf_placeholder': 0} for _ in range(remainder)]
        crf_features.extend(empty_features)
    elif len(crf_features) > len(df):
        # 多余部分裁剪
        crf_features = crf_features[:len(df)]
    
    # 加载预训练的CRF模型
    try:
        crf = load_pickle(crf_model_path)
        print(f"成功加载CRF模型: {crf_model_path}")
        
        # 为每个位置提取概率特征
        X_crf = []
        
        for i in range(0, len(df) - sequence_length + 1):
            if i + sequence_length <= len(crf_features):
                X_crf.append(crf_features[i:i+sequence_length])
        
        if X_crf:
            # 提取每个位置的概率特征
            crf_probs = []
            try:
                for seq_features in X_crf:
                    seq_probs = crf.predict_marginals_single(seq_features)
                    crf_probs.append(seq_probs)
                
                # 确定最大类别数
                all_probs = []
                for probs in crf_probs:
                    all_probs.extend(list(probs.keys()))
                
                # 尝试从所有键中提取数字类别
                digit_classes = []
                for k in all_probs:
                    try:
                        if k.isdigit():
                            digit_classes.append(int(k))
                    except:
                        continue
                
                if digit_classes:
                    max_classes = max(digit_classes) + 1
                else:
                    # 如果没有找到数字类别，默认使用10类
                    max_classes = 10
                    print(f"警告: 无法从CRF模型确定类别数量，使用默认值: {max_classes}")
                
                print(f"CRF特征类别数: {max_classes}")
                
                # 将CRF概率展平为每个音符的固定大小向量
                crf_vectors = []
                
                for probs in crf_probs:
                    vector = []
                    for i in range(max_classes):
                        class_key = str(i)
                        vector.append(probs.get(class_key, 0.0))
                    crf_vectors.append(vector)
                
                # 对于没有CRF特征的音符用零填充
                zero_vector = [0.0] * max_classes
                
                # 创建与df长度相同的向量列表
                all_vectors = []
                for i in range(len(df)):
                    if i < len(crf_vectors):
                        all_vectors.append(crf_vectors[i])
                    else:
                        all_vectors.append(zero_vector)
                
                # 将CRF向量添加到DataFrame
                df['crf_feature'] = all_vectors
                print(f"成功提取CRF特征，维度: {max_classes}")
            except Exception as e:
                print(f"CRF特征提取出错: {e}")
                df['crf_feature'] = [[0.0] * 10] * len(df)
        else:
            print("警告: 没有足够的数据生成CRF特征")
            df['crf_feature'] = [[0.0] * 10] * len(df)
    except Exception as e:
        print(f"CRF模型加载或使用出错: {e}")
        df['crf_feature'] = [[0.0] * 10] * len(df)
    
    return df