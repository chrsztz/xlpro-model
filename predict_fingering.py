# predict_fingering.py

import os
import sys
import torch
import numpy as np
import pandas as pd
from music21 import converter
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler

from musicxml_utils import extract_all_features, apply_predicted_fingering
from crf_feature_extractor import extract_crf_features
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
from models import EnhancedFingeringModel, CNNWithAttention, BiLSTMWithAttention


def preprocess_score_data(score_path, model_dir='.'):
    """预处理乐谱数据，确保与训练数据格式完全一致"""
    print(f"开始处理乐谱: {score_path}")
    
    # 1. 加载乐谱和初始特征
    score = converter.parse(score_path)
    df = extract_all_features(score)
    print(f"初始特征数量: {df.shape}")

    # 2. 加载所有预处理器
    try:
        le_pitch = load_pickle(os.path.join(model_dir, 'le_pitch.pkl'))
        le_duration = load_pickle(os.path.join(model_dir, 'le_duration.pkl'))
        le_hand = load_pickle(os.path.join(model_dir, 'le_hand.pkl'))
        le_fingering = load_pickle(os.path.join(model_dir, 'le_fingering.pkl'))
        word2vec_model = Word2Vec.load(os.path.join(model_dir, "word2vec_cbow.model"))
        scaler = load_pickle(os.path.join(model_dir, 'scaler.pkl'))
        print("成功加载所有预处理器")
    except FileNotFoundError as e:
        print(f"加载预处理器错误: {e}")
        raise

    # 3. 基础特征处理
    print("处理基础特征...")
    
    # 标准化音高
    df['normalized_spelled_pitch'] = df['note'].apply(normalize_spelled_pitch)

    # 计算MIDI编号并添加black_key特征
    df['midi_number'] = df['normalized_spelled_pitch'].apply(get_midi_number)
    df['black_key'] = df['midi_number'].apply(is_black_key)

    # 计算MIDI差异和速度特征
    df = calculate_midi_diff(df)
    
    # 从现有训练数据中获取note_density的分布信息
    try:
        train_df = pd.read_pickle(os.path.join(model_dir, 'df.pkl'))
        max_density = train_df['note_density'].max()
        min_density = train_df['note_density'].min()
    except FileNotFoundError:
        print("警告：找不到训练数据，使用默认密度范围")
        max_density = 10
        min_density = 0
    
    # 计算音符密度
    def calculate_density(df, row_idx, window=1.0):
        """计算音符密度并归一化"""
        current_time = df.iloc[row_idx]['onset_time']
        end_time = current_time + window
        
        count = df[(df['onset_time'] >= current_time) & 
                   (df['onset_time'] < end_time)].shape[0]
        
        # 归一化到训练集范围
        if max_density > min_density:
            normalized_count = (count - min_density) / (max_density - min_density) * 10
        else:
            normalized_count = count
            
        return normalized_count
    
    # 应用密度计算
    df['note_density'] = [calculate_density(df, i) for i in range(len(df))]
    
    # 确保chord特征存在
    if 'chord' not in df.columns:
        df['chord'] = 0
    
    # 4. 特征编码
    try:
        df['pitch_encoded'] = le_pitch.transform(df['normalized_spelled_pitch'])
        df['duration_encoded'] = le_duration.transform(df['duration'].astype(str))
        df['hand_encoded'] = le_hand.transform(df['hand'])
        print("特征编码成功")
    except ValueError as e:
        print(f"特征编码错误: {e}")
        
        # 处理未见过的值
        print("正在处理未见过的值...")
        
        # 处理未见过的音高
        for i, pitch in enumerate(df['normalized_spelled_pitch']):
            if pitch not in le_pitch.classes_:
                print(f"未见过的音高: {pitch}，替换为C4")
                df.at[i, 'normalized_spelled_pitch'] = 'C4'
        
        # 处理未见过的时值
        for i, duration in enumerate(df['duration'].astype(str)):
            if duration not in le_duration.classes_:
                print(f"未见过的时值: {duration}，替换为1.0")
                df.at[i, 'duration'] = '1.0'
        
        # 重新尝试编码
        df['pitch_encoded'] = le_pitch.transform(df['normalized_spelled_pitch'])
        df['duration_encoded'] = le_duration.transform(df['duration'].astype(str))
        df['hand_encoded'] = le_hand.transform(df['hand'])
        print("特征编码重试成功")

    # 5. 创建word列并获取Word2Vec特征
    feature_columns = [
        'pitch_encoded', 'duration_encoded', 'hand_encoded',
        'midi_diff_processed', 'real_duration', 'note_density',
        'black_key', 'chord'
    ]
    
    df = create_word_column(df, feature_columns)
    tokenized_sentences = df['word'].apply(lambda x: x.split()).tolist()
    
    # 6. 获取融合特征（Word2Vec）
    df = get_fused_features(df, word2vec_model, tokenized_sentences)
    
    # 确保fused_feature是numpy数组
    fused_features = np.vstack(df['fused_feature'].values)
    print(f"Fused feature dimension: {fused_features.shape[1]}")
    
    # 7. 添加CRF特征
    print("提取CRF特征...")
    
    # 创建零向量作为CRF特征占位符 - 与训练数据中的维度保持一致
    try:
        # 首先尝试加载CRF模型
        crf_model_path = os.path.join(model_dir, 'crf_model.pkl')
        crf = load_pickle(crf_model_path)
        print("成功加载CRF模型")
        
        # 使用CRF特征提取器
        df = extract_crf_features(df, crf_model_path=crf_model_path)
    except FileNotFoundError:
        print(f"警告: 找不到CRF模型，使用零向量替代")
        # 创建零向量占位符，确保维度正确（默认使用10类）
        df['crf_feature'] = [[0.0] * 10] * len(df)
    
    print("CRF特征提取完成")

    # 8. 检查特征维度是否与训练数据一致
    # 获取训练数据的特征维度
    try:
        sample_scaler_data = scaler.mean_.shape[0]
        expected_features = sample_scaler_data
        print(f"训练数据特征维度：{expected_features}")
        
        # 计算当前特征维度
        combined_dim = len(feature_columns) + fused_features.shape[1]
        if 'crf_feature' in df.columns:
            crf_dim = len(df['crf_feature'].iloc[0])
            combined_dim += crf_dim
        print(f"当前特征维度：{combined_dim}")
        
        # 检查是否匹配
        if combined_dim != expected_features:
            print(f"警告：特征维度不匹配！训练：{expected_features}，当前：{combined_dim}")
            
            # 调整CRF特征维度以匹配
            required_crf_dim = expected_features - len(feature_columns) - fused_features.shape[1]
            print(f"调整CRF特征维度为：{required_crf_dim}")
            if required_crf_dim > 0:
                df['crf_feature'] = [[0.0] * required_crf_dim] * len(df)
    except Exception as e:
        print(f"检查特征维度时出错：{e}")

    # 9. 组合所有特征
    print("组合所有特征...")
    # 使用修改后的combine_features函数
    combined_features = []
    
    for i, row in df.iterrows():
        # 获取原始特征
        orig_features = [float(row[col]) for col in feature_columns]
        
        # 获取融合特征
        fused_feature = row['fused_feature'] if 'fused_feature' in row else []
        
        # 获取CRF特征
        crf_feature = row['crf_feature'] if 'crf_feature' in row else []
        
        # 结合所有特征
        combined = np.concatenate([orig_features, fused_feature, crf_feature])
        combined_features.append(combined)
    
    df['combined_features'] = combined_features

    # 10. 提取并标准化特征
    X = np.stack(df['combined_features'].values)
    print(f"组合特征形状: {X.shape}")
    
    # 再次检查维度是否匹配
    if X.shape[1] != sample_scaler_data:
        raise ValueError(f"特征维度不匹配: 当前 {X.shape[1]}, 预期 {sample_scaler_data}")
    
    # 应用标准化
    X_scaled = scaler.transform(X)
    
    # 11. 创建序列
    sequence_length = 10
    X_seq = []
    
    # 处理边界条件
    padded_X = np.zeros((sequence_length-1 + len(X_scaled), X_scaled.shape[1]))
    padded_X[sequence_length-1:] = X_scaled
    
    for i in range(len(X_scaled)):
        X_seq.append(padded_X[i:i+sequence_length])
    
    X_seq = np.array(X_seq, dtype=np.float32)
    print(f"最终序列形状: {X_seq.shape}")

    return score, X_seq, df


def predict_fingering(score_path, model_path='./results_cnn/bilstm_cnn_attention_model_best.pth', model_dir='.'):
    """预测指法并应用到乐谱"""
    print(f"开始预测指法: {score_path}")
    
    # 预处理数据
    score, X_seq, df = preprocess_score_data(score_path, model_dir)
    
    # 确定输入特征维度
    input_size = X_seq.shape[2]
    print(f"模型输入特征维度: {input_size}")
    
    # 加载模型
    try:
        # 确定模型类型
        model_name = os.path.basename(model_path)
        
        # 首先加载模型权重以获取其维度
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 检查模型的输入维度
        model_input_size = None
        
        # 尝试从权重中推断输入维度
        if 'lstm.weight_ih_l0' in state_dict:
            print("从LSTM权重推断输入维度...")
            lstm_weight = state_dict['lstm.weight_ih_l0']
            if lstm_weight.dim() > 1:
                model_input_size = lstm_weight.size(1)
                print(f"模型原始输入维度: {model_input_size}")
        
        # 检查CNN的隐藏层维度
        cnn_hidden_size = 128
        cnn_input_channels = input_size
        if 'conv_layers.0.weight' in state_dict:
            print("从CNN权重推断输入通道数...")
            cnn_weight = state_dict['conv_layers.0.weight']
            if cnn_weight.dim() > 0:
                cnn_hidden_size = cnn_weight.size(0)
                cnn_input_channels = cnn_weight.size(1)
                print(f"CNN隐藏层维度: {cnn_hidden_size}, 输入通道数: {cnn_input_channels}")
        elif 'conv1.weight' in state_dict:
            print("从CNN权重推断隐藏层维度...")
            cnn_weight = state_dict['conv1.weight']
            if cnn_weight.dim() > 0:
                cnn_hidden_size = cnn_weight.size(0)
                print(f"CNN隐藏层维度: {cnn_hidden_size}")
        
        # 如果输入维度不匹配，需要调整特征
        if model_input_size is not None and model_input_size != input_size:
            print(f"警告: 特征维度不匹配，模型期望 {model_input_size}，实际为 {input_size}")
            print("调整特征维度以匹配模型...")
            
            if model_input_size < input_size:
                # 如果模型期望更少的特征，裁剪特征
                print(f"裁剪特征从 {input_size} 到 {model_input_size}")
                X_seq = X_seq[:, :, :model_input_size]
            else:
                # 如果模型期望更多的特征，填充为零
                print(f"填充特征从 {input_size} 到 {model_input_size}")
                padding = torch.zeros((X_seq.shape[0], X_seq.shape[1], model_input_size - input_size), dtype=torch.float32)
                X_seq = torch.cat([torch.tensor(X_seq, dtype=torch.float32), padding], dim=2)
            
            # 更新输入尺寸
            input_size = model_input_size
            print(f"调整后的特征维度: {X_seq.shape}")
        
        # 根据模型名称创建适当的模型
        if 'cnn_attention' in model_name:
            print("使用CNN+Attention模型")
            model = CNNWithAttention(
                input_size=input_size,
                hidden_size=cnn_hidden_size,  # 使用从模型中推断的隐藏层大小
                num_classes=10,
                dropout=0.5
            )
        elif 'bilstm' in model_name.lower():
            print("使用BiLSTM+Attention模型")
            model = BiLSTMWithAttention(
                input_size=input_size,
                hidden_size=512,
                num_layers=3,
                num_classes=10,
                dropout=0.5,
                bidirectional=True
            )
        else:
            print("使用增强型指法模型")
            model = EnhancedFingeringModel(
                input_size=cnn_input_channels, # 使用检测到的输入通道数，而不是特征维度
                hidden_size=128,  # 从512改为128，以匹配保存的模型参数
                num_classes=10,
                num_heads=4
            )
            
        # 加载模型权重
        model.load_state_dict(state_dict)
        model.eval()
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise
    
    # 加载指法编码器
    fingering_encoder = load_pickle(os.path.join(model_dir, 'le_fingering.pkl'))
    
    # 预测指法
    predicted_fingerings = []
    confidence_scores = []
    
    print("开始预测...")
    with torch.no_grad():
        for i, x in enumerate(X_seq):
            x_tensor = torch.tensor(x).unsqueeze(0)
            
            # 如果使用的是增强型模型，需要调整输入形状以适应CNN的输入通道
            if isinstance(model, EnhancedFingeringModel):
                batch_size, seq_len, features = x_tensor.size()
                
                # 如果特征维度与CNN输入通道不匹配，需要调整
                if features != cnn_input_channels:
                    print(f"调整张量形状以匹配CNN输入通道数，将{features}特征压缩到{cnn_input_channels}通道")
                    # 将特征维度从128压缩为4通道，通过特征分组
                    if features > cnn_input_channels:
                        # 简单的方法是取均值收缩，将特征分成cnn_input_channels组
                        group_size = features // cnn_input_channels
                        x_reshaped = x_tensor.view(batch_size, seq_len, cnn_input_channels, group_size)
                        x_tensor = x_reshaped.mean(dim=3)
                        print(f"调整后的张量形状: {x_tensor.shape}")
            
            output = model(x_tensor)
            
            # 获取预测类别和概率
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
            
            finger_idx = predicted.item()
            confidence_val = confidence.item()
            
            # 转换为实际指法
            try:
                # 根据指法编码器转换为实际指法值
                actual_fingering = fingering_encoder.inverse_transform([finger_idx])[0]
                
                # 检查当前音符是否为和弦的一部分
                is_chord = False
                if i < len(df) and 'chord' in df.columns:
                    is_chord = df.iloc[i]['chord'] == 1
                    
                # 如果是和弦，根据音符在和弦中的位置可能需要调整指法
                # 这里我们简单处理，确保每个和弦中的音符都有指法
                predicted_fingerings.append(actual_fingering)
                confidence_scores.append(confidence_val)
            except:
                print(f"警告: 无法转换指法索引 {finger_idx}，使用默认值1")
                predicted_fingerings.append(1)  # 使用默认值
                confidence_scores.append(confidence_val)
            
            # 打印进度
            if (i+1) % 100 == 0 or i+1 == len(X_seq):
                print(f"已处理 {i+1}/{len(X_seq)} 个音符")
    
    # 打印预测统计
    print("\n预测统计:")
    unique_fingers, counts = np.unique(predicted_fingerings, return_counts=True)
    for finger, count in zip(unique_fingers, counts):
        print(f"指法 {finger}: {count} 次 ({count/len(predicted_fingerings)*100:.2f}%)")
    
    print(f"\n平均预测置信度: {np.mean(confidence_scores):.4f}")
    
    # 优化预测的指法，修正不合理的连续相同指法
    print("优化预测结果，修正不合理的连续指法...")
    
    # 导入指法优化器
    from fingering_optimizer import FingeringOptimizer
    optimizer = FingeringOptimizer()
    
    # 创建临时 DataFrame 用于指法优化
    df_for_opt = df.copy()
    df_for_opt['finger_number'] = predicted_fingerings
    
    # 优化指法
    optimized_fingerings = optimizer.postprocess_predicted_fingerings(predicted_fingerings, df)
    
    # 打印优化后的统计
    print("\n优化后的指法统计:")
    unique_fingers, counts = np.unique(optimized_fingerings, return_counts=True)
    for finger, count in zip(unique_fingers, counts):
        print(f"指法 {finger}: {count} 次 ({count/len(optimized_fingerings)*100:.2f}%)")
    
    # 应用优化后的指法到乐谱
    modified_score = apply_predicted_fingering(score, optimized_fingerings)
    
    return modified_score, optimized_fingerings, confidence_scores


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='钢琴指法预测工具')
    parser.add_argument('input_file', type=str, help='输入的MusicXML文件路径')
    parser.add_argument('--output_file', type=str, help='输出的MusicXML文件路径')
    parser.add_argument('--model', type=str, default='./results_cnn/bilstm_cnn_attention_model_best.pth', 
                        help='模型文件路径')
    parser.add_argument('--model_dir', type=str, default='.', 
                        help='模型相关文件所在目录')
    
    args = parser.parse_args()
    
    # 确定输出文件名
    if args.output_file:
        output_path = args.output_file
    else:
        input_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_path = f"{input_name}_with_fingering.mxl"
    
    # 预测指法
    try:
        modified_score, fingerings, confidences = predict_fingering(
            args.input_file, 
            model_path=args.model,
            model_dir=args.model_dir
        )
        
        # 保存结果
        modified_score.write('mxl', fp=output_path)
        print(f"指法预测完成。已保存到 {output_path}")
        
        # 保存指法和置信度数据
        confidence_path = f"{os.path.splitext(output_path)[0]}_confidences.csv"
        pd.DataFrame({
            'fingering': fingerings,
            'confidence': confidences
        }).to_csv(confidence_path, index=False)
        print(f"置信度数据已保存到 {confidence_path}")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()