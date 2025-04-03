#!/usr/bin/env python
# 使用物理特征增强型模型预测钢琴指法

import os
import sys
import torch
import numpy as np
import pandas as pd
from music21 import converter
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from musicxml_utils import extract_all_features, apply_predicted_fingering
from models import PhysicalEnhancedFingeringModel
from data_utils import (
    load_pickle,
    normalize_spelled_pitch,
    calculate_midi_diff,
    calculate_speed_features,
    is_black_key,
    get_midi_number,
    calculate_physical_constraint_features,
    white_key_distance
)
from train_physical_model import extract_physical_features


def preprocess_score_data(score_path, model_dir='.'):
    """预处理乐谱数据，提取优化后的物理特征"""
    print(f"开始处理乐谱: {score_path}")
    
    # 1. 加载乐谱和初始特征
    score = converter.parse(score_path)
    df = extract_all_features(score)
    print(f"初始特征数量: {df.shape}")

    # 2. 加载预处理器
    try:
        le_pitch = load_pickle(os.path.join(model_dir, 'le_pitch.pkl'))
        le_duration = load_pickle(os.path.join(model_dir, 'le_duration.pkl'))
        le_hand = load_pickle(os.path.join(model_dir, 'le_hand.pkl'))
        le_fingering = load_pickle(os.path.join(model_dir, 'le_fingering.pkl'))
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
    df = calculate_speed_features(df)
    
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

    # 5. 确保chord特征存在
    if 'chord' not in df.columns:
        df['chord'] = 0

    # 6. 计算物理约束特征 - 使用优化的计算方法
    print("计算物理约束特征...")
    df = calculate_physical_constraint_features(df, window_size=3)
    
    # 7. 构建特征向量 - 按照训练时的特征分组
    base_features = [
        'pitch_encoded', 'duration_encoded', 'hand_encoded',
        'midi_diff_processed', 'real_duration',
        'note_density', 'black_key', 'chord'
    ]
    
    # 获取物理约束特征列
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
    
    # 8. 提取并合并特征
    X_features = []
    for i, row in df.iterrows():
        # 提取所有数值特征
        feature_values = np.array([float(row[col]) for col in feature_columns])
        X_features.append(feature_values)
    
    # 转换为numpy数组
    X = np.array(X_features)
    print(f"特征矩阵形状: {X.shape}")
    
    # 9. 处理无限值和NaN
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 10. 应用标准化
    # 检查特征维度是否与训练数据一致
    try:
        sample_scaler_data = scaler.mean_.shape[0]
        expected_features = sample_scaler_data
        print(f"训练数据特征维度：{expected_features}")
        
        # 检查是否匹配
        if X.shape[1] != expected_features:
            print(f"警告：特征维度不匹配！训练：{expected_features}，当前：{X.shape[1]}")
            
            # 调整特征维度以匹配
            if X.shape[1] < expected_features:
                # 填充
                padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                X = np.concatenate([X, padding], axis=1)
            else:
                # 裁剪
                X = X[:, :expected_features]
            
            print(f"调整后特征维度: {X.shape}")
    except Exception as e:
        print(f"检查特征维度时出错：{e}")
    
    # 标准化特征
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


def predict_fingering_physical_model(score_path, model_path='./results_physical_optimized/fingering_physical_model_best.pth', model_dir='.', num_physical_features=9):
    """使用物理特征增强型模型预测指法"""
    print(f"使用物理特征增强型模型分析乐谱: {score_path}")
    
    # 预处理数据
    score, X_seq, df = preprocess_score_data(score_path, model_dir)
    
    # 确定输入特征维度
    input_size = X_seq.shape[2]
    print(f"模型输入特征维度: {input_size}")
    
    # 加载模型
    try:
        # 检查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        # 加载模型权重
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 创建模型
        model = PhysicalEnhancedFingeringModel(
            input_size=input_size,
            hidden_size=128,
            num_classes=10,
            num_physical_features=num_physical_features,
            physical_weight=0.3,
            dropout=0.3
        )
        
        # 加载权重
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
        for i, x in enumerate(tqdm(X_seq)):
            x_tensor = torch.tensor(x).unsqueeze(0)
            
            # 提取物理特征
            physical_features = extract_physical_features(x_tensor, num_physical_features=num_physical_features)
            
            # 使用前向规划进行预测
            if hasattr(model, 'forward_planning'):
                # 更详细的预测，包含物理约束评估
                predicted_action = model.forward_planning(x_tensor)
                finger_idx = predicted_action.item()
                confidence_val = 0.9  # 前向规划不返回置信度，使用默认值
            else:
                # 标准预测
                output, combined_probs = model(x_tensor, physical_features)
                confidence, predicted = torch.max(combined_probs, dim=1)
                finger_idx = predicted.item()
                confidence_val = confidence.item()
            
            # 转换为实际指法
            try:
                # 根据指法编码器转换为实际指法值
                actual_fingering = fingering_encoder.inverse_transform([finger_idx])[0]
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
    
    # 优化预测的指法
    print("优化预测结果，修正不合理的连续指法...")
    
    # 导入指法优化器（如果有的话）
    try:
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
    except ImportError:
        print("未找到指法优化器，跳过优化步骤")
        optimized_fingerings = predicted_fingerings
    
    # 应用指法到乐谱
    modified_score = apply_predicted_fingering(score, optimized_fingerings)
    
    return modified_score, optimized_fingerings, confidence_scores


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用物理特征增强型模型预测钢琴指法')
    parser.add_argument('input_file', help='输入乐谱文件路径（MusicXML格式）')
    parser.add_argument('--model', default='./results_physical_optimized/fingering_physical_model_best.pth', 
                       help='模型文件路径')
    parser.add_argument('--model_dir', type=str, default='.', 
                       help='模型相关文件所在目录')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--num_physical_features', type=int, default=9,
                       help='物理特征数量')
    
    args = parser.parse_args()
    
    # 预测指法
    modified_score, fingerings, _ = predict_fingering_physical_model(
        args.input_file, 
        model_path=args.model,
        model_dir=args.model_dir,
        num_physical_features=args.num_physical_features
    )
    
    # 保存结果
    if args.output:
        output_path = args.output
    else:
        # 生成默认输出路径
        base_name = os.path.splitext(args.input_file)[0]
        output_path = f"{base_name}_physical_fingering.musicxml"
    
    # 保存带有指法的乐谱
    modified_score.write('musicxml', fp=output_path)
    print(f"带有物理约束指法的乐谱已保存到: {output_path}")


if __name__ == "__main__":
    main() 