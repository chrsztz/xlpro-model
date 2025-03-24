#!/usr/bin/env python3
"""
简单诊断工具: 检查钢琴指法编码分布，不需要PyTorch
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse
from data_utils import CustomFingeringEncoder

def visualize_encodings(df_path):
    """可视化指法编码分布"""
    try:
        print(f"从 {df_path} 加载数据...")
        with open(df_path, 'rb') as f:
            df = pickle.load(f)
            
        print(f"加载了 {len(df)} 条记录")
            
        # 统计原始指法
        finger_counts = df['finger_number'].value_counts().sort_index()
        
        # 统计编码后的指法
        encoded_counts = df['fingering_encoded'].value_counts().sort_index()
        
        # 打印统计
        print("\n原始指法统计:")
        print(finger_counts)
        
        print("\n编码后的指法分布:")
        print(encoded_counts)
        
        # 可视化
        plt.figure(figsize=(16, 10))
        
        plt.subplot(2, 2, 1)
        finger_counts.plot(kind='bar', color='skyblue')
        plt.title('原始指法分布 (Raw Fingering Distribution)')
        plt.xlabel('指法 (-5 ~ 5)')
        plt.ylabel('计数')
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(2, 2, 2)
        encoded_counts.plot(kind='bar', color='salmon')
        plt.title('编码后的指法分布 (Encoded Fingering Distribution)')
        plt.xlabel('编码类别 (0-9)')
        plt.ylabel('计数')
        plt.grid(axis='y', alpha=0.3)
        
        # 检查左手指法编码
        left_hand_fingers = [i for i in finger_counts.index if i < 0]
        left_hand_counts = finger_counts[left_hand_fingers] if left_hand_fingers else pd.Series()
        
        plt.subplot(2, 2, 3)
        if not left_hand_counts.empty:
            left_hand_counts.plot(kind='bar', color='lightgreen')
            plt.title('左手指法分布 (Left Hand Fingering)')
            plt.xlabel('左手指法 (-5 ~ -1)')
            plt.ylabel('计数')
            plt.grid(axis='y', alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No left hand fingerings found!', 
                     ha='center', va='center', fontsize=14, color='red')
            plt.title('Left Hand Fingering - Missing Data')
            plt.axis('off')
        
        # 检查左手指法编码是否正确映射到类别0-4
        left_encoded_classes = []
        for i in range(-5, 0):
            if i in df['finger_number'].values:
                encoded = df[df['finger_number'] == i]['fingering_encoded'].iloc[0]
                left_encoded_classes.append((i, encoded))
        
        plt.subplot(2, 2, 4)
        if left_encoded_classes:
            indices, values = zip(*left_encoded_classes)
            plt.bar(range(len(indices)), values, color='lightcoral')
            plt.xticks(range(len(indices)), indices)
            plt.title('左手指法的编码值 (Left Hand Encoding)')
            plt.xlabel('原始指法')
            plt.ylabel('编码值 (应为0-4)')
            plt.grid(axis='y', alpha=0.3)
            
            # 检查是否正确映射
            correct_mapping = all(0 <= enc <= 4 for _, enc in left_encoded_classes)
            if not correct_mapping:
                plt.text(0.5, 0.9, 'WARNING: Incorrect encoding!', 
                         ha='center', transform=plt.gca().transAxes, 
                         fontsize=14, color='red')
        else:
            plt.text(0.5, 0.5, 'Cannot check left hand encoding - no data!', 
                     ha='center', va='center', fontsize=14, color='red')
            plt.title('Left Hand Encoding Check - Missing Data')
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig('fingering_encoding_check.png', dpi=300)
        print("编码分布图已保存为 'fingering_encoding_check.png'")
        
        # 打印映射关系
        print("\n指法编码映射关系:")
        for original in sorted(df['finger_number'].unique()):
            encoded = df[df['finger_number'] == original]['fingering_encoded'].iloc[0]
            count = len(df[df['finger_number'] == original])
            print(f"原始指法: {original:2d} -> 编码类别: {encoded:2d} (样本数: {count})")
            
        # 创建一个参考编码器并验证映射
        encoder = CustomFingeringEncoder()
        print("\n参考编码映射 (从CustomFingeringEncoder):")
        for k, v in sorted(encoder.mapping.items()):
            print(f"指法: {k:2d} -> 编码: {v:2d}")
            
        # 检查是否有缺失的指法类别
        missing_fingers = set(range(-5, 6)) - set(df['finger_number'].unique())
        if missing_fingers:
            print(f"\n警告: 数据中缺少以下指法: {missing_fingers}")
            
        # 检查是否有缺失的编码类别
        missing_encoded = set(range(10)) - set(df['fingering_encoded'].unique())
        if missing_encoded and missing_encoded != {10}:  # 10是忽略的类别
            print(f"\n警告: 数据中缺少以下编码类别: {missing_encoded}")
            
        # 按手指统计
        hand_stats = df['hand'].value_counts()
        print("\n左右手统计:")
        print(hand_stats)
        
        # 检查左手指法正确性
        left_hand_df = df[df['hand'] == 'left']
        incorrect_left = left_hand_df[left_hand_df['finger_number'] >= 0]
        if len(incorrect_left) > 0:
            print(f"\n⚠️ 警告: 发现 {len(incorrect_left)} 条左手指法不是负数!")
            
        # 另一种检查方式 - 按手和指法交叉统计
        print("\n按手和指法交叉统计:")
        hand_finger_cross = pd.crosstab(df['hand'], df['finger_number'])
        print(hand_finger_cross)
            
    except Exception as e:
        print(f"可视化编码分布时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="钢琴指法编码诊断工具")
    parser.add_argument('--df_path', type=str, default='df.pkl', 
                        help='DataFrame pickle 文件路径')
    
    args = parser.parse_args()
    
    # 可视化编码
    visualize_encodings(args.df_path)
    
if __name__ == "__main__":
    main() 