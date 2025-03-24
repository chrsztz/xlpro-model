#!/usr/bin/env python3
"""
检查指法编码分布的脚本
用于验证我们的修复是否正确
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
from data_utils import CustomFingeringEncoder

def load_pickle(filename):
    """加载pickle文件"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    # 检查数据文件是否存在
    if not os.path.exists('df.pkl'):
        print("Error: df.pkl not found. Please run dataset_prep.py first.")
        sys.exit(1)
    
    try:
        # 加载数据
        print("Loading data...")
        df = load_pickle('df.pkl')
        print(f"数据集大小: {len(df)} 条记录")
        
        # 检查指法分布
        print("\n原始指法分布:")
        finger_counts = df['finger_number'].value_counts().sort_index()
        print(finger_counts)
        
        print("\n编码后的指法分布:")
        encoded_counts = df['fingering_encoded'].value_counts().sort_index()
        print(encoded_counts)
        
        # 可视化指法分布
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        finger_counts.plot(kind='bar', color='skyblue')
        plt.title('原始指法分布')
        plt.xlabel('指法')
        plt.ylabel('计数')
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        encoded_counts.plot(kind='bar', color='salmon')
        plt.title('编码后的指法分布')
        plt.xlabel('编码类别')
        plt.ylabel('计数')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fingering_distribution.png')
        print("\n指法分布图已保存为 'fingering_distribution.png'")
        
        # 打印编码映射
        encoder = CustomFingeringEncoder()
        print("\n指法编码映射:")
        for original, encoded in encoder.mapping.items():
            print(f"原始指法: {original:2d} -> 编码类别: {encoded:2d}")
            
        # 验证编码是否均匀
        if 0 not in encoded_counts or 1 not in encoded_counts:
            print("\n⚠️ 警告: 左手指法编码不正确，类别0或1缺失!")
        if 5 not in encoded_counts or 6 not in encoded_counts:
            print("\n⚠️ 警告: 右手指法编码不正确，类别5或6缺失!")
        
        # 检查编码范围
        if set(encoded_counts.index) - set(range(10)) - {10}:
            print(f"\n⚠️ 警告: 意外的类别编码: {set(encoded_counts.index) - set(range(10)) - {10}}")
        
        # 总结
        left_hand_count = sum(finger_counts.get(i, 0) for i in range(-5, 0))
        right_hand_count = sum(finger_counts.get(i, 0) for i in range(1, 6))
        
        print(f"\n总结:")
        print(f"- 左手指法数量: {left_hand_count}")
        print(f"- 右手指法数量: {right_hand_count}")
        print(f"- 未标注指法(0): {finger_counts.get(0, 0)}")
        
        # 分析训练/验证集划分
        print("\n数据集划分:")
        train_df = df[df['train'] == 1]
        val_df = df[df['train'] == 0]
        print(f"- 训练集: {len(train_df)} 条记录 ({len(train_df)/len(df):.1%})")
        print(f"- 验证集: {len(val_df)} 条记录 ({len(val_df)/len(df):.1%})")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 