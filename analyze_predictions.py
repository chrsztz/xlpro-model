#!/usr/bin/env python3
import pandas as pd
import numpy as np

# 读取置信度文件
df = pd.read_csv('output_example_bilstm_cnn_confidences.csv')

# 计算各指法的统计信息
fingering_stats = df.groupby('fingering').agg({
    'confidence': ['count', 'mean', 'std', 'min', 'max']
}).reset_index()

# 重命名列
fingering_stats.columns = ['fingering', 'count', 'mean_confidence', 'std_confidence', 'min_confidence', 'max_confidence']

# 按指法数量降序排列
fingering_stats = fingering_stats.sort_values('count', ascending=False)

# 计算总体统计信息
total_count = len(df)
mean_confidence = df['confidence'].mean()
median_confidence = df['confidence'].median()

# 打印结果
print('指法分布统计：')
for _, row in fingering_stats.iterrows():
    print(f'指法 {row["fingering"]}: {row["count"]} 个音符 ({row["count"]/total_count*100:.2f}%), ' +
          f'平均置信度: {row["mean_confidence"]:.4f}, 最小值: {row["min_confidence"]:.4f}, 最大值: {row["max_confidence"]:.4f}')

print(f'\n总体统计：')
print(f'总音符数: {total_count}')
print(f'平均置信度: {mean_confidence:.4f}')
print(f'中位数置信度: {median_confidence:.4f}')

# 打印置信度分布
confidence_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
confidence_counts = pd.cut(df['confidence'], bins=confidence_bins).value_counts().sort_index()
print('\n置信度分布：')
for interval, count in confidence_counts.items():
    print(f'{interval}: {count} 个音符 ({count/total_count*100:.2f}%)')

# 比较与其他模型的预测结果（如果存在）
try:
    df_bilstm = pd.read_csv('output_example_with_fingering_confidences.csv')
    
    # 计算预测一致性
    merged_df = pd.merge(df, df_bilstm, left_index=True, right_index=True, suffixes=('_cnn', '_bilstm'))
    
    # 计算预测一致和不一致的数量
    consistent_count = (merged_df['fingering_cnn'] == merged_df['fingering_bilstm']).sum()
    inconsistent_count = len(merged_df) - consistent_count
    
    print('\n模型比较：')
    print(f'一致预测数: {consistent_count} ({consistent_count/len(merged_df)*100:.2f}%)')
    print(f'不一致预测数: {inconsistent_count} ({inconsistent_count/len(merged_df)*100:.2f}%)')
    
    # 置信度比较
    print(f'CNN模型平均置信度: {merged_df["confidence_cnn"].mean():.4f}')
    print(f'BiLSTM模型平均置信度: {merged_df["confidence_bilstm"].mean():.4f}')
    
except FileNotFoundError:
    print('\n没有找到其他模型的预测结果进行比较。') 