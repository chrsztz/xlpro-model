#!/usr/bin/env python3
"""
诊断工具: 测试使用简化模型的钢琴指法预测，重点关注编码问题
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm
import argparse

class SimpleFingeringDataset(Dataset):
    """简化的指法数据集类，不使用序列窗口"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleModel(nn.Module):
    """简化的模型，使用单层前馈网络而非复杂的BiLSTM"""
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 添加Dropout防止过拟合
        self.bn = nn.BatchNorm1d(hidden_size)  # 添加批归一化提高稳定性
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 对于非序列数据
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def compute_class_weights(y_train):
    """计算类别权重以处理数据不平衡问题"""
    class_counts = np.bincount(y_train)
    n_samples = len(y_train)
    n_classes = len(class_counts)
    
    weights = n_samples / (n_classes * class_counts)
    return torch.FloatTensor(weights)

def create_weighted_sampler(y_train):
    """创建加权采样器，确保每批次中类别均衡"""
    class_counts = np.bincount(y_train)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_train),
        replacement=True
    )
    return sampler

def load_data(df_path):
    """加载数据并进行必要的处理"""
    print(f"从 {df_path} 加载数据...")
    
    try:
        with open(df_path, 'rb') as f:
            df = pickle.load(f)
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None, None, None, None
    
    print(f"加载了 {len(df)} 条记录")
    
    # 获取基本特征列（非序列特征）
    base_features = [
        'pitch_encoded', 'duration_encoded', 
        'hand_encoded', 'midi_number'
    ]
    
    # 检查是否有这些列
    missing_cols = [col for col in base_features if col not in df.columns]
    if missing_cols:
        print(f"警告: 缺少以下列: {missing_cols}")
        base_features = [col for col in base_features if col in df.columns]
    
    # 创建更丰富的特征集
    X = df[base_features].values
    y = df['fingering_encoded'].values
    
    # 显示类别分布
    class_distribution = np.bincount(y)
    print("原始类别分布:")
    for i, count in enumerate(class_distribution):
        print(f"类别 {i}: {count} 个样本")
    
    # 划分训练和验证集
    train_mask = df['train'] == 1
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[~train_mask], y[~train_mask]
    
    print(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")
    
    # 检查训练集每个类别的样本数
    train_dist = np.bincount(y_train)
    print("\n训练集类别分布:")
    for i, count in enumerate(train_dist):
        if i < len(train_dist):
            print(f"类别 {i}: {count} 样本")
    
    return X_train, y_train, X_val, y_val

def visualize_encodings(df_path):
    """可视化指法编码分布"""
    try:
        with open(df_path, 'rb') as f:
            df = pickle.load(f)
            
        # 统计原始指法
        finger_counts = df['finger_number'].value_counts().sort_index()
        
        # 统计编码后的指法
        encoded_counts = df['fingering_encoded'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        finger_counts.plot(kind='bar', color='skyblue')
        plt.title('原始指法分布')
        plt.xlabel('指法 (-5 ~ 5)')
        plt.ylabel('计数')
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        encoded_counts.plot(kind='bar', color='salmon')
        plt.title('编码后的指法分布')
        plt.xlabel('编码类别 (0-9)')
        plt.ylabel('计数')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fingering_encoding_visualization.png')
        print("编码分布图已保存为 'fingering_encoding_visualization.png'")
        
        # 打印映射关系
        print("\n指法编码映射关系:")
        for original in sorted(df['finger_number'].unique()):
            encoded = df[df['finger_number'] == original]['fingering_encoded'].iloc[0]
            count = len(df[df['finger_number'] == original])
            print(f"原始指法: {original:2d} -> 编码类别: {encoded:2d} (样本数: {count})")
            
    except Exception as e:
        print(f"可视化编码分布时出错: {e}")

def train_and_evaluate(X_train, y_train, X_val, y_val):
    """训练简单模型并评估性能"""
    # 模型参数
    input_size = X_train.shape[1]
    hidden_size = 128
    num_classes = len(np.unique(y_train))
    
    # 创建数据集和数据加载器
    train_dataset = SimpleFingeringDataset(X_train, y_train)
    val_dataset = SimpleFingeringDataset(X_val, y_val)
    
    # 使用加权采样器处理不平衡问题
    sampler = create_weighted_sampler(y_train)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256,
        sampler=sampler, 
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=2
    )
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 
                          'cpu')
    print(f"使用设备: {device}")
    
    model = SimpleModel(input_size, hidden_size, num_classes).to(device)
    
    # 计算类别权重并创建损失函数
    class_weights = compute_class_weights(y_train)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=10)
    
    # 使用带有权重衰减的Adam优化器减少过拟合
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 训练参数
    num_epochs = 30
    best_val_acc = 0
    patience = 10
    counter = 0
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    class_accs = {i: [] for i in range(num_classes)}
    
    # 训练循环
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        # 训练一个epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # 计算训练指标
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_targets, all_preds)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())
        
        # 计算验证指标
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_targets, all_val_preds)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 计算每个类别的准确率
        val_targets = np.array(all_val_targets)
        val_preds = np.array(all_val_preds)
        
        per_class_acc = []
        for i in range(num_classes):
            idx = (val_targets == i)
            if np.sum(idx) > 0:
                acc = np.mean(val_preds[idx] == i)
                class_accs[i].append(acc)
                per_class_acc.append(f"Class {i}: {acc:.4f}")
            else:
                class_accs[i].append(0)
                per_class_acc.append(f"Class {i}: N/A")
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print("每类准确率:", ", ".join(per_class_acc))
        
        # 学习率调整
        scheduler.step(val_acc)
        
        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), 'best_diagnostic_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"触发早停机制，停止训练！(patience={patience})")
                break
    
    # 训练结束，计算总时间
    total_time = time.time() - start_time
    print(f"\n训练完成，总时间: {total_time:.2f}秒")
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_diagnostic_model.pth'))
    
    # 评估最佳模型
    model.eval()
    y_pred = []
    
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 绘制训练曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    
    # 绘制每类准确率曲线
    plt.subplot(1, 3, 3)
    for i in range(num_classes):
        if len(class_accs[i]) > 0:  # 确保有数据
            plt.plot(class_accs[i], label=f'Class {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Per-class Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('diagnostic_training_curves.png')
    print("训练曲线已保存为 'diagnostic_training_curves.png'")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵已保存为 'confusion_matrix.png'")
    
    # 输出最佳性能
    print(f"\n最佳验证准确率: {best_val_acc:.4f}")
    
    # 返回最佳指标
    return best_val_acc, model

def main():
    parser = argparse.ArgumentParser(description="钢琴指法诊断工具")
    parser.add_argument('--df_path', type=str, default='df.pkl', 
                        help='DataFrame pickle 文件路径')
    parser.add_argument('--visualize_only', action='store_true',
                        help='只可视化编码分布，不训练模型')
    
    args = parser.parse_args()
    
    # 可视化编码
    visualize_encodings(args.df_path)
    
    if not args.visualize_only:
        # 加载并训练模型
        X_train, y_train, X_val, y_val = load_data(args.df_path)
        
        if X_train is not None:
            train_and_evaluate(X_train, y_train, X_val, y_val)
    
if __name__ == "__main__":
    main() 