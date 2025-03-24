#!/usr/bin/env python3
"""
稳定版钢琴指法预测模型训练
使用改进的架构和训练策略，解决类别不平衡问题
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import argparse
import time
from tqdm import tqdm


class FocalLoss(nn.Module):
    """
    焦点损失 (Focal Loss)
    修改标准交叉熵损失，增加对难分类样本的关注
    gamma: 聚焦参数，增加对错误分类的惩罚
    alpha: 类别权重，处理类别不平衡
    """
    def __init__(self, gamma=2, alpha=None, ignore_index=10, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha, 
            ignore_index=self.ignore_index, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SequenceDataset(Dataset):
    """
    序列数据集，创建序列窗口用于BiLSTM训练
    为每个样本创建一个前后文窗口
    """
    def __init__(self, X, y, window_size=5):
        self.X = X.astype(np.float32)  # 确保数据类型一致
        self.y = y
        self.window_size = window_size
        
        # 为序列数据添加前后文
        self.windowed_X, self.windowed_y = self._create_windows()
        
    def _create_windows(self):
        n_samples, n_features = self.X.shape
        padded_X = np.vstack([
            np.zeros((self.window_size, n_features)), 
            self.X, 
            np.zeros((self.window_size, n_features))
        ])
        
        windowed_X = []
        windowed_y = []
        
        # 为每个样本创建一个包含前后文的窗口
        for i in range(n_samples):
            window = padded_X[i:i+2*self.window_size+1]
            windowed_X.append(window)
            windowed_y.append(self.y[i])
        
        return np.array(windowed_X), np.array(windowed_y)
    
    def __len__(self):
        return len(self.windowed_y)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.windowed_X[idx], dtype=torch.float32)
        y = torch.tensor(self.windowed_y[idx], dtype=torch.long)
        return X, y


class CNNBiLSTMAttention(nn.Module):
    """
    融合CNN, BiLSTM和注意力机制的指法预测模型
    1. CNN提取局部特征
    2. BiLSTM处理序列上下文
    3. 注意力机制关注重要时间步
    4. 残差连接和规范化层提高稳定性
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNNBiLSTMAttention, self).__init__()
        
        # CNN特征提取层
        self.conv1 = nn.Conv1d(input_size, hidden_size//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size//2, hidden_size//2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size//2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size//2, 
            hidden_size=hidden_size//2,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True
        )
        
        # 注意力层
        self.attention = nn.Linear(hidden_size, 1)
        
        # 输出层
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_output = nn.Linear(hidden_size, num_classes)
        
        # 残差连接所需参数
        self.residual_conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # CNN特征提取
        x_cnn = x.transpose(1, 2)  # [batch, features, seq_len]
        x_cnn = self.conv1(x_cnn)
        x_cnn = F.relu(x_cnn)
        x_cnn = self.bn1(x_cnn)
        x_cnn = self.conv2(x_cnn)
        x_cnn = F.relu(x_cnn)
        x_cnn = self.pool(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)  # [batch, seq_len, hidden_size//2]
        
        # BiLSTM处理
        lstm_out, _ = self.lstm(x_cnn)  # [batch, seq_len, hidden_size]
        
        # 注意力机制
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 残差连接
        residual = self.residual_conv(x.transpose(1, 2))
        residual = residual.mean(dim=2)
        
        # 最终输出
        output = self.fc_hidden(context_vector)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)  # 残差连接
        output = self.fc_output(output)
        
        return output


def compute_class_weights(y_train):
    """计算类别权重以处理不平衡问题"""
    class_counts = np.bincount(y_train)
    n_samples = len(y_train)
    n_classes = len(class_counts)
    
    weights = n_samples / (n_classes * class_counts)
    return torch.FloatTensor(weights)


def load_data(df_path, window_size=5):
    """
    加载并预处理数据，创建序列窗口
    """
    print(f"从 {df_path} 加载数据...")
    
    try:
        with open(df_path, 'rb') as f:
            df = pickle.load(f)
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None, None, None, None, None, None
    
    print(f"加载了 {len(df)} 条记录")
    
    # 获取基本特征
    feature_columns = [
        'pitch_encoded', 'duration_encoded', 
        'hand_encoded', 'midi_number'
    ]
    
    # 提取特征和标签
    X = df[feature_columns].values
    y = df['fingering_encoded'].values
    
    # 打印类别分布
    class_counts = np.bincount(y)
    print("\n类别分布:")
    for i, count in enumerate(class_counts):
        if i < len(class_counts):
            print(f"类别 {i}: {count} 样本")
    
    # 划分训练集和验证集
    train_mask = df['train'] == 1
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[~train_mask]
    y_val = y[~train_mask]
    
    print(f"训练集: {len(X_train)} 样本, 验证集: {len(X_val)} 样本")
    
    # 创建序列数据集
    train_dataset = SequenceDataset(X_train, y_train, window_size=window_size)
    val_dataset = SequenceDataset(X_val, y_val, window_size=window_size)
    
    # 创建加权采样器
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[y_train]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(X_train),
        replacement=True
    )
    
    return train_dataset, val_dataset, sampler, X_train, y_train, X_val, y_val


def train_model(train_dataset, val_dataset, sampler, y_train, args):
    """
    训练模型，使用改进的训练策略和正则化技术
    """
    # 模型参数
    input_size = 4  # 基本特征数
    hidden_size = args.hidden_size
    num_classes = 10  # 0-9类别
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 
                          'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = CNNBiLSTMAttention(input_size, hidden_size, num_classes).to(device)
    
    # 损失函数和优化器
    class_weights = compute_class_weights(y_train).to(device)
    
    # 使用Focal Loss
    criterion = FocalLoss(
        gamma=args.focal_gamma,
        alpha=class_weights, 
        ignore_index=10
    )
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器 - CosineAnnealing更稳定
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5, 
        T_mult=2,
        eta_min=1e-6
    )
    
    # 训练参数
    num_epochs = args.epochs
    best_val_acc = 0
    patience = args.patience
    counter = 0
    
    # 记录训练过程
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'class_acc': {i: [] for i in range(num_classes)}
    }
    
    # 训练循环
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(output, 1)
            
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(target.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(), 
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # 学习率调整
        scheduler.step()
        
        # 计算训练指标
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, preds = torch.max(output, 1)
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        # 计算验证指标
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 计算每个类别的准确率
        val_targets_np = np.array(val_targets)
        val_preds_np = np.array(val_preds)
        
        per_class_acc = []
        for i in range(num_classes):
            idx = (val_targets_np == i)
            if np.sum(idx) > 0:
                acc = np.mean(val_preds_np[idx] == i)
                history['class_acc'][i].append(acc)
                per_class_acc.append(f"Class {i}: {acc:.4f}")
            else:
                history['class_acc'][i].append(0)
                per_class_acc.append(f"Class {i}: N/A")
        
        # 打印本轮结果
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print("每类准确率:", ", ".join(per_class_acc))
        
        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, args.model_path)
            
            print(f"✓ 保存最佳模型 (验证准确率: {val_acc:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"触发早停机制，停止训练！(patience={patience})")
                break
    
    # 计算训练总时间
    total_time = time.time() - start_time
    print(f"\n训练完成，总时间: {total_time:.2f}秒")
    
    return model, history, best_val_acc


def evaluate_model(model, X_val, y_val, window_size=5, device='cpu'):
    """
    评估模型性能，生成混淆矩阵和可视化
    """
    # 创建验证数据集和加载器
    val_dataset = SequenceDataset(X_val, y_val, window_size=window_size)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=4
    )
    
    # 切换到评估模式
    model.eval()
    
    all_preds = []
    
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_val, all_preds)
    
    # 计算每个类别的准确率
    per_class_acc = []
    for i in range(len(np.unique(y_val))):
        idx = (y_val == i)
        if np.sum(idx) > 0:
            acc = np.mean(np.array(all_preds)[idx] == i)
            per_class_acc.append(f"Class {i}: {acc:.4f}")
        else:
            per_class_acc.append(f"Class {i}: N/A")
    
    return cm, per_class_acc


def plot_results(history, cm):
    """
    绘制训练结果图表：
    1. 训练和验证的损失曲线
    2. 训练和验证的准确率曲线
    3. 每个类别的准确率曲线
    4. 混淆矩阵
    """
    num_classes = len(history['class_acc'])
    
    # 创建包含4个子图的图表
    plt.figure(figsize=(20, 15))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制每个类别的准确率曲线
    plt.subplot(2, 2, 3)
    for i in range(num_classes):
        plt.plot(history['class_acc'][i], label=f'Class {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Per-class Accuracy')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    
    # 绘制混淆矩阵
    plt.subplot(2, 2, 4)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, np.arange(num_classes))
    plt.yticks(tick_marks, np.arange(num_classes))
    
    # 添加数字标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("训练结果图表已保存为 'training_results.png'")


def predict_with_model(model, input_data, window_size=5, device='cpu'):
    """使用训练好的模型进行指法预测"""
    # 创建适合模型输入的窗口数据
    seq_dataset = SequenceDataset(input_data, np.zeros(len(input_data)), window_size)
    
    # 创建数据加载器
    loader = DataLoader(seq_dataset, batch_size=64, shuffle=False)
    
    # 预测
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    
    return np.array(predictions)


def main():
    parser = argparse.ArgumentParser(description="训练稳定版钢琴指法预测模型")
    
    # 数据参数
    parser.add_argument('--df_path', type=str, default='df.pkl',
                        help='DataFrame pickle文件路径')
    parser.add_argument('--window_size', type=int, default=5,
                        help='输入序列窗口大小')
    
    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='隐藏层大小')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss的gamma参数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减系数')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停轮数')
    
    # 输出参数
    parser.add_argument('--model_path', type=str, default='stabilized_fingering_model.pth',
                        help='模型保存路径')
    
    args = parser.parse_args()
    
    # 加载数据
    train_dataset, val_dataset, sampler, X_train, y_train, X_val, y_val = load_data(
        args.df_path, window_size=args.window_size
    )
    
    if train_dataset is None:
        print("数据加载失败，终止训练")
        return
    
    # 训练模型
    model, history, best_val_acc = train_model(
        train_dataset, val_dataset, sampler, y_train, args
    )
    
    # 使用最佳模型
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 
                          'cpu')
    
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估模型
    cm, per_class_acc = evaluate_model(
        model, X_val, y_val, window_size=args.window_size, device=device
    )
    
    # 绘制结果
    plot_results(history, cm)
    
    # 打印最终结果
    print(f"\n最佳验证准确率: {best_val_acc:.4f}")
    print("每类准确率:")
    for acc in per_class_acc:
        print(acc)
    
    print(f"\n✅ 成功完成! 最终模型已保存至 '{args.model_path}'")

if __name__ == "__main__":
    main() 