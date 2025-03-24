#!/usr/bin/env python3
"""
增强版钢琴指法预测模型 - 基于处理过的数据
使用预处理和增强后的NumPy数据文件，而非原始df.pkl
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import pickle
import argparse
import time
from tqdm import tqdm


class ClassBalancedLoss(nn.Module):
    """
    类平衡损失函数，结合Focal Loss和类权重
    专门针对严重类不平衡的场景
    """
    def __init__(self, beta=0.9999, gamma=2.0, samples_per_class=None, num_classes=10, ignore_index=10):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        if samples_per_class is not None:
            # 计算有效样本数
            effective_num = 1.0 - np.power(beta, samples_per_class)
            weights = (1.0 - beta) / np.array(effective_num)
            # 归一化权重
            weights = weights / np.sum(weights) * num_classes
            self.weights = torch.tensor(weights).float()
        else:
            self.weights = None
            
    def forward(self, logits, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(
            logits, targets, 
            weight=self.weights.to(logits.device) if self.weights is not None else None,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # 计算概率和焦点权重
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # 最终损失
        loss = focal_weight * ce_loss
        return loss.mean()


class ProcessedSequenceDataset(Dataset):
    """
    序列数据集，专门处理已经预处理为3D的数据
    (samples, sequence_length, features)
    """
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return X, y


class EnhancedFingeringModel(nn.Module):
    """
    增强版指法预测模型
    多头注意力+ResNet风格的残差连接+Dropout+BatchNorm+LayerNorm
    专门针对3D输入格式 (batch_size, sequence_length, features)
    """
    def __init__(self, input_size, hidden_size, num_classes=10, num_heads=4):
        super(EnhancedFingeringModel, self).__init__()
        
        # CNN特征提取
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size//2,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True
        )
        
        # 多头自注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads,
            dropout=0.1
        )
        
        # 输出层
        self.global_attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size//2, num_classes)
        )
        
        # 分类器 - 单独处理每个类
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_classes)
        ])
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # CNN特征提取
        x_cnn = x.transpose(1, 2)  # [batch, features, seq_len]
        x_cnn = self.conv_layers(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)  # [batch, seq_len, hidden_size]
        
        # BiLSTM处理
        lstm_out, _ = self.lstm(x_cnn)  # [batch, seq_len, hidden_size]
        
        # 多头自注意力
        lstm_out_t = lstm_out.transpose(0, 1)  # [seq_len, batch, hidden_size]
        attn_out, _ = self.multihead_attn(lstm_out_t, lstm_out_t, lstm_out_t)
        attn_out = attn_out.transpose(0, 1)  # [batch, seq_len, hidden_size]
        
        # 残差连接和层归一化
        combined = lstm_out + attn_out  # 残差连接
        combined = F.layer_norm(combined, [combined.size(-1)])  # 层归一化
        
        # 全局注意力
        attn_weights = self.global_attn(combined)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * combined, dim=1)
        
        # 主输出 - 共享特征
        main_logits = self.fc_layers(context)
        
        # 单独的分类器
        class_logits = torch.cat([cls(context) for cls in self.classifiers], dim=1)
        
        # 混合主输出和专用分类器 (主要使用主输出，但给予专用分类器一些权重)
        mixed_logits = main_logits * 0.7 + class_logits * 0.3
        
        return mixed_logits


def load_processed_data(args):
    """
    加载预处理好的NumPy数据文件 - 已经是3D格式
    """
    print("加载预处理的数据文件...")
    
    try:
        # 加载训练集和验证集
        X_train = np.load(args.train_features_path)
        y_train = np.load(args.train_labels_path)
        X_val = np.load(args.val_features_path)
        y_val = np.load(args.val_labels_path)
        
        print(f"加载了 {len(X_train)} 个训练样本和 {len(X_val)} 个验证样本")
        
        # 打印特征维度
        print(f"数据形状: {X_train.shape}")
        
        # 打印类别分布
        class_counts = np.bincount(y_train)
        print("\n类别分布:")
        for i, count in enumerate(class_counts):
            if i < len(class_counts):
                print(f"类别 {i}: {count} 样本 ({count/len(y_train)*100:.2f}%)")
        
        # 创建序列数据集 - 使用已经是3D格式的数据
        train_dataset = ProcessedSequenceDataset(X_train, y_train)
        val_dataset = ProcessedSequenceDataset(X_val, y_val)
        
        # 计算采样权重
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[y_train]
        
        return train_dataset, val_dataset, sample_weights, X_train, y_train, X_val, y_val, class_counts
    
    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None


def evaluate_model(model, dataloader, criterion, device):
    """评估模型性能，详细计算每个类别的指标"""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算整体准确率
    accuracy = accuracy_score(all_targets, all_preds)
    
    # 计算每个类别的准确率
    class_accuracies = {}
    for cls in range(10):  # 假设有10个类别
        cls_mask = np.array(all_targets) == cls
        if np.sum(cls_mask) > 0:  # 确保有该类的样本
            cls_acc = accuracy_score(
                np.array(all_targets)[cls_mask], 
                np.array(all_preds)[cls_mask]
            )
            class_accuracies[cls] = cls_acc
        else:
            class_accuracies[cls] = 0.0
    
    return total_loss / len(all_targets), accuracy, class_accuracies, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description="增强版钢琴指法预测模型训练 - 预处理数据版")
    
    # 数据路径参数
    parser.add_argument('--train_features_path', type=str, default='X_train.npy', 
                        help='训练特征文件路径')
    parser.add_argument('--train_labels_path', type=str, default='y_train.npy', 
                        help='训练标签文件路径')
    parser.add_argument('--val_features_path', type=str, default='X_val.npy', 
                        help='验证特征文件路径')
    parser.add_argument('--val_labels_path', type=str, default='y_val.npy', 
                        help='验证标签文件路径')
    
    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=128, 
                        help='隐藏层大小')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0003, 
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='权重衰减系数')
    parser.add_argument('--dropout', type=float, default=0.3, 
                        help='Dropout比例')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='训练轮数')
    parser.add_argument('--patience', type=int, default=10, 
                        help='早停轮数')
    parser.add_argument('--model_path', type=str, default='fingering_processed_enhanced_model.pth', 
                        help='模型保存路径')
    parser.add_argument('--use_mixup', action='store_true',
                        help='是否使用Mixup数据增强')
    parser.add_argument('--beta', type=float, default=0.9999,
                        help='类平衡损失函数的beta参数')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='类平衡损失函数的gamma参数')
    
    args = parser.parse_args()
    
    # 加载数据
    train_dataset, val_dataset, sample_weights, X_train, y_train, X_val, y_val, class_counts = load_processed_data(args)
    
    if train_dataset is None:
        print("数据加载失败，退出训练")
        return
    
    # 准备数据加载器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(X_train),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else 
                          "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型 - 对于3D输入 (batch_size, seq_len, features)
    # 输入特征维度是每个时间步的特征数
    input_size = X_train.shape[2]  # features
    model = EnhancedFingeringModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_classes=10  # 0-9类别
    ).to(device)
    
    # 创建损失函数
    criterion = ClassBalancedLoss(
        samples_per_class=class_counts,
        num_classes=10,
        beta=args.beta,
        gamma=args.gamma
    )
    
    # 两阶段优化器策略
    # 第一阶段: 基础参数
    base_params = list(model.conv_layers.parameters()) + list(model.lstm.parameters())
    # 第二阶段: 分类器参数
    classifier_params = (list(model.global_attn.parameters()) + 
                       list(model.fc_layers.parameters()) +
                       [p for m in model.classifiers for p in m.parameters()])
    
    # 为分类器参数使用更高的学习率
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': args.learning_rate},
        {'params': classifier_params, 'lr': args.learning_rate * 2.0}
    ], weight_decay=args.weight_decay)
    
    # 学习率调度器: 余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,  # 初始周期长度
        T_mult=2,  # 每次重启后周期长度倍增
        eta_min=1e-6  # 最小学习率
    )
    
    # 训练模型
    print("\n开始训练...")
    start_time = time.time()
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # 每个epoch记录每个类别的准确率
        class_corrects = {i: 0 for i in range(10)}
        class_totals = {i: 0 for i in range(10)}
        
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Mixup数据增强
            if args.use_mixup and epoch > 5:
                lam = np.random.beta(0.2, 0.2)
                index = torch.randperm(inputs.size(0)).to(device)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                inputs = mixed_inputs
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # 更新每个类别的统计
            for c in range(10):
                class_mask = (targets == c)
                class_totals[c] += class_mask.sum().item()
                class_corrects[c] += ((predicted == targets) & class_mask).sum().item()
            
            # 更新进度条
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f"{loss.item():.2f}", 'lr': f"{lr:.6f}"})
        
        # 更新学习率
        scheduler.step()
        
        # 计算训练指标
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 评估验证集
        model.eval()
        val_loss, val_acc, class_accs, all_preds, all_targets = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 打印结果
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 打印每个类别的验证准确率
        print("每类准确率:", end=" ")
        for c in range(10):
            print(f"Class {c}: {class_accs[c]:.4f}", end=", ")
        print()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), args.model_path)
            print(f"✓ 保存最佳模型 (验证准确率: {best_val_acc:.4f})")
            
            # 保存混淆矩阵
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            plt.savefig('confusion_matrix_processed.png', dpi=300)
            plt.close()
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"触发早停机制，停止训练！(patience={args.patience})")
                break
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"\n训练完成，总时间: {total_time:.2f}秒")
    
    # 绘制训练和验证损失/准确率曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_results_processed.png')
    plt.close()
    
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(args.model_path))
    _, val_acc, class_accs, all_preds, all_targets = evaluate_model(model, val_loader, criterion, device)
    
    print(f"\n最佳验证准确率: {best_val_acc:.4f}")
    print("每类准确率:")
    for c in range(10):
        print(f"Class {c}: {class_accs[c]:.4f}")
    
    # 输出分类报告
    target_names = [f"Class {i}" for i in range(10)]
    report = classification_report(all_targets, all_preds, target_names=target_names)
    print("\n分类报告:")
    print(report)
    
    # 保存分类报告到文件
    with open("classification_report_processed.txt", "w") as f:
        f.write(report)
    
    print(f"\n✅ 成功完成! 使用处理后的数据训练的最终模型已保存至 '{args.model_path}'")


if __name__ == "__main__":
    main() 