#!/usr/bin/env python
# 训练物理约束增强的指法预测模型

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from models import PhysicalEnhancedFingeringModel
from data_utils import load_pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Train Physical-Enhanced Fingering Model')
    parser.add_argument('--input_size', type=int, default=None,
                        help='Input feature dimension')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden layer size')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of output classes')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--use_augmented', action='store_true',
                        help='Use augmented data')
    parser.add_argument('--physical_weight', type=float, default=0.3,
                        help='Weight for physical constraint loss')
    parser.add_argument('--num_physical_features', type=int, default=12,
                        help='Number of physical constraint features')
    parser.add_argument('--output_dir', type=str, default='results_physical',
                        help='Directory to save results')
    return parser.parse_args()

def extract_physical_features(X, num_physical_features=9):
    """从输入特征中提取关键物理约束特征
    
    智能选择最重要的几个物理特征，为模型提供更有针对性的约束信息
    
    Args:
        X: 输入特征序列 [batch_size, seq_len, input_size]
        num_physical_features: 提取的物理特征数量
        
    Returns:
        key_physical_features: 关键物理特征 [batch_size, num_physical_features]
    """
    batch_size = X.shape[0]
    
    # 假设基础特征占据前10个维度，物理特征位于向量的后部分
    if X.shape[2] <= 10:
        # 处理特征维度不足的情况
        print(f"警告: 输入特征维度({X.shape[2]})不足，无法提取物理特征")
        return torch.zeros(batch_size, num_physical_features, device=X.device)
    
    # 使用最后一个时间步，选择最重要的物理约束特征
    # 根据特征含义选择关键指标
    last_timestep = X[:, -1, :]
    
    # 关键物理特征索引（示例索引，需要根据实际特征顺序调整）
    key_indices = []
    
    # 1. 伸展率相关特征（选择3个）
    stretch_indices = [i for i in range(10, X.shape[2]) if i % 7 == 0]  # 示例规则，选择一些与伸展相关的特征
    key_indices.extend(stretch_indices[:3])
    
    # 2. 交叉指法特征（选择2个）
    cross_indices = [i for i in range(10, X.shape[2]) if i % 7 == 1]    # 示例规则，选择一些与交叉相关的特征
    key_indices.extend(cross_indices[:2])
    
    # 3. 指法自然度特征（选择2个）
    natural_indices = [i for i in range(10, X.shape[2]) if i % 7 == 2]  # 指法自然度特征
    key_indices.extend(natural_indices[:2])
    
    # 4. 指法强度特征（选择2个）
    strength_indices = [i for i in range(10, X.shape[2]) if i % 7 == 3] # 指法强度/难度特征
    key_indices.extend(strength_indices[:2])
    
    # 确保我们有足够的索引
    while len(key_indices) < num_physical_features and len(key_indices) < X.shape[2] - 10:
        remaining = set(range(10, X.shape[2])) - set(key_indices)
        if not remaining:
            break
        key_indices.append(min(remaining))
    
    # 选择关键特征
    if len(key_indices) >= num_physical_features:
        key_indices = key_indices[:num_physical_features]
        key_features = last_timestep[:, key_indices]
    else:
        # 特征不足时，填充到所需数量
        key_features = torch.zeros(batch_size, num_physical_features, device=X.device)
        if key_indices:
            key_features[:, :len(key_indices)] = last_timestep[:, key_indices]
    
    return key_features

class PhysicalConstraintLoss(nn.Module):
    """物理约束损失函数，重点惩罚不符合人体工程学的指法"""
    def __init__(self, num_classes=10):
        super(PhysicalConstraintLoss, self).__init__()
        self.base_loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        
    def forward(self, logits, targets, physical_features):
        # 基础分类损失
        base_loss = self.base_loss(logits, targets)
        
        # 从物理特征中提取不同的约束类型
        # 假设物理特征的顺序是:
        # [伸展特征, 交叉特征, 指法自然度特征, 强度特征]
        if physical_features.size(1) >= 4:
            # 提取各类约束特征
            stretch_features = physical_features[:, 0]  # 伸展率
            cross_features = physical_features[:, 1]    # 交叉难度
            natural_violation = physical_features[:, 2] # 指法自然度违反
            strength_violation = physical_features[:, 3] # 指法强度违反
            
            # 获取预测类别
            _, predicted = torch.max(logits, 1)
            
            # 检查预测是否正确
            correct_mask = (predicted == targets)
            
            # 更精细的物理约束权重
            physical_weights = torch.ones_like(stretch_features)
            
            # 对错误预测应用不同的约束惩罚
            # 1. 高伸展率的惩罚
            physical_weights = torch.where(
                ~correct_mask & (stretch_features > 0.7),
                physical_weights * (1.0 + 2.0 * stretch_features),
                physical_weights
            )
            
            # 2. 交叉困难的惩罚
            physical_weights = torch.where(
                ~correct_mask & (cross_features > 0.5),
                physical_weights * (1.0 + 1.5 * cross_features),
                physical_weights
            )
            
            # 3. 指法自然度违反的惩罚
            physical_weights = torch.where(
                ~correct_mask & (natural_violation > 0.5),
                physical_weights * 1.3,
                physical_weights
            )
            
            # 4. 指法强度违反的惩罚
            physical_weights = torch.where(
                ~correct_mask & (strength_violation > 0.5),
                physical_weights * 1.2,
                physical_weights
            )
            
            # 应用物理权重到损失
            weighted_loss = base_loss * torch.mean(physical_weights)
            
            return weighted_loss
        else:
            # 如果物理特征不足，退回到简单加权方案
            # 使用平均物理特征值作为权重
            mean_physical = torch.mean(physical_features, dim=1)
            
            # 获取预测类别
            _, predicted = torch.max(logits, 1)
            
            # 检查预测是否正确
            correct_mask = (predicted == targets)
            
            # 对错误预测应用简单惩罚
            physical_weights = torch.ones_like(mean_physical)
            physical_weights[~correct_mask] = 1.0 + mean_physical[~correct_mask]
            
            # 应用物理权重到损失
            weighted_loss = base_loss * torch.mean(physical_weights)
            
            return weighted_loss

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, 
                device, epochs, patience, physical_weight, output_dir):
    """训练物理约束增强模型"""
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # 物理约束损失函数
    physical_loss_fn = PhysicalConstraintLoss(num_classes=model.classifier[-1].out_features)
    
    # 获取物理特征数量
    num_physical_features = model.num_physical_features
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # 提取物理特征
            physical_features = extract_physical_features(inputs, num_physical_features)
            
            # 前向传播
            logits, combined_probs = model(inputs, physical_features)
            
            # 计算损失
            base_loss = criterion(logits, targets)
            phys_loss = physical_loss_fn(logits, targets, physical_features)
            
            # 组合损失
            loss = (1 - physical_weight) * base_loss + physical_weight * phys_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 计算准确率（使用主分类器输出）
            _, predicted = torch.max(logits.data, 1)
            
            # 统计
            train_loss += loss.item()
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # 计算平均训练损失和准确率
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 提取物理特征
                physical_features = extract_physical_features(inputs, num_physical_features)
                
                # 前向传播
                if hasattr(model, 'forward_planning'):
                    # 使用前向规划
                    predicted = model.forward_planning(inputs)
                    # 为了计算损失，仍然需要logits
                    logits, _ = model(inputs, physical_features)
                else:
                    # 常规推理
                    logits, _ = model(inputs, physical_features)
                    _, predicted = torch.max(logits.data, 1)
                
                # 计算损失
                loss = criterion(logits, targets)
                
                # 统计
                val_loss += loss.item()
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # 计算平均验证损失和准确率
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印统计
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 早停检查
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%")
            best_val_acc = val_acc
            patience_counter = 0
            
            # 保存最佳模型
            model_path = os.path.join(output_dir, 'fingering_physical_model_best.pth')
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            print(f"Validation accuracy did not improve. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    # 加载最佳模型
    model_path = os.path.join(output_dir, 'fingering_physical_model_best.pth')
    model.load_state_dict(torch.load(model_path))
    
    return model, history

def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    
    # 获取物理特征数量
    num_physical_features = model.num_physical_features
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 提取物理特征
            physical_features = extract_physical_features(inputs, num_physical_features)
            
            # 使用前向规划
            if hasattr(model, 'forward_planning'):
                predicted = model.forward_planning(inputs)
            else:
                # 常规推理，使用物理特征
                logits, combined_probs = model(inputs, physical_features)
                _, predicted = torch.max(combined_probs.data, 1)  # 使用物理增强后的概率
            
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # 计算分类报告
    class_report = classification_report(all_targets, all_preds, output_dict=True)
    
    return test_acc, conf_matrix, class_report

def plot_training_history(history, output_dir):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle('Physical-Enhanced Fingering Model Training History')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(conf_matrix, num_classes, output_dir):
    """绘制混淆矩阵"""
    import seaborn as sns
    
    # 归一化混淆矩阵
    norm_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(norm_conf_matrix, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix - Physical-Enhanced Model')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据加载
    if args.use_augmented:
        print("Loading augmented data...")
        X_train = np.load('X_train_aug.npy')
        y_train = np.load('y_train_aug.npy')
        X_val = np.load('X_val_aug.npy')
        y_val = np.load('y_val_aug.npy')
    else:
        print("Loading regular data...")
        try:
            X_train = np.load('X_train.npy')
            y_train = np.load('y_train.npy')
            X_val = np.load('X_val.npy')
            y_val = np.load('y_val.npy')
        except FileNotFoundError:
            print("ERROR: Data files not found. Please run data_process.py first.")
            return
    
    # 检测物理特征
    try:
        feature_size = X_train.shape[2]
        args.num_physical_features = min(args.num_physical_features, feature_size)
        print(f"Using {args.num_physical_features} physical features")
    except:
        print(f"Could not detect physical features, using default value: {args.num_physical_features}")
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # 转换为PyTorch数据集
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 设置输入大小
    if args.input_size is None:
        args.input_size = X_train.shape[2]
    
    # 设置类别数
    if args.num_classes is None:
        args.num_classes = len(np.unique(y_train))
    
    print(f"Input size: {args.input_size}")
    print(f"Number of classes: {args.num_classes}")
    
    # 创建模型
    model = PhysicalEnhancedFingeringModel(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes,
        num_physical_features=args.num_physical_features,
        physical_weight=args.physical_weight,
        dropout=args.dropout
    )
    
    # 将模型移至设备
    model = model.to(device)
    print("Created Physical-Enhanced Fingering Model")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练模型
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        physical_weight=args.physical_weight,
        output_dir=args.output_dir
    )
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'fingering_physical_model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # 评估模型
    test_acc, conf_matrix, class_report = evaluate_model(model, val_loader, device)
    
    # 保存评估结果
    results = {
        'accuracy': test_acc,
        'class_report': class_report
    }
    
    import json
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 绘制训练历史
    plot_training_history(history, args.output_dir)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(conf_matrix, args.num_classes, args.output_dir)
    
    print(f"Model saved as '{final_model_path}'")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"All results saved to '{args.output_dir}'")

if __name__ == '__main__':
    main() 