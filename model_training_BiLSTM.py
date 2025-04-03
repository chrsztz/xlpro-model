# model_training.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm 
from data_utils import load_pickle
from models import BiLSTMWithAttention,CNNWithAttention,EnhancedFingeringModel, PhysicalEnhancedFingeringModel
import argparse
import torch.optim as optim
from torch.utils.data import TensorDataset
import pandas as pd

# 定义改进的 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing  # 添加标签平滑

    def forward(self, inputs, targets):
        # 应用标签平滑
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            smooth_targets = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smooth_targets = smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            BCE_loss = -torch.sum(smooth_targets * F.log_softmax(inputs, dim=1), dim=1)
        else:
            BCE_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# 定义 Dataset
class FingeringDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # 输入特征
        self.y = torch.tensor(y, dtype=torch.long)  # 指法标签

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 评估模型性能
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(data_loader, desc="Evaluation", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    val_loss /= len(data_loader.dataset)
    
    # 计算总体准确率
    correct = sum(1 for p, t in zip(all_preds, all_labels) if p == t)
    accuracy = correct / len(all_labels)
    
    # 计算每个类别的准确率
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    return val_loss, accuracy, class_accuracies, conf_matrix, all_preds, all_labels

# 修改 train_epoch 函数以支持 OneCycleLR 调度器
def train_epoch(model, train_loader, optimizer, criterion, device, clip_value=1.0):
    model.train()
    train_loss = 0
    all_labels = []
    all_preds = []
    grad_norms = []
    
    for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # 反向传播
        loss.backward()
        
        # 计算梯度范数（用于监控梯度）
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        # 梯度裁剪防止梯度爆炸
        if clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        # 更新学习率 - 批次级别更新
        if isinstance(optimizer.param_groups[0]['lr'], torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        train_loss += loss.item() * X_batch.size(0)
        
        # 记录预测结果
        _, preds = torch.max(outputs, 1)
        all_labels.extend(y_batch.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
        # 计算当前批次每个类别的准确率
        batch_labels = y_batch.cpu().numpy()
        batch_preds = preds.cpu().numpy()
        
        # 每100个批次检查一次类别分布
        if len(all_preds) % (100 * X_batch.size(0)) == 0:
            # 计算目前为止的类别分布
            pred_counts = np.bincount(all_preds[-1000:] if len(all_preds) > 1000 else all_preds, 
                                      minlength=len(np.unique(all_labels)))
            
            # 如果预测严重偏向某些类别（大于70%），则警告
            max_pred_class = np.argmax(pred_counts)
            max_pred_ratio = pred_counts[max_pred_class] / pred_counts.sum()
            
            print(f"当前预测类别分布: {pred_counts}, 梯度范数: {np.mean(grad_norms):.4f}")
            
            if max_pred_ratio > 0.7:
                print(f"警告: 检测到模式崩溃 - 类别 {max_pred_class} 占比 {max_pred_ratio:.4f}")
                
                # 检查损失值是否异常
                if not np.isfinite(loss.item()):
                    print(f"警告: 损失值异常 ({loss.item()})")
                
                # 检查梯度是否异常
                if np.mean(grad_norms) > 10 or np.mean(grad_norms) < 1e-6:
                    print(f"警告: 梯度范数异常 ({np.mean(grad_norms):.4f})")
            
            # 重置梯度范数列表避免内存过大
            grad_norms = []
    
    train_loss /= len(train_loader.dataset)
    
    # 计算训练准确率
    correct = sum(1 for p, t in zip(all_preds, all_labels) if p == t)
    accuracy = correct / len(all_labels)
    
    # 计算每个类别的准确率
    class_accs = []
    unique_classes = np.unique(all_labels)
    for cls in unique_classes:
        cls_indices = [i for i, l in enumerate(all_labels) if l == cls]
        if cls_indices:
            cls_correct = sum(1 for i in cls_indices if all_preds[i] == all_labels[i])
            cls_acc = cls_correct / len(cls_indices)
            class_accs.append((cls, cls_acc))
    
    # 排序并输出每个类别的准确率
    class_accs.sort(key=lambda x: x[0])
    for cls, acc in class_accs:
        print(f"  类别 {cls} 训练准确率: {acc:.4f}")
    
    return train_loss, accuracy, all_preds, all_labels

# 可视化训练历史
def plot_training_history(history, save_path='results'):
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制学习率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['learning_rates'])
    plt.title('Learning Rate Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    # 绘制每个类别的最终准确率
    plt.subplot(2, 2, 4)
    class_accs = history['class_accs'][-1]
    plt.bar(range(len(class_accs)), class_accs)
    plt.title('Per-Class Accuracy (Final)')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(class_accs)))
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/training_history.png')
    plt.close()

# 绘制混淆矩阵
def plot_confusion_matrix(conf_matrix, save_path='results'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/confusion_matrix.png')
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train BiLSTM model for piano fingering')
    parser.add_argument('--model_type', type=str, default='physical_enhanced',
                        choices=['bilstm', 'transformer', 'bigru', 'bilstm_attention', 
                                'cnn_attention', 'enhanced', 'physical_enhanced'],
                        help='Model type to use')
    parser.add_argument('--input_size', type=int, default=136,
                        help='Input feature dimension')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM/GRU layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads for Transformer')
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
    return parser.parse_args()

def extract_physical_features(X):
    """
    从输入特征中提取物理约束相关特征
    
    Args:
        X: 输入特征序列 [batch_size, seq_len, input_size]
        
    Returns:
        物理特征 [batch_size, num_physical_features]
    """
    # 假设物理特征是输入向量的最后12个维度（伸展率、交叉指法距离、白键距离等）
    # 使用序列的最后一个时间步
    return X[:, -1, -12:]

class PhysicalConstraintLoss(nn.Module):
    """
    基于物理约束的损失函数
    重点惩罚不符合人体工程学的指法
    """
    def __init__(self, num_classes=10):
        super(PhysicalConstraintLoss, self).__init__()
        self.base_loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        
    def forward(self, logits, targets, physical_features):
        """
        Args:
            logits: 模型输出的 logits [batch_size, num_classes]
            targets: 真实标签 [batch_size]
            physical_features: 物理约束特征 [batch_size, num_physical_features]
            
        Returns:
            combined_loss: 合并基础损失和物理约束损失
        """
        # 基础分类损失
        base_loss = self.base_loss(logits, targets)
        
        # 物理约束损失 - 使用物理特征加权
        # 提取与伸展率相关的特征 (假设是前几个特征)
        stretch_features = physical_features[:, :6]  # 伸展率特征
        
        # 计算平均伸展率
        mean_stretch = torch.mean(stretch_features, dim=1)
        
        # 获取模型预测的类别
        _, predicted = torch.max(logits, 1)
        
        # 检查预测是否正确
        correct_mask = (predicted == targets)
        
        # 对于错误预测，增加物理惩罚
        # 构建物理惩罚权重 - 错误预测但符合物理约束的惩罚较轻
        physical_weights = torch.ones_like(mean_stretch)
        physical_weights[~correct_mask] = 1.0 + mean_stretch[~correct_mask]
        
        # 应用物理权重到损失
        weighted_loss = base_loss * torch.mean(physical_weights)
        
        return weighted_loss

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, 
          device, epochs, patience, model_type='bilstm', physical_weight=0.3):
    """
    训练模型
    
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        criterion: 损失函数
        device: 训练设备 (CPU/GPU)
        epochs: 训练轮数
        patience: 早停耐心值
        model_type: 模型类型
        physical_weight: 物理约束损失权重
        
    Returns:
        trained_model: 训练好的模型
        history: 训练历史 (loss, accuracy等)
    """
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    physical_loss_fn = None
    if model_type == 'physical_enhanced':
        physical_loss_fn = PhysicalConstraintLoss(num_classes=model.classifier[-1].out_features)
    
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
            
            # 处理不同模型类型的前向传播
            if model_type == 'physical_enhanced':
                # 提取物理特征
                physical_features = extract_physical_features(inputs)
                
                # 前向传播
                logits, combined_probs = model(inputs, physical_features)
                
                # 组合损失
                base_loss = criterion(logits, targets)
                
                if physical_loss_fn:
                    phys_loss = physical_loss_fn(logits, targets, physical_features)
                    loss = (1 - physical_weight) * base_loss + physical_weight * phys_loss
                else:
                    loss = base_loss
                
                # 使用主分类器的logits计算准确率
                _, predicted = torch.max(logits.data, 1)
            else:
                # 标准前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
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
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 处理不同模型类型的前向传播
                if model_type == 'physical_enhanced':
                    # 验证时使用前向规划功能
                    physical_features = extract_physical_features(inputs)
                    logits, _ = model(inputs, physical_features)
                    
                    # 可选：使用前向规划获取最佳动作
                    if hasattr(model, 'forward_planning'):
                        predicted = model.forward_planning(inputs)
                    else:
                        _, predicted = torch.max(logits.data, 1)
                    
                    # 计算损失
                    loss = criterion(logits, targets)
                else:
                    # 标准前向传播
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    _, predicted = torch.max(outputs.data, 1)
                
                # 统计
                val_loss += loss.item()
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # 计算平均验证损失和准确率
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 更新学习率
        if scheduler:
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
            torch.save(model.state_dict(), 'fingering_model_best.pth')
        else:
            patience_counter += 1
            print(f"Validation accuracy did not improve. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('fingering_model_best.pth'))
    
    return model, history

def evaluate_model(model, test_loader, device, model_type='bilstm'):
    """
    评估模型
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备 (CPU/GPU)
        model_type: 模型类型
        
    Returns:
        test_acc: 测试准确率
        conf_matrix: 混淆矩阵
    """
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if model_type == 'physical_enhanced':
                # 使用前向规划
                if hasattr(model, 'forward_planning'):
                    predicted = model.forward_planning(inputs)
                else:
                    # 常规推理
                    logits, _ = model(inputs)
                    _, predicted = torch.max(logits.data, 1)
            else:
                # 标准前向传播
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
            
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    return test_acc, conf_matrix

def main():
    args = parse_args()
    
    # 设置随机种子确保结果可重复
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 创建结果目录
    results_dir = 'results_cnn'
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载数据
    try:
        # 尝试加载增强后的数据
        X_train = np.load('X_train_aug.npy')
        y_train = np.load('y_train_aug.npy')
        X_val = np.load('X_val_aug.npy')
        y_val = np.load('y_val_aug.npy')
    except FileNotFoundError:
        try:
            # 尝试加载常规训练数据
            X_train = np.load('X_train.npy')
            y_train = np.load('y_train.npy')
            X_val = np.load('X_val.npy')
            y_val = np.load('y_val.npy')
        except FileNotFoundError as e:
            print(f"无法加载数据文件: {e}")
            print("请确保已运行 'dataset_prep.py' 和 'data_process.py' 并生成所需的 .npy 文件。")
            return
    
    # 检查特征维度是否一致
    if X_train.shape[2] != X_val.shape[2]:
        print(f"警告: 训练集和验证集特征维度不匹配 - 训练集: {X_train.shape[2]}, 验证集: {X_val.shape[2]}")
        print("重新创建训练集和验证集，确保特征维度一致...")
        
        # 结合所有数据重新划分
        all_X = X_train
        all_y = y_train
        
        # 从训练数据重新划分
        indices = np.random.permutation(len(all_X))
        train_size = int(0.8 * len(all_X))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train = all_X[train_indices]
        y_train = all_y[train_indices]
        X_val = all_X[val_indices]
        y_val = all_y[val_indices]
        
        print(f"新的训练集/验证集大小: {X_train.shape} / {X_val.shape}")
    
    print(f"训练集: X shape {X_train.shape}, y shape {y_train.shape}")
    print(f"验证集: X shape {X_val.shape}, y shape {y_val.shape}")
    
    # 输出类别分布
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    print("训练集类别分布:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  类别 {cls}: {count} 样本 ({100.0 * count / len(y_train):.2f}%)")
    
    # 检查数据中是否存在NaN或无穷大的值
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("警告：训练集中存在NaN或无穷大值，将替换为0...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.isnan(X_val).any() or np.isinf(X_val).any():
        print("警告：验证集中存在NaN或无穷大值，将替换为0...")
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 使用更稳健的方式计算均值和标准差
    print("应用稳健的特征标准化...")
    X_flat_train = X_train.reshape(-1, X_train.shape[2])
    
    # 计算均值和标准差，使用更稳健的方法处理离群值
    q25 = np.percentile(X_flat_train, 25, axis=0)
    q75 = np.percentile(X_flat_train, 75, axis=0)
    iqr = q75 - q25
    
    # 将超出 IQR 范围 3 倍的值视为离群值并替换
    upper_bound = q75 + 3 * iqr
    lower_bound = q25 - 3 * iqr
    
    for i in range(X_flat_train.shape[1]):
        X_flat_train[:, i] = np.clip(X_flat_train[:, i], lower_bound[i], upper_bound[i])
    
    # 计算新的均值和标准差
    mean = np.mean(X_flat_train, axis=0)
    std = np.std(X_flat_train, axis=0)
    
    # 避免除零，确保标准差最小值
    std = np.maximum(std, 1e-6)
    
    # 应用标准化
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    
    # 参数设置
    input_size = X_train.shape[2]  # 特征数量
    hidden_size = 512  # 增加隐藏层大小以适应CNN架构
    num_classes = len(np.unique(y_train))
    num_layers = 3  # 增加LSTM层数
    dropout = 0.5
    
    print(f"使用特征数量: {input_size}, 隐藏层大小: {hidden_size}, 类别数: {num_classes}")
    
    # 创建 Dataset 和 DataLoader
    batch_size = 128  # 增大batch size以适应CNN训练特性
    
    train_dataset = FingeringDataset(X_train, y_train)
    val_dataset = FingeringDataset(X_val, y_val)
    
    # 计算类别权重以处理数据不平衡
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    print("类别权重:")
    for i, weight in enumerate(class_weights):
        print(f"  类别 {i}: {weight:.4f}")
    
    # 创建加权采样器 - 修改为更平衡的采样方法
    sample_weights = np.ones_like(y_train, dtype=np.float32)
    for i, cls in enumerate(np.unique(y_train)):
        sample_weights[y_train == cls] = class_weights[i].item()
    
    # 归一化权重
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    sampler = WeightedRandomSampler(
        sample_weights, 
        len(sample_weights), 
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Verify training batches for class distribution
    print("验证训练批次的类分布...")
    class_counts = np.zeros(num_classes, dtype=np.int32)
    for i, (_, y_batch) in enumerate(train_loader):
        for cls in range(num_classes):
            class_counts[cls] += (y_batch == cls).sum().item()
        if i >= 5:  # 只检查前5个批次
            break
    
    print("前5个批次的类分布:")
    for cls in range(num_classes):
        print(f"  类别 {cls}: {class_counts[cls]} 样本")
    
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化CNN模型
    model = BiLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    print(model)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 定义优化器 - 使用 AdamW 并调整参数
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.001,  # 增加学习率以适应CNN架构
        weight_decay=0.0005,  # 轻微权重衰减
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 使用改进的 Focal Loss
    criterion = FocalLoss(
        alpha=class_weights.to(device),
        gamma=2.0,  # 恢复标准gamma值
        label_smoothing=0.1  # 轻微标签平滑
    )
    
    # 使用 One Cycle 学习率调度器
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=steps_per_epoch,
        epochs=30,
        pct_start=0.3,  # 用30%的时间来提高学习率
        anneal_strategy='cos',
        final_div_factor=100
    )
    
    # 训练参数
    num_epochs = 30
    best_val_acc = 0.0
    patience = 10
    counter = 0
    best_model_state = None
    
    # 历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'class_accs': [],
        'learning_rates': [],
    }
    
    print(f"开始训练 {num_epochs} 个 epoch...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 训练一个 epoch
        train_loss, train_acc, _, _ = train_epoch(
            model, train_loader, optimizer, criterion, device, clip_value=1.0
        )
        
        # 评估模型
        val_loss, val_acc, class_accs, conf_matrix, _, _ = evaluate_model(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        # 注意：在这里不调用scheduler.step()，因为我们在每个batch之后调用
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['class_accs'].append(class_accs)
        history['learning_rates'].append(current_lr)
        
        # 显示训练结果
        print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
        print(f"当前学习率: {current_lr:.6f}")
        print("每个类别的准确率:")
        for i, acc in enumerate(class_accs):
            print(f"  类别 {i}: {acc:.4f}")
        
        # 检查是否为最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            counter = 0
            
            # 保存混淆矩阵
            plot_confusion_matrix(conf_matrix, results_dir)
            
            print(f"新的最佳模型! 验证准确率: {val_acc:.4f}")
        else:
            counter += 1
            print(f"早停计数器: {counter}/{patience}")
            
            # 早停
            if counter >= patience:
                print(f"早停! 在 epoch {epoch + 1} 停止训练。")
                break
    
    # 绘制训练历史
    plot_training_history(history, results_dir)
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"已加载最佳模型 (验证准确率: {best_val_acc:.4f})")
    
    # 最终评估
    _, final_acc, final_class_accs, final_conf_matrix, all_preds, all_labels = evaluate_model(
        model, val_loader, criterion, device
    )
    
    # 生成分类报告
    class_report = classification_report(all_labels, all_preds, digits=4)
    print("\n分类报告:")
    print(class_report)
    
    # 保存分类报告
    with open(f'{results_dir}/classification_report.txt', 'w') as f:
        f.write("钢琴指法预测模型 (CNN+Attention) - 分类报告\n")
        f.write("="*50 + "\n\n")
        f.write(f"验证准确率: {final_acc:.4f}\n\n")
        f.write("每个类别的准确率:\n")
        for i, acc in enumerate(final_class_accs):
            f.write(f"类别 {i}: {acc:.4f}\n")
        f.write("\n详细分类报告:\n")
        f.write(class_report)
    
    # 保存模型
    torch.save(model.state_dict(), f'{results_dir}/cnn_attention_model_best.pth')
    print(f"模型已保存到 {results_dir}/cnn_attention_model_best.pth")

if __name__ == "__main__":
    main()