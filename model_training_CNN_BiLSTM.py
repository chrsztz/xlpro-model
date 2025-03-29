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
from models import TransformerModel,BiGRU,BiLSTM,BiLSTMWithAttention,CNNWithAttention,EnhancedFingeringModel

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

def main():
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
    model = EnhancedFingeringModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
    ).to(device)
    
    print(model)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 定义优化器 - 使用 AdamW 并调整参数，提高基础学习率
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.003,  # 提高初始学习率，从0.001增加到0.003
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
    
    # 使用 One Cycle 学习率调度器，调整参数以提高初始学习率
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,  # 提高最大学习率，从0.001增加到0.003
        steps_per_epoch=steps_per_epoch,
        epochs=30,
        pct_start=0.3,  # 用30%的时间来提高学习率
        div_factor=10,  # 降低除数因子，使初始学习率更高 (初始lr = max_lr/div_factor = 0.0003)
        final_div_factor=100,
        anneal_strategy='cos'
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