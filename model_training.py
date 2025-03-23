import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import pickle
import os
from data_utils import load_pickle  # 用于加载 LabelEncoder
from models import HierarchicalAttentionFingeringModel  # 新添加的模型
from torch.cuda.amp import GradScaler, autocast


# Weighted Focal Loss 定义，用于处理类别不平衡和难分类样本
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', ignore_index=10):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.alpha = alpha  # 类别权重

    def forward(self, inputs, targets):
        # 创建忽略标签的掩码
        mask = targets != self.ignore_index
        inputs_masked = inputs[mask]
        targets_masked = targets[mask]
        
        if inputs_masked.size(0) == 0:
            return torch.tensor(0.0, device=inputs.device)

        # 计算带权重的交叉熵损失
        if self.alpha is not None:
            # 确保alpha是一个与输入设备相同的张量
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(inputs.device)
            else:
                alpha = torch.tensor(self.alpha, device=inputs.device)
                
            # 获取每个样本对应的权重
            batch_alpha = alpha.gather(0, targets_masked)
            BCE_loss = F.cross_entropy(inputs_masked, targets_masked, reduction='none')
            pt = torch.exp(-BCE_loss)
            
            # 应用Focal Loss公式
            F_loss = batch_alpha * (1 - pt) ** self.gamma * BCE_loss
        else:
            BCE_loss = F.cross_entropy(inputs_masked, targets_masked, reduction='none')
            pt = torch.exp(-BCE_loss)
            F_loss = (1 - pt) ** self.gamma * BCE_loss

        # 根据reduction参数返回结果
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# 增强的Dataset类，支持复杂的数据增强
class EnhancedFingeringDataset(Dataset):
    def __init__(self, X, y, le_hand, hand_index=2, mirror_prob=0.5, 
                 transpose_prob=0.2, transpose_range=(-2, 2), 
                 noise_prob=0.1, noise_level=0.02):
        self.X = X
        self.y = y
        self.le_hand = le_hand
        self.hand_index = hand_index
        self.mirror_prob = mirror_prob
        self.transpose_prob = transpose_prob
        self.transpose_range = transpose_range
        self.noise_prob = noise_prob
        self.noise_level = noise_level

        # 打印形状信息以便调试
        print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")
        print(f"X dtype: {self.X.dtype}, y dtype: {self.y.dtype}")
        print(f"Sample X[0] shape: {self.X[0].shape}, y[0]: {self.y[0]}")

        # 计算类别分布，用于后续加权采样
        self.class_counts = np.bincount(self.y[self.y != 10], minlength=10)
        print(f"Class distribution: {self.class_counts}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_sample = self.X[idx].copy()
        y_sample = int(self.y[idx])  # 将y_sample转换为整数，因为它是一个单一的标签值

        # 1. 镜像增强
        if np.random.rand() < self.mirror_prob:
            # 检查最后一个时间步的手属性
            if X_sample[-1, self.hand_index] == self.le_hand.transform(['left'])[0]:
                # 将所有时间步的手属性改为右手
                X_sample[:, self.hand_index] = self.le_hand.transform(['right'])[0]
            else:
                # 将所有时间步的手属性改为左手
                X_sample[:, self.hand_index] = self.le_hand.transform(['left'])[0]

            # 指法翻转规则 - 直接应用到单一的标签值
            finger_flip = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 9, 6: 8, 7: 7, 8: 6, 9: 5}
            y_sample = finger_flip.get(y_sample, y_sample)

        # 2. 音符平移（模拟移调）- 不会影响指法
        if np.random.rand() < self.transpose_prob and y_sample != 10:
            # 通常钢琴音符在MIDI中是以midi_number表示
            # 假设X的第一个特征是音高特征，适当修改索引
            pitch_index = 0  # 修改为实际音高特征的索引
            transpose_amount = np.random.randint(
                self.transpose_range[0], self.transpose_range[1] + 1)
            
            # 对所有时间步的音高应用平移
            X_sample[:, pitch_index] = X_sample[:, pitch_index] + transpose_amount
            
            # 确保音高保持在合理范围内（例如MIDI音符范围21-108）
            X_sample[:, pitch_index] = np.clip(X_sample[:, pitch_index], 21, 108)

        # 3. 添加随机噪声
        if np.random.rand() < self.noise_prob and y_sample != 10:
            # 对除了分类特征（如手、指法）之外的所有特征添加噪声
            # 假设最后几列是分类特征，前面的是数值特征
            numeric_features = list(range(X_sample.shape[1]))
            numeric_features.remove(self.hand_index)  # 移除手部特征索引
            
            # 添加高斯噪声
            noise = np.random.normal(0, self.noise_level, size=(X_sample.shape[0], len(numeric_features)))
            X_sample[:, numeric_features] += noise

        # 返回转换为张量的样本
        return torch.tensor(X_sample, dtype=torch.float32), torch.tensor(y_sample, dtype=torch.long)

    def get_sample_weights(self):
        """获取每个样本的权重，用于加权采样"""
        weights = np.ones(len(self.y))
        for i, y_val in enumerate(self.y):
            if y_val != 10:  # 忽略未标注的样本
                # 根据类别频率的倒数计算权重
                weights[i] = 1.0 / self.class_counts[y_val]
                
        # 归一化权重
        if weights.sum() > 0:
            weights = weights / weights.sum() * len(weights)
        return weights


def train_epoch(model, loader, criterion, optimizer, device, use_amp=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    scaler = GradScaler() if use_amp else None
    
    for X_batch, y_batch in tqdm(loader, desc="Training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                # 这里假设hand_index=2，实际情况可能需要调整
                outputs = model(X_batch, hand_indices=2)
                if outputs.dim() > 2:  # 如果模型返回序列预测，取最后一个时间步
                    outputs = outputs[:, -1, :]
                loss = criterion(outputs, y_batch)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(X_batch, hand_indices=2)
            if outputs.dim() > 2:  # 如果模型返回序列预测，取最后一个时间步
                outputs = outputs[:, -1, :]
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        
        # 计算准确率
        mask = y_batch != 10  # 忽略未标注样本
        if mask.sum() > 0:
            _, predicted = torch.max(outputs[mask], 1)
            total += mask.sum().item()
            correct += (predicted == y_batch[mask]).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch, hand_indices=2)
            
            if outputs.dim() > 2:  # 如果模型返回序列预测，取最后一个时间步
                outputs = outputs[:, -1, :]
                
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            # 只评估有标注的样本
            mask = y_batch != 10
            if mask.sum() > 0:
                _, predicted = torch.max(outputs[mask], 1)
                total += mask.sum().item()
                correct += (predicted == y_batch[mask]).sum().item()
                
                # 收集预测结果和标签，用于详细分析
                all_predictions.append(predicted.cpu().numpy())
                all_targets.append(y_batch[mask].cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0
    
    # 合并所有预测结果和标签
    if all_predictions:
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # 计算每个类别的准确率
        class_accuracies = []
        for cls in range(10):  # 假设有10个类别
            mask = all_targets == cls
            if mask.sum() > 0:
                cls_acc = (all_predictions[mask] == cls).mean()
                class_accuracies.append(cls_acc)
            else:
                class_accuracies.append(0.0)
        
        return avg_loss, accuracy, class_accuracies
    else:
        return avg_loss, accuracy, [0.0] * 10


def main():
    # 数据路径
    train_X_path = 'X_train_combined.npy'
    train_y_path = 'y_train_combined.npy'
    val_X_path = 'X_val_combined.npy'
    val_y_path = 'y_val_combined.npy'

    # 打印数据信息
    print(f"加载训练和验证数据...")

    # 加载数据
    X_train = np.load(train_X_path, mmap_mode='r')
    y_train = np.load(train_y_path, mmap_mode='r')
    X_val = np.load(val_X_path, mmap_mode='r')
    y_val = np.load(val_y_path, mmap_mode='r')

    print(f"数据加载完成。X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # 加载 LabelEncoders
    le_hand = load_pickle('le_hand.pkl')
    le_fingering = load_pickle('le_fingering.pkl')

    # 分离标注和未标注数据（未标注数据标记为10）
    labeled_idx = y_train != 10
    X_train_labeled = X_train[labeled_idx]
    y_train_labeled = y_train[labeled_idx]
    X_train_unlabeled = X_train[~labeled_idx]

    print(f"已标注数据: {len(X_train_labeled)}, 未标注数据: {len(X_train_unlabeled)}")

    # 模型参数
    input_size = X_train.shape[2]
    hidden_size = 256
    num_layers = 3
    num_classes = 10
    dropout = 0.3
    num_heads = 4

    print(f"模型参数 - input_size: {input_size}, hidden_size: {hidden_size}, num_classes: {num_classes}")

    # 初始化模型 - 使用新的层次化注意力模型
    model = HierarchicalAttentionFingeringModel(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout=dropout,
        num_layers=num_layers
    )
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"使用设备: {device}")

    # 创建加强版数据集
    train_dataset = EnhancedFingeringDataset(
        X_train_labeled, 
        y_train_labeled, 
        le_hand, 
        hand_index=2,
        mirror_prob=0.5,
        transpose_prob=0.2,
        transpose_range=(-2, 2),
        noise_prob=0.1
    )
    
    val_dataset = EnhancedFingeringDataset(
        X_val, y_val, le_hand, 
        hand_index=2, 
        mirror_prob=0.0,  # 验证集不做增强
        transpose_prob=0.0,
        noise_prob=0.0
    )
    
    # 计算类别权重用于加权损失函数
    class_counts = np.bincount(y_train_labeled, minlength=10)
    # 平滑处理类别权重，避免极端值
    smoothed_class_counts = class_counts + 1
    class_weights = 1.0 / (smoothed_class_counts / smoothed_class_counts.sum())
    # 限制权重范围，避免过度惩罚稀有类别
    class_weights = np.clip(class_weights, 0.5, 2.0)
    # 归一化权重
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"类别权重: {class_weights}")
    
    # 使用WeightedRandomSampler进行平衡采样
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 优化器、损失函数和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = WeightedFocalLoss(alpha=torch.tensor(class_weights, dtype=torch.float32), 
                                 gamma=2, ignore_index=10)
    
    # 使用更先进的学习率调度器
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        steps_per_epoch=len(train_loader),
        epochs=50,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='cos'
    )

    # 训练参数
    num_epochs = 50
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 8
    trigger_times = 0
    use_amp = True if device.type == 'cuda' else False  # 仅GPU支持混合精度训练

    # 创建模型保存目录
    os.makedirs('models', exist_ok=True)
    
    print("第一阶段训练：仅使用标注数据...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_amp)
        val_loss, val_accuracy, class_accuracies = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        print(f"每类准确率: {', '.join([f'Class {i}: {acc:.4f}' for i, acc in enumerate(class_accuracies)])}")

        # 早停机制 - 监控验证损失和验证准确率
        if val_loss < best_val_loss or val_accuracy > best_val_acc:
            # 保存最佳损失和最佳准确率模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'models/best_loss_model.pth')
                
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(model.state_dict(), 'models/best_acc_model.pth')
                
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"触发早停机制，停止训练！(patience={patience})")
                break

    # 自训练阶段
    if len(X_train_unlabeled) > 0:
        print("\n第二阶段训练：加入伪标签进行自训练...")
        
        # 加载验证准确率最高的模型
        model.load_state_dict(torch.load('models/best_acc_model.pth'))
        model.eval()

        # 处理未标注数据的批次大小
        batch_size = 128
        all_pseudo_X = []
        all_pseudo_y = []
        confidence_thresholds = [0.95, 0.90, 0.85]  # 逐渐降低置信度阈值
        
        # 分批处理未标注数据以避免内存问题，并使用不同的置信度阈值
        for threshold in confidence_thresholds:
            print(f"使用置信度阈值 {threshold} 生成伪标签...")
            for i in range(0, len(X_train_unlabeled), batch_size):
                end = min(i + batch_size, len(X_train_unlabeled))
                batch_X = X_train_unlabeled[i:end]

                with torch.no_grad():
                    X_unlabeled_tensor = torch.tensor(batch_X, dtype=torch.float32).to(device)
                    outputs = model(X_unlabeled_tensor, hand_indices=2)
                    if outputs.dim() > 2:  # 如果模型返回序列预测，取最后一个时间步
                        outputs = outputs[:, -1, :]
                    
                    probabilities, predicted = torch.max(F.softmax(outputs, dim=1), 1)
                    high_conf_idx = probabilities > threshold  # 置信度阈值

                    # 收集高置信度的样本
                    if high_conf_idx.sum().item() > 0:
                        pseudo_X = batch_X[high_conf_idx.cpu().numpy()]
                        pseudo_y = predicted[high_conf_idx].cpu().numpy()

                        all_pseudo_X.append(pseudo_X)
                        all_pseudo_y.append(pseudo_y)
            
            # 如果已经收集了足够多的伪标签样本，就退出循环
            total_pseudo_samples = sum(len(x) for x in all_pseudo_X) if all_pseudo_X else 0
            if total_pseudo_samples >= len(X_train_unlabeled) * 0.3:  # 如果已经标记了30%以上的未标注样本
                print(f"已收集 {total_pseudo_samples} 个伪标签样本，停止降低置信度阈值")
                break

        # 合并所有伪标签样本
        if all_pseudo_X:
            pseudo_X = np.concatenate(all_pseudo_X, axis=0)
            pseudo_y = np.concatenate(all_pseudo_y, axis=0)

            print(f"添加 {len(pseudo_X)} 个伪标签样本到训练集")
            
            # 分析伪标签分布
            pseudo_class_counts = np.bincount(pseudo_y, minlength=10)
            print(f"伪标签类别分布: {pseudo_class_counts}")
            
            # 合并标注数据和伪标签数据
            X_train_new = np.concatenate([X_train_labeled, pseudo_X], axis=0)
            y_train_new = np.concatenate([y_train_labeled, pseudo_y], axis=0)
            
            # 创建新的数据集和数据加载器
            train_dataset = EnhancedFingeringDataset(X_train_new, y_train_new, le_hand, 
                                                   hand_index=2, mirror_prob=0.3)
            
            # 重新计算样本权重，但给伪标签样本较低的权重
            sample_weights = train_dataset.get_sample_weights()
            # 降低伪标签样本的权重
            for i in range(len(X_train_labeled), len(sample_weights)):
                sample_weights[i] *= 0.7  # 伪标签样本权重降低30%
                
            sampler = WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(train_dataset), 
                replacement=True
            )
            
            train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

            # 重置优化器和学习率调度器
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=0.00001)
            
            # 重置早停参数
            best_val_loss = float('inf')
            best_val_acc = 0.0
            trigger_times = 0
            patience = 5  # 自训练阶段使用更短的patience

            # 继续训练
            print("\n开始第二阶段训练（含伪标签）...")
            for epoch in range(15):  # 自训练15个epoch
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_amp)
                val_loss, val_accuracy, class_accuracies = evaluate(model, val_loader, criterion, device)
                
                print(f"Self-Training Epoch [{epoch + 1}/15], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                print(f"每类准确率: {', '.join([f'Class {i}: {acc:.4f}' for i, acc in enumerate(class_accuracies)])}")
                
                scheduler.step()
                
                # 早停机制
                if val_loss < best_val_loss or val_accuracy > best_val_acc:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), 'models/best_selftrain_loss_model.pth')
                        
                    if val_accuracy > best_val_acc:
                        best_val_acc = val_accuracy
                        torch.save(model.state_dict(), 'models/best_selftrain_acc_model.pth')
                        
                    trigger_times = 0
                else:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print(f"触发早停机制，停止训练！")
                        break

    # 评估最终模型性能
    print("\n评估模型性能...")
    
    # 确定最佳模型文件
    best_model_files = [
        'models/best_acc_model.pth',
        'models/best_selftrain_acc_model.pth'
    ]
    
    best_model_path = None
    best_accuracy = 0.0
    
    for model_file in best_model_files:
        if os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file))
            _, accuracy, class_accs = evaluate(model, val_loader, criterion, device)
            
            print(f"模型 {model_file} 验证集准确率: {accuracy:.4f}")
            print(f"每类准确率: {', '.join([f'Class {i}: {acc:.4f}' for i, acc in enumerate(class_accs)])}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = model_file
    
    # 保存最终模型
    if best_model_path:
        print(f"\n最佳模型: {best_model_path}, 准确率: {best_accuracy:.4f}")
        model.load_state_dict(torch.load(best_model_path))
        torch.save(model.state_dict(), 'fingering_model_final.pth')
        print("训练完成，最终模型已保存至 'fingering_model_final.pth'")
    else:
        torch.save(model.state_dict(), 'fingering_model_final.pth')
        print("训练完成，最终模型已保存")


if __name__ == "__main__":
    main()