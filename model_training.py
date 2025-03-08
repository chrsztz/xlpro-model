import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pickle
from data_utils import load_pickle  # 用于加载 LabelEncoder
from models import BiLSTMWithAttention  # 示例模型，可以替换为其他模型
from torch.cuda.amp import GradScaler, autocast


# Focal Loss 定义，用于处理类别不平衡和难分类样本
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=10):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]
        if inputs.size(0) == 0:
            return torch.tensor(0.0, device=inputs.device)

        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# 自定义 Dataset，支持动态镜像增强
class FingeringDataset(Dataset):
    def __init__(self, X, y, le_hand, hand_index=2, mirror_prob=0.5):
        self.X = X
        self.y = y
        self.le_hand = le_hand
        self.hand_index = hand_index
        self.mirror_prob = mirror_prob

        # 打印形状信息以便调试
        print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")
        # 输出数据类型信息
        print(f"X dtype: {self.X.dtype}, y dtype: {self.y.dtype}")
        # 输出一个样本示例
        print(f"Sample X[0] shape: {self.X[0].shape}, y[0]: {self.y[0]}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_sample = self.X[idx].copy()
        y_sample = int(self.y[idx])  # 将y_sample转换为整数，因为它是一个单一的标签值

        # 动态镜像增强
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

        # 返回转换为张量的样本
        return torch.tensor(X_sample, dtype=torch.float32), torch.tensor(y_sample, dtype=torch.long)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    # scaler = GradScaler()
    # # 在 train_epoch 中：
    # with autocast():
    #     outputs = model(X_batch)
    #     loss = criterion(outputs, y_batch)
    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()
    for X_batch, y_batch in tqdm(loader, desc="Training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += (y_batch != 10).sum().item()
            correct += (predicted == y_batch).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


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
    num_layers = 2
    num_classes = 10
    dropout = 0.3

    print(f"模型参数 - input_size: {input_size}, hidden_size: {hidden_size}, num_classes: {num_classes}")

    # 初始化模型
    model = BiLSTMWithAttention(input_size, hidden_size, num_layers, num_classes, dropout)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"使用设备: {device}")

    # 数据加载器
    train_dataset = FingeringDataset(X_train_labeled, y_train_labeled, le_hand, mirror_prob=0.5)
    val_dataset = FingeringDataset(X_val, y_val, le_hand, mirror_prob=0.0)  # 验证集不增强
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 优化器、损失函数和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = FocalLoss(alpha=1, gamma=2, ignore_index=10)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)

    # 训练参数
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 5
    trigger_times = 0

    print("开始初始训练...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        scheduler.step()

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("触发早停机制，停止训练！")
                break

    # 自训练阶段
    if len(X_train_unlabeled) > 0:
        print("开始自训练...")
        model.eval()

        # 处理未标注数据的批次大小
        batch_size = 256
        all_pseudo_X = []
        all_pseudo_y = []

        # 分批处理未标注数据以避免内存问题
        for i in range(0, len(X_train_unlabeled), batch_size):
            end = min(i + batch_size, len(X_train_unlabeled))
            batch_X = X_train_unlabeled[i:end]

            with torch.no_grad():
                X_unlabeled_tensor = torch.tensor(batch_X, dtype=torch.float32).to(device)
                outputs = model(X_unlabeled_tensor)
                probabilities, predicted = torch.max(F.softmax(outputs, dim=1), 1)
                high_conf_idx = probabilities > 0.9  # 置信度阈值

                # 收集高置信度的样本
                if high_conf_idx.sum().item() > 0:
                    pseudo_X = batch_X[high_conf_idx.cpu().numpy()]
                    pseudo_y = predicted[high_conf_idx].cpu().numpy()

                    all_pseudo_X.append(pseudo_X)
                    all_pseudo_y.append(pseudo_y)

        # 合并所有伪标签样本
        if all_pseudo_X:
            pseudo_X = np.concatenate(all_pseudo_X, axis=0)
            pseudo_y = np.concatenate(all_pseudo_y, axis=0)

            print(f"添加 {len(pseudo_X)} 个伪标签样本到训练集")
            X_train_new = np.concatenate([X_train_labeled, pseudo_X], axis=0)
            y_train_new = np.concatenate([y_train_labeled, pseudo_y], axis=0)
            train_dataset = FingeringDataset(X_train_new, y_train_new, le_hand, mirror_prob=0.5)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            # 继续训练
            model.load_state_dict(torch.load('best_model.pth'))  # 加载最佳模型
            for epoch in range(10):  # 自训练10个额外epoch
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
                print(f"Self-Training Epoch [{epoch + 1}/10], Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                scheduler.step()

    # 保存最终模型
    torch.save(model.state_dict(), 'fingering_model_final.pth')
    print("训练完成，模型已保存至 'fingering_model_final.pth'")


if __name__ == "__main__":
    main()