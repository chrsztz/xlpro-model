# model_training.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from data_utils import load_pickle
from models import TransformerModel,BiGRU,BiLSTM,BiLSTMWithAttention

# 定义 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

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

def main():
    # 加载增强后的数据
    try:
        X_train_aug = np.load('X_train_aug.npy')
        X_val_aug = np.load('X_val_aug.npy')
        y_train_aug = np.load('y_train_aug.npy')
        y_val_aug = np.load('y_val_aug.npy')
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("请确保已运行 'dataset_prep.py' 和 'data_process.py' 并生成所需的 .npy 文件。")
        return

    # 加载 LabelEncoders
    try:
        le_fingering = load_pickle('le_fingering.pkl')
        le_hand = load_pickle('le_hand.pkl')
    except FileNotFoundError as e:
        print(f"Error loading LabelEncoders: {e}")
        print("请确保已运行 'dataset_prep.py' 并生成相关的 .pkl 文件。")
        return

    # 参数设置
    input_size = X_train_aug.shape[2]  # 特征数量
    hidden_size = 256
    num_layers = 3
    num_classes = len(le_fingering.classes_)
    dropout = 0.5

    # 初始化模型
    # 选择您要使用的模型，取消相应的注释
    model = BiLSTMWithAttention(input_size, hidden_size, num_layers, num_classes, dropout)
    # model = BiLSTM(input_size, hidden_size, num_layers, num_classes, dropout)
    # model = BiGRU(input_size, hidden_size, num_layers, num_classes, dropout)
    # model = TransformerModel(input_size, hidden_size_tr, num_heads, num_layers_tr, num_classes, dropout)

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    print(model)

    # 创建 Dataset 和 DataLoader
    batch_size = 64

    train_dataset = FingeringDataset(X_train_aug, y_train_aug)
    val_dataset = FingeringDataset(X_val_aug, y_val_aug)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 计算类别权重
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_aug), y=y_train_aug)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 使用 Focal Loss
    criterion = FocalLoss(alpha=1, gamma=2)

    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    num_epochs = 100
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # 前向传播
            outputs = model(X_batch)  # [batch_size, num_classes]
            loss = criterion(outputs, y_batch)  # [batch_size]

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 调整学习率
        scheduler.step(val_loss)

        # 获取并打印当前学习率
        current_lrs = scheduler.get_last_lr()
        print(f"当前学习率: {current_lrs}")

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"Trigger Times: {trigger_times}")
            if trigger_times >= patience:
                print("Early stopping!")
                break

    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)

    # 保存模型
    torch.save(model.state_dict(), 'fingering_bilstm_model.pth')
    print("Model saved as fingering_bilstm_model.pth")

if __name__ == "__main__":
    main()
