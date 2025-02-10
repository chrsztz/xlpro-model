# model_training.py
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import DataLoader, IterableDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm 
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


class FingeringIterableDataset(IterableDataset):
    """
    使用内存映射的 npy 文件，并按 batch_size 切片迭代，不一次性加载整个数据。
    """
    def __init__(self, X_path, y_path, batch_size=64, mmap_mode='r'):
        super().__init__()
        # 使用内存映射方式读取
        self.X = np.load(X_path, mmap_mode=mmap_mode)
        self.y = np.load(y_path, mmap_mode=mmap_mode)

        self.batch_size = batch_size
        self.n_samples = self.X.shape[0]
        assert self.X.shape[0] == self.y.shape[0], "X 与 y 行数不一致！"

    def __iter__(self):
        # 按 batch_size 切片顺序返回
        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            # 此处只在需要时才将对应切片加载到内存
            X_slice = torch.tensor(self.X[start:end], dtype=torch.float32)
            y_slice = torch.tensor(self.y[start:end], dtype=torch.long)
            yield X_slice, y_slice

def main():
    # 加载增强后的数据（使用内存映射 + 只在IterableDataset中读取）
    train_X_path = 'data/X_train_aug.npy'
    train_y_path = 'data/y_train_aug.npy'
    val_X_path   = 'data/X_val_aug.npy'
    val_y_path   = 'data/y_val_aug.npy'

    # 确保这几个文件存在
    for f in [train_X_path, train_y_path, val_X_path, val_y_path]:
        if not os.path.exists(f):
            print(f"未找到文件: {f}")
            return

    # 加载 LabelEncoders
    try:
        le_fingering = load_pickle('data/le_fingering.pkl')
        le_hand = load_pickle('data/le_hand.pkl')
    except FileNotFoundError as e:
        print(f"Error loading LabelEncoders: {e}")
        print("请确保已运行 'dataset_prep.py' 并生成相关的 .pkl 文件。")
        return

    # 参数设置
    # 初始化网络参数
    # 这里先简单用内存映射看看 X_train_aug 的第三维度（特征数），读取一小部分
    X_train_tmp = np.load(train_X_path, mmap_mode='r')
    input_size = X_train_tmp.shape[2]  # 读取 shape 以确定特征维度
    print("input_size =", input_size)
    hidden_size = 512
    num_layers = 3
    num_classes = 10
    dropout = 0.5

    # 初始化模型
    # 选择您要使用的模型，取消相应的注释
    model = BiLSTMWithAttention(input_size, hidden_size, num_layers, num_classes, dropout)
    # model = BiLSTM(input_size, hidden_size, num_layers, num_classes, dropout)
    # model = BiGRU(input_size, hidden_size, num_layers, num_classes, dropout)
    # model = TransformerModel(input_size, hidden_size_tr, num_heads, num_layers_tr, num_classes, dropout)

    # 选择设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    print(model)

    # 使用 IterableDataset 构建 DataLoader
    batch_size = 64
    train_dataset = FingeringIterableDataset(train_X_path, train_y_path, batch_size=batch_size, mmap_mode='r')
    val_dataset = FingeringIterableDataset(val_X_path, val_y_path, batch_size=batch_size, mmap_mode='r')

    # DataLoader 传入 batch_size=None，因为 IterableDataset 本身已经按 batch 产出
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 计算类别权重
    # 若需要 class_weight，请先把 y_train_aug 以内存映射或分批取值统计
    # 这里简单演示：只要能一次性读取 y_train_aug 即可
    y_train_full = np.load(train_y_path, mmap_mode='r')
    class_values = np.unique(y_train_full)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=class_values, y=y_train_full)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 假设有效指法为 1,2,3,4,5，未标注记为0, ignore_index=10
    criterion = nn.CrossEntropyLoss(ignore_index=10)
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
        train_count = 0

        # 遍历 train_loader，每个批次由 IterableDataset 产出
        for X_batch, y_batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False, ncols=100,
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_count += X_batch.size(0)

        train_loss /= train_count

        # 验证阶段
        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc="Validation", leave=False, ncols=100,
                                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_count += X_batch.size(0)

        val_loss /= val_count

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 调整学习率
        scheduler.step(val_loss)

        current_lrs = scheduler.get_last_lr()
        print(f"当前学习率: {current_lrs}")

        # Early stopping
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

    if best_model_state:
        model.load_state_dict(best_model_state)

    torch.save(model.state_dict(), 'fingering_bilstm_model.pth')
    print("Model saved as fingering_bilstm_model.pth")

if __name__ == "__main__":
    main()
