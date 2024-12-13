import torch
import torch.nn as nn
# from torchcrf import CRF  # 需要安装 torchcrf 库
# 安装命令：pip install torchcrf
import torch.nn.functional as F
import data_process
import numpy as np
from data_process import le_fingering, train_loader, val_loader
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight


# Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, num_classes, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads,
                                                    dim_feedforward=hidden_size * 4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.input_fc(x)  # [batch_size, seq_length, hidden_size]
        x = x.permute(1, 0, 2)  # [seq_length, batch_size, hidden_size]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # [batch_size, seq_length, hidden_size]
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc(x)  # [batch_size, num_classes]
        return x


# BiLSTM
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向 LSTM

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_length, hidden_size*2]
        final_feature = lstm_out[:, -1, :]  # [batch_size, hidden_size*2]
        out = self.fc(final_feature)  # [batch_size, num_classes]
        return out


# BiGRU
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向 GRU

    def forward(self, x):
        gru_out, _ = self.gru(x)  # [batch_size, seq_length, hidden_size*2]
        final_feature = gru_out[:, -1, :]  # [batch_size, hidden_size*2]
        out = self.fc(final_feature)  # [batch_size, num_classes]
        return out


# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.v = nn.Parameter(torch.rand(hidden_size * 2))

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size*2]
        # encoder_outputs: [batch_size, seq_length, hidden_size*2]

        # 计算注意力分数
        attn_scores = torch.tanh(self.attn(encoder_outputs))  # [batch_size, seq_length, hidden_size*2]
        attn_scores = attn_scores @ self.v  # [batch_size, seq_length]

        # 归一化为概率
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, seq_length]

        # 加权求和
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, hidden_size*2]
        context = context.squeeze(1)  # [batch_size, hidden_size*2]

        return context


class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(BiLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向 LSTM

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_length, hidden_size*2]
        context = self.attention(lstm_out[:, -1, :], lstm_out)  # [batch_size, hidden_size*2]
        out = self.fc(context)  # [batch_size, num_classes]
        return out

#文件导入
# 将序列数据保存为 npy 文件
X_train_aug = np.save('X_train_aug.npy')
X_val_aug = np.save('X_val_aug.npy')
y_train_aug = np.save('y_train_aug.npy')
y_val_aug = np.save('y_val_aug.npy')

# 参数设置
input_size = X_train_aug.shape[2]  # 特征数量
# 初始化模型
hidden_size = 256  # 从128减少到64
num_layers = 3  # 从2减少到1
num_classes = len(le_fingering.classes_)
dropout = 0.5

# 初始化 Transformer 模型
hidden_size_tr = 64
num_heads = 4
num_layers_tr = 2

# 初始化模型
# model = BiLSTM(input_size, hidden_size, num_layers, num_classes, dropout)
# model = BiGRU(input_size, hidden_size, num_layers, num_classes, dropout)
# model = TransformerModel(input_size, hidden_size_tr, num_heads, num_layers_tr, num_classes, dropout)
model = BiLSTMWithAttention(input_size, hidden_size, num_layers, num_classes, dropout)

# 选择设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps")
model.to(device)

print(model)

# 假设您已经定义了 FingeringDataset 和 DataLoader
# train_loader 和 val_loader 已经定义

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 添加 weight_decay

# 计算类别权重
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_aug), y=y_train_aug)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)


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


# 使用Focal Loss
criterion = FocalLoss(alpha=1, gamma=2)

# # 更新损失函数
# criterion = nn.CrossEntropyLoss(weight=class_weights)

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