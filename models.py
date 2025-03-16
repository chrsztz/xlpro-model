# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformer 模型
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

# CNN 模型
class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(CNNModel, self).__init__()

        # Keep the same parameters interface as BiLSTM for easy swapping
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        # CNN layers
        self.conv_layers = nn.ModuleList()

        # First conv layer takes input features
        self.conv_layers.append(nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
        ))

        # Add additional conv layers based on num_layers
        for i in range(1, num_layers):
            channels = hidden_size * (2 if i > 1 else 1)
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(hidden_size, channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
            ))
            hidden_size = channels

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # Input x shape: [batch, seq_len, features]
        # Transpose for Conv1d: [batch, features, seq_len]
        x = x.transpose(1, 2)

        # Apply CNN layers
        for conv in self.conv_layers:
            x = conv(x)

        # Global pooling
        x = self.global_pool(x)  # Shape: [batch, channels, 1]
        x = x.squeeze(-1)  # Shape: [batch, channels]

        # Fully connected layers for classification
        x = self.fc(x)

        return x

# BiLSTM 模型
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

# BiGRU 模型
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

# Attention 机制
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

# BiLSTM 与 Attention 结合的模型
import torch.nn as nn
import torch

class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, bidirectional=True):
        super(BiLSTMWithAttention, self).__init__()
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention Layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = AttentionMechanism(lstm_output_size)
        
        # Fully Connected Layer
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # print(f"LSTM Output Shape: {lstm_out.shape}")
        context_vector, attn_weights = self.attention(lstm_out)
        # print(f"Context Vector Shape: {context_vector.shape}")
        output = self.fc(context_vector)
        return output



class AttentionMechanism(nn.Module):
    def __init__(self, lstm_output_size):
        super(AttentionMechanism, self).__init__()
        self.attn = nn.Linear(lstm_output_size, lstm_output_size, bias=True)
        self.v = nn.Parameter(torch.randn(lstm_output_size))

    def forward(self, lstm_out):
        # lstm_out: [batch_size, seq_len, lstm_output_size]
        
        # Compute attention scores
        attn_scores = torch.tanh(self.attn(lstm_out))  # [batch_size, seq_len, lstm_output_size]
        attn_scores = torch.matmul(attn_scores, self.v)  # [batch_size, seq_len]

        # Normalize attention scores to probabilities
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch_size, seq_len]

        # Compute the context vector
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * lstm_out, dim=1)  # [batch_size, lstm_output_size]
        
        return context_vector, attn_weights

