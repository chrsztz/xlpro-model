# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, input_size, cnn_channels, lstm_hidden_size, num_classes):
        super().__init__()
        # CNN特征提取模块
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(10)  # 动态调整序列长度
        )

        # BiLSTM时序建模
        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,
            num_heads=4,
            dropout=0.3
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 输入x形状: [batch_size, seq_len, input_size]

        # CNN处理
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        cnn_features = self.cnn(x)  # [batch, cnn_channels*2, 10]
        cnn_features = cnn_features.permute(0, 2, 1)  # [batch, 10, cnn_channels*2]

        # LSTM处理
        lstm_out, _ = self.lstm(cnn_features)  # [batch, 10, lstm_hidden*2]

        # 注意力聚合
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out
        )  # [batch, 10, lstm_hidden*2]

        # 取最后时间步
        final_feature = attn_out[:, -1, :]

        # 分类

        return self.classifier(final_feature)


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


# Positional Encoding for Transformer-based models
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x expected shape: [batch_size, seq_len, embedding_dim]
        pos_encoding = self.pe[:x.size(1), :].transpose(0, 1)
        x = x + pos_encoding
        return self.dropout(x)


# New Improved Model with Simplified Architecture and Hand-Specific Processing
class ImprovedCNNBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=10, dropout=0.3):
        super().__init__()
        
        # Layer normalization for input stabilization
        self.input_norm = nn.LayerNorm(input_size)
        
        # CNN for local pattern extraction - simpler architecture
        self.cnn_layers = nn.Sequential(
            # First CNN block - extract simple patterns
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second CNN block - more complex patterns
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # BiLSTM for sequential modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        
        # Simple attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )
        
        # Classification layers with skip connection
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hand_indices=None):
        batch_size, seq_len, features = x.size()
        
        # Normalize input for stable training
        x = self.input_norm(x)
        
        # CNN feature extraction
        x_cnn = x.transpose(1, 2)  # [batch, features, seq_len]
        x_cnn = self.cnn_layers(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)  # [batch, seq_len, hidden]
        
        # Hand-specific adjustment if hand indices provided
        if hand_indices is not None:
            # Get hand information from the specified index
            if isinstance(hand_indices, int):
                hand_feature = x[:, -1, hand_indices].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
                # Adjust channel importance based on hand (simple attention)
                hand_weight = torch.sigmoid(hand_feature)
                # Broadcast properly to match x_cnn dimensions [batch, seq_len, hidden]
                hand_weight = hand_weight.expand(-1, x_cnn.size(1), 1)
                x_cnn = x_cnn * (1.0 + 0.5 * hand_weight)
        
        # BiLSTM processing
        lstm_out, _ = self.lstm(x_cnn)  # [batch, seq, hidden*2]
        
        # Simple attention mechanism
        attn_scores = self.attention(lstm_out)  # [batch, seq, 1]
        attn_weights = attn_scores / (attn_scores.sum(dim=1, keepdim=True) + 1e-8)  # Add epsilon for numerical stability
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_out)  # [batch, 1, hidden*2]
        context = context.squeeze(1)  # [batch, hidden*2]
        
        # Classification with skip connection for better gradient flow
        out = self.fc1(context)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Store attention weights for visualization
        self.last_attention_weights = attn_weights.detach()
        
        return out

# New Hierarchical Attention Fingering Model
class HierarchicalAttentionFingeringModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=256, num_heads=4, 
                 dropout=0.3, num_layers=3):
        super().__init__()
        
        # Embedding layer
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Local context modeling with CNN
        self.local_context = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
        )
        
        # Sequential context with Transformer
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        # Hierarchical attention
        self.note_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Hand-specific processing
        self.hand_gate = nn.Linear(hidden_size, 2)  # Gate for left/right hand features
        self.hand_specific_left = nn.Linear(hidden_size, hidden_size//2)
        self.hand_specific_right = nn.Linear(hidden_size, hidden_size//2)
        
        # Ergonomic constraints layer
        self.ergonomic_layer = nn.Linear(hidden_size, hidden_size)
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_size//2, num_classes)
        )
        
    def forward(self, x, hand_indices=None):
        # x shape: [batch_size, seq_len, features]
        
        # Initial embedding
        x = self.input_embedding(x)  # [batch, seq_len, hidden_size]
        
        # Local context with CNN
        local_x = x.transpose(1, 2)  # [batch, hidden_size, seq_len]
        local_x = self.local_context(local_x)
        local_x = local_x.transpose(1, 2)  # [batch, seq_len, hidden_size]
        
        # Add positional encoding
        x = self.pos_encoder(local_x)
        
        # Transformer for sequential context
        x = self.transformer_encoder(x)
        
        # Note-level attention
        attn_output, _ = self.note_attention(x, x, x)
        
        # Hand-specific processing
        # Assume hand_indices=2 (index of hand feature in input)
        if hand_indices is not None:
            # Extract hand information (assuming normalized to 0=left, 1=right)
            hand_info = torch.zeros((x.size(0), x.size(1), 2), device=x.device)
            
            # Check if hand_indices is a batch of indices
            if isinstance(hand_indices, torch.Tensor) and hand_indices.dim() > 0:
                for i, idx in enumerate(hand_indices):
                    hand_val = x[i, -1, idx].item()  # Use last timestep's hand info
                    # Set left or right gate based on hand value
                    if hand_val < 0.5:  # Assuming left hand is encoded as 0
                        hand_info[i, :, 0] = 1.0  # Activate left hand gate
                    else:
                        hand_info[i, :, 1] = 1.0  # Activate right hand gate
            else:
                # Use a fixed index for the entire batch
                for i in range(x.size(0)):
                    hand_val = x[i, -1, hand_indices].item()
                    if hand_val < 0.5:
                        hand_info[i, :, 0] = 1.0
                    else:
                        hand_info[i, :, 1] = 1.0
            
            # Process the hand gating information
            hand_gates = torch.sigmoid(self.hand_gate(x))
            # Override with our hand info where available
            hand_gates = hand_gates + hand_info
            
            # Process separately for each hand
            left_features = self.hand_specific_left(x) * hand_gates[:, :, 0].unsqueeze(-1)
            right_features = self.hand_specific_right(x) * hand_gates[:, :, 1].unsqueeze(-1)
            
            # Combine hand-specific features
            combined_features = torch.cat([left_features, right_features], dim=-1)
        else:
            combined_features = x
            
        # Ergonomic constraints
        x = self.ergonomic_layer(combined_features)
        x = F.gelu(x + attn_output)  # Residual connection
        
        # Final classification - predict for each position in the sequence
        predictions = self.classifier(x)
        
        return predictions

