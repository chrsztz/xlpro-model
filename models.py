# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加物理约束增强模型
class PhysicalEnhancedFingeringModel(nn.Module):
    """
    物理约束增强型指法预测模型
    
    结合了深度学习的特征提取和物理约束评估，使用双阶段训练和前向规划
    """
    def __init__(self, input_size, hidden_size, num_classes=10, num_physical_features=12, 
                 physical_weight=0.3, dropout=0.4):
        super(PhysicalEnhancedFingeringModel, self).__init__()
        
        self.physical_weight = physical_weight  # 物理约束损失权重
        
        # CNN+BiLSTM 特征提取部分
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1)
        )
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 物理约束评估网络 - 接收物理特征输入
        # 确保输入维度匹配，默认使用9个物理特征
        self.num_physical_features = num_physical_features
        self.physical_net = nn.Sequential(
            nn.Linear(num_physical_features, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, num_classes),
            nn.Softmax(dim=1)
        )
        
        # 主分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x, physical_features=None):
        """
        Args:
            x: 输入特征序列 [batch_size, seq_len, input_size]
            physical_features: 物理约束特征 [batch_size, num_physical_features]
                            如果为None，则仅使用主分类器
        """
        batch_size, seq_len, features = x.size()
        
        # CNN特征提取
        x_reshaped = x.transpose(1, 2)  # [batch, input_size, seq_len]
        conv_out = self.conv_block(x_reshaped)
        conv_out = conv_out.transpose(1, 2)  # [batch, seq_len, hidden_size]
        
        # BiLSTM处理
        lstm_out, _ = self.lstm(conv_out)  # [batch, seq_len, hidden_size]
        
        # 注意力机制
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden_size]
        
        # 主分类器预测
        main_logits = self.classifier(context)
        
        if physical_features is not None:
            # 物理约束分类器预测
            physical_logits = self.physical_net(physical_features)
            
            # 融合两个分类器的结果
            # main_logits是原始logits，physical_logits是softmax后的概率
            # 转换main_logits为概率
            main_probs = F.softmax(main_logits, dim=1)
            
            # 加权融合
            combined_probs = (1 - self.physical_weight) * main_probs + self.physical_weight * physical_logits
            
            # 返回logits和概率
            return main_logits, combined_probs
        else:
            # 仅使用主分类器
            return main_logits, F.softmax(main_logits, dim=1)
    
    def extract_physical_features(self, x):
        """
        从输入序列中提取与物理约束相关的特征
        实现前向规划能力，评估多个可能的指法序列
        
        Args:
            x: 输入特征序列 [batch_size, seq_len, input_size]
            
        Returns:
            物理特征 [batch_size, num_physical_features]
        """
        # 这里实现从输入特征提取物理约束相关特征的逻辑
        # 在实际应用中，可以使用滑动窗口提取特征或直接从特征向量中选择物理约束相关维度
        
        # 确保提取正确数量的物理特征，与模型期望的num_physical_features匹配
        batch_size = x.size(0)
        # 获取最后一个时间步的最后n个特征
        if x.size(2) >= self.num_physical_features:
            return x[:, -1, -self.num_physical_features:]
        else:
            # 如果特征不足，填充到必要的数量
            features = x[:, -1, :]  # 获取所有可用特征
            padding = torch.zeros(batch_size, self.num_physical_features - x.size(2), device=x.device)
            return torch.cat([features, padding], dim=1)
    
    def forward_planning(self, x, top_k=3, max_depth=3):
        """
        实现前向规划，类似于RL的规划过程
        
        Args:
            x: 输入特征序列 [batch_size, seq_len, input_size]
            top_k: 每步考虑的最佳动作数
            max_depth: 最大规划深度
            
        Returns:
            最佳指法预测 [batch_size]
        """
        # 阶段1：通过主网络获取初始预测
        main_logits, main_probs = self.forward(x)
        
        # 获取每个样本top_k个最可能的指法
        _, top_indices = torch.topk(main_probs, top_k, dim=1)
        
        batch_size = x.size(0)
        best_scores = torch.zeros(batch_size).to(x.device)
        best_actions = torch.zeros(batch_size, dtype=torch.long).to(x.device)
        
        # 阶段2：对每个样本的top_k个指法进行前向规划和评估
        for b in range(batch_size):
            max_score = float('-inf')
            best_action = -1
            
            # 提取物理特征
            physical_features = self.extract_physical_features(x[b:b+1])
            
            # 确保物理特征的形状符合预期
            if physical_features.size(1) != self.num_physical_features:
                # 如果维度不匹配，调整为正确的维度
                if physical_features.size(1) > self.num_physical_features:
                    physical_features = physical_features[:, :self.num_physical_features]
                else:
                    padding = torch.zeros(1, self.num_physical_features - physical_features.size(1), device=x.device)
                    physical_features = torch.cat([physical_features, padding], dim=1)
            
            # 评估每个候选指法
            for action_idx in top_indices[b]:
                # 这里可以扩展为多步前向规划，递归评估多个步骤
                # 简化实现：直接使用物理网络评估单步分数
                action_score = main_probs[b, action_idx] * (1 + self.physical_net(physical_features)[0, action_idx])
                
                if action_score > max_score:
                    max_score = action_score
                    best_action = action_idx
            
            best_scores[b] = max_score
            best_actions[b] = best_action
        
        return best_actions

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

# 定义新的 CNN 模型
class CNNWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.5):
        super(CNNWithAttention, self).__init__()
        
        # 输入标准化
        self.input_norm = nn.LayerNorm(input_size)
        
        # CNN架构 - 使用不同大小的卷积核捕获不同尺度的特征
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_size, hidden_size, kernel_size=7, padding=3)
        
        # 批量归一化 - 加速训练并提高稳定性
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        
        # 注意力层 - 增强对序列中重要部分的关注
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入形状: [batch_size, seq_length, input_size]
        batch_size, seq_len, features = x.size()
        
        # 应用输入标准化
        x = self.input_norm(x)
        
        # 转换为卷积所需的形状
        x_conv = x.transpose(1, 2)  # [batch_size, input_size, seq_length]
        
        # 应用不同尺度的卷积
        conv1_out = F.relu(self.bn1(self.conv1(x_conv)))  # [batch_size, hidden_size, seq_length]
        conv2_out = F.relu(self.bn2(self.conv2(x_conv)))
        conv3_out = F.relu(self.bn3(self.conv3(x_conv)))
        
        # 转回序列形式
        conv1_out = conv1_out.transpose(1, 2)  # [batch_size, seq_length, hidden_size]
        conv2_out = conv2_out.transpose(1, 2)
        conv3_out = conv3_out.transpose(1, 2)
        
        # 组合不同尺度的卷积特征
        conv_cat = torch.cat([conv1_out, conv2_out, conv3_out], dim=2)  # [batch_size, seq_length, hidden_size*3]
        
        # 应用注意力机制
        attn_weights = self.attention(conv_cat)  # [batch_size, seq_length, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * conv_cat, dim=1)  # [batch_size, hidden_size*3]
        
        # 输出分类结果
        output = self.fc(context)
        
        return output
    
# Enhanced fingering model with CNN, BiLSTM, and attention
class EnhancedFingeringModel(nn.Module):
    """
    增强版指法预测模型
    多头注意力+ResNet风格的残差连接+Dropout+BatchNorm+LayerNorm
    """
    def __init__(self, input_size, hidden_size, num_classes=10, num_heads=4):
        super(EnhancedFingeringModel, self).__init__()
        
        # CNN特征提取
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size//2,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True
        )
        
        # 多头自注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads,
            dropout=0.1
        )
        
        # 输出层
        self.global_attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size//2, num_classes)
        )
        
        # 分类器 - 单独处理每个类
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_classes)
        ])
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # CNN特征提取
        x_cnn = x.transpose(1, 2)  # [batch, features, seq_len]
        x_cnn = self.conv_layers(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)  # [batch, seq_len, hidden_size]
        
        # BiLSTM处理
        lstm_out, _ = self.lstm(x_cnn)  # [batch, seq_len, hidden_size]
        
        # 多头自注意力
        lstm_out_t = lstm_out.transpose(0, 1)  # [seq_len, batch, hidden_size]
        attn_out, _ = self.multihead_attn(lstm_out_t, lstm_out_t, lstm_out_t)
        attn_out = attn_out.transpose(0, 1)  # [batch, seq_len, hidden_size]
        
        # 残差连接和层归一化
        combined = lstm_out + attn_out  # 残差连接
        combined = F.layer_norm(combined, [combined.size(-1)])  # 层归一化
        
        # 全局注意力
        attn_weights = self.global_attn(combined)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * combined, dim=1)
        
        # 主输出 - 共享特征
        main_logits = self.fc_layers(context)
        
        # 单独的分类器
        class_logits = torch.cat([cls(context) for cls in self.classifiers], dim=1)
        
        # 混合主输出和专用分类器 (主要使用主输出，但给予专用分类器一些权重)
        mixed_logits = main_logits * 0.7 + class_logits * 0.3
        
        return mixed_logits