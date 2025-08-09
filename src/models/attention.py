"""
注意力机制

实现论文Section 3.5中的注意力机制:
- 动态聚焦于输入序列的最相关部分
- 计算注意力权重
- 生成上下文向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionLayer(nn.Module):
    """
    注意力层
    
    实现论文中的注意力机制:
    et = tanh(Wa * H_lstm,t + ba)
    αt = exp(va^T * et) / Σj exp(va^T * ej)
    c = Σt αt * H_lstm,t
    """
    
    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int = None
    ):
        """
        初始化注意力层
        
        Args:
            hidden_dim: LSTM隐藏状态维度
            attention_dim: 注意力计算维度 (默认为hidden_dim)
        """
        super(AttentionLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim or hidden_dim
        
        # 注意力参数
        self.W_a = nn.Linear(hidden_dim, self.attention_dim, bias=True)
        self.v_a = nn.Linear(self.attention_dim, 1, bias=False)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.zeros_(self.W_a.bias)
        nn.init.xavier_uniform_(self.v_a.weight)
    
    def forward(
        self,
        lstm_output: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            lstm_output: LSTM输出 [batch_size, seq_len, hidden_dim]
            mask: 序列掩码 [batch_size, seq_len] (可选)
            
        Returns:
            attended_output: 加权后的输出 [batch_size, seq_len, hidden_dim]
            attention_weights: 注意力权重 [batch_size, seq_len]
        """
        batch_size, seq_len, hidden_dim = lstm_output.shape
        
        # 1. 计算注意力能量
        # et = tanh(Wa * H_lstm,t + ba)
        energy = torch.tanh(self.W_a(lstm_output))
        # energy: [batch_size, seq_len, attention_dim]
        
        # 2. 计算注意力分数
        # score = va^T * et
        attention_scores = self.v_a(energy).squeeze(-1)
        # attention_scores: [batch_size, seq_len]
        
        # 3. 应用掩码 (如果提供)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 4. 计算注意力权重
        # αt = softmax(score)
        attention_weights = F.softmax(attention_scores, dim=-1)
        # attention_weights: [batch_size, seq_len]
        
        # 5. 计算加权输出
        # 为每个时间步应用注意力权重
        attended_output = lstm_output * attention_weights.unsqueeze(-1)
        # attended_output: [batch_size, seq_len, hidden_dim]
        
        return attended_output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (可选的增强版本)
    
    提供更强的建模能力，虽然论文中未使用，但可用于实验
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1
    ):
        """
        初始化多头注意力
        
        Args:
            hidden_dim: 隐藏状态维度
            num_heads: 注意力头数
            dropout_rate: Dropout率
        """
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 线性变换层
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 缩放因子
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询 [batch_size, seq_len, hidden_dim]
            key: 键 (默认与query相同)
            value: 值 (默认与query相同)
            mask: 注意力掩码
            
        Returns:
            output: 注意力输出
            attention_weights: 注意力权重
        """
        if key is None:
            key = query
        if value is None:
            value = query
        
        batch_size, seq_len, _ = query.shape
        
        # 1. 线性变换
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 形状: [batch_size, num_heads, seq_len, head_dim]
        
        # 3. 计算注意力
        attention_output, attention_weights = self._scaled_dot_product_attention(
            Q, K, V, mask
        )
        
        # 4. 连接多头输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        # 5. 最终线性变换
        output = self.W_o(attention_output)
        
        return output, attention_weights
    
    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """缩放点积注意力"""
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(Q.device)
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # 扩展维度匹配多头
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    """
    自注意力机制
    
    简化版本，适用于序列内部的依赖建模
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float = 0.1
    ):
        """
        初始化自注意力
        
        Args:
            hidden_dim: 隐藏维度
            dropout_rate: Dropout率
        """
        super(SelfAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        
        # 自注意力参数
        self.W_self = nn.Linear(hidden_dim, hidden_dim)
        self.v_self = nn.Linear(hidden_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.zeros_(self.W_self.bias)
        nn.init.xavier_uniform_(self.v_self.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入 [batch_size, seq_len, hidden_dim]
            mask: 序列掩码
            
        Returns:
            output: 自注意力输出
            weights: 注意力权重
        """
        # 自注意力计算
        energy = torch.tanh(self.W_self(x))
        attention_scores = self.v_self(energy).squeeze(-1)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 全局上下文向量
        context = torch.sum(attention_weights.unsqueeze(-1) * x, dim=1, keepdim=True)
        context = context.expand_as(x)
        
        # 门控机制
        gate = torch.sigmoid(self.W_self(x))
        output = gate * x + (1 - gate) * context
        
        return output, attention_weights 