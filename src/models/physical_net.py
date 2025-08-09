"""
物理约束网络

实现论文Section 3.4中的物理约束分类器:
PhysNet(Φ) = softmax(WΦ2 · ReLU(WΦ1 · Φ + bΦ1) + bΦ2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PhysicalConstraintNet(nn.Module):
    """
    物理约束分类器网络
    
    专门处理物理约束特征，输出指法预测
    """
    
    def __init__(
        self,
        input_dim: int = 5,          # 物理约束特征维度
        hidden_dim: int = 64,        # 隐藏层维度
        output_dim: int = 11,        # 输出类别数 (-5到5)
        dropout_rate: float = 0.3,   # Dropout率
        use_batch_norm: bool = True  # 是否使用批归一化
    ):
        """
        初始化物理约束网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出类别数
            dropout_rate: Dropout概率
            use_batch_norm: 是否使用批归一化
        """
        super(PhysicalConstraintNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        
        # 构建网络层
        layers = []
        
        # 第一层: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate))
        
        # 第二层: hidden_dim -> hidden_dim
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate))
        
        # 输出层: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 物理约束特征 [batch_size, seq_len, input_dim]
            
        Returns:
            logits: 分类结果 [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 重塑为2D用于网络处理
        x_flat = x.view(-1, input_dim)
        
        # 处理BatchNorm1d的特殊情况
        if self.use_batch_norm and x_flat.size(0) == 1:
            # 如果batch size为1，跳过BatchNorm
            output_flat = self._forward_without_batchnorm(x_flat)
        else:
            output_flat = self.network(x_flat)
        
        # 重塑回3D
        output = output_flat.view(batch_size, seq_len, self.output_dim)
        
        return output
    
    def _forward_without_batchnorm(self, x: torch.Tensor) -> torch.Tensor:
        """无BatchNorm的前向传播 (用于batch_size=1的情况)"""
        for layer in self.network:
            if not isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
        return x


class EnhancedPhysicalConstraintNet(nn.Module):
    """
    增强版物理约束网络
    
    包含更复杂的架构和特征融合
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        output_dim: int = 11,
        num_layers: int = 3,
        dropout_rate: float = 0.3,
        use_residual: bool = True,
        use_attention: bool = False
    ):
        """
        初始化增强版物理约束网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出类别数
            num_layers: 隐藏层数量
            dropout_rate: Dropout概率
            use_residual: 是否使用残差连接
            use_attention: 是否使用内部注意力机制
        """
        super(EnhancedPhysicalConstraintNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )
            self.hidden_layers.append(layer)
        
        # 特征注意力 (可选)
        if use_attention:
            self.feature_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 物理约束特征 [batch_size, seq_len, input_dim]
            
        Returns:
            logits: 分类结果 [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x_flat = x.view(-1, self.input_dim)
        h = self.input_projection(x_flat)
        
        # 隐藏层处理
        for layer in self.hidden_layers:
            if self.use_residual and h.size(-1) == self.hidden_dim:
                h_new = layer(h)
                h = h + h_new  # 残差连接
            else:
                h = layer(h)
        
        # 特征注意力 (可选)
        if self.use_attention:
            attention_weights = self.feature_attention(h)
            h = h * attention_weights
        
        # 输出层
        output_flat = self.output_layer(h)
        
        # 重塑回3D
        output = output_flat.view(batch_size, seq_len, self.output_dim)
        
        return output


class PhysicalConstraintLoss(nn.Module):
    """
    物理约束损失函数
    
    结合交叉熵损失和物理合理性惩罚
    """
    
    def __init__(
        self,
        constraint_weight: float = 0.1,
        smoothness_weight: float = 0.05
    ):
        """
        初始化物理约束损失
        
        Args:
            constraint_weight: 物理约束惩罚权重
            smoothness_weight: 平滑性惩罚权重
        """
        super(PhysicalConstraintLoss, self).__init__()
        
        self.constraint_weight = constraint_weight
        self.smoothness_weight = smoothness_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=5)  # ignore 0 (无标注)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        physical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算损失
        
        Args:
            logits: 预测结果 [batch_size, seq_len, num_classes]
            targets: 真实标签 [batch_size, seq_len]
            physical_features: 物理约束特征 [batch_size, seq_len, constraint_dim]
            
        Returns:
            总损失值
        """
        # 主要交叉熵损失
        target_indices = targets + 5  # -5到5 -> 0到10
        target_indices = torch.clamp(target_indices, 0, 10)
        
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_indices.view(-1)
        
        ce_loss = self.ce_loss(logits_flat, targets_flat)
        
        # 物理约束惩罚
        constraint_penalty = self._compute_constraint_penalty(
            logits, physical_features
        )
        
        # 平滑性惩罚 (相邻预测的一致性)
        smoothness_penalty = self._compute_smoothness_penalty(logits)
        
        # 总损失
        total_loss = (ce_loss + 
                     self.constraint_weight * constraint_penalty +
                     self.smoothness_weight * smoothness_penalty)
        
        return total_loss
    
    def _compute_constraint_penalty(
        self,
        logits: torch.Tensor,
        physical_features: torch.Tensor
    ) -> torch.Tensor:
        """计算物理约束惩罚"""
        # 获取预测概率
        probs = F.softmax(logits, dim=-1)
        
        # 提取物理约束特征
        stretching = physical_features[:, :, 0]
        crossing = physical_features[:, :, 1]
        natural_violation = physical_features[:, :, 3]
        strength_violation = physical_features[:, :, 4]
        
        # 计算总的物理违反程度
        total_violation = (stretching + crossing + 
                          natural_violation + strength_violation)
        
        # 对违反程度高的预测给予更大惩罚
        penalty = torch.mean(total_violation * torch.max(probs, dim=-1)[0])
        
        return penalty
    
    def _compute_smoothness_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        """计算平滑性惩罚"""
        if logits.size(1) <= 1:
            return torch.tensor(0.0, device=logits.device)
        
        # 相邻时间步的预测差异
        diff = logits[:, 1:, :] - logits[:, :-1, :]
        smoothness_loss = torch.mean(torch.sum(diff ** 2, dim=-1))
        
        return smoothness_loss 