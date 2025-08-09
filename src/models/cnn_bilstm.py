"""
CNN-BiLSTM混合模型

实现论文Section 3.4中的混合架构:
- CNN特征提取 (局部模式)
- BiLSTM序列建模 (长期依赖)
- 注意力机制
- 双分类器方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from loguru import logger

from .attention import AttentionLayer
from .physical_net import PhysicalConstraintNet


class CNNBiLSTMModel(nn.Module):
    """
    CNN-BiLSTM混合模型
    
    架构按照论文Figure 2:
    Input → CNN → BiLSTM → Attention → Dual Classifiers → Output
    """
    
    def __init__(
        self,
        input_dim: int = 5,           # 基础特征维度
        hidden_size: int = 128,       # 隐藏层大小 (论文使用128)
        num_cnn_layers: int = 2,      # CNN层数
        num_lstm_layers: int = 2,     # BiLSTM层数
        num_classes: int = 11,        # 指法类别数 (-5到5)
        dropout_rate: float = 0.3,    # Dropout率
        physical_features_dim: int = 5, # 物理约束特征维度
        physical_weight: float = 0.3,   # 物理约束权重 λ
        use_attention: bool = True,     # 是否使用注意力机制
        use_batch_norm: bool = True     # 是否使用批归一化
    ):
        """
        初始化CNN-BiLSTM模型
        
        Args:
            input_dim: 输入特征维度
            hidden_size: 隐藏层大小
            num_cnn_layers: CNN层数
            num_lstm_layers: BiLSTM层数  
            num_classes: 输出类别数
            dropout_rate: Dropout概率
            physical_features_dim: 物理约束特征维度
            physical_weight: 物理约束权重
            use_attention: 是否使用注意力
            use_batch_norm: 是否使用批归一化
        """
        super(CNNBiLSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.physical_weight = physical_weight
        self.use_attention = use_attention
        
        # 1. CNN特征提取层
        self.cnn_layers = self._build_cnn_layers(
            input_dim, hidden_size, num_cnn_layers, 
            dropout_rate, use_batch_norm
        )
        
        # 2. BiLSTM序列建模层
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0
        )
        
        # BiLSTM输出维度是2*hidden_size (双向)
        lstm_output_dim = hidden_size * 2
        
        # 3. 注意力机制 (可选)
        if self.use_attention:
            self.attention = AttentionLayer(lstm_output_dim)
            classifier_input_dim = lstm_output_dim
        else:
            classifier_input_dim = lstm_output_dim
        
        # 4. 主分类器
        self.main_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 5. 物理约束分类器
        self.physical_classifier = PhysicalConstraintNet(
            input_dim=physical_features_dim,
            hidden_dim=hidden_size // 2,
            output_dim=num_classes,
            dropout_rate=dropout_rate
        )
        
        # 权重初始化
        self._initialize_weights()
        
        logger.info(f"CNNBiLSTM模型初始化完成: "
                   f"input_dim={input_dim}, hidden_size={hidden_size}, "
                   f"classes={num_classes}")
    
    def _build_cnn_layers(
        self,
        input_dim: int,
        hidden_size: int, 
        num_layers: int,
        dropout_rate: float,
        use_batch_norm: bool
    ) -> nn.ModuleList:
        """构建CNN特征提取层"""
        layers = nn.ModuleList()
        
        # 第一层：input_dim -> hidden_size
        conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )
        layers.append(conv1)
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Dropout(dropout_rate))
        
        # 后续层：hidden_size -> hidden_size
        for i in range(1, num_layers):
            conv = nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=3,
                padding=1
            )
            layers.append(conv)
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
        
        return layers
    
    def _initialize_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for param in module.parameters():
                    if len(param.shape) >= 2:
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        physical_features: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            physical_features: 物理约束特征 [batch_size, seq_len, physical_dim] 
            return_attention: 是否返回注意力权重
            
        Returns:
            包含预测结果的字典
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. CNN特征提取
        # 转换维度: [batch, seq, features] -> [batch, features, seq]
        x_cnn = x.transpose(1, 2)
        
        for layer in self.cnn_layers:
            x_cnn = layer(x_cnn)
        
        # 转换回: [batch, features, seq] -> [batch, seq, features]
        x_cnn = x_cnn.transpose(1, 2)
        
        # 2. BiLSTM序列建模
        lstm_out, (h_n, c_n) = self.bilstm(x_cnn)
        # lstm_out: [batch_size, seq_len, 2*hidden_size]
        
        # 3. 注意力机制 (可选)
        attention_weights = None
        if self.use_attention:
            attended_out, attention_weights = self.attention(lstm_out)
            # attended_out: [batch_size, seq_len, 2*hidden_size]
            classifier_input = attended_out
        else:
            classifier_input = lstm_out
        
        # 4. 主分类器预测
        main_logits = self.main_classifier(classifier_input)
        # main_logits: [batch_size, seq_len, num_classes]
        
        # 5. 物理约束分类器预测 (如果提供了物理特征)
        if physical_features is not None:
            physical_logits = self.physical_classifier(physical_features)
            # physical_logits: [batch_size, seq_len, num_classes]
            
            # 6. 组合预测 (按照论文公式)
            main_probs = F.softmax(main_logits, dim=-1)
            physical_probs = F.softmax(physical_logits, dim=-1)
            
            # ŷ_final = (1-λ) * softmax(ŷ_main) + λ * ŷ_phys
            final_probs = ((1 - self.physical_weight) * main_probs + 
                          self.physical_weight * physical_probs)
            final_logits = torch.log(final_probs + 1e-8)  # 避免log(0)
        else:
            physical_logits = None
            final_logits = main_logits
        
        # 组织输出结果
        outputs = {
            'logits': final_logits,           # 最终预测结果
            'main_logits': main_logits,       # 主分类器结果
            'physical_logits': physical_logits, # 物理约束分类器结果
            'lstm_features': lstm_out,        # LSTM特征 (用于可视化分析)
            'cnn_features': x_cnn            # CNN特征 (用于可视化分析)
        }
        
        if return_attention and attention_weights is not None:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def predict(
        self,
        x: torch.Tensor,
        physical_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测指法
        
        Args:
            x: 输入特征
            physical_features: 物理约束特征
            
        Returns:
            预测的指法 [batch_size, seq_len]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, physical_features)
            predictions = torch.argmax(outputs['logits'], dim=-1)
        
        return predictions
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        physical_features: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        获取注意力权重 (用于可视化)
        
        Args:
            x: 输入特征
            physical_features: 物理约束特征
            
        Returns:
            注意力权重 [batch_size, seq_len, seq_len] 或 None
        """
        if not self.use_attention:
            return None
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, physical_features, return_attention=True)
            return outputs.get('attention_weights')
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        physical_features: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失函数
        
        Args:
            logits: 模型输出logits
            targets: 真实标签 [batch_size, seq_len]
            physical_features: 物理约束特征
            class_weights: 类别权重
            
        Returns:
            损失值字典
        """
        # 将指法标签转换为类别索引 (-5到5 -> 0到10)
        target_indices = targets + 5
        target_indices = torch.clamp(target_indices, 0, self.num_classes - 1)
        
        # 主要交叉熵损失
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=5)  # ignore 0 (无标注)
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=5)  # ignore 0 (无标注)
        
        # 重塑张量用于损失计算
        logits_flat = logits.view(-1, self.num_classes)
        targets_flat = target_indices.view(-1)
        
        main_loss = criterion(logits_flat, targets_flat)
        
        loss_dict = {
            'total_loss': main_loss,
            'main_loss': main_loss
        }
        
        return loss_dict


def create_model(config: Dict) -> CNNBiLSTMModel:
    """
    根据配置创建模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        初始化的CNN-BiLSTM模型
    """
    model = CNNBiLSTMModel(
        input_dim=config.get('input_dim', 5),
        hidden_size=config.get('hidden_size', 128),
        num_cnn_layers=config.get('num_cnn_layers', 2),
        num_lstm_layers=config.get('num_lstm_layers', 2),
        num_classes=config.get('num_classes', 11),
        dropout_rate=config.get('dropout_rate', 0.3),
        physical_features_dim=config.get('physical_features_dim', 5),
        physical_weight=config.get('physical_weight', 0.3),
        use_attention=config.get('use_attention', True),
        use_batch_norm=config.get('use_batch_norm', True)
    )
    
    return model 