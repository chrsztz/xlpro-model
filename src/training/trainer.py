"""
训练器

实现完整的模型训练pipeline，包括训练循环、验证、早停等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from loguru import logger
import numpy as np
from tqdm import tqdm

from ..data.dataset import PianoFingeringDataset, collate_fn
from ..data.physical_constraints import create_physical_features
from ..models.cnn_bilstm import CNNBiLSTMModel
from ..models.physical_net import PhysicalConstraintLoss


class Trainer:
    """
    模型训练器
    
    负责模型的训练、验证、保存等功能
    """
    
    def __init__(
        self,
        model: CNNBiLSTMModel,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        初始化训练器
        
        Args:
            model: 待训练的模型
            config: 训练配置
            device: 训练设备
        """
        self.model = model
        self.config = config
        self.device = device or self._get_device()
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 设置优化器
        self.optimizer = self._setup_optimizer()
        
        # 设置学习率调度器
        self.scheduler = self._setup_scheduler()
        
        # 设置损失函数
        self.criterion = self._setup_criterion()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
        # 设置保存目录
        self.save_dir = Path(config['experiment']['save_dir']) / config['experiment']['name']
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.log_dir = Path(config['experiment']['log_dir']) / config['experiment']['name']
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        logger.info(f"训练器初始化完成，设备: {self.device}")
    
    def _get_device(self) -> torch.device:
        """自动选择训练设备"""
        device_config = self.config['experiment']['device']
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"自动选择CUDA设备: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("自动选择MPS设备")
            else:
                device = torch.device('cpu')
                logger.info("自动选择CPU设备")
        else:
            device = torch.device(device_config)
            logger.info(f"使用指定设备: {device}")
        
        return device
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """设置优化器"""
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        logger.info(f"优化器设置: {optimizer_name}, lr={lr}, weight_decay={weight_decay}")
        return optimizer
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """设置学习率调度器"""
        scheduler_name = self.config['training'].get('lr_scheduler', 'plateau')
        
        if scheduler_name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config['training']['lr_factor'],
                patience=self.config['training']['lr_patience'],
                verbose=True
            )
        elif scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training'].get('step_size', 10),
                gamma=self.config['training']['lr_factor']
            )
        else:
            scheduler = None
        
        logger.info(f"学习率调度器: {scheduler_name}")
        return scheduler
    
    def _setup_criterion(self) -> nn.Module:
        """设置损失函数"""
        # 计算类别权重 (处理数据不平衡)
        class_weights = self._compute_class_weights()
        
        if self.config['model'].get('use_physical_loss', True):
            criterion = PhysicalConstraintLoss(
                constraint_weight=0.1,
                smoothness_weight=0.05
            )
        else:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=5  # 忽略 0 (无标注)
            )
        
        return criterion.to(self.device)
    
    def _compute_class_weights(self) -> Optional[torch.Tensor]:
        """计算类别权重"""
        # 简化版本：为所有类别设置相等权重
        # 在实际应用中，可以基于数据分布计算权重
        num_classes = self.config['model']['num_classes']
        weights = torch.ones(num_classes)
        return weights.to(self.device)
    
    def train(
        self,
        train_dataset: PianoFingeringDataset,
        val_dataset: PianoFingeringDataset
    ) -> Dict:
        """
        开始训练
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            
        Returns:
            训练历史字典
        """
        logger.info("开始训练...")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['evaluation']['eval_batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # 开始训练循环
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # 训练阶段
            train_metrics = self._train_epoch(train_loader)
            
            # 验证阶段
            if epoch % self.config['training']['validate_every'] == 0:
                val_metrics = self._validate_epoch(val_loader)
            else:
                val_metrics = {'loss': float('inf'), 'accuracy': 0.0}
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # 记录到tensorboard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印训练信息
            logger.info(
                f"Epoch {epoch+1}/{self.config['training']['num_epochs']} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # 保存最佳模型
            is_best = val_metrics['accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_val_accuracy = val_metrics['accuracy']
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
                logger.info(f"新的最佳模型！验证准确率: {val_metrics['accuracy']:.4f}")
            else:
                self.patience_counter += 1
            
            # 定期保存模型
            if epoch % self.config['training']['save_every'] == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # 早停检查
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                logger.info(f"早停触发！连续{self.patience_counter}个epoch无改善")
                break
        
        logger.info("训练完成！")
        logger.info(f"最佳验证准确率: {self.best_val_accuracy:.4f}")
        
        # 关闭writer
        self.writer.close()
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移动到设备
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 计算物理约束特征
            physical_features = self._extract_physical_features(batch)
            
            # 前向传播
            outputs = self.model(features, physical_features)
            
            # 计算损失
            loss_dict = self._compute_loss(outputs, labels, physical_features)
            loss = loss_dict['total_loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config['training'].get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
            
            self.optimizer.step()
            
            # 计算准确率
            predictions = torch.argmax(outputs['logits'], dim=-1)
            # 转换标签 (-5到5 -> 0到10)
            target_indices = labels + 5
            target_indices = torch.clamp(target_indices, 0, 10)
            
            # 忽略无标注的位置 (原始标签为0，转换后为5)
            mask = target_indices != 5
            if mask.sum() > 0:
                correct = (predictions[mask] == target_indices[mask]).sum().item()
                samples = mask.sum().item()
                
                total_correct += correct
                total_samples += samples
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{total_correct/max(total_samples, 1):.4f}"
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_correct / max(total_samples, 1)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation Epoch {self.current_epoch+1}")
            
            for batch in pbar:
                # 数据移动到设备
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 计算物理约束特征
                physical_features = self._extract_physical_features(batch)
                
                # 前向传播
                outputs = self.model(features, physical_features)
                
                # 计算损失
                loss_dict = self._compute_loss(outputs, labels, physical_features)
                loss = loss_dict['total_loss']
                
                # 计算准确率
                predictions = torch.argmax(outputs['logits'], dim=-1)
                target_indices = labels + 5
                target_indices = torch.clamp(target_indices, 0, 10)
                
                mask = target_indices != 5
                if mask.sum() > 0:
                    correct = (predictions[mask] == target_indices[mask]).sum().item()
                    samples = mask.sum().item()
                    
                    total_correct += correct
                    total_samples += samples
                
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{total_correct/max(total_samples, 1):.4f}"
                })
        
        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_correct / max(total_samples, 1)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def _extract_physical_features(self, batch: Dict) -> Optional[torch.Tensor]:
        """从批次数据中提取物理约束特征"""
        if not self.config['model'].get('use_physical_constraints', True):
            return None
        
        # 直接从batch中获取原始notes
        if 'notes' in batch and len(batch['notes']) > 0 and len(batch['notes'][0]) > 0:
            batch_notes = batch['notes']
            return create_physical_features(batch_notes).to(self.device)
        
        # 回退：若不存在notes，则返回零特征
        batch_size, seq_len = batch['features'].shape[:2]
        return torch.zeros(batch_size, seq_len, 5).to(self.device)
    
    def _compute_loss(
        self,
        outputs: Dict,
        labels: torch.Tensor,
        physical_features: Optional[torch.Tensor]
    ) -> Dict:
        """计算损失"""
        logits = outputs['logits']
        
        if isinstance(self.criterion, PhysicalConstraintLoss):
            # 使用物理约束损失
            if physical_features is not None:
                loss = self.criterion(logits, labels, physical_features)
            else:
                # 如果没有物理特征，退回到标准交叉熵
                target_indices = labels + 5
                target_indices = torch.clamp(target_indices, 0, 10)
                
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = target_indices.view(-1)
                
                ce_loss = nn.CrossEntropyLoss(ignore_index=5)
                loss = ce_loss(logits_flat, targets_flat)
        else:
            # 标准交叉熵损失
            target_indices = labels + 5
            target_indices = torch.clamp(target_indices, 0, 10)
            
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_indices.view(-1)
            
            loss = self.criterion(logits_flat, targets_flat)
        
        return {
            'total_loss': loss,
            'main_loss': loss
        }
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config
        }
        
        # 保存最新的检查点
        latest_path = self.save_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
        
        # 保存定期检查点
        if epoch % self.config['training']['save_every'] == 0:
            epoch_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        
        logger.info(f"加载检查点: epoch {self.current_epoch}, "
                   f"最佳验证准确率: {self.best_val_accuracy:.4f}")
        
        return checkpoint


def create_trainer(model: CNNBiLSTMModel, config: Dict) -> Trainer:
    """
    创建训练器
    
    Args:
        model: 待训练的模型
        config: 配置字典
        
    Returns:
        初始化的训练器
    """
    trainer = Trainer(model, config)
    return trainer 