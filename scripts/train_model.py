"""
模型训练脚本

功能:
1. 加载预处理后的数据集
2. 创建CNN-BiLSTM模型
3. 配置训练参数和优化器
4. 执行完整的训练循环
5. 保存最佳模型和训练历史
6. 支持断点续训

运行:
  python scripts/train_model.py --data_dir PIGdata --experiment_name my_experiment
  python scripts/train_model.py --resume checkpoints/my_experiment/latest_checkpoint.pth
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import yaml
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from loguru import logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import PianoFingeringDataset, collate_fn
from src.models.cnn_bilstm import create_model
from src.training.trainer import create_trainer


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """设置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    
    # 控制台输出
    logger.add(sys.stderr, level=level, 
              format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    
    # 文件输出(可选)
    if log_file:
        logger.add(log_file, level="DEBUG", 
                  format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                  rotation="10 MB")


def load_config(config_path: str = None) -> Dict:
    """加载配置文件"""
    if config_path is None:
        config_path = project_root / "configs" / "model_config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def update_config_for_experiment(config: Dict, args) -> Dict:
    """根据命令行参数更新配置"""
    # 更新实验名称
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    
    # 更新数据路径
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    
    # 更新设备
    if args.device:
        config['experiment']['device'] = args.device
    
    # 更新训练参数
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    return config


def create_datasets(config: Dict) -> tuple:
    """创建训练和验证数据集"""
    data_dir = Path(config['data']['data_dir'])
    sequence_length = config['data']['sequence_length']
    
    logger.info(f"加载数据集: {data_dir}")
    
    # 创建数据集
    train_dataset = PianoFingeringDataset(
        data_dir=data_dir,
        split="train",
        sequence_length=sequence_length
    )
    
    val_dataset = PianoFingeringDataset(
        data_dir=data_dir,
        split="val",
        sequence_length=sequence_length
    )
    
    test_dataset = PianoFingeringDataset(
        data_dir=data_dir,
        split="test",
        sequence_length=sequence_length
    )
    
    # 输出数据集信息
    logger.info(f"训练集: {len(train_dataset)} 序列")
    logger.info(f"验证集: {len(val_dataset)} 序列")
    logger.info(f"测试集: {len(test_dataset)} 序列")
    
    # 验证数据集不为空
    if len(train_dataset) == 0:
        raise ValueError("训练集为空")
    if len(val_dataset) == 0:
        raise ValueError("验证集为空")
    
    return train_dataset, val_dataset, test_dataset


def save_experiment_config(config: Dict, save_dir: Path):
    """保存实验配置"""
    config_path = save_dir / 'experiment_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"实验配置已保存: {config_path}")


def print_model_info(model, config: Dict):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("🧠 模型架构信息:")
    logger.info(f"  输入维度: {config['model']['input_dim']}")
    logger.info(f"  隐藏层大小: {config['model']['hidden_size']}")
    logger.info(f"  CNN层数: {config['model']['num_cnn_layers']}")
    logger.info(f"  BiLSTM层数: {config['model']['num_lstm_layers']}")
    logger.info(f"  输出类别数: {config['model']['num_classes']}")
    logger.info(f"  物理约束权重: {config['model']['physical_weight']}")
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")


def main():
    parser = argparse.ArgumentParser(description="CNN-BiLSTM模型训练")
    
    # 数据和模型参数
    parser.add_argument('--data_dir', type=str, default='PIGdata',
                       help='数据集目录路径')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学习率')
    parser.add_argument('--device', type=str, default=None,
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='训练设备')
    
    # 续训和输出
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_file', type=str, default=None,
                       help='日志文件路径')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细日志输出')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose, args.log_file)
    
    logger.info("🎹 开始CNN-BiLSTM模型训练")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
    
    try:
        # 加载配置
        config = load_config(args.config)
        config = update_config_for_experiment(config, args)
        
        # 设置实验名称和输出目录
        if not args.experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config['experiment']['name'] = f"cnn_bilstm_{timestamp}"
        
        # 更新保存和日志路径
        config['experiment']['save_dir'] = args.output_dir
        config['experiment']['log_dir'] = f"{args.output_dir}/logs"
        
        logger.info(f"实验名称: {config['experiment']['name']}")
        
        # 创建输出目录
        save_dir = Path(args.output_dir) / config['experiment']['name']
        save_dir.mkdir(parents=True, exist_ok=True)
        
        log_dir = Path(config['experiment']['log_dir']) / config['experiment']['name']
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存实验配置
        save_experiment_config(config, save_dir)
        
        # 创建数据集
        logger.info("📚 准备数据集...")
        train_dataset, val_dataset, test_dataset = create_datasets(config)
        
        # 创建模型
        logger.info("🏗️ 创建模型...")
        model = create_model(config['model'])
        print_model_info(model, config)
        
        # 创建训练器
        logger.info("🎯 配置训练器...")
        trainer = create_trainer(model, config)
        
        # 从检查点恢复(可选)
        start_epoch = 0
        if args.resume:
            logger.info(f"📂 从检查点恢复: {args.resume}")
            checkpoint = trainer.load_checkpoint(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"从第 {start_epoch} 轮继续训练")
        
        # 开始训练
        logger.info("🚀 开始训练...")
        logger.info("=" * 80)
        
        try:
            history = trainer.train(train_dataset, val_dataset)
            
            # 保存训练历史
            history_path = save_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                # 转换numpy类型为Python原生类型
                serializable_history = {}
                for key, values in history.items():
                    if isinstance(values, list):
                        serializable_history[key] = [float(v) if hasattr(v, 'item') else v for v in values]
                    else:
                        serializable_history[key] = values
                
                json.dump(serializable_history, f, indent=2)
            
            logger.info(f"训练历史已保存: {history_path}")
            
            # 输出最终结果
            logger.info("=" * 80)
            logger.info("🎉 训练完成!")
            logger.info(f"最佳验证准确率: {trainer.best_val_accuracy:.4f}")
            logger.info(f"最佳模型保存在: {save_dir / 'best_model.pth'}")
            logger.info(f"训练历史保存在: {history_path}")
            
            return 0
            
        except KeyboardInterrupt:
            logger.info("❌ 训练被用户中断")
            trainer._save_checkpoint(trainer.current_epoch, is_best=False)
            logger.info("💾 已保存当前训练状态")
            return 1
            
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        logger.exception("详细错误信息:")
        return 1


if __name__ == "__main__":
    exit(main())