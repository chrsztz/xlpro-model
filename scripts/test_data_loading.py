"""
测试数据加载脚本

验证PIG数据集加载和特征提取是否正常工作
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import yaml
from torch.utils.data import DataLoader
from loguru import logger

from src.data.dataset import PianoFingeringDataset, collate_fn
from src.data.physical_constraints import PhysicalConstraints, create_physical_features


def load_config(config_path: str = None):
    """加载配置文件"""
    if config_path is None:
        config_path = project_root / "configs" / "model_config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def test_pig_dataset(config):
    """测试PIG数据集加载"""
    logger.info("=== 测试PIG数据集加载 ===")
    
    data_dir = project_root / config['data']['data_dir']
    
    try:
        # 创建训练数据集
        train_dataset = PianoFingeringDataset(
            data_dir=data_dir,
            split="train",
            sequence_length=config['data']['sequence_length']
        )
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        
        # 获取数据集统计信息
        stats = train_dataset.get_stats()
        logger.info("数据集统计信息:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # 测试单个样本
        sample = train_dataset[0]
        logger.info(f"样本结构: {sample.keys()}")
        logger.info(f"音符数量: {len(sample['notes'])}")
        
        # 测试批处理
        dataloader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        batch = next(iter(dataloader))
        logger.info(f"批处理特征形状: {batch['features'].shape}")
        logger.info(f"批处理标签形状: {batch['labels'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        return False


def test_physical_constraints(config):
    """测试物理约束计算"""
    logger.info("=== 测试物理约束计算 ===")
    
    try:
        # 创建物理约束计算器
        constraints = PhysicalConstraints(
            velocity_threshold=config['physical_constraints']['velocity_threshold'],
            finger_weights=config['physical_constraints']['finger_weights']
        )
        
        # 创建测试音符序列
        test_notes = [
            {
                'midi_number': 60, 'channel': 0, 'finger': 1,
                'onset_velocity': 70, 'is_black_key': False
            },
            {
                'midi_number': 62, 'channel': 0, 'finger': 2,
                'onset_velocity': 75, 'is_black_key': False
            },
            {
                'midi_number': 64, 'channel': 0, 'finger': 3,
                'onset_velocity': 80, 'is_black_key': False
            },
        ]
        
        # 计算约束特征
        constraint_features = constraints.compute_sequence_constraints(test_notes)
        logger.info(f"约束特征形状: {constraint_features.shape}")
        logger.info(f"约束特征示例:\n{constraint_features}")
        
        # 测试单个约束计算
        constraints_dict = constraints.compute_all_constraints(test_notes, 1)
        logger.info("单个音符约束:")
        for key, value in constraints_dict.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"物理约束计算失败: {e}")
        return False


def test_model_creation(config):
    """测试模型创建"""
    logger.info("=== 测试模型创建 ===")
    
    try:
        from src.models.cnn_bilstm import create_model
        
        # 创建模型
        model = create_model(config['model'])
        logger.info(f"模型创建成功")
        logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        batch_size = 4
        seq_len = config['data']['sequence_length']
        input_dim = config['model']['input_dim']
        
        # 创建测试输入
        test_input = torch.randn(batch_size, seq_len, input_dim)
        test_physical = torch.randn(batch_size, seq_len, config['model']['physical_features_dim'])
        
        # 前向传播
        with torch.no_grad():
            outputs = model(test_input, test_physical)
        
        logger.info(f"模型输出形状: {outputs['logits'].shape}")
        logger.info("模型前向传播测试成功")
        
        return True
        
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        return False


def main():
    """主测试函数"""
    logger.info("开始运行数据加载和模型测试")
    
    # 加载配置
    config = load_config()
    logger.info("配置文件加载成功")
    
    # 设置随机种子
    torch.manual_seed(config['experiment']['seed'])
    
    # 运行测试
    tests = [
        ("PIG数据集加载", lambda: test_pig_dataset(config)),
        ("物理约束计算", lambda: test_physical_constraints(config)), 
        ("模型创建", lambda: test_model_creation(config))
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"运行测试: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
            
            if success:
                logger.info(f"✅ {test_name} - 通过")
            else:
                logger.error(f"❌ {test_name} - 失败")
                
        except Exception as e:
            logger.error(f"❌ {test_name} - 异常: {e}")
            results[test_name] = False
    
    # 输出总结
    logger.info(f"\n{'='*50}")
    logger.info("测试总结")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！项目基础框架工作正常。")
    else:
        logger.warning("⚠️  部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    main() 