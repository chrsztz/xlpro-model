"""
数据集预处理脚本

功能:
1. 加载和验证PIG数据集
2. 数据质量检查和统计分析
3. 创建数据集分割(train/val/test)
4. 保存预处理后的数据集信息
5. 生成数据报告

运行:
  python scripts/prepare_data.py --data_dir PIGdata --output_dir processed_data --report
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import yaml
from collections import Counter
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from loguru import logger
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PianoFingeringDataset, collate_fn
from src.data.physical_constraints import PhysicalConstraints


def setup_logging(verbose: bool = False):
    """设置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


def load_config(config_path: str = None) -> Dict:
    """加载配置文件"""
    if config_path is None:
        config_path = project_root / "configs" / "model_config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_data_directory(data_dir: Path) -> bool:
    """验证数据目录结构"""
    logger.info(f"验证数据目录: {data_dir}")
    
    required_files = [
        "List.csv",
        "FingeringFiles"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.error(f"缺少必需文件: {missing_files}")
        return False
    
    # 检查指法文件数量
    fingering_dir = data_dir / "FingeringFiles"
    fingering_files = list(fingering_dir.glob("*_fingering.txt"))
    logger.info(f"发现 {len(fingering_files)} 个指法文件")
    
    if len(fingering_files) == 0:
        logger.error("未找到指法文件")
        return False
    
    logger.info("✅ 数据目录结构验证通过")
    return True


def analyze_dataset_splits(data_dir: Path, config: Dict) -> Dict:
    """分析数据集分割情况"""
    logger.info("分析数据集分割...")
    
    splits_info = {}
    total_sequences = 0
    total_notes = 0
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = PianoFingeringDataset(
                data_dir=data_dir,
                split=split,
                sequence_length=config['data']['sequence_length']
            )
            
            stats = dataset.get_stats()
            splits_info[split] = {
                'num_sequences': len(dataset),
                'total_notes': stats['total_notes'],
                'avg_sequence_length': stats['avg_sequence_length'],
                'finger_distribution': stats['finger_distribution'],
                'hand_distribution': stats['hand_distribution']
            }
            
            total_sequences += len(dataset)
            total_notes += stats['total_notes']
            
            logger.info(f"  {split}: {len(dataset)} 序列, {stats['total_notes']} 音符")
            
        except Exception as e:
            logger.error(f"处理 {split} 分割时出错: {e}")
            splits_info[split] = {'error': str(e)}
    
    splits_info['total'] = {
        'sequences': total_sequences,
        'notes': total_notes
    }
    
    return splits_info


def test_data_loading(data_dir: Path, config: Dict) -> bool:
    """测试数据加载功能"""
    logger.info("测试数据加载...")
    
    try:
        # 创建小样本数据集用于测试
        test_dataset = PianoFingeringDataset(
            data_dir=data_dir,
            split="train",
            sequence_length=config['data']['sequence_length'],
            max_pieces=2  # 只加载2个作品用于测试
        )
        
        if len(test_dataset) == 0:
            logger.error("测试数据集为空")
            return False
        
        # 测试数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # 测试批处理
        batch = next(iter(test_loader))
        expected_keys = ['features', 'labels', 'metadata', 'notes']
        
        for key in expected_keys:
            if key not in batch:
                logger.error(f"批处理缺少键: {key}")
                return False
        
        # 验证张量形状
        features_shape = batch['features'].shape
        labels_shape = batch['labels'].shape
        
        logger.info(f"  特征张量形状: {features_shape}")
        logger.info(f"  标签张量形状: {labels_shape}")
        logger.info(f"  批大小: {len(batch['metadata'])}")
        
        # 测试物理约束计算
        if 'notes' in batch and len(batch['notes']) > 0:
            from src.data.physical_constraints import create_physical_features
            physical_features = create_physical_features(batch['notes'])
            logger.info(f"  物理特征张量形状: {physical_features.shape}")
        
        logger.info("✅ 数据加载测试通过")
        return True
        
    except Exception as e:
        logger.error(f"数据加载测试失败: {e}")
        return False


def generate_data_report(splits_info: Dict, output_dir: Path):
    """生成数据报告"""
    logger.info("生成数据报告...")
    
    report = {
        'dataset_summary': {
            'total_sequences': splits_info['total']['sequences'],
            'total_notes': splits_info['total']['notes'],
            'splits': {}
        },
        'finger_distribution': {},
        'hand_distribution': {},
        'quality_checks': []
    }
    
    # 收集各分割的统计信息
    for split in ['train', 'val', 'test']:
        if split in splits_info and 'error' not in splits_info[split]:
            split_info = splits_info[split]
            report['dataset_summary']['splits'][split] = {
                'sequences': split_info['num_sequences'],
                'notes': split_info['total_notes'],
                'avg_length': round(split_info['avg_sequence_length'], 2)
            }
            
            # 合并指法分布
            for finger, count in split_info['finger_distribution'].items():
                if finger not in report['finger_distribution']:
                    report['finger_distribution'][finger] = 0
                report['finger_distribution'][finger] += count
            
            # 合并手部分布
            for hand, count in split_info['hand_distribution'].items():
                if hand not in report['hand_distribution']:
                    report['hand_distribution'][hand] = 0
                report['hand_distribution'][hand] += count
    
    # 保存报告
    report_path = output_dir / 'data_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    logger.info("📊 数据集摘要:")
    logger.info(f"  总序列数: {report['dataset_summary']['total_sequences']:,}")
    logger.info(f"  总音符数: {report['dataset_summary']['total_notes']:,}")
    
    for split, info in report['dataset_summary']['splits'].items():
        logger.info(f"  {split}: {info['sequences']:,} 序列, {info['notes']:,} 音符")
    
    logger.info(f"📄 详细报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="PIG数据集预处理")
    parser.add_argument('--data_dir', type=str, default='PIGdata', 
                       help='PIG数据集目录路径')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                       help='输出目录')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--report', action='store_true',
                       help='生成详细数据报告')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细日志输出')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    logger.info("🎹 开始PIG数据集预处理")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 验证数据目录
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return 1
    
    if not validate_data_directory(data_dir):
        logger.error("数据目录验证失败")
        return 1
    
    # 加载配置
    try:
        config = load_config(args.config)
        logger.info("配置文件加载成功")
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        return 1
    
    # 测试数据加载
    if not test_data_loading(data_dir, config):
        logger.error("数据加载测试失败")
        return 1
    
    # 分析数据集分割
    try:
        splits_info = analyze_dataset_splits(data_dir, config)
    except Exception as e:
        logger.error(f"数据集分析失败: {e}")
        return 1
    
    # 生成报告
    if args.report:
        try:
            generate_data_report(splits_info, output_dir)
        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            return 1
    
    # 保存预处理配置
    prep_config = {
        'data_dir': str(data_dir.absolute()),
        'sequence_length': config['data']['sequence_length'],
        'splits': splits_info,
        'timestamp': str(datetime.now())
    }
    
    config_path = output_dir / 'preprocessing_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(prep_config, f, indent=2, ensure_ascii=False)
    
    logger.info("✅ 数据预处理完成!")
    logger.info(f"配置已保存: {config_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
