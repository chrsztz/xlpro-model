"""
数据处理审计脚本

用途:
- 对比原始raw notes与进入网络前(features/physical_features)的分布是否合理
- 输出关键统计(均值/方差/分位数/极值、类别占比、异常计数)

运行:
  python scripts/audit_data_pipeline.py --split train --max_batches 20
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import yaml
from collections import Counter, defaultdict

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from torch.utils.data import DataLoader
from loguru import logger

from src.data.dataset import PianoFingeringDataset, collate_fn
from src.data.physical_constraints import create_physical_features


def load_config():
    config_path = project_root / "configs" / "model_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def format_number(num, precision=4):
    """格式化数字显示"""
    if abs(num) >= 1000:
        return f"{num:.1f}"
    elif abs(num) >= 10:
        return f"{num:.2f}"
    elif abs(num) >= 1:
        return f"{num:.3f}"
    else:
        return f"{num:.{precision}f}"


def print_section_header(title: str):
    """打印章节标题"""
    print()
    logger.info("=" * 80)
    logger.info(f"  {title}")
    logger.info("=" * 80)


def describe_continuous_variable(name: str, arr: np.ndarray, expected_range=None):
    """描述连续变量的统计信息"""
    arr = arr.astype(float)
    if arr.size == 0:
        logger.info(f"📊 {name}: (empty)")
        return
    
    # 基础统计
    stats = {
        'count': arr.size,
        'mean': np.mean(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'p25': np.percentile(arr, 25),
        'p50': np.percentile(arr, 50),
        'p75': np.percentile(arr, 75),
        'max': np.max(arr),
    }
    
    # 检查异常值
    warnings = []
    if expected_range:
        min_exp, max_exp = expected_range
        if stats['min'] < min_exp or stats['max'] > max_exp:
            warnings.append(f"⚠️ 超出预期范围 [{min_exp}, {max_exp}]")
    
    # 检查数据质量
    if np.any(np.isnan(arr)):
        warnings.append("⚠️ 包含NaN值")
    if np.any(np.isinf(arr)):
        warnings.append("⚠️ 包含无穷值")
    
    # 格式化输出
    logger.info(f"📊 {name} (n={stats['count']:,})")
    logger.info(f"   范围: [{format_number(stats['min'])}, {format_number(stats['max'])}]")
    logger.info(f"   中心: 均值={format_number(stats['mean'])}, 中位数={format_number(stats['p50'])}")
    logger.info(f"   分布: Q1={format_number(stats['p25'])}, Q3={format_number(stats['p75'])}, 标准差={format_number(stats['std'])}")
    
    # 简单的分布可视化
    create_simple_histogram(arr, name)
    
    # 输出警告
    for warning in warnings:
        logger.info(f"   {warning}")


def describe_categorical_variable(name: str, arr: np.ndarray):
    """描述分类变量的分布"""
    if arr.size == 0:
        logger.info(f"📊 {name}: (empty)")
        return
    
    # 统计各类别
    unique_vals, counts = np.unique(arr, return_counts=True)
    total = arr.size
    
    logger.info(f"📊 {name} (n={total:,}, {len(unique_vals)} 类别)")
    
    # 按频率排序显示
    sorted_indices = np.argsort(counts)[::-1]
    for i in sorted_indices:
        val = unique_vals[i]
        count = counts[i]
        pct = 100.0 * count / total
        
        # 创建简单的条形图
        bar_length = min(30, int(30 * count / counts.max()))
        bar = "█" * bar_length + "░" * (30 - bar_length)
        
        logger.info(f"   {str(val).rjust(8)}: {bar} {count:6,} ({pct:5.1f}%)")


def create_simple_histogram(arr: np.ndarray, name: str, bins=20):
    """创建简单的ASCII直方图"""
    try:
        hist, edges = np.histogram(arr, bins=bins)
        if hist.max() == 0:
            return
        
        # 选择合适的字符
        chars = " ▁▂▃▄▅▆▇█"
        max_count = hist.max()
        
        # 缩放到字符范围
        scaled = hist * (len(chars) - 1) / max_count
        histogram_str = ''.join(chars[int(val)] for val in scaled)
        
        logger.info(f"   分布: {histogram_str}")
        
    except Exception:
        pass  # 如果直方图创建失败，静默跳过


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', choices=['train','val','test'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_batches', type=int, default=20)
    args = parser.parse_args()

    config = load_config()
    data_dir = project_root / config['data']['data_dir']

    dataset = PianoFingeringDataset(
        data_dir=data_dir,
        split=args.split,
        sequence_length=config['data']['sequence_length']
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    logger.info(f"开始审计: split={args.split}, 批大小={args.batch_size}, 最大批数={args.max_batches}")

    # 原始级别聚合
    midi_numbers = []
    durations = []
    onset_velocities = []
    is_black_keys = []
    channels = []
    fingers = []

    # 处理后特征聚合 (features: [*, 5])
    feat_midi = []
    feat_duration = []
    feat_black = []
    feat_velocity = []
    feat_channel = []

    # 物理特征聚合 ([*, 5])
    phys_stretch = []
    phys_cross = []
    phys_pos = []
    phys_nat = []
    phys_str = []

    batches_done = 0
    for batch in loader:
        # 原始notes
        for notes in batch['notes']:
            for n in notes:
                midi_numbers.append(n['midi_number'])
                durations.append(n['duration'])
                onset_velocities.append(n['onset_velocity'])
                is_black_keys.append(float(n['is_black_key']))
                channels.append(n['channel'])
                fingers.append(n.get('finger', 0))

        # 进入网络前的基础特征
        feats = batch['features'].numpy()  # [B, L, 5]
        feat_midi.append(feats[:, :, 0].ravel())
        feat_duration.append(feats[:, :, 1].ravel())
        feat_black.append(feats[:, :, 2].ravel())
        feat_velocity.append(feats[:, :, 3].ravel())
        feat_channel.append(feats[:, :, 4].ravel())

        # 物理特征(用notes在线计算，保证与训练一致)
        phys = create_physical_features(batch['notes']).numpy()  # [B, L, 5]
        phys_stretch.append(phys[:, :, 0].ravel())
        phys_cross.append(phys[:, :, 1].ravel())
        phys_pos.append(phys[:, :, 2].ravel())
        phys_nat.append(phys[:, :, 3].ravel())
        phys_str.append(phys[:, :, 4].ravel())

        batches_done += 1
        if batches_done >= args.max_batches:
            break

    # 转为np数组
    midi_numbers = np.array(midi_numbers)
    durations = np.array(durations)
    onset_velocities = np.array(onset_velocities)
    is_black_keys = np.array(is_black_keys)
    channels = np.array(channels)
    fingers = np.array(fingers)

    feat_midi = np.concatenate(feat_midi, axis=0) if feat_midi else np.array([])
    feat_duration = np.concatenate(feat_duration, axis=0) if feat_duration else np.array([])
    feat_black = np.concatenate(feat_black, axis=0) if feat_black else np.array([])
    feat_velocity = np.concatenate(feat_velocity, axis=0) if feat_velocity else np.array([])
    feat_channel = np.concatenate(feat_channel, axis=0) if feat_channel else np.array([])

    phys_stretch = np.concatenate(phys_stretch, axis=0) if phys_stretch else np.array([])
    phys_cross = np.concatenate(phys_cross, axis=0) if phys_cross else np.array([])
    phys_pos = np.concatenate(phys_pos, axis=0) if phys_pos else np.array([])
    phys_nat = np.concatenate(phys_nat, axis=0) if phys_nat else np.array([])
    phys_str = np.concatenate(phys_str, axis=0) if phys_str else np.array([])

    # ========================================
    # 原始数据分析
    # ========================================
    print_section_header("原始音符数据分析")
    
    # 连续变量
    describe_continuous_variable("MIDI音符号", midi_numbers, expected_range=(0, 127))
    describe_continuous_variable("音符时长(秒)", durations, expected_range=(0, 10))
    describe_continuous_variable("起始力度", onset_velocities, expected_range=(0, 127))
    
    # 分类变量
    describe_categorical_variable("黑键标识", is_black_keys)
    describe_categorical_variable("声道(0=右手,1=左手)", channels)
    describe_categorical_variable("指法标签(-5到5)", fingers)

    # ========================================
    # 数据质量检查
    # ========================================
    print_section_header("数据质量检查")
    
    quality_issues = []
    
    # MIDI音符号检查
    oob_midi = int(np.sum((midi_numbers < 0) | (midi_numbers > 127)))
    if oob_midi > 0:
        quality_issues.append(f"🔴 MIDI音符号越界: {oob_midi} 个")
    else:
        logger.info("✅ MIDI音符号范围正常 (0-127)")
    
    # 时长检查
    neg_dur = int(np.sum(durations < 0))
    zero_dur = int(np.sum(durations == 0))
    if neg_dur > 0:
        quality_issues.append(f"🔴 负时长音符: {neg_dur} 个")
    if zero_dur > 0:
        quality_issues.append(f"🟡 零时长音符: {zero_dur} 个")
    if neg_dur == 0 and zero_dur == 0:
        logger.info("✅ 音符时长范围正常")
    
    # 力度检查
    vel_oob = int(np.sum((onset_velocities < 0) | (onset_velocities > 127)))
    if vel_oob > 0:
        quality_issues.append(f"🔴 力度值越界: {vel_oob} 个")
    else:
        logger.info("✅ 力度值范围正常 (0-127)")
    
    # 指法标签检查
    unlabeled = int(np.sum(fingers == 0))
    unlabeled_pct = 100.0 * unlabeled / len(fingers) if len(fingers) > 0 else 0
    if unlabeled_pct > 50:
        quality_issues.append(f"🟡 无标注指法过多: {unlabeled_pct:.1f}%")
    elif unlabeled_pct > 10:
        logger.info(f"🟡 无标注指法比例: {unlabeled_pct:.1f}% (可接受)")
    else:
        logger.info(f"✅ 指法标注完整度: {100-unlabeled_pct:.1f}%")
    
    # 输出质量问题汇总
    if quality_issues:
        logger.info("🔍 发现的质量问题:")
        for issue in quality_issues:
            logger.info(f"   {issue}")
    else:
        logger.info("🎉 数据质量良好，无明显问题")

    # ========================================
    # 网络输入特征分析
    # ========================================
    print_section_header("网络输入特征分析")
    
    describe_continuous_variable("归一化MIDI号", feat_midi, expected_range=(0, 1))
    describe_continuous_variable("音符时长", feat_duration)
    describe_categorical_variable("黑键特征", feat_black)
    describe_continuous_variable("归一化力度", feat_velocity, expected_range=(0, 1))
    describe_categorical_variable("声道特征", feat_channel)

    # ========================================
    # 物理约束特征分析
    # ========================================
    print_section_header("物理约束特征分析")
    
    describe_continuous_variable("拉伸率", phys_stretch, expected_range=(0, 2))
    describe_continuous_variable("交叉距离", phys_cross, expected_range=(0, 2))
    describe_continuous_variable("手部位置", phys_pos, expected_range=(0, 1))
    describe_continuous_variable("自然违反", phys_nat, expected_range=(0, 1))
    describe_continuous_variable("力度违反", phys_str, expected_range=(0, 1))

    # ========================================
    # 审计总结
    # ========================================
    print_section_header("审计总结和建议")
    
    logger.info("🔍 关键检查点:")
    logger.info("   ✓ 归一化特征是否在 [0,1] 范围内")
    logger.info("   ✓ 物理约束特征是否在合理范围 (通常0-2)")
    logger.info("   ✓ 无标注指法(0)比例是否过高")
    logger.info("   ✓ 时长分布是否存在异常值")
    
    logger.info("\n💡 如发现问题:")
    logger.info("   • 归一化越界 → 检查数据预处理逻辑")
    logger.info("   • 物理特征异常 → 调整约束计算参数")
    logger.info("   • 时长分布异常 → 考虑对数缩放或截断")
    logger.info("   • 指法标注不足 → 检查数据集质量")
    
    total_samples = len(midi_numbers)
    logger.info(f"\n📊 本次审计样本: {total_samples:,} 个音符，来自 {batches_done} 个批次")
    logger.info("🎯 审计完成！请检查上述统计信息和警告")


if __name__ == "__main__":
    main()


