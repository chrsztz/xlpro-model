"""
指法预测脚本

使用训练好的模型预测MIDI文件的指法
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import argparse
from loguru import logger

from src.inference.predictor import create_predictor


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='预测钢琴MIDI文件的指法')
    
    parser.add_argument(
        'input_midi',
        type=str,
        help='输入的MIDI文件路径'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='experiments/cnn_bilstm_physical_constraints/best_model.pth',
        help='训练好的模型路径'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_config.yaml',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径 (默认为输入文件名_fingering.mid)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='midi',
        choices=['midi', 'xml', 'json', 'csv', 'html'],
        help='输出格式'
    )
    
    parser.add_argument(
        '--no-hand-separation',
        action='store_true',
        help='禁用手部分离 (适用于已经分离的MIDI)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='推理设备 (auto, cpu, cuda, mps)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 检查输入文件
    input_path = Path(args.input_midi)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        return
    
    # 设置输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_fingering.{args.format}"
    
    logger.info(f"🎹 开始预测指法")
    logger.info(f"📁 输入文件: {input_path}")
    logger.info(f"💾 输出文件: {output_path}")
    logger.info(f"📋 输出格式: {args.format}")
    
    try:
        # 创建预测器
        logger.info("🔄 加载模型...")
        predictor = create_predictor(
            model_path=args.model,
            config_path=args.config,
            device=args.device
        )
        
        # 预测指法
        logger.info("🧠 预测指法...")
        result = predictor.predict_midi(
            midi_path=str(input_path),
            use_hand_separation=not args.no_hand_separation
        )
        
        # 保存结果
        logger.info("💾 保存结果...")
        saved_path = predictor.save_result(result, output_path, args.format)
        
        # 打印统计信息
        fingerings = result['fingerings']
        logger.info("📊 预测统计:")
        logger.info(f"  总音符数: {len(fingerings)}")
        
        # 统计左右手分布
        right_count = sum(1 for f in fingerings if f['hand'] == 'right')
        left_count = sum(1 for f in fingerings if f['hand'] == 'left')
        logger.info(f"  右手音符: {right_count}")
        logger.info(f"  左手音符: {left_count}")
        
        # 统计指法分布
        finger_counts = {}
        for f in fingerings:
            finger = abs(f['finger']) if f['finger'] != 0 else 0
            finger_counts[finger] = finger_counts.get(finger, 0) + 1
        
        logger.info("  指法分布:")
        for finger in sorted(finger_counts.keys()):
            if finger == 0:
                logger.info(f"    无标注: {finger_counts[finger]}")
            else:
                logger.info(f"    {finger}指: {finger_counts[finger]}")
        
        logger.info(f"✅ 完成！结果已保存到: {saved_path}")
        
        # 如果是HTML格式，同时生成可视化
        if args.format == 'html':
            predictor.output_formatter.create_visualization_html(
                result, 
                output_path.with_suffix('.html')
            )
            logger.info("🎨 可视化HTML已生成")
        
    except Exception as e:
        logger.error(f"❌ 预测失败: {e}")
        raise


if __name__ == "__main__":
    main() 