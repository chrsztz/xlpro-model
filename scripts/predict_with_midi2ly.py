"""
基于midi2ly的完整预测脚本

流程: MIDI输入 → 模型预测指法 → midi2ly转换 → 添加指法 → PDF输出
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from loguru import logger
import torch

from src.inference.predictor import create_predictor
from src.inference.midi2ly_processor import Midi2LyProcessor


def setup_logging():
    """设置日志格式"""
    logger.remove()
    logger.add(sys.stderr, level="INFO", 
              format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


def main():
    parser = argparse.ArgumentParser(description="基于midi2ly的指法预测")
    
    parser.add_argument('input_midi', type=str, nargs='?', help='输入MIDI文件')
    parser.add_argument('--model', type=str, 
                       default='experiments/cnn_bilstm_physical_constraints/best_model.pth',
                       help='模型文件路径')
    parser.add_argument('--config', type=str, 
                       default='configs/model_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output', type=str, default='midi2ly_output',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                       help='推理设备')
    parser.add_argument('--test-only', action='store_true',
                       help='只测试midi2ly工具，不进行预测')
    
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info("🎼 基于midi2ly的指法预测")
    logger.info("=" * 60)
    
    # 测试模式
    if args.test_only:
        logger.info("🧪 测试模式: 只验证midi2ly功能")
        
        try:
            processor = Midi2LyProcessor()
            output_dir = Path("test_midi2ly")
            
            if processor.test_conversion(output_dir):
                logger.info("🎉 midi2ly测试成功!")
                return 0
            else:
                logger.error("❌ midi2ly测试失败")
                return 1
                
        except Exception as e:
            logger.error(f"❌ 测试异常: {e}")
            return 1
    
    # 检查输入文件 (测试模式不需要)
    if not args.test_only:
        if not args.input_midi:
            logger.error("❌ 请提供输入MIDI文件")
            return 1
            
        input_path = Path(args.input_midi)
        if not input_path.exists():
            logger.error(f"❌ 输入文件不存在: {input_path}")
            return 1
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"❌ 模型文件不存在: {model_path}")
        logger.info("💡 请先训练模型:")
        logger.info("   python scripts/train_model.py --epochs 5")
        return 1
    
    logger.info(f"📁 输入MIDI: {input_path}")
    logger.info(f"🧠 使用模型: {model_path}")
    logger.info(f"📂 输出目录: {args.output}")
    
    try:
        # 步骤1: 初始化处理器
        logger.info("\n🔧 初始化处理器...")
        processor = Midi2LyProcessor()
        
        # 步骤2: 加载模型并预测指法
        logger.info("\n🧠 加载模型并预测指法...")
        predictor = create_predictor(
            model_path=str(model_path),
            config_path=args.config,
            device=args.device
        )
        
        # 预测指法
        result = predictor.predict_midi(
            midi_path=str(input_path),
            use_hand_separation=True
        )
        
        fingerings = result['fingerings']
        logger.info(f"✅ 指法预测完成: {len(fingerings)} 个音符")
        
        # 统计信息
        right_count = sum(1 for f in fingerings if f['hand'] == 'right')
        left_count = sum(1 for f in fingerings if f['hand'] == 'left')
        logger.info(f"   右手: {right_count} 音符")
        logger.info(f"   左手: {left_count} 音符")
        
        # 步骤3: 生成LilyPond乐谱
        logger.info("\n📝 生成LilyPond乐谱...")
        output_dir = Path(args.output)
        
        ly_path, pdf_path = processor.generate_fingered_score(
            midi_path=str(input_path),
            fingerings=fingerings,
            output_dir=output_dir,
            base_name=input_path.stem
        )
        
        # 步骤4: 输出结果
        logger.info("\n🎉 完整流程执行成功!")
        logger.info("=" * 60)
        logger.info("📁 生成的文件:")
        logger.info(f"   LilyPond源码: {ly_path}")
        
        if pdf_path:
            logger.info(f"   PDF乐谱: {pdf_path}")
            logger.info("\n📖 使用建议:")
            logger.info("1. 打开PDF文件查看带指法的乐谱")
            logger.info("2. 检查指法标记是否正确显示")
            logger.info("3. 如需修改，可编辑LilyPond源码后重新编译")
        else:
            logger.warning("⚠️ PDF生成失败，但LilyPond代码已生成")
            logger.info("💡 可以手动编译:")
            logger.info(f"   lilypond --pdf {ly_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("❌ 用户中断")
        return 1
    except Exception as e:
        logger.error(f"❌ 执行失败: {e}")
        logger.exception("详细错误:")
        return 1


if __name__ == "__main__":
    exit(main())
