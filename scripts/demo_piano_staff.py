"""
钢琴大谱表演示脚本

展示单谱表 vs 钢琴大谱表的对比效果
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from loguru import logger
from src.inference.midi2ly_processor import Midi2LyProcessor


def demo_piano_staff_comparison():
    """演示钢琴大谱表对比效果"""
    logger.remove()
    logger.add(sys.stderr, level="INFO", 
              format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    
    logger.info("🎹 钢琴大谱表格式演示")
    logger.info("=" * 60)
    
    # 创建测试用的简单LilyPond内容
    test_content = '''\\version "2.14.0"

\\header {
  title = "钢琴大谱表演示"
}

testMusic = \\relative c' {
  % 右手高音
  c'4-1 d-2 e-3 f-4 g-5 a-1 b-2 c-3
  % 左手低音  
  c,4-5 d-4 e-3 f-2 g-1 a-5 b-4 c-3
}

\\score {
  \\context Staff \\testMusic
  \\layout {}
  \\midi {}
}'''
    
    processor = Midi2LyProcessor()
    
    logger.info("📝 原始单谱表格式:")
    logger.info("-" * 40)
    logger.info("\\score {")
    logger.info("  \\context Staff \\testMusic  % 所有音符在一个谱表")
    logger.info("}")
    
    # 转换为钢琴大谱表
    piano_staff_content = processor.staff_processor.process_to_piano_staff(test_content)
    
    logger.info("\n🎼 钢琴大谱表格式:")
    logger.info("-" * 40)
    lines = piano_staff_content.split('\n')
    for line in lines[20:35]:  # 显示关键部分
        if line.strip():
            logger.info(line)
    
    logger.info("\n✨ 主要改进:")
    logger.info("  ✅ 右手: 高音谱号 (treble clef)")
    logger.info("  ✅ 左手: 低音谱号 (bass clef)")
    logger.info("  ✅ 音符按音高自动分离")
    logger.info("  ✅ 标准钢琴乐谱格式")
    logger.info("  ✅ 更好的视觉效果")
    
    logger.info("\n📊 格式对比:")
    logger.info("=" * 60)
    
    comparison = [
        ("特性", "单谱表", "钢琴大谱表"),
        ("视觉效果", "❌ 混乱", "✅ 清晰"),
        ("音符分布", "❌ 全部挤在一起", "✅ 按手分离"),
        ("谱号", "❌ 只有高音谱号", "✅ 高音+低音谱号"),
        ("专业性", "❌ 不标准", "✅ 符合标准"),
        ("可读性", "❌ 难以阅读", "✅ 易于阅读"),
    ]
    
    for feature, single, piano in comparison:
        logger.info(f"  {feature:<10} | {single:<15} | {piano}")
    
    logger.info("\n🎯 使用效果:")
    logger.info("  📚 学习: 学生更容易理解左右手分工")
    logger.info("  🎼 教学: 老师可以清楚指导每只手的演奏")
    logger.info("  📖 演奏: 演奏者可以专注于各手的指法")
    logger.info("  💻 制谱: 符合专业乐谱制作标准")
    
    return True


def main():
    """主函数"""
    try:
        success = demo_piano_staff_comparison()
        
        if success:
            logger.info("\n🎉 钢琴大谱表演示完成!")
            logger.info("现在您的MIDI文件将自动生成专业的钢琴大谱表格式!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 演示失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

