"""
测试midi2ly功能

验证midi2ly工具的安装和基本功能
"""

import sys
import subprocess
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from loguru import logger
import pretty_midi


def check_midi2ly_installation():
    """检查midi2ly安装"""
    logger.info("🔍 检查midi2ly安装...")
    
    try:
        # 检查midi2ly命令
        result = subprocess.run(['midi2ly', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version_info = result.stdout.strip()
            logger.info(f"✅ midi2ly可用: {version_info}")
            return True
        else:
            logger.error("❌ midi2ly命令执行失败")
            return False
            
    except FileNotFoundError:
        logger.error("❌ midi2ly未找到")
        logger.info("解决方案:")
        logger.info("  macOS: brew install lilypond")
        logger.info("  Ubuntu: sudo apt-get install lilypond")
        logger.info("  Windows: 从 https://lilypond.org 下载")
        return False


def create_test_midi(output_path: Path):
    """创建测试MIDI文件"""
    logger.info("🎵 创建测试MIDI...")
    
    midi = pretty_midi.PrettyMIDI()
    
    # 创建钢琴轨道
    piano = pretty_midi.Instrument(program=0, name='Piano')
    
    # 右手旋律 - C大调音阶
    right_notes = [
        (60, 0.0, 0.5),   # C4
        (62, 0.5, 1.0),   # D4
        (64, 1.0, 1.5),   # E4
        (65, 1.5, 2.0),   # F4
        (67, 2.0, 2.5),   # G4
        (69, 2.5, 3.0),   # A4
        (71, 3.0, 3.5),   # B4
        (72, 3.5, 4.0),   # C5
    ]
    
    # 左手伴奏
    left_notes = [
        (48, 0.0, 2.0),   # C3 (长音)
        (55, 2.0, 4.0),   # G3 (长音)
    ]
    
    # 添加右手音符
    for pitch, start, end in right_notes:
        note = pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
        piano.notes.append(note)
    
    # 添加左手音符
    for pitch, start, end in left_notes:
        note = pretty_midi.Note(velocity=70, pitch=pitch, start=start, end=end)
        piano.notes.append(note)
    
    midi.instruments.append(piano)
    midi.write(str(output_path))
    
    logger.info(f"✅ 测试MIDI已创建: {output_path}")


def test_midi2ly_conversion(midi_path: Path, output_dir: Path):
    """测试midi2ly转换"""
    logger.info("🔄 测试midi2ly转换...")
    
    ly_output = output_dir / "converted.ly"
    
    try:
        # 运行midi2ly
        result = subprocess.run([
            'midi2ly',
            '--output', str(ly_output),
            '--duration-quant', '8',
            str(midi_path)
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info(f"✅ midi2ly转换成功: {ly_output}")
            
            # 读取并显示部分内容
            with open(ly_output, 'r') as f:
                content = f.read()
            
            logger.info("📄 生成的LilyPond代码预览:")
            lines = content.split('\n')
            for i, line in enumerate(lines[:20]):  # 显示前20行
                logger.info(f"  {i+1:2d}: {line}")
            
            if len(lines) > 20:
                logger.info(f"  ... (还有 {len(lines)-20} 行)")
            
            return True
        else:
            logger.error(f"❌ midi2ly转换失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ midi2ly转换超时")
        return False
    except Exception as e:
        logger.error(f"❌ midi2ly转换异常: {e}")
        return False


def add_fingerings_demo(ly_path: Path, output_dir: Path):
    """演示指法添加"""
    logger.info("🖐️ 演示指法添加...")
    
    # 读取LilyPond代码
    with open(ly_path, 'r') as f:
        content = f.read()
    
    # 简单的指法添加 (演示用)
    # 匹配音符模式并添加指法
    import re
    
    # 为右手音符添加指法 (1-5循环)
    finger_cycle = [1, 2, 3, 4, 5]
    finger_idx = 0
    
    def add_finger(match):
        nonlocal finger_idx
        note = match.group(1)
        finger = finger_cycle[finger_idx % len(finger_cycle)]
        finger_idx += 1
        return f"{note}-{finger}"
    
    # 匹配音符模式 (简化版)
    note_pattern = r"([a-g](?:is|es)?[,']*\d+)"
    modified_content = re.sub(note_pattern, add_finger, content)
    
    # 保存带指法的版本
    fingered_ly_path = output_dir / "fingered.ly"
    with open(fingered_ly_path, 'w') as f:
        f.write(modified_content)
    
    logger.info(f"✅ 指法演示完成: {fingered_ly_path}")
    
    # 显示修改后的代码片段
    logger.info("📄 添加指法后的代码预览:")
    lines = modified_content.split('\n')
    for i, line in enumerate(lines[:15]):
        if any(note in line for note in ['c', 'd', 'e', 'f', 'g', 'a', 'b']):
            logger.info(f"  {i+1:2d}: {line}")


def test_pdf_compilation(ly_path: Path, output_dir: Path):
    """测试PDF编译"""
    logger.info("🔨 测试PDF编译...")
    
    try:
        result = subprocess.run([
            'lilypond',
            '--pdf',
            '--output', str(output_dir),
            str(ly_path)
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            pdf_path = ly_path.with_suffix('.pdf')
            if pdf_path.exists():
                logger.info(f"✅ PDF编译成功: {pdf_path}")
                
                # 检查文件大小
                size_kb = pdf_path.stat().st_size / 1024
                logger.info(f"   PDF大小: {size_kb:.1f} KB")
                
                return True
            else:
                logger.warning("⚠️ PDF文件未生成")
                return False
        else:
            logger.error(f"❌ PDF编译失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ PDF编译超时")
        return False
    except Exception as e:
        logger.error(f"❌ PDF编译异常: {e}")
        return False


def main():
    """主测试函数"""
    logger.info("🎼 测试MIDI2LY功能")
    logger.info("=" * 50)
    
    # 创建测试目录
    test_dir = Path("test_midi2ly_output")
    test_dir.mkdir(exist_ok=True)
    
    # 测试步骤
    tests = [
        ("检查midi2ly安装", lambda: check_midi2ly_installation()),
        ("创建测试MIDI", lambda: create_test_midi(test_dir / "test.mid")),
        ("测试midi2ly转换", lambda: test_midi2ly_conversion(test_dir / "test.mid", test_dir)),
        ("演示指法添加", lambda: add_fingerings_demo(test_dir / "converted.ly", test_dir)),
        ("测试PDF编译", lambda: test_pdf_compilation(test_dir / "fingered.ly", test_dir))
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name} - 成功")
            else:
                logger.error(f"❌ {test_name} - 失败")
                
        except Exception as e:
            logger.error(f"❌ {test_name} - 异常: {e}")
            results.append((test_name, False))
    
    # 总结
    logger.info("\n" + "=" * 50)
    logger.info("📊 测试结果总结:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        logger.info(f"  {status} {test_name}")
    
    logger.info(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过! midi2ly功能正常")
        logger.info(f"📁 查看测试文件: {test_dir}")
        logger.info("📄 重点查看: fingered.ly 和 fingered.pdf")
    else:
        logger.warning("⚠️ 部分测试失败，请检查LilyPond安装")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())


