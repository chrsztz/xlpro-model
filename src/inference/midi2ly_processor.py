"""
基于midi2ly的LilyPond处理器

实现流程: MIDI → midi2ly转换 → 解析并添加指法 → PDF编译
"""

import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import pretty_midi
from .smart_fingering_injector import SmartFingeringInjector
from .piano_staff_processor import PianoStaffProcessor


class Midi2LyProcessor:
    """
    基于midi2ly的指法乐谱生成器
    
    优势: 使用LilyPond官方工具确保音乐理论正确性
    """
    
    def __init__(self):
        """初始化处理器"""
        self.lilypond_path = self._find_tool('lilypond')
        self.midi2ly_path = self._find_tool('midi2ly')
        self.fingering_injector = SmartFingeringInjector()
        self.staff_processor = PianoStaffProcessor()
        
        if not self.lilypond_path:
            raise RuntimeError("未找到LilyPond，请安装: brew install lilypond")
        
        if not self.midi2ly_path:
            raise RuntimeError("未找到midi2ly，请检查LilyPond安装")
        
        logger.info(f"✅ LilyPond: {self.lilypond_path}")
        logger.info(f"✅ midi2ly: {self.midi2ly_path}")
    
    def _find_tool(self, tool_name: str) -> Optional[str]:
        """查找工具路径"""
        try:
            result = subprocess.run(['which', tool_name], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def generate_fingered_score(
        self,
        midi_path: str,
        fingerings: List[Dict],
        output_dir: Path,
        base_name: str = None
    ) -> Tuple[str, Optional[str]]:
        """
        完整流程: MIDI → midi2ly → 添加指法 → PDF
        
        Args:
            midi_path: 原始MIDI文件路径
            fingerings: 指法预测结果
            output_dir: 输出目录
            base_name: 文件基名
            
        Returns:
            (lilypond_path, pdf_path)
        """
        if base_name is None:
            base_name = Path(midi_path).stem
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎼 开始生成指法乐谱: {base_name}")
        
        # 步骤1: 使用midi2ly转换MIDI
        base_ly_path = self._convert_midi_to_lilypond(midi_path, output_dir, base_name)
        
        # 步骤2: 解析LilyPond并添加指法
        fingered_ly_path = self._add_fingerings_to_score(
            base_ly_path, fingerings, output_dir, base_name
        )
        
        # 步骤3: 编译为PDF
        pdf_path = self._compile_to_pdf(fingered_ly_path, output_dir)
        
        # 步骤4: 清理临时文件
        try:
            base_ly_path.unlink()
        except:
            pass
        
        return str(fingered_ly_path), pdf_path
    
    def _convert_midi_to_lilypond(
        self, 
        midi_path: str, 
        output_dir: Path, 
        base_name: str
    ) -> Path:
        """
        使用midi2ly转换MIDI文件
        """
        logger.info("🔄 使用midi2ly转换MIDI...")
        
        base_ly_path = output_dir / f"{base_name}_base.ly"
        
        try:
            # 调用midi2ly工具
            # midi2ly的正确语法: midi2ly [options] file.mid
            result = subprocess.run([
                self.midi2ly_path,
                '--duration-quant=8',     # 八分音符量化
                '--allow-tuplet=4*2/3',   # 允许连音符
                '--output=' + str(base_ly_path),  # 输出文件
                '--verbose',              # 详细输出
                str(midi_path)            # 输入MIDI文件
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.error(f"midi2ly错误: {result.stderr}")
                raise RuntimeError(f"midi2ly转换失败: {result.stderr}")
            
            if not base_ly_path.exists():
                raise RuntimeError("midi2ly未生成输出文件")
            
            logger.info(f"✅ MIDI转换完成: {base_ly_path}")
            return base_ly_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("midi2ly转换超时")
        except Exception as e:
            raise RuntimeError(f"midi2ly转换失败: {e}")
    
    def _add_fingerings_to_score(
        self,
        base_ly_path: Path,
        fingerings: List[Dict],
        output_dir: Path,
        base_name: str
    ) -> Path:
        """
        解析LilyPond代码并添加指法信息
        """
        logger.info("🖐️ 添加指法信息...")
        
        # 读取基础LilyPond代码
        with open(base_ly_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用智能指法注入器
        modified_content = self.fingering_injector.inject_fingerings(content, fingerings)
        
        # 转换为钢琴大谱表格式
        modified_content = self.staff_processor.process_to_piano_staff(modified_content)
        
        # 改进标题
        modified_content = self._enhance_header(modified_content, base_name)
        
        # 保存带指法的文件
        fingered_ly_path = output_dir / f"{base_name}_with_fingerings.ly"
        with open(fingered_ly_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        logger.info(f"✅ 指法添加完成: {fingered_ly_path}")
        return fingered_ly_path
    
    def _create_smart_fingering_map(self, fingerings: List[Dict]) -> Dict:
        """
        创建智能指法映射
        
        基于时间和音高创建精确的指法映射
        """
        # 按时间和音高创建映射
        finger_map = {}
        
        for f in fingerings:
            # 创建唯一键: 时间(秒) + 音高
            time_key = round(f['onset_time'], 2)  # 精确到0.01秒
            pitch_key = f['midi_number']
            
            unique_key = f"{time_key}_{pitch_key}"
            finger_map[unique_key] = {
                'finger': abs(f['finger']) if f['finger'] != 0 else 1,
                'hand': f['hand'],
                'pitch_name': f.get('pitch_name', ''),
                'confidence': f.get('confidence', 1.0)
            }
        
        logger.info(f"创建了 {len(finger_map)} 个指法映射")
        return finger_map
    
    def _inject_fingerings(self, content: str, fingering_map: Dict) -> str:
        """
        将指法注入到LilyPond音符中
        
        使用正确的LilyPond指法语法: note-digit
        """
        logger.info("🔍 分析LilyPond代码结构...")
        
        # 获取排序后的指法列表
        right_fingerings = [f for f in fingering_map.values() if f['hand'] == 'right']
        left_fingerings = [f for f in fingering_map.values() if f['hand'] == 'left']
        
        logger.info(f"   右手指法: {len(right_fingerings)} 个")
        logger.info(f"   左手指法: {len(left_fingerings)} 个")
        
        # 指法分配计数器
        right_idx = 0
        left_idx = 0
        
        def add_fingering_to_note(match):
            nonlocal right_idx, left_idx
            
            full_match = match.group(0)
            note_name = match.group(1)
            octave_marks = match.group(2) if match.group(2) else ""
            duration = match.group(3) if match.group(3) else ""
            
            # 判断声部：高音区(有')通常是右手，低音区(有,)通常是左手
            context_before = match.string[max(0, match.start()-300):match.start()]
            
            # 更精确的声部判断
            is_right_hand = False
            if "'" in octave_marks:  # 高音区
                is_right_hand = True
            elif "," in octave_marks:  # 低音区
                is_right_hand = False
            elif 'voiceB' in context_before or 'trackB' in context_before:
                is_right_hand = True
            elif 'voiceC' in context_before or any(voice in context_before for voice in ['voiceC', 'voiceD', 'voiceE']):
                is_right_hand = False
            else:
                # 默认判断：中音区根据音符名称
                is_right_hand = note_name.lower() in ['e', 'f', 'g', 'a', 'b']
            
            # 分配指法
            if is_right_hand:
                if right_idx < len(right_fingerings):
                    finger = right_fingerings[right_idx]['finger']
                    right_idx += 1
                else:
                    finger = (right_idx % 5) + 1  # 循环使用1-5指
                    right_idx += 1
            else:
                if left_idx < len(left_fingerings):
                    finger = left_fingerings[left_idx]['finger']
                    left_idx += 1
                else:
                    finger = 5 - (left_idx % 5)  # 循环使用5-1指
                    left_idx += 1
            
            # 确保指法在1-5范围内
            finger = max(1, min(5, finger))
            
            # 构造带指法的音符 (LilyPond语法: note-digit)
            return f"{note_name}{octave_marks}{duration}-{finger}"
        
        # 改进的音符匹配模式
        # 只在音乐内容中匹配音符，避免LilyPond命令
        # 匹配: 音符名 + 可选升降号 + 可选八度标记 + 可选时值
        note_pattern = r'(?<!\\key\s)(?<!\\relative\s)(?<!\\clef\s)\b([a-g](?:is|es)?)((?:[,\']*)?)((?:\d+)?(?:\.[^-\s]*)?)\b(?=\s|$|[|\]}])(?!\s*\\major)(?!\s*\\minor)(?!\s*\\relative)'
        
        # 应用指法标记，但要避免已经有指法的音符
        def safe_add_fingering(match):
            full_match = match.group(0)
            # 检查是否已经有指法标记
            if '-' in full_match and re.search(r'-[1-5]', full_match):
                return full_match  # 已经有指法，不修改
            return add_fingering_to_note(match)
        
        # 先保护调号等LilyPond命令，避免被误修改
        def protect_lilypond_commands(content):
            # 保护调号命令
            protected_content = re.sub(
                r'(\\key\s+[a-g](?:is|es)?)\s*(\\major|\\minor)',
                r'\1 \2',
                content
            )
            return protected_content
        
        # 保护LilyPond命令
        modified_content = protect_lilypond_commands(content)
        
        # 先处理和弦外的单音符
        modified_content = re.sub(note_pattern, safe_add_fingering, modified_content)
        
        # 处理和弦内的音符 (在 < > 之间)
        def process_chord(chord_match):
            chord_content = chord_match.group(1)
            # 为和弦内的每个音符添加指法
            chord_modified = re.sub(note_pattern, safe_add_fingering, chord_content)
            return f"<{chord_modified}>"
        
        # 处理和弦
        chord_pattern = r'<([^>]+)>'
        modified_content = re.sub(chord_pattern, process_chord, modified_content)
        
        logger.info("✅ 指法注入完成")
        return modified_content
    
    def _enhance_header(self, content: str, title: str) -> str:
        """改进LilyPond文件头"""
        # 如果没有header，添加一个
        if '\\header' not in content:
            # 使用字符串拼接避免转义问题
            header_lines = [
                '\\header {',
                f'  title = "{title}"',
                '  subtitle = "钢琴指法 (AI自动生成)"',
                '  tagline = "Generated by XLPro Fingering System"',
                '}',
                ''
            ]
            header = '\n'.join(header_lines) + '\n'
            
            # 在version之后插入header
            version_pattern = r'(\\version[^\n]*\n)'
            if re.search(version_pattern, content):
                # 使用字符串操作避免正则表达式转义问题
                match = re.search(version_pattern, content)
                if match:
                    insert_pos = match.end()
                    content = content[:insert_pos] + header + content[insert_pos:]
            else:
                # 如果没有version，直接在开头添加
                content = header + content
        else:
            # 修改现有header (简化处理)
            content = content.replace('\\header', '\\header\n  % AI Generated')
        
        return content
    
    def _compile_to_pdf(self, ly_path: Path, output_dir: Path) -> Optional[str]:
        """
        编译LilyPond文件为PDF
        """
        logger.info("🔨 编译PDF...")
        
        try:
            result = subprocess.run([
                self.lilypond_path,
                '--pdf',
                '--output', str(output_dir),
                str(ly_path)
            ], capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                pdf_path = ly_path.with_suffix('.pdf')
                if pdf_path.exists():
                    logger.info(f"✅ PDF编译成功: {pdf_path}")
                    return str(pdf_path)
                else:
                    logger.warning("PDF文件未找到")
                    return None
            else:
                logger.error(f"LilyPond编译失败:")
                logger.error(f"错误信息: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("LilyPond编译超时")
            return None
        except Exception as e:
            logger.error(f"编译异常: {e}")
            return None
    
    def test_conversion(self, output_dir: Path) -> bool:
        """
        测试midi2ly转换功能
        """
        logger.info("🧪 测试midi2ly功能...")
        
        # 确保输出目录存在
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建简单的测试MIDI
        test_midi = self._create_test_midi(output_dir)
        
        try:
            # 转换为LilyPond
            base_ly = self._convert_midi_to_lilypond(str(test_midi), output_dir, "test")
            
            # 读取结果
            with open(base_ly, 'r') as f:
                content = f.read()
            
            logger.info("✅ midi2ly转换测试成功")
            logger.info(f"生成的LilyPond代码长度: {len(content)} 字符")
            
            # 清理
            test_midi.unlink()
            base_ly.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ midi2ly测试失败: {e}")
            return False
    
    def _create_test_midi(self, output_dir: Path) -> Path:
        """创建测试MIDI文件"""
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        
        # 添加简单的音符序列
        notes_data = [
            (60, 0.0, 0.5),  # C4
            (62, 0.5, 1.0),  # D4
            (64, 1.0, 1.5),  # E4
        ]
        
        for pitch, start, end in notes_data:
            note = pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
            piano.notes.append(note)
        
        midi.instruments.append(piano)
        
        test_path = output_dir / "test_midi.mid"
        midi.write(str(test_path))
        
        return test_path
