"""
智能指法注入器

专门处理LilyPond代码的指法添加，避免语法错误
"""

import re
from typing import Dict, List
from loguru import logger


class SmartFingeringInjector:
    """
    智能指法注入器
    
    解决LilyPond语法问题，确保指法正确添加
    """
    
    def __init__(self):
        """初始化注入器"""
        pass
    
    def inject_fingerings(self, content: str, fingerings: List[Dict]) -> str:
        """
        智能注入指法到LilyPond代码
        
        策略:
        1. 识别音乐内容区域（排除命令和元数据）
        2. 按声部分离指法
        3. 安全地添加指法标记
        """
        logger.info("🎯 开始智能指法注入...")
        
        # 创建指法队列
        right_fingerings = [f for f in fingerings if f['hand'] == 'right']
        left_fingerings = [f for f in fingerings if f['hand'] == 'left']
        
        # 按时间排序
        right_fingerings.sort(key=lambda x: x['onset_time'])
        left_fingerings.sort(key=lambda x: x['onset_time'])
        
        logger.info(f"   右手指法队列: {len(right_fingerings)}")
        logger.info(f"   左手指法队列: {len(left_fingerings)}")
        
        # 逐行处理，避免破坏LilyPond结构
        lines = content.split('\n')
        processed_lines = []
        
        right_idx = 0
        left_idx = 0
        
        for line in lines:
            # 跳过命令行和注释
            if self._is_command_line(line):
                processed_lines.append(line)
                continue
            
            # 处理音乐内容行
            if self._contains_musical_notes(line):
                # 判断是右手还是左手声部
                is_right_hand = self._detect_hand_from_context(line, processed_lines)
                
                # 添加指法
                if is_right_hand and right_idx < len(right_fingerings):
                    processed_line = self._add_fingerings_to_line(
                        line, right_fingerings, right_idx, 'right'
                    )
                    # 更新索引
                    notes_in_line = len(re.findall(r'\b[a-g](?:is|es)?[,\']*\d*\b', line))
                    right_idx += notes_in_line
                elif not is_right_hand and left_idx < len(left_fingerings):
                    processed_line = self._add_fingerings_to_line(
                        line, left_fingerings, left_idx, 'left'
                    )
                    # 更新索引
                    notes_in_line = len(re.findall(r'\b[a-g](?:is|es)?[,\']*\d*\b', line))
                    left_idx += notes_in_line
                else:
                    processed_line = line
                
                processed_lines.append(processed_line)
            else:
                processed_lines.append(line)
        
        result = '\n'.join(processed_lines)
        
        # 后处理：修复可能的语法问题
        result = self._fix_syntax_issues(result)
        
        logger.info("✅ 智能指法注入完成")
        return result
    
    def _is_command_line(self, line: str) -> bool:
        """判断是否为LilyPond命令行"""
        stripped = line.strip()
        
        # 空行和注释
        if not stripped or stripped.startswith('%'):
            return True
        
        # LilyPond命令
        command_patterns = [
            r'\\version',
            r'\\header',
            r'\\layout',
            r'\\score',
            r'\\key\s',
            r'\\time\s',
            r'\\tempo\s',
            r'\\clef\s',
            r'\\relative\s',
            r'\\skip\s',
            r'\\set\s',
            r'\\context\s',
            r'\\new\s',
            r'\\voice',
            r'\\midi',
            r'\\override',
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, stripped):
                return True
        
        # 变量定义
        if re.match(r'^[a-zA-Z][a-zA-Z0-9]*\s*=', stripped):
            return True
        
        # 括号和分隔符行
        if stripped in ['{', '}', '<<', '>>', '|']:
            return True
        
        return False
    
    def _contains_musical_notes(self, line: str) -> bool:
        """判断行是否包含音符"""
        # 查找音符模式
        note_pattern = r'\b[a-g](?:is|es)?[,\']*\d*\b'
        return bool(re.search(note_pattern, line))
    
    def _detect_hand_from_context(self, line: str, previous_lines: List[str]) -> bool:
        """
        从上下文检测声部（右手=True，左手=False）
        """
        # 检查最近的几行上下文
        context_lines = previous_lines[-10:] if len(previous_lines) >= 10 else previous_lines
        context = ' '.join(context_lines).lower()
        
        # 明确的声部标识
        if 'voiceone' in context or 'right' in context:
            return True
        elif 'voicetwo' in context or 'left' in context or 'bass' in context:
            return False
        
        # 根据音符高度判断
        high_notes = re.findall(r'[a-g][,\']*\'', line)  # 带'的高音
        low_notes = re.findall(r'[a-g][,\']*,', line)    # 带,的低音
        
        if high_notes and not low_notes:
            return True  # 高音区 = 右手
        elif low_notes and not high_notes:
            return False  # 低音区 = 左手
        
        # 默认判断：奇数声部=右手，偶数声部=左手
        voice_matches = re.findall(r'voice([a-z])', context)
        if voice_matches:
            last_voice = voice_matches[-1]
            # a, c, e, g... = 右手; b, d, f, h... = 左手
            return ord(last_voice) % 2 == 1
        
        # 最后的默认值：根据trackB通常是右手
        return 'trackb' in context
    
    def _add_fingerings_to_line(
        self, 
        line: str, 
        fingering_queue: List[Dict], 
        start_idx: int, 
        hand: str
    ) -> str:
        """
        为一行音符添加指法
        """
        # 简单策略：为每个音符添加指法
        finger_idx = start_idx
        
        def add_finger_to_note(match):
            nonlocal finger_idx
            
            note_str = match.group(0)
            
            # 检查是否已有指法
            if '-' in note_str and re.search(r'-[1-5]', note_str):
                return note_str
            
            # 获取指法
            if finger_idx < len(fingering_queue):
                finger = fingering_queue[finger_idx]['finger']
                finger_idx += 1
            else:
                # 默认指法
                finger = 1 if hand == 'right' else 5
            
            # 确保指法有效
            finger = max(1, min(5, abs(finger)))
            
            # 添加指法（在时值之后）
            if re.search(r'\d+', note_str):
                # 有时值的情况: c4 -> c4-1
                return f"{note_str}-{finger}"
            else:
                # 无时值的情况: c -> c-1
                return f"{note_str}-{finger}"
        
        # 匹配音符（排除和弦）
        note_pattern = r'\b[a-g](?:is|es)?[,\']*\d*(?:\.[^<>\s]*)?'
        
        # 先处理非和弦音符
        result_line = re.sub(note_pattern, add_finger_to_note, line)
        
        return result_line
    
    def _fix_syntax_issues(self, content: str) -> str:
        """
        修复常见的LilyPond语法问题
        """
        # 修复调号问题: \key des-2 \major -> \key des \major
        content = re.sub(r'\\key\s+([a-g](?:is|es)?)-\d+\s+(\\major|\\minor)', 
                        r'\\key \1 \2', content)
        
        # 修复相对音高问题: \relative c-digit -> \relative c
        content = re.sub(r'\\relative\s+([a-g](?:is|es)?[,\']*)-\d+', 
                        r'\\relative \1', content)
        
        # 修复时值乘法问题: note*digit -> note (移除无效的乘法)
        content = re.sub(r'([a-g](?:is|es)?[,\']*\d*)-(\d+)\*(\d+)', 
                        r'\1-\2', content)
        
        logger.info("🔧 语法问题修复完成")
        return content


