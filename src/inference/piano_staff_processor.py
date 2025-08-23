"""
钢琴大谱表处理器

将midi2ly生成的单谱表转换为标准钢琴大谱表格式：
- 上方谱表：右手，高音谱号
- 下方谱表：左手，低音谱号
"""

import re
from typing import Dict, List, Tuple
from loguru import logger


class PianoStaffProcessor:
    """
    钢琴大谱表处理器
    
    功能：
    1. 分析音符音高和指法
    2. 分离右手/左手音符到不同谱表
    3. 重构LilyPond代码为钢琴大谱表格式
    """
    
    def __init__(self):
        """初始化处理器"""
        # 中央C (C4) = MIDI 60，作为分界线
        self.split_point = 60  # 中央C
        
    def process_to_piano_staff(self, content: str) -> str:
        """
        将单谱表转换为钢琴大谱表
        
        Args:
            content: 原始LilyPond代码
            
        Returns:
            转换后的钢琴大谱表代码
        """
        logger.info("🎹 开始转换为钢琴大谱表格式...")
        
        # 解析原始结构
        header, music_content, footer = self._parse_structure(content)
        
        # 提取音乐数据
        tracks = self._extract_tracks(music_content)
        
        # 按音高分离左右手
        right_hand_tracks, left_hand_tracks = self._separate_hands(tracks)
        
        # 生成钢琴大谱表代码
        piano_staff_content = self._generate_piano_staff(
            header, right_hand_tracks, left_hand_tracks, footer
        )
        
        logger.info("✅ 钢琴大谱表转换完成")
        return piano_staff_content
    
    def _parse_structure(self, content: str) -> Tuple[str, str, str]:
        """解析LilyPond文件结构"""
        lines = content.split('\n')
        
        header_lines = []
        music_lines = []
        footer_lines = []
        
        current_section = 'header'
        
        for line in lines:
            if line.strip().startswith('track') and '=' in line:
                current_section = 'music'
            elif line.strip().startswith('\\score'):
                current_section = 'footer'
            
            if current_section == 'header':
                header_lines.append(line)
            elif current_section == 'music':
                music_lines.append(line)
            else:
                footer_lines.append(line)
        
        return '\n'.join(header_lines), '\n'.join(music_lines), '\n'.join(footer_lines)
    
    def _extract_tracks(self, music_content: str) -> List[Dict]:
        """提取音轨数据"""
        tracks = []
        
        # 匹配音轨定义
        track_pattern = r'(track\w+)\s*=\s*([^}]+})'
        
        for match in re.finditer(track_pattern, music_content, re.DOTALL):
            track_name = match.group(1)
            track_content = match.group(2)
            
            # 跳过只有skip和tempo的控制轨道
            if self._is_control_track(track_content):
                continue
            
            tracks.append({
                'name': track_name,
                'content': track_content,
                'notes': self._extract_notes_from_track(track_content)
            })
        
        logger.info(f"提取了 {len(tracks)} 个音轨")
        return tracks
    
    def _is_control_track(self, content: str) -> bool:
        """判断是否为控制轨道（只有tempo、skip等）"""
        # 移除注释和空白
        clean_content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        # 检查是否只包含控制命令
        control_commands = [
            r'\\tempo', r'\\skip', r'\\time', r'\\key', 
            r'\\set', r'\\clef', r'\|', r'r\d*', r'r4\*'
        ]
        
        # 移除所有控制命令
        for cmd in control_commands:
            clean_content = re.sub(cmd + r'[^\\]*', '', clean_content)
        
        # 移除数字、括号、等号等
        clean_content = re.sub(r'[0-9\{\}\(\)\=\*\/\.\s\|]+', '', clean_content)
        
        # 如果剩下的内容很少，说明主要是控制命令
        return len(clean_content.strip()) < 10
    
    def _extract_notes_from_track(self, track_content: str) -> List[Dict]:
        """从音轨中提取音符信息"""
        notes = []
        
        # 匹配音符模式（包括和弦）
        # 单音符: c4-1, des'-2
        # 和弦: <c-1 e-2 g-3>4
        note_patterns = [
            r'<([^>]+)>(\d*)',  # 和弦
            r'\b([a-g](?:is|es)?[,\']*\d*)(?:-(\d+))?',  # 单音符
        ]
        
        for pattern in note_patterns:
            for match in re.finditer(pattern, track_content):
                if '<' in match.group(0):  # 和弦
                    chord_notes = match.group(1)
                    duration = match.group(2) if match.group(2) else '4'
                    
                    # 解析和弦中的每个音符
                    chord_pattern = r'([a-g](?:is|es)?[,\']*\d*)(?:-(\d+))?'
                    for chord_match in re.finditer(chord_pattern, chord_notes):
                        note_name = chord_match.group(1)
                        fingering = chord_match.group(2)
                        
                        notes.append({
                            'name': note_name,
                            'fingering': fingering,
                            'duration': duration,
                            'midi_pitch': self._note_to_midi(note_name),
                            'is_chord': True,
                            'original': match.group(0)
                        })
                else:  # 单音符
                    note_name = match.group(1)
                    fingering = match.group(2)
                    
                    # 提取时值
                    duration_match = re.search(r'\d+', note_name)
                    if duration_match:
                        duration = duration_match.group(0)
                        note_name = re.sub(r'\d+', '', note_name)
                    else:
                        duration = '4'
                    
                    notes.append({
                        'name': note_name,
                        'fingering': fingering,
                        'duration': duration,
                        'midi_pitch': self._note_to_midi(note_name),
                        'is_chord': False,
                        'original': match.group(0)
                    })
        
        return notes
    
    def _note_to_midi(self, note_name: str) -> int:
        """将音符名转换为MIDI音高"""
        # 基础音高映射
        base_pitches = {
            'c': 0, 'cis': 1, 'des': 1, 'd': 2, 'dis': 3, 'ees': 3,
            'e': 4, 'f': 5, 'fis': 6, 'ges': 6, 'g': 7, 'gis': 8,
            'aes': 8, 'a': 9, 'ais': 10, 'bes': 10, 'b': 11
        }
        
        # 解析音符名
        note_match = re.match(r'([a-g](?:is|es)?)(.*)', note_name.lower())
        if not note_match:
            return 60  # 默认中央C
        
        base_note = note_match.group(1)
        octave_marks = note_match.group(2)
        
        # 计算八度
        octave = 4  # 默认第4八度
        
        # 计算撇号和逗号
        apostrophes = octave_marks.count("'")
        commas = octave_marks.count(",")
        
        octave += apostrophes - commas
        
        # 计算MIDI音高
        midi_pitch = base_pitches.get(base_note, 0) + (octave * 12)
        
        return max(0, min(127, midi_pitch))  # 限制在MIDI范围内
    
    def _separate_hands(self, tracks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """按音高分离左右手"""
        right_hand_tracks = []
        left_hand_tracks = []
        
        for track in tracks:
            right_notes = []
            left_notes = []
            
            for note in track['notes']:
                # 根据MIDI音高和指法判断左右手
                if self._is_right_hand_note(note):
                    right_notes.append(note)
                else:
                    left_notes.append(note)
            
            if right_notes:
                right_hand_tracks.append({
                    'name': f"{track['name']}_RH",
                    'content': track['content'],
                    'notes': right_notes
                })
            
            if left_notes:
                left_hand_tracks.append({
                    'name': f"{track['name']}_LH", 
                    'content': track['content'],
                    'notes': left_notes
                })
        
        logger.info(f"分离结果: 右手 {len(right_hand_tracks)} 轨道, 左手 {len(left_hand_tracks)} 轨道")
        return right_hand_tracks, left_hand_tracks
    
    def _is_right_hand_note(self, note: Dict) -> bool:
        """判断音符是否属于右手"""
        midi_pitch = note['midi_pitch']
        fingering = note['fingering']
        
        # 主要根据音高判断
        if midi_pitch >= self.split_point:  # 中央C及以上
            return True
        elif midi_pitch < self.split_point - 12:  # 中央C以下一个八度
            return False
        else:
            # 中央C附近，根据指法判断
            if fingering:
                # 正数指法通常是右手
                try:
                    finger_num = int(fingering)
                    return finger_num > 0
                except:
                    pass
            
            # 默认：中央C附近偏向右手
            return midi_pitch >= self.split_point - 6
    
    def _generate_piano_staff(
        self, 
        header: str, 
        right_tracks: List[Dict], 
        left_tracks: List[Dict], 
        footer: str
    ) -> str:
        """生成钢琴大谱表代码"""
        
        # 重构音轨内容
        right_content = self._reconstruct_tracks(right_tracks, 'treble')
        left_content = self._reconstruct_tracks(left_tracks, 'bass')
        
        # 生成钢琴大谱表结构
        piano_staff = f"""
% 右手声部
rightHand = {{
  \\clef treble
  \\key des \\major
  \\time 9/8
{right_content}
}}

% 左手声部  
leftHand = {{
  \\clef bass
  \\key des \\major
  \\time 9/8
{left_content}
}}

% 钢琴大谱表
\\score {{
  \\new PianoStaff <<
    \\new Staff = "right" {{
      \\set Staff.instrumentName = #"Piano"
      \\rightHand
    }}
    \\new Staff = "left" {{
      \\leftHand
    }}
  >>
  \\layout {{
    \\context {{
      \\PianoStaff
      \\accepts "Staff"
    }}
  }}
  \\midi {{}}
}}
"""
        
        return header + piano_staff
    
    def _reconstruct_tracks(self, tracks: List[Dict], clef: str) -> str:
        """重构音轨内容"""
        if not tracks:
            return "  r1*100  % 空声部"
        
        # 根据音符重构内容
        combined_notes = []
        
        for track in tracks:
            for note in track['notes']:
                # 重构音符字符串
                note_str = note['name']
                if note['duration'] and note['duration'] != '4':
                    note_str += note['duration']
                if note['fingering']:
                    note_str += f"-{note['fingering']}"
                
                combined_notes.append(note_str)
        
        if not combined_notes:
            return "  r1*100  % 空声部"
        
        # 将音符分组，每行不超过8个
        lines = []
        for i in range(0, len(combined_notes), 8):
            line_notes = combined_notes[i:i+8]
            lines.append(f"  {' '.join(line_notes)}")
            if len(lines) > 20:  # 限制行数
                break
        
        return '\n'.join(lines)
    
    def _clean_content_for_hand(self, content: str, hand: str) -> str:
        """为特定手部清理内容"""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 跳过控制行
            if any(cmd in line for cmd in ['\\tempo', '\\skip', '\\set', '\\time']):
                continue
                
            # 只保留有音符的行
            if re.search(r'[a-g](?:is|es)?[,\']*\d*(?:-\d+)?', line):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines[:20])  # 限制长度
