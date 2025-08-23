"""
MIDI处理器

负责MIDI文件的加载、解析、手部分离等预处理功能
"""

import numpy as np
import pretty_midi
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import subprocess
import tempfile
import os


class MIDIProcessor:
    """
    MIDI文件处理器
    
    负责MIDI文件的各种预处理任务
    """
    
    def __init__(self):
        """初始化MIDI处理器"""
        pass
    
    def process_midi(
        self,
        midi_path: Path,
        use_hand_separation: bool = True,
        sequence_length: Optional[int] = None
    ) -> Dict:
        """
        处理MIDI文件
        
        Args:
            midi_path: MIDI文件路径
            use_hand_separation: 是否使用手部分离
            sequence_length: 目标序列长度
            
        Returns:
            处理后的数据字典
        """
        logger.info(f"处理MIDI文件: {midi_path}")
        
        # 1. 加载MIDI文件
        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            raise ValueError(f"无法加载MIDI文件: {e}")
        
        # 2. 手部分离 (如果需要)
        if use_hand_separation:
            separated_midi = self._separate_hands(midi_path)
            if separated_midi:
                midi_data = separated_midi
        
        # 3. 提取音符
        notes = self._extract_notes(midi_data)
        
        # 4. 序列长度处理
        # 在推理时，我们希望处理所有音符，不截断
        # 只有当音符数量太少时才进行填充
        if sequence_length and len(notes) < sequence_length:
            notes = self._pad_sequence(notes, sequence_length)
        
        # 注意：在推理阶段，我们处理所有音符，不限制序列长度
        # 模型会分批处理长序列
        
        return {
            'notes': notes,
            'original_midi': midi_data,
            'metadata': {
                'file_path': str(midi_path),
                'total_notes': len(notes),
                'duration': midi_data.get_end_time(),
                'hand_separation_used': use_hand_separation
            }
        }
    
    def _separate_hands(self, midi_path: Path) -> Optional[pretty_midi.PrettyMIDI]:
        """
        使用PianoHands.jl进行手部分离
        
        Args:
            midi_path: 输入MIDI文件路径
            
        Returns:
            分离后的MIDI数据，如果失败返回None
        """
        try:
            # 检查PianoHands.jl是否可用
            pianohands_dir = midi_path.parent.parent / "PianoHands.jl"
            if not pianohands_dir.exists():
                logger.warning("PianoHands.jl不可用，跳过手部分离")
                return None
            
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(suffix='_separated.mid', delete=False) as temp_file:
                temp_output = temp_file.name
            
            # 调用Julia脚本
            julia_cmd = [
                'julia', '--project=' + str(pianohands_dir),
                '-e', f'''
                using PianoHands
                generate_midi("{midi_path}", output_file="{temp_output}")
                '''
            ]
            
            result = subprocess.run(
                julia_cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30秒超时
            )
            
            if result.returncode == 0 and os.path.exists(temp_output):
                # 加载分离后的MIDI
                separated_midi = pretty_midi.PrettyMIDI(temp_output)
                logger.info("手部分离成功")
                
                # 清理临时文件
                os.unlink(temp_output)
                
                return separated_midi
            else:
                logger.warning(f"手部分离失败: {result.stderr}")
                # 清理临时文件
                if os.path.exists(temp_output):
                    os.unlink(temp_output)
                return None
                
        except Exception as e:
            logger.warning(f"手部分离异常: {e}")
            return None
    
    def _extract_notes(self, midi_data: pretty_midi.PrettyMIDI) -> List[Dict]:
        """
        从MIDI数据中提取音符信息
        
        Args:
            midi_data: MIDI数据
            
        Returns:
            音符信息列表
        """
        notes = []
        
        for instrument_idx, instrument in enumerate(midi_data.instruments):
            # 跳过鼓轨
            if instrument.is_drum:
                continue
            
            # 确定手部 (假设前2个轨道分别是右手和左手)
            if len(midi_data.instruments) >= 2:
                channel = 0 if instrument_idx == 0 else 1  # 0=右手, 1=左手
            else:
                # 单轨道：根据音高判断
                channel = self._infer_hand_from_pitch(instrument.notes)
            
            for note in instrument.notes:
                note_info = {
                    'onset_time': note.start,
                    'offset_time': note.end,
                    'duration': note.end - note.start,
                    'midi_number': note.pitch,
                    'pitch_name': pretty_midi.note_number_to_name(note.pitch),
                    'onset_velocity': note.velocity,
                    'offset_velocity': note.velocity,  # PrettyMIDI没有offset velocity
                    'channel': channel,
                    'is_black_key': self._is_black_key(note.pitch),
                    'instrument_idx': instrument_idx
                }
                notes.append(note_info)
        
        # 按时间排序
        notes.sort(key=lambda x: x['onset_time'])
        
        logger.info(f"提取了{len(notes)}个音符")
        return notes
    
    def _infer_hand_from_pitch(self, notes: List) -> int:
        """
        根据音高推断手部
        
        Args:
            notes: 音符列表
            
        Returns:
            手部标识 (0=右手, 1=左手)
        """
        if not notes:
            return 0
        
        # 计算平均音高
        avg_pitch = sum(note.pitch for note in notes) / len(notes)
        
        # 中央C (60) 作为分界线
        return 0 if avg_pitch >= 60 else 1
    
    def _is_black_key(self, midi_number: int) -> bool:
        """判断是否为黑键"""
        return (midi_number % 12) in {1, 3, 6, 8, 10}
    
    def _split_into_sequences(
        self,
        notes: List[Dict],
        sequence_length: int
    ) -> List[Dict]:
        """
        将长序列分割成固定长度的序列
        
        Args:
            notes: 音符列表
            sequence_length: 目标序列长度
            
        Returns:
            第一个序列的音符 (简化版本)
        """
        # 简化版本：只返回前sequence_length个音符
        # 在完整实现中，应该返回多个序列
        return notes[:sequence_length]
    
    def _pad_sequence(
        self,
        notes: List[Dict],
        sequence_length: int
    ) -> List[Dict]:
        """
        填充序列到指定长度
        
        Args:
            notes: 音符列表
            sequence_length: 目标序列长度
            
        Returns:
            填充后的音符列表
        """
        if len(notes) >= sequence_length:
            return notes
        
        # 使用最后一个音符进行填充
        if notes:
            last_note = notes[-1].copy()
            padding_notes = [last_note.copy() for _ in range(sequence_length - len(notes))]
            return notes + padding_notes
        else:
            # 如果没有音符，创建默认音符
            default_note = {
                'onset_time': 0.0,
                'offset_time': 0.5,
                'duration': 0.5,
                'midi_number': 60,  # 中央C
                'pitch_name': 'C4',
                'onset_velocity': 64,
                'offset_velocity': 64,
                'channel': 0,
                'is_black_key': False,
                'instrument_idx': 0
            }
            return [default_note.copy() for _ in range(sequence_length)]
    
    def analyze_midi(self, midi_path: Path) -> Dict:
        """
        分析MIDI文件的基本信息
        
        Args:
            midi_path: MIDI文件路径
            
        Returns:
            分析结果
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            
            # 统计信息
            total_notes = sum(len(instr.notes) for instr in midi_data.instruments if not instr.is_drum)
            
            # 音高范围
            all_pitches = []
            for instr in midi_data.instruments:
                if not instr.is_drum:
                    all_pitches.extend([note.pitch for note in instr.notes])
            
            pitch_range = (min(all_pitches), max(all_pitches)) if all_pitches else (0, 0)
            
            analysis = {
                'file_path': str(midi_path),
                'duration': midi_data.get_end_time(),
                'num_instruments': len([instr for instr in midi_data.instruments if not instr.is_drum]),
                'total_notes': total_notes,
                'pitch_range': pitch_range,
                'tempo_changes': len(getattr(midi_data, 'tempo_changes', [])),
                'time_signature_changes': len(getattr(midi_data, 'time_signature_changes', [])),
                'key_signature_changes': len(getattr(midi_data, 'key_signature_changes', []))
            }
            
            return analysis
            
        except Exception as e:
            raise ValueError(f"无法分析MIDI文件: {e}")
    
    def validate_midi(self, midi_path: Path) -> bool:
        """
        验证MIDI文件是否有效
        
        Args:
            midi_path: MIDI文件路径
            
        Returns:
            是否有效
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            
            # 检查是否有非鼓轨道
            has_melodic = any(not instr.is_drum for instr in midi_data.instruments)
            
            # 检查是否有音符
            has_notes = any(len(instr.notes) > 0 for instr in midi_data.instruments if not instr.is_drum)
            
            return has_melodic and has_notes
            
        except Exception:
            return False 