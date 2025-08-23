"""
PIG数据集加载器

处理PIG格式的指法数据文件，按照论文中的数据表示方法。
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset
from loguru import logger


class PianoFingeringDataset(Dataset):
    """
    PIG数据集加载器
    
    数据格式按照论文Section 3.1:
    - 音符序列 T notes
    - 指法标签 yt: 1-5(右手), -1到-5(左手), 0(无标注)
    - 音高表示 xt
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        sequence_length: int = 75,
        transform=None,
        max_pieces: Optional[int] = None
    ):
        """
        初始化数据集
        
        Args:
            data_dir: PIG数据集目录路径
            split: 数据分割 ('train', 'val', 'test')
            sequence_length: 序列长度(论文中使用75)
            transform: 数据变换函数
            max_pieces: 最大作品数量(用于演示)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.transform = transform
        self.max_pieces = max_pieces
        
        # 加载数据文件列表
        self.fingering_files = self._get_file_list()
        
        # 加载所有数据
        self.data = self._load_data()
        
        logger.info(f"加载了 {len(self.data)} 个序列，分割: {split}")
    
    def _get_file_list(self) -> List[Path]:
        """获取指法文件列表"""
        fingering_dir = self.data_dir / "FingeringFiles"
        if not fingering_dir.exists():
            raise FileNotFoundError(f"找不到指法文件目录: {fingering_dir}")
        
        # 获取所有.txt文件
        files = list(fingering_dir.glob("*_fingering.txt"))
        
        # 按文件名排序以确保一致性
        files.sort()
        
        # 限制作品数量 (用于演示)
        if self.max_pieces:
            files = files[:self.max_pieces]
        
        # 分割数据集 (70% train, 15% val, 15% test)
        n_files = len(files)
        train_end = int(0.7 * n_files)
        val_end = int(0.85 * n_files)
        
        if self.split == "train":
            return files[:train_end]
        elif self.split == "val":
            return files[train_end:val_end]
        elif self.split == "test":
            return files[val_end:]
        else:
            raise ValueError(f"无效的分割: {self.split}")
    
    def _load_data(self) -> List[Dict]:
        """加载所有数据文件"""
        all_sequences = []
        
        for file_path in self.fingering_files:
            try:
                sequences = self._parse_pig_file(file_path)
                all_sequences.extend(sequences)
            except Exception as e:
                logger.warning(f"解析文件失败 {file_path}: {e}")
                continue
        
        return all_sequences
    
    def _parse_pig_file(self, file_path: Path) -> List[Dict]:
        """
        解析单个PIG文件
        
        PIG文件格式:
        (note_id) (onset_time) (offset_time) (spelled_pitch) (onset_velocity) (offset_velocity) (channel) (finger)
        """
        notes = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                
                # 跳过注释和空行
                if line.startswith("//") or not line:
                    continue
                
                try:
                    parts = line.split('\t')
                    if len(parts) >= 8:  # 确保有所有必需字段
                        note = {
                            'note_id': int(parts[0]),
                            'onset_time': float(parts[1]),
                            'offset_time': float(parts[2]),
                            'spelled_pitch': parts[3],
                            'onset_velocity': int(parts[4]),
                            'offset_velocity': int(parts[5]),
                            'channel': int(parts[6]),  # 0=右手, 1=左手
                            'finger': self._parse_finger(parts[7]) if len(parts) > 7 else 0
                        }
                        
                        # 计算基础特征
                        note['midi_number'] = self._pitch_to_midi(note['spelled_pitch'])
                        note['duration'] = note['offset_time'] - note['onset_time']
                        note['is_black_key'] = self._is_black_key(note['midi_number'])
                        
                        notes.append(note)
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"解析行失败 {file_path}:{line_num}: {line}, 错误: {e}")
                    continue
        
        # 按时间排序
        notes.sort(key=lambda x: x['onset_time'])
        
        # 分割成固定长度的序列
        sequences = self._create_sequences(notes, file_path.stem)
        
        return sequences
    
    def _parse_finger(self, finger_str: str) -> int:
        """
        解析指法字符串
        
        格式: 数字 (1-5, -1到-5) 或 数字_数字 (指法替换)
        """
        try:
            # 处理指法替换 (如 "3_1")
            if '_' in finger_str:
                finger_str = finger_str.split('_')[0]  # 取第一个指法
            
            finger = int(finger_str)
            
            # 验证指法范围
            if finger == 0 or (1 <= abs(finger) <= 5):
                return finger
            else:
                return 0  # 无效指法标记为0
                
        except (ValueError, AttributeError):
            return 0
    
    def _pitch_to_midi(self, pitch_name: str) -> int:
        """将音高名称转换为MIDI音符号"""
        # 音高映射表
        note_mapping = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        try:
            # 解析音高名称
            if len(pitch_name) >= 2:
                if '#' in pitch_name:
                    note = pitch_name[:-2] + '#'
                    octave = int(pitch_name[-1])
                elif 'b' in pitch_name:
                    note = pitch_name[:-2] + 'b'
                    octave = int(pitch_name[-1])
                else:
                    note = pitch_name[:-1]
                    octave = int(pitch_name[-1])
                
                # 计算MIDI音符号
                midi_num = note_mapping.get(note, 0) + (octave + 1) * 12
                return max(0, min(127, midi_num))  # 限制在MIDI范围内
        except:
            pass
        
        return 60  # 默认中央C
    
    def _is_black_key(self, midi_number: int) -> bool:
        """判断是否为黑键"""
        return (midi_number % 12) in {1, 3, 6, 8, 10}
    
    def _create_sequences(self, notes: List[Dict], piece_name: str) -> List[Dict]:
        """将音符列表分割成固定长度的序列"""
        sequences = []
        
        if len(notes) < self.sequence_length:
            # 如果音符数不足，进行填充
            padded_notes = notes + [notes[-1]] * (self.sequence_length - len(notes))
            sequences.append({
                'notes': padded_notes,
                'piece_name': piece_name,
                'sequence_id': 0
            })
        else:
            # 使用滑动窗口创建多个序列
            stride = self.sequence_length // 2  # 50% 重叠
            for i in range(0, len(notes) - self.sequence_length + 1, stride):
                sequence_notes = notes[i:i + self.sequence_length]
                sequences.append({
                    'notes': sequence_notes,
                    'piece_name': piece_name,
                    'sequence_id': i // stride
                })
        
        return sequences
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        sequence = self.data[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence
    
    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        total_notes = sum(len(seq['notes']) for seq in self.data)
        
        # 统计指法分布
        finger_counts = {}
        hand_counts = {'left': 0, 'right': 0}
        
        for sequence in self.data:
            for note in sequence['notes']:
                finger = note['finger']
                if finger != 0:
                    finger_counts[finger] = finger_counts.get(finger, 0) + 1
                    
                    if finger > 0:
                        hand_counts['right'] += 1
                    else:
                        hand_counts['left'] += 1
        
        return {
            'total_sequences': len(self.data),
            'total_notes': total_notes,
            'avg_sequence_length': total_notes / len(self.data),
            'finger_distribution': finger_counts,
            'hand_distribution': hand_counts
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    批处理整理函数
    
    将批量数据转换为tensor格式
    """
    # 提取特征和标签
    sequences = []
    labels = []
    metadata = []
    raw_notes_batch = []
    
    for item in batch:
        notes = item['notes']
        
        # 提取特征向量
        features = []
        targets = []
        
        for note in notes:
            # 基础特征 (按照论文Section 3.2)
            feature_vector = [
                note['midi_number'] / 127.0,  # 归一化MIDI号
                note['duration'],              # 时长
                float(note['is_black_key']),   # 黑键标识
                note['onset_velocity'] / 127.0, # 归一化力度
                note['channel']                # 声道
            ]
            
            features.append(feature_vector)
            targets.append(note['finger'])
        
        sequences.append(features)
        labels.append(targets)
        metadata.append({
            'piece_name': item['piece_name'],
            'sequence_id': item['sequence_id']
        })
        # 保留原始notes用于下游物理约束在线计算
        raw_notes_batch.append(notes)
    
    # 转换为tensor
    features_tensor = torch.FloatTensor(sequences)
    labels_tensor = torch.LongTensor(labels)
    
    return {
        'features': features_tensor,
        'labels': labels_tensor,
        'metadata': metadata,
        'notes': raw_notes_batch
    } 