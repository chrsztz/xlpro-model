"""
指法预测器

实现MIDI文件的指法预测和结果输出功能
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
from loguru import logger

# MIDI处理
import pretty_midi
import mido
from music21 import stream, note, meter, key, tempo, metadata

from ..models.cnn_bilstm import CNNBiLSTMModel, create_model
from ..models.forward_planning import ForwardPlanner, BeamSearchPlanner
from ..data.physical_constraints import PhysicalConstraints
from .midi_processor import MIDIProcessor
from .output_formatter import OutputFormatter


class FingeringPredictor:
    """
    指法预测器
    
    负责加载模型、处理MIDI文件、预测指法、输出结果
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: Optional[torch.device] = None
    ):
        """
        初始化指法预测器
        
        Args:
            model_path: 训练好的模型路径
            config_path: 配置文件路径
            device: 推理设备
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = device or self._get_device()
        
        # 加载配置
        self.config = self._load_config()
        
        # 加载模型
        self.model = self._load_model()
        
        # 初始化组件
        self.midi_processor = MIDIProcessor()
        self.physical_constraints = PhysicalConstraints()
        self.output_formatter = OutputFormatter()
        
        # 设置前向规划器
        planning_config = self.config.get('forward_planning', {})
        if planning_config.get('use_planning', True):
            self.planner = BeamSearchPlanner(
                planning_depth=planning_config.get('planning_depth', 3),
                top_k_candidates=planning_config.get('top_k_candidates', 3),
                physical_weight=self.config['model']['physical_weight'],
                beam_width=5
            )
        else:
            self.planner = None
        
        logger.info(f"指法预测器初始化完成，设备: {self.device}")
    
    def _get_device(self) -> torch.device:
        """自动选择推理设备"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _load_model(self) -> CNNBiLSTMModel:
        """加载训练好的模型"""
        # 创建模型
        model = create_model(self.config['model'])
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 移动到设备并设置为评估模式
        model.to(self.device)
        model.eval()
        
        logger.info(f"模型加载完成: {self.model_path}")
        return model
    
    def predict_midi(
        self,
        midi_path: str,
        use_hand_separation: bool = True,
        sequence_length: Optional[int] = None
    ) -> Dict:
        """
        预测MIDI文件的指法
        
        Args:
            midi_path: MIDI文件路径
            use_hand_separation: 是否使用手部分离
            sequence_length: 序列长度 (None表示使用配置中的默认值)
            
        Returns:
            包含指法预测结果的字典
        """
        midi_path = Path(midi_path)
        logger.info(f"开始预测指法: {midi_path}")
        
        # 1. 处理MIDI文件
        processed_data = self.midi_processor.process_midi(
            midi_path,
            use_hand_separation=use_hand_separation,
            sequence_length=sequence_length or self.config['data']['sequence_length']
        )
        
        # 2. 提取特征
        features = self._extract_features(processed_data['notes'])
        physical_features = self._extract_physical_features(processed_data['notes'])
        
        # 3. 模型预测 (支持长序列分批处理)
        seq_len = len(processed_data['notes'])
        max_seq_len = sequence_length or self.config['data']['sequence_length']
        
        with torch.no_grad():
            if seq_len <= max_seq_len:
                # 序列不长，直接处理
                features_batch = features.unsqueeze(0).to(self.device)
                physical_batch = physical_features.unsqueeze(0).to(self.device) if physical_features is not None else None
                
                outputs = self.model(features_batch, physical_batch)
                
                if self.planner:
                    planned_fingerings = self.planner.plan_sequence(
                        self.model, features_batch, physical_batch
                    )
                    fingering_predictions = planned_fingerings[0][:seq_len]  # 只取实际长度
                else:
                    logits = outputs['logits'][0][:seq_len]  # 只取实际长度
                    fingering_predictions = torch.argmax(logits, dim=-1)
            else:
                # 长序列，分批处理
                logger.info(f"长序列检测到({seq_len}音符)，使用分批处理...")
                fingering_predictions = self._predict_long_sequence(
                    features, physical_features, max_seq_len
                )
        
        # 4. 转换预测结果
        fingering_sequence = self._convert_predictions(
            fingering_predictions, 
            processed_data['notes']
        )
        
        # 5. 构建结果
        result = {
            'input_file': str(midi_path),
            'notes': processed_data['notes'],
            'fingerings': fingering_sequence,
            'metadata': {
                'model_config': self.config['model'],
                'sequence_length': len(processed_data['notes']),
                'hand_separation_used': use_hand_separation,
                'planning_used': self.planner is not None
            }
        }
        
        logger.info(f"指法预测完成，共{len(fingering_sequence)}个音符")
        return result
    
    def _extract_features(self, notes: List[Dict]) -> torch.Tensor:
        """提取基础特征"""
        features = []
        
        for note in notes:
            feature_vector = [
                note['midi_number'] / 127.0,     # 归一化MIDI号
                note['duration'],                # 时长
                float(note['is_black_key']),     # 黑键标识
                note['onset_velocity'] / 127.0,  # 归一化力度
                note['channel']                  # 声道
            ]
            features.append(feature_vector)
        
        return torch.FloatTensor(features)
    
    def _extract_physical_features(self, notes: List[Dict]) -> Optional[torch.Tensor]:
        """提取物理约束特征"""
        if not self.config['model'].get('use_physical_constraints', True):
            return None
        
        return self.physical_constraints.compute_sequence_constraints(notes)
    
    def _convert_predictions(
        self,
        predictions: torch.Tensor,
        notes: List[Dict]
    ) -> List[Dict]:
        """
        转换预测结果为指法序列
        
        Args:
            predictions: 模型预测结果 [seq_len]
            notes: 原始音符数据
            
        Returns:
            指法序列
        """
        fingering_sequence = []
        
        for i, (pred, note) in enumerate(zip(predictions, notes)):
            # 转换预测索引到实际指法 (0-10 -> -5到5)
            finger_index = pred.item()
            actual_finger = finger_index - 5
            
            # 如果预测为0（无标注），根据手部和音高进行启发式分配
            if actual_finger == 0:
                actual_finger = self._heuristic_finger_assignment(note)
            
            fingering_info = {
                'note_id': i,
                'onset_time': note['onset_time'],
                'offset_time': note['offset_time'],
                'midi_number': note['midi_number'],
                'pitch_name': note.get('pitch_name', ''),
                'channel': note['channel'],
                'hand': 'right' if note['channel'] == 0 else 'left',
                'finger': actual_finger,
                'confidence': float(F.softmax(torch.tensor([finger_index], dtype=torch.float), dim=0)[0])
            }
            
            fingering_sequence.append(fingering_info)
        
        return fingering_sequence
    
    def _predict_long_sequence(self, features: torch.Tensor, physical_features: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        """
        对长序列进行分批预测
        
        Args:
            features: 输入特征 [seq_len, feature_dim]
            physical_features: 物理约束特征 [seq_len, phys_dim]
            max_seq_len: 最大序列长度
            
        Returns:
            预测的指法序列 [seq_len]
        """
        seq_len = features.shape[0]
        all_predictions = []
        
        # 使用滑动窗口处理长序列
        overlap = max_seq_len // 4  # 25%重叠避免边界效应
        
        for start_idx in range(0, seq_len, max_seq_len - overlap):
            end_idx = min(start_idx + max_seq_len, seq_len)
            
            # 提取当前窗口的特征
            window_features = features[start_idx:end_idx]
            window_physical = physical_features[start_idx:end_idx] if physical_features is not None else None
            
            # 如果窗口太小，进行填充
            if window_features.shape[0] < max_seq_len:
                pad_size = max_seq_len - window_features.shape[0]
                feature_pad = torch.zeros(pad_size, window_features.shape[1], device=features.device)
                window_features = torch.cat([window_features, feature_pad], dim=0)
                
                if window_physical is not None:
                    phys_pad = torch.zeros(pad_size, window_physical.shape[1], device=physical_features.device)
                    window_physical = torch.cat([window_physical, phys_pad], dim=0)
            
            # 添加batch维度并预测
            features_batch = window_features.unsqueeze(0).to(self.device)
            physical_batch = window_physical.unsqueeze(0).to(self.device) if window_physical is not None else None
            
            outputs = self.model(features_batch, physical_batch)
            
            if self.planner:
                planned_fingerings = self.planner.plan_sequence(
                    self.model, features_batch, physical_batch
                )
                window_predictions = planned_fingerings[0]
            else:
                logits = outputs['logits'][0]
                window_predictions = torch.argmax(logits, dim=-1)
            
            # 确定要保留的部分 (避免重叠)
            if start_idx == 0:
                # 第一个窗口，保留全部
                keep_end = min(max_seq_len - overlap // 2, end_idx - start_idx)
                all_predictions.append(window_predictions[:keep_end])
            elif end_idx == seq_len:
                # 最后一个窗口，从重叠中间开始保留
                keep_start = overlap // 2
                actual_len = end_idx - start_idx
                all_predictions.append(window_predictions[keep_start:actual_len])
            else:
                # 中间窗口，保留中间部分
                keep_start = overlap // 2
                keep_end = max_seq_len - overlap // 2
                all_predictions.append(window_predictions[keep_start:keep_end])
        
        # 拼接所有预测结果
        return torch.cat(all_predictions, dim=0)
    
    def _heuristic_finger_assignment(self, note: Dict) -> int:
        """
        启发式指法分配 (当模型无法确定时)
        
        Args:
            note: 音符信息
            
        Returns:
            指法编号
        """
        midi_num = note['midi_number']
        channel = note['channel']
        
        # 简单的启发式规则
        if channel == 0:  # 右手
            if midi_num >= 72:      # 高音区
                return 2  # 食指
            elif midi_num >= 60:    # 中音区
                return 3  # 中指
            else:                   # 低音区
                return 1  # 拇指
        else:  # 左手
            if midi_num <= 48:      # 低音区
                return -2  # 左手食指
            elif midi_num <= 60:    # 中音区
                return -3  # 左手中指
            else:                   # 高音区
                return -1  # 左手拇指
    
    def save_result(
        self,
        result: Dict,
        output_path: str,
        format: str = 'midi'
    ) -> str:
        """
        保存预测结果
        
        Args:
            result: 预测结果
            output_path: 输出文件路径
            format: 输出格式 ('midi', 'xml', 'json')
            
        Returns:
            实际保存的文件路径
        """
        output_path = Path(output_path)
        
        if format.lower() == 'midi':
            saved_path = self.output_formatter.save_as_midi(result, output_path)
        elif format.lower() == 'xml' or format.lower() == 'musicxml':
            saved_path = self.output_formatter.save_as_musicxml(result, output_path)
        elif format.lower() == 'json':
            saved_path = self.output_formatter.save_as_json(result, output_path)
        elif format.lower() == 'html':
            saved_path = self.output_formatter.create_visualization_html(result, output_path)
        elif format.lower() == 'csv':
            saved_path = self.output_formatter.save_as_csv(result, output_path)
        else:
            raise ValueError(f"不支持的输出格式: {format}. 支持的格式: midi, json, html, csv, xml")
        
        logger.info(f"结果已保存: {saved_path}")
        return str(saved_path)
    
    def batch_predict(
        self,
        midi_files: List[str],
        output_dir: str,
        format: str = 'midi'
    ) -> List[str]:
        """
        批量预测多个MIDI文件
        
        Args:
            midi_files: MIDI文件路径列表
            output_dir: 输出目录
            format: 输出格式
            
        Returns:
            输出文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        
        for midi_file in midi_files:
            try:
                # 预测指法
                result = self.predict_midi(midi_file)
                
                # 生成输出文件名
                input_name = Path(midi_file).stem
                output_name = f"{input_name}_fingering.{format}"
                output_path = output_dir / output_name
                
                # 保存结果
                saved_path = self.save_result(result, output_path, format)
                output_files.append(saved_path)
                
                logger.info(f"完成: {midi_file} -> {saved_path}")
                
            except Exception as e:
                logger.error(f"处理失败 {midi_file}: {e}")
                continue
        
        logger.info(f"批量预测完成，成功处理 {len(output_files)}/{len(midi_files)} 个文件")
        return output_files
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_path': str(self.model_path),
            'config': self.config,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        }


def create_predictor(
    model_path: str,
    config_path: str,
    device: Optional[str] = None
) -> FingeringPredictor:
    """
    创建指法预测器
    
    Args:
        model_path: 模型文件路径
        config_path: 配置文件路径
        device: 设备名称
        
    Returns:
        初始化的预测器
    """
    # 处理设备参数
    if device == "auto" or device is None:
        # 自动选择设备
        if torch.cuda.is_available():
            device_obj = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_obj = torch.device("mps")
        else:
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)
    
    predictor = FingeringPredictor(model_path, config_path, device_obj)
    return predictor 