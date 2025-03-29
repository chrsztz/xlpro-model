# musicxml_utils.py

from music21 import converter, note, chord, articulations, clef, tempo
import music21
import pandas as pd
import numpy as np
from data_utils import load_pickle

TEMPO_MAPPING = {
    'Larghissimo': 24,
    'Grave': 40,
    'Largo': 40,
    'Larghetto': 50,
    'Adagio': 66,
    'Adagietto': 68,
    'Andante': 76,
    'Andantino': 80,
    'Moderato': 108,
    'Allegretto': 112,
    'Allegro': 120,
    'Vivace': 156,
    'Presto': 168,
    'Prestissimo': 200
}


def get_tempo(score, default_bpm=120):
    """获取乐谱的速度标记"""
    # 首先检查显式的速度标记
    for t in score.flat.getElementsByClass(tempo.MetronomeMark):
        if t.number is not None:
            return t.number
        elif t.text is not None:
            return TEMPO_MAPPING.get(t.text, default_bpm)

    # 如果没有速度标记，通过分析音符时值推断
    durations = [n.duration.quarterLength for n in score.flat.notes]
    if not durations:
        return default_bpm

    most_common_duration = max(set(durations), key=durations.count)
    inferred_bpm = 60 / most_common_duration
    return min(max(inferred_bpm, 40), 200)  # 限制在合理范围内


def convert_to_seconds(quarterLength, bpm):
    """将四分音符长度转换为秒数"""
    return (quarterLength * 60) / bpm


def determine_hands(score):
    """
    确定右手和左手部分，通过分析整个乐谱的音高分布
    """
    parts = score.parts
    part_info = []

    for part in parts:
        notes = []
        for element in part.recurse().notes:
            if isinstance(element, note.Note):
                notes.append(element)
            elif isinstance(element, chord.Chord):
                notes.extend(element.notes)

        if not notes:
            continue

        # 计算这个声部的统计信息
        avg_pitch = sum(n.pitch.midi for n in notes) / len(notes)
        avg_octave = sum(n.octave for n in notes) / len(notes)
        highest_pitch = max(n.pitch.midi for n in notes)
        lowest_pitch = min(n.pitch.midi for n in notes)

        part_info.append({
            'part': part,
            'avg_pitch': avg_pitch,
            'avg_octave': avg_octave,
            'highest_pitch': highest_pitch,
            'lowest_pitch': lowest_pitch,
            'note_count': len(notes)
        })

    # 按平均音高排序
    part_info.sort(key=lambda x: x['avg_pitch'], reverse=True)

    # 创建手型映射字典
    hand_mapping = {}
    for i, info in enumerate(part_info):
        if i == 0:
            hand_mapping[info['part']] = 'right'
        else:
            hand_mapping[info['part']] = 'left'

    return hand_mapping

def extract_note_features(element, measure, current_time, bpm, part_hand=None):
    """提取单个音符的特征，确保与训练数据格式一致"""
    # 转换为秒数
    onset_time = convert_to_seconds(current_time, bpm)
    offset_time = convert_to_seconds(current_time + element.duration.quarterLength, bpm)

    # 计算duration和real_duration（在训练集中是一样的）
    duration = round(offset_time - onset_time, 2)
    real_duration = duration  # 保持一致

    features = {
        'onset_time': round(onset_time, 3),
        'offset_time': round(offset_time, 3),
        'duration': duration,
        'real_duration': real_duration,  # 添加real_duration
        'hand': part_hand,
        'chord': 0,
        'fingering': -1
    }

    if isinstance(element, note.Note):
        features['note'] = f"{element.pitch.name}{element.pitch.octave}"
        features['midi_number'] = element.pitch.midi
    else:
        features['note'] = 'rest'
        features['midi_number'] = 60

    return features

def process_chord_notes(chord_element, measure, current_time, bpm, part_hand=None):
    """处理和弦中的音符，返回每个音符的特征"""
    sorted_notes = sorted(chord_element.notes, key=lambda x: x.pitch.midi)
    features_list = []

    onset_time = convert_to_seconds(current_time, bpm)
    offset_time = convert_to_seconds(current_time + chord_element.duration.quarterLength, bpm)
    duration = round(offset_time - onset_time, 2)
    real_duration = duration  # 保持一致

    for note_obj in sorted_notes:
        features = extract_note_features(note_obj, measure, current_time, bpm, part_hand)
        features['chord'] = 1
        features['onset_time'] = round(onset_time, 3)
        features['offset_time'] = round(offset_time, 3)
        features['duration'] = duration
        features['real_duration'] = real_duration  # 确保添加real_duration
        features_list.append(features)

    return features_list


def extract_all_features(score):
    """从整个乐谱中提取所有特征"""
    bpm = get_tempo(score)
    print(f"Using tempo: {bpm} BPM")

    # 首先确定手型映射
    hand_mapping = determine_hands(score)
    print("Hand assignment:")
    for part, hand in hand_mapping.items():
        print(f"Part {part.id if hasattr(part, 'id') else 'Unknown'}: {hand}")

    all_features = []
    current_time = 0.0

    for part in score.parts:
        # 获取当前声部的手型
        part_hand = hand_mapping.get(part, 'right')

        for measure in part.getElementsByClass('Measure'):
            for element in measure.notesAndRests:
                if isinstance(element, note.Note):
                    features = extract_note_features(element, measure, current_time, bpm, part_hand)
                    all_features.append(features)
                elif isinstance(element, chord.Chord):
                    chord_features = process_chord_notes(element, measure, current_time, bpm, part_hand)
                    all_features.extend(chord_features)
                current_time += element.duration.quarterLength

    return pd.DataFrame(all_features)


def apply_predicted_fingering(score, predicted_fingerings):
    """应用预测的指法到乐谱，确保包括和弦音符在内的所有音符都有正确的指法"""
    from music21 import articulations
    # 获取手部信息
    hand_mapping = determine_hands(score)
    
    # 首先收集所有音符及其索引，包括和弦中的音符
    all_notes = []
    fingering_idx = 0
    
    # 清除旧指法并收集所有音符
    for part in score.parts:
        for measure in part.getElementsByClass('Measure'):
            for element in measure.notesAndRests:
                if isinstance(element, note.Note):
                    # 清除当前指法
                    element.articulations = [a for a in element.articulations if not isinstance(a, articulations.Fingering)]
                    all_notes.append((element, part))
                    fingering_idx += 1
                elif isinstance(element, chord.Chord):
                    # 对和弦中的音符按音高排序
                    for n in sorted(element.notes, key=lambda x: x.pitch.midi):
                        # 清除当前指法
                        n.articulations = [a for a in n.articulations if not isinstance(a, articulations.Fingering)]
                        all_notes.append((n, part))
                        fingering_idx += 1
    
    # 确保预测的指法数量与音符数量匹配
    if len(predicted_fingerings) < len(all_notes):
        print(f"警告：预测指法数量 ({len(predicted_fingerings)}) 少于音符数量 ({len(all_notes)})")
        # 补齐缺失的指法
        predicted_fingerings = list(predicted_fingerings) + [1] * (len(all_notes) - len(predicted_fingerings))
    elif len(predicted_fingerings) > len(all_notes):
        print(f"警告：预测指法数量 ({len(predicted_fingerings)}) 多于音符数量 ({len(all_notes)})")
        predicted_fingerings = predicted_fingerings[:len(all_notes)]
    
    # 应用指法
    chord_count = 0
    chord_notes_count = 0
    
    # 统计和弦信息
    for part in score.parts:
        for measure in part.getElementsByClass('Measure'):
            for element in measure.notesAndRests:
                if isinstance(element, chord.Chord):
                    chord_count += 1
                    chord_notes_count += len(element.notes)
    
    # 应用指法
    for i, (note_obj, part) in enumerate(all_notes):
        if i < len(predicted_fingerings):
            fingering = predicted_fingerings[i]
            part_hand = hand_mapping.get(part, 'right')
            
            # 根据手部调整指法符号
            if part_hand == 'left' and fingering > 0:
                fingering = -fingering  # 左手使用负号
            elif part_hand == 'right' and fingering < 0:
                fingering = abs(fingering)  # 右手使用正号
            
            # 创建指法对象并添加到音符
            finger_obj = articulations.Fingering(fingerNumber=fingering)
            note_obj.articulations.append(finger_obj)
    
    # 强制确保和弦音符的指法被正确标记
    chord_note_count_with_fingering = 0
    for part in score.parts:
        for measure in part.getElementsByClass('Measure'):
            for element in measure.getElementsByClass('Chord'):
                for n in element.notes:
                    has_fingering = False
                    for a in n.articulations:
                        if isinstance(a, articulations.Fingering):
                            has_fingering = True
                            chord_note_count_with_fingering += 1
                            break
                    
                    if not has_fingering:
                        # 如果没有指法，尝试添加一个默认指法
                        part_hand = hand_mapping.get(part, 'right')
                        default_fingering = 1 if part_hand == 'right' else -1
                        finger_obj = articulations.Fingering(fingerNumber=default_fingering)
                        n.articulations.append(finger_obj)
                        chord_note_count_with_fingering += 1
    
    print(f"指法应用完成。共处理 {chord_count} 个和弦，{chord_notes_count} 个和弦音符")
    print(f"和弦音符指法覆盖率: {chord_note_count_with_fingering}/{chord_notes_count} ({chord_note_count_with_fingering/chord_notes_count*100:.2f}% 如果至少有一个音符有指法)")
    
    return score