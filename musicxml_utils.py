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
    """应用预测的指法到乐谱"""
    fingering_idx = 0

    for part in score.parts:
        for measure in part.getElementsByClass('Measure'):
            for element in measure.notesAndRests:
                if isinstance(element, note.Note):
                    if fingering_idx < len(predicted_fingerings):
                        fingering = predicted_fingerings[fingering_idx]
                        # 因为已经是实际指法值了，不需要再转换
                        element.articulations.append(articulations.Fingering(fingering))
                        fingering_idx += 1
                elif isinstance(element, chord.Chord):
                    for n in element.notes:
                        if fingering_idx < len(predicted_fingerings):
                            fingering = predicted_fingerings[fingering_idx]
                            n.articulations.append(articulations.Fingering(fingering))
                            fingering_idx += 1

    return score