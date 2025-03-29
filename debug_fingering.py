#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from music21 import converter, note, chord, articulations
import pandas as pd
import numpy as np

def analyze_fingering(file_path):
    """分析MusicXML文件中的指法标记"""
    score = converter.parse(file_path)
    
    for part_idx, part in enumerate(score.parts):
        print(f"\n分析声部 {part_idx + 1}:")
        
        for measure in part.getElementsByClass('Measure'):
            measure_number = measure.number
            print(f"  小节 {measure_number}:")
            
            for element in measure.notesAndRests:
                if isinstance(element, note.Note):
                    # 获取指法
                    fingering = None
                    for articulation in element.articulations:
                        if articulation.name == 'fingering':
                            fingering = articulation.fingerNumber
                            break
                    
                    if fingering is not None:
                        print(f"    音符 {element.nameWithOctave}: 指法={fingering}")
                    else:
                        print(f"    音符 {element.nameWithOctave}: 无指法")
                        
                elif isinstance(element, chord.Chord):
                    for n in element.notes:
                        # 获取指法
                        fingering = None
                        for articulation in n.articulations:
                            if articulation.name == 'fingering':
                                fingering = articulation.fingerNumber
                                break
                        
                        if fingering is not None:
                            print(f"    和弦音符 {n.nameWithOctave}: 指法={fingering}")
                        else:
                            print(f"    和弦音符 {n.nameWithOctave}: 无指法")

def analyze_consecutive_fingerings(file_path):
    """分析连续相同指法问题"""
    score = converter.parse(file_path)
    
    # 各手的指法收集
    right_hand_fingerings = []
    left_hand_fingerings = []
    
    for part_idx, part in enumerate(score.parts):
        # 简单判断手型：第一个声部通常是右手，第二个是左手
        is_right_hand = part_idx == 0
        
        for element in part.recurse().notes:
            if isinstance(element, note.Note):
                fingering = None
                for articulation in element.articulations:
                    if articulation.name == 'fingering':
                        fingering = articulation.fingerNumber
                        break
                
                if fingering is not None:
                    if is_right_hand and fingering > 0:
                        right_hand_fingerings.append(fingering)
                    elif not is_right_hand and fingering < 0:
                        left_hand_fingerings.append(fingering)
            
            elif isinstance(element, chord.Chord):
                for n in element.notes:
                    fingering = None
                    for articulation in n.articulations:
                        if articulation.name == 'fingering':
                            fingering = articulation.fingerNumber
                            break
                    
                    if fingering is not None:
                        if is_right_hand and fingering > 0:
                            right_hand_fingerings.append(fingering)
                        elif not is_right_hand and fingering < 0:
                            left_hand_fingerings.append(fingering)
    
    print("\n分析连续指法问题:", file_path)
    
    # 分析右手指法
    print("\n右手指法分析:")
    if right_hand_fingerings:
        # 指法分布
        finger_counts = {}
        for f in right_hand_fingerings:
            finger_counts[f] = finger_counts.get(f, 0) + 1
        
        print("指法分布:")
        for finger, count in sorted(finger_counts.items()):
            print(f"  指法 {finger}: {count} 次 ({count/len(right_hand_fingerings)*100:.2f}%)")
        
        # 查找最长连续相同指法
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(right_hand_fingerings)):
            if right_hand_fingerings[i] == right_hand_fingerings[i-1]:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
        max_consecutive = max(max_consecutive, current_consecutive)
        
        print(f"最长连续相同指法: {max_consecutive}")
        
        # 输出长连续段
        print("连续相同指法段 (长度>=3):")
        current_consecutive = 1
        current_finger = right_hand_fingerings[0]
        
        for i in range(1, len(right_hand_fingerings)):
            if right_hand_fingerings[i] == current_finger:
                current_consecutive += 1
            else:
                if current_consecutive >= 3:
                    print(f"  指法 {current_finger}: 连续 {current_consecutive} 次")
                current_finger = right_hand_fingerings[i]
                current_consecutive = 1
        
        if current_consecutive >= 3:
            print(f"  指法 {current_finger}: 连续 {current_consecutive} 次")
    
    # 分析左手指法
    print("\n左手指法分析:")
    if left_hand_fingerings:
        # 指法分布
        finger_counts = {}
        for f in left_hand_fingerings:
            finger_counts[f] = finger_counts.get(f, 0) + 1
        
        print("指法分布:")
        for finger, count in sorted(finger_counts.items()):
            print(f"  指法 {finger}: {count} 次 ({count/len(left_hand_fingerings)*100:.2f}%)")
        
        # 查找最长连续相同指法
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(left_hand_fingerings)):
            if left_hand_fingerings[i] == left_hand_fingerings[i-1]:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
        max_consecutive = max(max_consecutive, current_consecutive)
        
        print(f"最长连续相同指法: {max_consecutive}")
        
        # 输出长连续段
        print("连续相同指法段 (长度>=3):")
        current_consecutive = 1
        current_finger = left_hand_fingerings[0]
        
        for i in range(1, len(left_hand_fingerings)):
            if left_hand_fingerings[i] == current_finger:
                current_consecutive += 1
            else:
                if current_consecutive >= 3:
                    print(f"  指法 {current_finger}: 连续 {current_consecutive} 次")
                current_finger = left_hand_fingerings[i]
                current_consecutive = 1
        
        if current_consecutive >= 3:
            print(f"  指法 {current_finger}: 连续 {current_consecutive} 次")

def count_fingerings(file_path):
    """统计音符和和弦的指法情况"""
    score = converter.parse(file_path)
    
    total_notes = 0
    notes_with_fingering = 0
    total_chords = 0
    chords_with_at_least_one_fingering = 0
    chord_notes_total = 0
    chord_notes_with_fingering = 0
    
    for part in score.parts:
        for measure in part.getElementsByClass('Measure'):
            for element in measure.notesAndRests:
                if isinstance(element, note.Note):
                    total_notes += 1
                    
                    # 检查是否有指法
                    for articulation in element.articulations:
                        if articulation.name == 'fingering':
                            notes_with_fingering += 1
                            break
                            
                elif isinstance(element, chord.Chord):
                    total_chords += 1
                    chord_has_fingering = False
                    
                    for n in element.notes:
                        chord_notes_total += 1
                        
                        # 检查是否有指法
                        for articulation in n.articulations:
                            if articulation.name == 'fingering':
                                chord_notes_with_fingering += 1
                                chord_has_fingering = True
                                break
                    
                    if chord_has_fingering:
                        chords_with_at_least_one_fingering += 1
    
    print("\n统计结果:")
    print(f"总音符数: {total_notes}")
    print(f"有指法的音符数: {notes_with_fingering} ({notes_with_fingering/total_notes*100:.2f}%)")
    print(f"总和弦数: {total_chords}")
    print(f"有指法的和弦数: {chords_with_at_least_one_fingering} ({chords_with_at_least_one_fingering/total_chords*100:.2f}% 如果至少有一个音符有指法)")
    print(f"和弦内总音符数: {chord_notes_total}")
    print(f"和弦内有指法的音符数: {chord_notes_with_fingering} ({chord_notes_with_fingering/chord_notes_total*100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python debug_fingering.py <musicxml_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyze_fingering(file_path)
    count_fingerings(file_path)
    analyze_consecutive_fingerings(file_path) 