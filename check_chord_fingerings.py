from music21 import converter, articulations, chord, note
import sys

def check_chord_fingerings(file_path):
    """检查MusicXML文件中和弦的指法标记情况"""
    print(f"分析文件: {file_path}")
    s = converter.parse(file_path)
    
    chord_count = 0
    chord_notes_count = 0
    chord_fingerings_count = 0
    
    for part_idx, part in enumerate(s.parts):
        print(f"\n分析声部 {part_idx + 1}:")
        
        for m_idx, m in enumerate(part.getElementsByClass('Measure')):
            chords_in_measure = list(m.getElementsByClass('Chord'))
            if chords_in_measure:
                print(f"  小节 {m.number}:")
                
                for chord_idx, c in enumerate(chords_in_measure):
                    chord_count += 1
                    notes_in_chord = c.notes
                    chord_notes_count += len(notes_in_chord)
                    
                    print(f"    和弦 {chord_idx + 1} (音符数: {len(notes_in_chord)}):")
                    
                    for note_idx, n in enumerate(notes_in_chord):
                        fingering = None
                        for a in n.articulations:
                            if a.name == 'fingering':
                                fingering = a.fingerNumber
                                chord_fingerings_count += 1
                                break
                        
                        if fingering is not None:
                            print(f"      音符 {n.nameWithOctave}: 指法={fingering}")
                        else:
                            print(f"      音符 {n.nameWithOctave}: 无指法")
    
    print("\n统计结果:")
    print(f"和弦总数: {chord_count}")
    print(f"和弦内音符总数: {chord_notes_count}")
    print(f"有指法的和弦音符数: {chord_fingerings_count}")
    if chord_notes_count > 0:
        print(f"和弦内音符指法覆盖率: {chord_fingerings_count/chord_notes_count*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python check_chord_fingerings.py <musicxml_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    check_chord_fingerings(file_path) 