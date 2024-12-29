from music21 import converter, instrument, note, chord
import matplotlib.pyplot as plt


def get_notes(part):
    """
    提取指定部分中的所有音符和和弦。
    """
    notes = []
    for element in part.recurse():
        if isinstance(element, note.Note):
            notes.append(element.nameWithOctave)
        elif isinstance(element, chord.Chord):
            # 将和弦中的各个音符用点连接
            notes.append('.'.join(n.nameWithOctave for n in element.notes))
    return notes


def identify_right_left(parts):
    """
    根据每个部分的名称或乐器信息来识别右手和左手。
    如果无法通过名称或乐器信息识别，则可以根据音域进行辅助判断。
    """
    right_hand = None
    left_hand = None

    for part in parts:
        part_instrument = part.getInstrument()
        part_name = part.partName.lower() if part.partName else ""
        instrument_name = part_instrument.instrumentName.lower() if part_instrument.instrumentName else ""

        # 检查部分名称或乐器名称中是否包含“right”或“left”
        if any(keyword in part_name for keyword in ["right", "treble", "rh"]) or \
                any(keyword in instrument_name for keyword in ["right", "treble"]):
            right_hand = part
        elif any(keyword in part_name for keyword in ["left", "bass", "lh"]) or \
                any(keyword in instrument_name for keyword in ["left", "bass"]):
            left_hand = part

    # 如果无法通过名称或乐器信息识别，使用音域辅助判断
    if not right_hand or not left_hand:
        print("无法通过部分名称或乐器信息完全识别右手和左手，尝试基于音域进行判断。")
        part_averages = []
        for part in parts:
            notes = []
            for element in part.recurse().notes:
                if isinstance(element, note.Note):
                    notes.append(element)
                elif isinstance(element, chord.Chord):
                    notes.extend(element.notes)
            if not notes:
                part_averages.append(0)
                continue
            avg_octave = sum(n.octave for n in notes) / len(notes)
            part_averages.append(avg_octave)

        # 排序部分，假设平均八度高的是右手，低的是左手
        sorted_parts = sorted(zip(parts, part_averages), key=lambda x: x[1], reverse=True)
        if not right_hand and len(sorted_parts) >= 1:
            right_hand = sorted_parts[0][0]
        if not left_hand and len(sorted_parts) >= 2:
            left_hand = sorted_parts[1][0]

    return right_hand, left_hand


def main():
    # 替换为你的 .mxl 文件路径
    file_path = 'example.mxl'  # 例如: 'E:/music/score.mxl'

    # 解析 .mxl 文件
    score = converter.parse(file_path)

    # 获取所有部分
    parts = score.parts

    if len(parts) < 2:
        print("乐谱中没有找到足够的部分来区分右手和左手。")
        return

    # 识别右手和左手
    right_hand, left_hand = identify_right_left(parts)

    # 提取音符
    right_notes = get_notes(right_hand) if right_hand else []
    left_notes = get_notes(left_hand) if left_hand else []

    # 计算数量
    right_count = len(right_notes)
    left_count = len(left_notes)
    total_count = right_count + left_count

    # 计算分布比例
    right_percentage = (right_count / total_count * 100) if total_count > 0 else 0
    left_percentage = (left_count / total_count * 100) if total_count > 0 else 0

    # 打印结果
    print("音符分布统计:")
    print(f"总音符数量: {total_count}")
    print(f"右手音符数量: {right_count} ({right_percentage:.2f}%)")
    print(f"左手音符数量: {left_count} ({left_percentage:.2f}%)")

    # 可选：打印所有右手和左手的音符
    print("\n右手音符:")
    for n in right_notes:
        print(n)

    print("\n左手音符:")
    for n in left_notes:
        print(n)

    # 可选：绘制饼图
    if total_count > 0:
        labels = ['右手', '左手']
        sizes = [right_count, left_count]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.05, 0)  # 突出显示右手

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=140)
        plt.title('右手与左手音符分布')
        plt.axis('equal')  # 确保饼图为圆形
        plt.show()


if __name__ == "__main__":
    main()
