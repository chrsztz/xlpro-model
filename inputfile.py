from music21 import converter, note, chord, key, articulations
from model_training import BiLSTM
import torch
import torch.nn as nn
import numpy as np

# 定义升号和降号的顺序
ORDER_SHARPS = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
ORDER_FLATS = ['B', 'E', 'A', 'D', 'G', 'C', 'F']

# 加载mxl文件
score = converter.parse('example.mxl')  # 替换为您的文件路径

# 分析调号
key_sig = score.analyze('key')
key_str = f"{key_sig.tonic.name} {key_sig.mode}"  # 例如 "A major"

# 根据调号中的升降符号数量确定哪些音符需要升降
sharps = set()
flats = set()

if key_sig.sharps > 0:
    sharps = set(ORDER_SHARPS[:key_sig.sharps])
elif key_sig.sharps < 0:
    flats = set(ORDER_FLATS[:abs(key_sig.sharps)])

# 初始化左右手音符列表
right_hand_notes = []
left_hand_notes = []

# 初始化数据列表用于后续的训练数据构造
data = []

# 改进后的函数：根据音高范围判断是右手还是左手
def get_hand(part):
    high_notes = 0
    low_notes = 0
    for n in part.flat.notes:
        if isinstance(n, note.Note):
            if n.pitch.midi >= 60:  # C4及以上
                high_notes += 1
            else:
                low_notes += 1
    if high_notes >= low_notes:
        hand = 'right'
    else:
        hand = 'left'
    print(f"Assigning part '{part.partName}' (ID: {part.id}) to {hand} hand based on pitch range.")
    return hand

# 遍历每个 part
for part_index, part in enumerate(score.parts):
    hand = get_hand(part)

    previous_note = None  # 存储前一个音符，用于处理连线

    for element in part.flat.notes:
        # 处理单个音符
        if isinstance(element, note.Note):
            base_name = element.pitch.name  # 例如 'C', 'F#'
            octave = element.octave

            # 根据调号自动添加升降符号
            if base_name in sharps:
                note_name = f"{base_name}#{octave}"
            elif base_name in flats:
                note_name = f"{base_name}b{octave}"
            else:
                note_name = f"{base_name}{octave}"

            # 如果音符有显式的升降符号，覆盖调号的标记
            if element.pitch.accidental:
                if element.pitch.accidental.name == 'sharp':
                    note_name = f"{base_name}#{octave}"
                elif element.pitch.accidental.name == 'flat':
                    note_name = f"{base_name}b{octave}"
                elif element.pitch.accidental.name == 'natural':
                    note_name = f"{base_name}{octave}"

            # 获取指法信息
            fingering = None
            for articulation in element.articulations:
                if isinstance(articulation, articulations.Fingering):
                    fingering = articulation.fingerNumber  # 获取指法号码
                    break  # 假设每个音符只有一个指法

            # 构建音符字符串并添加调号标记
            note_str = f"{key_str} {note_name}"
            if fingering:
                note_str += f" fingering={fingering}"

            # 处理连线，仅在 'continue' 和 'stop' 类型时添加 "finger stays"
            if element.tie:
                if element.tie.type in ['continue', 'stop']:
                    note_str += " (finger stays)"

            # 检查是否为同音连线的不需要弹奏的音符
            if (previous_note and
                previous_note.name == element.name and
                element.tie and
                element.tie.type == 'start'):
                note_str += " (do not play)"
            else:
                # 根据 hand 分配到左右手
                if hand == 'right':
                    right_hand_notes.append(note_str)
                elif hand == 'left':
                    left_hand_notes.append(note_str)

            # 将数据添加到训练数据列表中
            data.append({
                'note': note_name,
                'octave': octave,
                'duration': element.quarterLength,
                'hand': hand,
                'fingering': fingering
            })

            # 更新前一个音符
            previous_note = element

        # 处理和弦
        elif isinstance(element, chord.Chord):
            chord_notes = []
            do_not_play = False  # 标记是否有音符不需要弹奏

            for n in element.notes:
                base_name = n.pitch.name
                octave = n.octave

                # 根据调号自动添加升降符号
                if base_name in sharps:
                    note_name = f"{base_name}#{octave}"
                elif base_name in flats:
                    note_name = f"{base_name}b{octave}"
                else:
                    note_name = f"{base_name}{octave}"

                # 如果音符有显式的升降符号，覆盖调号的标记
                if n.pitch.accidental:
                    if n.pitch.accidental.name == 'sharp':
                        note_name = f"{base_name}#{octave}"
                    elif n.pitch.accidental.name == 'flat':
                        note_name = f"{base_name}b{octave}"
                    elif n.pitch.accidental.name == 'natural':
                        note_name = f"{base_name}{octave}"

                # 获取指法信息
                fingering = None
                for articulation in n.articulations:
                    if isinstance(articulation, articulations.Fingering):
                        fingering = articulation.fingerNumber  # 获取指法号码
                        break  # 假设每个音符只有一个指法

                # 构建音符字符串并添加调号标记
                note_str = f"{key_str} {note_name}"
                if fingering:
                    note_str += f" fingering={fingering}"

                # 处理连线，仅在 'continue' 和 'stop' 类型时添加 "finger stays"
                if n.tie:
                    if n.tie.type in ['continue', 'stop']:
                        note_str += " (finger stays)"

                # 检查是否为同音连线的不需要弹奏的音符
                if (previous_note and
                    previous_note.name == n.name and
                    n.tie and
                    n.tie.type == 'start'):
                    note_str += " (do not play)"
                    do_not_play = True

                chord_notes.append(note_str)

                # 将数据添加到训练数据列表中
                data.append({
                    'note': note_name,
                    'octave': octave,
                    'duration': n.quarterLength,
                    'hand': hand,
                    'fingering': fingering
                })

            # 构建和弦字符串并添加调号标记
            chord_str = f"{key_str} " + " ".join(chord_notes)

            if do_not_play:
                chord_str += " (do not play)"

            # 根据 hand 分配到左右手
            if hand == 'right':
                right_hand_notes.append(chord_str)
            elif hand == 'left':
                left_hand_notes.append(chord_str)

            # 更新前一个音符为和弦中的最后一个音符
            if element.notes:
                previous_note = element.notes[-1]
            else:
                previous_note = None

        # 处理休止符
        elif isinstance(element, note.Rest):
            rest_duration = element.quarterLength
            # 判断是否是常见的休止符时值
            if rest_duration in [0.25, 0.5, 1, 2, 4, 8]:
                rest_str = f'rest {rest_duration}'
                # 添加到左右手
                right_hand_notes.append(rest_str)
                left_hand_notes.append(rest_str)

# 输出左右手的音符、和弦和休止符
print("右手音符、和弦和休止符：")
for item in right_hand_notes:
    print(item)

print("\n左手音符、和弦和休止符：")
for item in left_hand_notes:
    print(item)

# 查看部分训练数据
print("\n部分训练数据样例：")
for entry in data[:10]:
    print(entry)

# 定义预处理函数
def preprocess(notes, sequence_length=100, note_to_int=None):
    """
    将音符序列转换为模型输入的张量格式。
    Args:
        notes (list): 音符的 MIDI 编号列表。
        sequence_length (int): 序列的固定长度。
        note_to_int (dict): 音符到整数的映射，如果没有则自动创建。
    Returns:
        torch.Tensor: 预处理后的输入张量。
        dict: 音符到整数的映射。
    """
    if note_to_int is None:
        unique_notes = sorted(list(set(notes)))
        note_to_int = {note: number for number, note in enumerate(unique_notes)}

    # 将音符转换为整数序列
    integer_sequence = [note_to_int[note] for note in notes]

    # 填充或截断序列
    if len(integer_sequence) < sequence_length:
        integer_sequence = [0] * (sequence_length - len(integer_sequence)) + integer_sequence
    else:
        integer_sequence = integer_sequence[-sequence_length:]

    # 转换为张量
    input_tensor = torch.tensor(integer_sequence, dtype=torch.long).unsqueeze(0)  # (1, sequence_length)

    return input_tensor, note_to_int


# 示例调用
sequence_length = 100  # 根据您的模型需求调整
input_tensor, note_to_int = preprocess([entry['note'] for entry in data], sequence_length)

# 加载模型
model = BiLSTM()
model.load_state_dict(torch.load('fingering_bilstm_model.pth', map_location=torch.device('cpu')))
model.eval()  # 设定为评估模式

# 进行预测
with torch.no_grad():
    output = model(input_tensor)  # 假设模型输出是指法的概率分布或直接的指法编号
    # 如果模型输出是概率分布，需要取最大值
    if isinstance(output, torch.Tensor):
        predicted_fingering = torch.argmax(output, dim=1).tolist()
    else:
        predicted_fingering = output  # 根据模型输出调整

# 确保预测的指法数量与音符数量一致
if len(predicted_fingering) != len(data):
    print("预测的指法数量与音符数量不一致！")
else:
    fingering_index = 0  # 指法索引
    for part in score.parts:
        for element in part.flat.notes:
            if isinstance(element, note.Note):
                # 添加预测的指法
                fingering_number = predicted_fingering[fingering_index]
                fingering_articulation = articulations.Fingering(fingering_number)
                element.articulations.append(fingering_articulation)
                fingering_index += 1
            elif isinstance(element, chord.Chord):
                for n in element.notes:
                    fingering_number = predicted_fingering[fingering_index]
                    fingering_articulation = articulations.Fingering(fingering_number)
                    n.articulations.append(fingering_articulation)
                    fingering_index += 1
# 导出为 MusicXML
modified_musicxml_path = 'modified_example.mxl'
