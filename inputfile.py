# inputfile.py

from music21 import converter, note, chord, articulations, tempo
from models import BiLSTMWithAttention
import torch
import numpy as np
import pickle
import pandas as pd
from data_utils import (
    load_pickle,
    get_fused_features,
    combine_features,
    get_midi_number,
    calculate_speed_features,
    calculate_midi_diff,
    create_word_column,
    is_black_key,
    normalize_spelled_pitch,        # 导入标准化函数
    denormalize_spelled_pitch     # 导入还原函数
)
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec

# 1. 添加 Tempo 标记到 BPM 的映射
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

def get_hand(part):
    """
    根据音高范围判断是右手还是左手。
    """
    high_notes = 0
    low_notes = 0
    for n in part.flat.notes:
        if isinstance(n, note.Note):
            if n.pitch.midi >= 60:  # C4及以上为右手阈值（可根据需要调整）
                high_notes += 1
            else:
                low_notes += 1
    hand = 'right' if high_notes >= low_notes else 'left'
    print(f"Assigning part '{part.partName}' (ID: {part.id}) to {hand} hand based on pitch range.")
    return hand

def infer_tempo(score):
    """
    根据乐谱中最常见的音符时值推断 Tempo。
    假设最常见的音符时值对应一个节拍单位。
    """
    durations = [element.duration.quarterLength for element in score.flat.notes if isinstance(element, (note.Note, chord.Chord))]

    if not durations:
        return 120  # 如果没有音符，返回默认 BPM

    duration_counts = pd.Series(durations).value_counts()
    most_common_duration = duration_counts.idxmax()

    # 假设最常见的音符时值对应一个节拍单位
    # 例如，如果最常见的 duration 是 1.0（四分音符），则 BPM = 60 / 1.0 = 60
    # 根据您的训练数据，您可能需要调整此计算方式
    inferred_bpm = 60 / most_common_duration

    # 设置 BPM 的合理范围
    inferred_bpm = max(60, min(inferred_bpm, 240))

    print(f"Inferred Tempo based on most common duration ({most_common_duration}): {inferred_bpm} BPM")
    return inferred_bpm

def get_tempi(score, default_bpm=120):
    """
    从乐谱中提取所有 Tempo 信息，并按位置排序。
    """
    tempos = score.flat.getElementsByClass(tempo.MetronomeMark)
    tempi = []
    for tm in tempos:
        if tm.number:
            bpm = tm.number
            tempi.append((tm.offset, bpm))
            print(f"Found Tempo Mark: {tm.number} BPM at offset {tm.offset}")
        elif tm.text:
            bpm = TEMPO_MAPPING.get(tm.text, default_bpm)
            tempi.append((tm.offset, bpm))
            print(f"Found Tempo Text: '{tm.text}' mapped to {bpm} BPM at offset {tm.offset}")
    # 按 offset 排序
    tempi = sorted(tempi, key=lambda x: x[0])
    return tempi

def get_tempo_at_offset(tempi, current_offset, default_bpm=120):
    """
    根据当前 offset 查找对应的 BPM。
    """
    current_bpm = default_bpm
    for offset, bpm in tempi:
        if current_offset >= offset:
            current_bpm = bpm
        else:
            break
    return current_bpm

def calculate_actual_duration(element, bpm):
    """
    根据 BPM 计算实际的音符时长（秒）。
    """
    beat_duration = 60.0 / bpm  # 一拍的时长（秒）
    actual_duration = element.duration.quarterLength * beat_duration
    return round(actual_duration,2)


# inputfile.py

def preprocess_input_data(data, le_pitch, le_duration, le_hand, le_fingering, word2vec_model, scaler,
                          sequence_length=10, window=1.0):
    """
    将音符数据转换为模型输入格式。
    """
    df = pd.DataFrame(data)
    # 在这里添加调试信息
    print("Initial DataFrame shape:", df.shape)
    print("Initial data sample:")
    print(df.head())

    # 删除指法缺失的音符
    df = df.dropna(subset=['fingering'])
    # 添加 dropna 后的调试信息
    print("Shape after dropna:", df.shape)
    # 确保指法为整数类型，填充缺失值为 -1
    df['fingering'] = df['fingering'].fillna(-1).astype(int)

    # 标准化 'note' 列
    df['normalized_spelled_pitch'] = df['note'].apply(normalize_spelled_pitch)

    # 计算 MIDI 编号，使用标准化后的音符
    df['midi_number'] = df['normalized_spelled_pitch'].apply(get_midi_number)

    # 对类别特征进行标签编码
    df['pitch_encoded'] = le_pitch.transform(df['normalized_spelled_pitch'])
    df['duration_encoded'] = le_duration.transform(df['duration'].astype(str))
    df['hand_encoded'] = le_hand.transform(df['hand'])

    # 对目标标签进行标签编码
    df['fingering_encoded'] = le_fingering.transform(df['fingering'])

    if 'chord' not in df.columns:
        df['chord'] = 0  # 默认不是和弦

    # 使用 calculate_midi_diff 计算 midi_diff_processed
    df = calculate_midi_diff(df)

    # 使用 calculate_speed_features 计算 real_duration 和 note_density
    df = calculate_speed_features(df, window=window)

    # 计算 black_key
    df['black_key'] = df['midi_number'].apply(is_black_key)

    # 创建 word 列
    feature_columns = [
        'pitch_encoded', 'duration_encoded', 'hand_encoded',
        'midi_diff_processed', 'real_duration',
        'note_density', 'black_key', 'chord'
    ]

    # 在获取融合特征之前添加调试信息
    print("Feature columns:", feature_columns)
    print("DataFrame shape:", df.shape)
    print("Sample of data before word creation:")
    print(df[feature_columns].head())

    # 创建 word 列（确保分词）
    df = create_word_column(df, feature_columns)
    print("\nSample words after creation:")
    print(df['word'].head())

    # **修改部分开始**
    # 将 'word' 列拆分为单词列表，并保持为 Pandas Series
    tokenized_sentences = df['word'].apply(lambda x: x.split())
    # **修改部分结束**

    # **添加调试信息：检查未在词汇表中的单词**
    all_tokens = set([token for sentence in tokenized_sentences for token in sentence])
    missing_words = all_tokens - set(word2vec_model.wv.index_to_key)
    if missing_words:
        print(f"Missing words in Word2Vec vocabulary: {missing_words}")
    else:
        print("All words are present in Word2Vec vocabulary.")

    # 获取融合特征
    try:
        # 传递分词后的句子列表给 get_fused_features
        df = get_fused_features(df, word2vec_model, tokenized_sentences)

        # 验证融合特征是否成功生成
        if df['fused_feature'].empty or len(df['fused_feature']) == 0:
            raise ValueError("No fused features were generated")

        # 检查融合特征的维度
        print(f"\nFused feature sample shape: {len(df['fused_feature'].iloc[0])}")
        print(f"Number of fused features: {len(df['fused_feature'])}")

        fused_features = np.vstack(df['fused_feature'].values)
        fused_features_scaled = scaler.transform(fused_features)
        df['fused_feature_scaled'] = list(fused_features_scaled)
    except Exception as e:
        print(f"Error in feature fusion process: {str(e)}")
        raise

    # 将融合特征与原始特征组合
    df = combine_features(df, feature_columns)

    X = np.stack(df['combined_features'].values)
    y = df['fingering_encoded'].values

    # 创建序列
    X_seq = []
    y_seq = []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    return X_seq, y_seq


def main():
    score_path = 'Fr_Elise.mxl'  # 替换为您的文件路径
    score = converter.parse(score_path)

    # 提取 Tempo 信息
    tempi = get_tempi(score, default_bpm=120)

    # 初始化 BPM 列表，按 offset 排序
    sorted_tempi = sorted(tempi, key=lambda x: x[0])
    tempo_index = 0
    bpm = sorted_tempi[0][1] if sorted_tempi else 120

    # 初始化当前时间
    current_time = 0.0

    key_sig = score.analyze('key')
    key_str = f"{key_sig.tonic.name} {key_sig.mode}"

    ORDER_SHARPS = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
    ORDER_FLATS = ['B', 'E', 'A', 'D', 'G', 'C', 'F']

    sharps = set()
    flats = set()

    if key_sig.sharps > 0:
        sharps = set(ORDER_SHARPS[:key_sig.sharps])
    elif key_sig.sharps < 0:
        flats = set(ORDER_FLATS[:abs(key_sig.sharps)])

    right_hand_notes = []
    left_hand_notes = []
    data = []

    def process_note_name(pitch, sharps, flats):
        base_step = pitch.step  # 纯字母音名，如：C, D, E, F, G, A, B
        octave = pitch.octave
        accidental = pitch.accidental

        # 根据调号确定初步升降号
        if base_step in sharps:
            note_name = f"{base_step}#{octave}"
        elif base_step in flats:
            note_name = f"{base_step}b{octave}"
        else:
            note_name = f"{base_step}{octave}"

        # 如果实际音中有显式的accidental，则覆盖上面由调号确定的符号
        if accidental:
            if accidental.name == 'sharp':
                note_name = f"{base_step}#{octave}"
            elif accidental.name == 'flat':
                note_name = f"{base_step}b{octave}"
            elif accidental.name == 'natural':
                note_name = f"{base_step}{octave}"

        return note_name

    # 处理每个音符，计算实际 Duration
    for part_index, part in enumerate(score.parts):
        hand = get_hand(part)
        previous_note = None

        for element in part.flat.notes:
            # 检查是否有新的 Tempo 标记
            while tempo_index < len(sorted_tempi) and element.offset >= sorted_tempi[tempo_index][0]:
                bpm = sorted_tempi[tempo_index][1]
                print(f"Tempo changed to {bpm} BPM at offset {sorted_tempi[tempo_index][0]}")
                tempo_index += 1

            # 计算实际 Duration
            actual_duration = calculate_actual_duration(element, bpm)

            # 设置 onset_time 和 offset_time
            onset_time = current_time
            offset_time = onset_time + actual_duration

            # 更新 current_time
            current_time = offset_time

            if isinstance(element, note.Note):
                # 处理指法
                fingering = None
                for articulation in element.articulations:
                    if isinstance(articulation, articulations.Fingering):
                        fingering = articulation.fingerNumber
                        break

                # 记录音符信息
                note_name = process_note_name(element.pitch, sharps, flats)
                data.append({
                    'note': note_name,
                    'octave': element.octave,
                    'duration': actual_duration,  # 使用实际时长
                    'hand': hand,
                    'fingering': fingering if fingering is not None else -1,
                    'is_chord': 0,
                    'chord': 0,
                    'onset_time': round(onset_time, 3),
                    'offset_time': round(offset_time, 3)
                })

                # 构建可视化字符串
                note_str = f"{key_str} {note_name} duration={actual_duration:.2f}s"
                if fingering:
                    note_str += f" fingering={fingering}"

                if element.tie and element.tie.type in ['continue', 'stop']:
                    note_str += " (finger stays)"

                if (previous_note and
                    previous_note.name == element.name and
                    element.tie and
                    element.tie.type == 'start'):
                    note_str += " (do not play)"
                else:
                    if hand == 'right':
                        right_hand_notes.append(note_str)
                    elif hand == 'left':
                        left_hand_notes.append(note_str)

                data.append({
                    'note': note_name,
                    'octave': element.octave,
                    'duration': actual_duration,
                    'hand': hand,
                    'fingering': fingering if fingering is not None else -1,
                    'is_chord': 0,
                    'chord': 0,
                    'onset_time': round(onset_time, 3),
                    'offset_time': round(offset_time, 3)
                })

                previous_note = element

            elif isinstance(element, chord.Chord):
                chord_notes = []
                do_not_play = False
                for n in element.notes:
                    # 处理指法
                    fingering = None
                    for articulation in n.articulations:
                        if isinstance(articulation, articulations.Fingering):
                            fingering = articulation.fingerNumber
                            break

                    # 记录和弦中每个音符的信息
                    note_name = process_note_name(n.pitch, sharps, flats)
                    data.append({
                        'note': note_name,
                        'octave': n.octave,
                        'duration': actual_duration,  # 使用实际时长
                        'hand': hand,
                        'fingering': fingering if fingering is not None else -1,
                        'is_chord': 1,
                        'chord': 1,
                        'onset_time': round(onset_time, 3),
                        'offset_time': round(offset_time, 3)
                    })

                    # 构建可视化字符串
                    note_str = f"{key_str} {note_name} duration={actual_duration:.2f}s"
                    if fingering:
                        note_str += f" fingering={fingering}"

                    if n.tie and n.tie.type in ['continue', 'stop']:
                        note_str += " (finger stays)"

                    if (previous_note and
                        previous_note.name == n.name and
                        n.tie and
                        n.tie.type == 'start'):
                        note_str += " (do not play)"
                        do_not_play = True

                    chord_notes.append(note_str)

                chord_str = f"{key_str} " + " ".join(chord_notes)
                if do_not_play:
                    chord_str += " (do not play)"

                if hand == 'right':
                    right_hand_notes.append(chord_str)
                elif hand == 'left':
                    left_hand_notes.append(chord_str)

                if element.notes:
                    previous_note = element.notes[-1]
                else:
                    previous_note = None

            elif isinstance(element, note.Rest):
                # 记录休止符信息
                data.append({
                    'note': 'rest',
                    'octave': -1,  # 休止符没有八度
                    'duration': actual_duration,  # 使用实际时长
                    'hand': hand,
                    'fingering': -1,  # 休止符没有指法
                    'is_chord': 0,
                    'chord': 0,
                    'onset_time': round(onset_time, 3),
                    'offset_time': round(offset_time, 3)
                })

                # 构建可视化字符串
                rest_str = f'rest {actual_duration:.2f}s'
                right_hand_notes.append(rest_str)
                left_hand_notes.append(rest_str)

        print("右手音符、和弦和休止符：")
        for item in right_hand_notes:
            print(item)

        print("\n左手音符、和弦和休止符：")
        for item in left_hand_notes:
            print(item)

        print("\n部分训练数据样例：")
        for entry in data[:10]:
            print(entry)

        # 加载 LabelEncoders
        try:
            le_pitch = load_pickle('le_pitch.pkl')
            le_duration = load_pickle('le_duration.pkl')
            le_hand = load_pickle('le_hand.pkl')
            le_fingering = load_pickle('le_fingering.pkl')
        except FileNotFoundError as e:
            print(f"Error loading LabelEncoders: {e}")
            print("请确保已运行 'dataset_prep.py' 并生成相关的 .pkl 文件。")
            exit(1)

        # 加载 Word2Vec 模型和标准化器
        try:
            word2vec_model = Word2Vec.load("word2vec_cbow.model")
            scaler = load_pickle('scaler.pkl')
        except FileNotFoundError as e:
            print(f"Error loading model or scaler: {e}")
            print("请确保已运行 'data_process.py' 并生成 word2vec_cbow.model 和 scaler.pkl 文件。")
            exit(1)

        # 预处理数据
        X_seq, y_seq = preprocess_input_data(
            data,
            le_pitch,
            le_duration,
            le_hand,
            le_fingering,
            word2vec_model,
            scaler,
            sequence_length=10,
            window=1.0
        )
        if X_seq is None:
            exit(1)  # 预处理失败

        class PredictionDataset(torch.utils.data.Dataset):
            def __init__(self, X):
                self.X = torch.tensor(X, dtype=torch.float32)
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx]

        prediction_dataset = PredictionDataset(X_seq)
        prediction_loader = torch.utils.data.DataLoader(prediction_dataset, batch_size=1, shuffle=False)

        # 加载模型
        try:
            model = BiLSTMWithAttention(input_size=X_seq.shape[2], hidden_size=256, num_layers=3, num_classes=len(le_fingering.classes_), dropout=0.5)
            model.load_state_dict(torch.load('fingering_bilstm_model.pth', map_location=torch.device('cpu')))
        except FileNotFoundError:
            print("Error: 'fingering_bilstm_model.pth' not found. 请先运行 'model_training.py'。")
            exit(1)

        model.eval()

        predicted_fingerings = []
        with torch.no_grad():
            for X_batch in prediction_loader:
                output = model(X_batch)
                predicted_fingering = torch.argmax(output, dim=1).item()
                predicted_fingerings.append(predicted_fingering)

        fingering_labels = le_fingering.inverse_transform(predicted_fingerings)

        note_count = sum(1 for element in score.flat.notes if isinstance(element, note.Note))
        chord_note_count = sum(len(element.notes) for element in score.flat.getElementsByClass(chord.Chord))
        total_notes = note_count + chord_note_count

        if len(fingering_labels) != total_notes:
            print(f"预测的指法数量 ({len(fingering_labels)}) 与音符数量 ({total_notes}) 不一致！")
        else:
            fingering_index = 0
            for part in score.parts:
                for element in part.flat.notes:
                    if isinstance(element, note.Note):
                        fingering_number = fingering_labels[fingering_index]
                        fingering_articulation = articulations.Fingering(fingering_number)
                        element.articulations.append(fingering_articulation)
                        fingering_index += 1
                    elif isinstance(element, chord.Chord):
                        for n in element.notes:
                            fingering_number = fingering_labels[fingering_index]
                            fingering_articulation = articulations.Fingering(fingering_number)
                            n.articulations.append(fingering_articulation)
                            fingering_index += 1

        modified_musicxml_path = 'modified_example.mxl'
        score.write('mxl', fp=modified_musicxml_path)
        print(f"预测完成，修改后的文件已保存为 '{modified_musicxml_path}'")

if __name__ == "__main__":
    main()
