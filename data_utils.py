import re
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from gensim.models import Word2Vec
import gc
from functools import lru_cache
import numba

def create_word_column_verbose(df, feature_columns):
    """
    创建 'word' 列的详细版本，会打印出每一步的信息，用于调试。
    """
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Requested feature columns: {feature_columns}")

    # Check which columns exist
    existing_cols = [col for col in feature_columns if col in df.columns]
    missing_cols = [col for col in feature_columns if col not in df.columns]

    print(f"Existing columns: {existing_cols}")
    print(f"Missing columns: {missing_cols}")

    if not existing_cols:
        print("No requested columns found in DataFrame!")
        print(f"Available columns: {df.columns.tolist()}")
        # Create placeholder
        df['word'] = 'placeholder'
        return df

    # Show sample data from each column
    for col in existing_cols:
        try:
            print(f"Column '{col}' - types: {df[col].dtype}, sample values: {df[col].head(3).tolist()}")
        except Exception as e:
            print(f"Error inspecting column '{col}': {e}")

    # Create the word column
    try:
        string_values = []
        for i, row in df.iloc[:100].iterrows():  # Process only first 100 rows for inspection
            parts = []
            for col in existing_cols:
                if pd.notna(row[col]):
                    parts.append(str(row[col]))
                else:
                    parts.append("0")
            string_values.append(' '.join(parts))

        print(f"Sample generated words (first 3): {string_values[:3]}")

        # Now process the entire DataFrame
        df['word'] = df[existing_cols].astype(str).agg(' '.join, axis=1)
        print(f"Word column created successfully. Sample: {df['word'].head(3).tolist()}")

    except Exception as e:
        print(f"Error creating word column: {e}")
        # Fallback to row-by-row processing
        print("Falling back to slower row-by-row processing...")
        string_values = []
        for i, row in df.iterrows():
            parts = []
            for col in existing_cols:
                if pd.notna(row[col]):
                    parts.append(str(row[col]))
                else:
                    parts.append("0")
            string_values.append(' '.join(parts))

        df['word'] = string_values

    return df

# Global mappings (unchanged)
ENHARMONIC_MAPPING = {
    # 双降音符
    'Cbb': 'Bb',
    'Dbb': 'C',
    'Ebb': 'D',
    'Fbb': 'Eb',
    'Gbb': 'F',
    'Abb': 'G',
    'Bbb': 'A',

    # 单降音符
    'Cb': 'B',
    'Db': 'C#',
    'Eb': 'D#',
    'Fb': 'E',
    'Gb': 'F#',
    'Ab': 'G#',
    'Bb': 'A#',

    # 纯音符
    'C': 'C',
    'D': 'D',
    'E': 'E',
    'F': 'F',
    'G': 'G',
    'A': 'A',
    'B': 'B',

    # 单升音符
    'C#': 'C#',
    'D#': 'D#',
    'E#': 'F',
    'F#': 'F#',
    'G#': 'G#',
    'A#': 'A#',
    'B#': 'C',

    # 双升音符
    'Cx': 'D',
    'Dx': 'E',
    'Ex': 'F#',
    'Fx': 'G',
    'Gx': 'A',
    'Ax': 'B',
    'Bx': 'C#',
}

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

# Pre-compute the reverse mapping
REVERSE_ENHARMONIC_MAPPING = {}
for key, value in ENHARMONIC_MAPPING.items():
    if value not in REVERSE_ENHARMONIC_MAPPING:
        REVERSE_ENHARMONIC_MAPPING[value] = []
    REVERSE_ENHARMONIC_MAPPING[value].append(key)

# Pre-compile regular expressions for better performance
PITCH_REGEX = re.compile(r'^([A-Ga-g][#b]{0,2})(\d+)$')


class CustomFingeringEncoder:
    """自定义指法编码器，处理钢琴指法的编码和解码。
       右手指法为正数，左手指法为负数；新数据集中未标注的指法（0）将被映射为10，
       并在训练时通过损失函数忽略（ignore_index=10）。
    """

    def __init__(self):
        self.mapping = {
            -5: 0, -4: 1, -3: 2, -2: 3, -1: 4,  # 左手指法：-5到-1映射到0-4
            0: 10,  # 未标注的指法（0）映射为10（ignore index）
            1: 5, 2: 6, 3: 7, 4: 8, 5: 9  # 右手指法：1到5映射到5-9
        }
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}

    def fit(self, y):
        # 验证数据中的指法是否都在映射范围内
        invalid_fingers = set(y) - set(self.mapping.keys())
        if invalid_fingers:
            raise ValueError(f"Found invalid fingerings: {invalid_fingers}")
        return self

    def transform(self, y):
        # Use numpy vectorize for better performance
        transform_func = np.vectorize(lambda x: self.mapping.get(x, -1))
        return transform_func(y)

    def inverse_transform(self, y):
        # Use numpy vectorize for better performance
        inverse_func = np.vectorize(lambda x: self.inverse_mapping.get(x, 0))
        return inverse_func(y)

    def __getstate__(self):
        return {'mapping': self.mapping, 'inverse_mapping': self.inverse_mapping}

    def __setstate__(self, state):
        self.mapping = state['mapping']
        self.inverse_mapping = state['inverse_mapping']


# Use LRU cache to avoid repeated calculations for common pitches
@lru_cache(maxsize=128)
def normalize_spelled_pitch(spelled_pitch):
    """标准化音高表示"""
    try:
        match = PITCH_REGEX.match(spelled_pitch)
        if not match:
            return "C4"  # Default value for invalid format
        pitch_name, octave = match.groups()
        normalized_pitch = ENHARMONIC_MAPPING.get(pitch_name, pitch_name)
        return f"{normalized_pitch}{octave}"
    except Exception:
        return "C4"  # Default value for any exception


# Use LRU cache to avoid repeated calculations
@lru_cache(maxsize=128)
def denormalize_spelled_pitch(normalized_pitch):
    """
    根据需要将标准化后的音符还原为原始的同音异名形式。
    """
    pitch_name = ''.join([c for c in normalized_pitch if c.isalpha() or c in ['#', 'b']])
    octave = ''.join([c for c in normalized_pitch if c.isdigit()])
    original_pitch = REVERSE_ENHARMONIC_MAPPING.get(pitch_name, [pitch_name])[0]  # 默认选第一个
    return f"{original_pitch}{octave}"


# Pre-compute note to semitone mapping
NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4, 'E#': 5,
    'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11, 'B#': 0
}


# Use LRU cache and regex for better performance
@lru_cache(maxsize=256)
def get_midi_number(spelled_pitch):
    """
    将标准化后的 spelled_pitch 转换为 MIDI 编号。
    """
    match = PITCH_REGEX.match(spelled_pitch)
    if not match:
        return 60  # Default C4

    pitch, octave = match.groups()
    octave = int(octave)
    semitone = NOTE_TO_SEMITONE.get(pitch, 0)
    midi_number = 12 * (octave + 1) + semitone
    return midi_number


# Use a set for faster lookups
BLACK_KEY_SET = {1, 3, 6, 8, 10}  # C#, D#, F#, G#, A#


def is_black_key(midi_number):
    """
    判断MIDI编号对应的音符是否为黑键。
    """
    return 1 if (midi_number % 12) in BLACK_KEY_SET else 0


# Use Numba to accelerate this computation-heavy function
@numba.jit(nopython=True)
def calculate_density_numba(onset_times, window=1.0):
    """
    Numba-accelerated version of calculate_density
    """
    n = len(onset_times)
    sorted_onsets = np.sort(onset_times)
    density = np.zeros(n, dtype=np.int32)

    for i in range(n):
        t = onset_times[i]
        # Count notes falling within the window
        count = 0
        for j in range(n):
            if sorted_onsets[j] >= t and sorted_onsets[j] <= t + window:
                count += 1
        density[i] = count

    return density


def calculate_density(df, window=1.0):
    """
    使用向量化的方法计算每个音符在给定时间窗口内的密度。
    """
    # 提取所有音符的 onset_time 并排序
    onset_times = df['onset_time'].values

    # Try to use the Numba-accelerated version when possible
    try:
        return calculate_density_numba(onset_times, window)
    except:
        # Fallback to numpy version
        sorted_onsets = np.sort(onset_times)
        lower_indices = np.searchsorted(sorted_onsets, onset_times, side='left')
        upper_indices = np.searchsorted(sorted_onsets, onset_times + window, side='right')
        return (upper_indices - lower_indices).tolist()


def calculate_speed_features(df, window=1.0):
    """计算速度相关特征"""
    df = df.copy()

    # 计算真实时值
    df['real_duration'] = df['offset_time'] - df['onset_time']

    # 计算稠密度（向量化实现）
    df['note_density'] = calculate_density(df, window)
    return df


def calculate_midi_diff(df):
    """
    计算 MIDI 差值和处理后的 MIDI 差值。
    """
    df = df.copy()

    # Use shift with efficient filling
    df['prev_midi_number'] = df['midi_number'].shift(1, fill_value=60)
    df['midi_diff'] = df['midi_number'] - df['prev_midi_number']

    # Add chord indicator if missing
    if 'chord' not in df.columns:
        df['chord'] = 0  # Default not a chord

    # Vectorize the processing of midi_diff
    def process_midi_diff(row):
        if row['chord']:
            # Based on formula (3), assuming k=0
            return 200 * 0 - row['midi_diff']
        else:
            if row['midi_diff'] < 100:
                return -row['midi_diff']
            else:
                return row['midi_diff']

    # Apply in chunks to save memory
    chunk_size = 100000
    result = []

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        processed = chunk.apply(process_midi_diff, axis=1)
        result.append(processed)

    df['midi_diff_processed'] = pd.concat(result)
    return df


def create_word_column(df, feature_columns):
    """
    创建 'word' 列，将多个特征组合成一个字符串，用于Word2Vec训练。
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Verify all columns exist, use only available columns
    available_cols = [col for col in feature_columns if col in df.columns]
    if not available_cols:
        print(f"Warning: None of the requested feature columns {feature_columns} found in DataFrame.")
        # Create a default word column with a placeholder
        df['word'] = 'placeholder'
        return df

    # Convert to string column by column and join to save memory
    string_values = []
    for i, row in df.iterrows():
        parts = []
        for col in available_cols:
            if pd.notna(row[col]):  # Handle NaN values
                parts.append(str(row[col]))
            else:
                parts.append("0")  # Use "0" as placeholder for missing values
        string_values.append(' '.join(parts))

    df['word'] = string_values
    return df


def train_word2vec(sentences, window=5, vector_size=64, min_count=5, workers=4):
    """训练Word2Vec模型，使用CBOW"""
    model = Word2Vec(sentences, window=window, vector_size=vector_size,
                     min_count=min_count, workers=workers, sg=0)  # sg=0表示CBOW
    return model


def get_fused_features(df, word2vec_model, tokenized_sentences):
    """
    获取融合特征，通过 Word2Vec 将 'word' 列转换为向量，确保最终向量维度为128。
    对于未在词汇表中的单词，使用全零向量代替。
    优化点：使用局部变量缓存词向量对象，减少函数调用开销。
    内存优化版本：按更小批次处理，使用float32减小内存占用。
    """
    wv = word2vec_model.wv
    vector_size = wv.vector_size

    # Process in very small batches to avoid memory issues
    batch_size = min(500, len(df))
    all_vectors = []

    # Force garbage collection before starting
    gc.collect()

    for i in range(0, len(tokenized_sentences), batch_size):
        # Free memory before processing
        gc.collect()

        end = min(i + batch_size, len(tokenized_sentences))
        batch_tokens = tokenized_sentences[i:end]

        # Process each sample in the batch
        batch_vectors = []
        for tokens in batch_tokens:
            # Get vectors for tokens in vocabulary
            token_vectors = []
            for token in tokens:
                if token in wv:
                    # Get vector and immediately convert to float32
                    token_vectors.append(wv[token].astype(np.float32))

            # Calculate mean vector or use zeros
            if token_vectors:
                # Use float32 to reduce memory usage
                base_vector = np.mean(token_vectors, axis=0).astype(np.float32)
            else:
                base_vector = np.zeros(vector_size, dtype=np.float32)

            # Ensure 128-dimensional output
            if base_vector.shape[0] > 128:
                result_vector = base_vector[:128]
            elif base_vector.shape[0] < 128:
                padding = np.zeros(128 - base_vector.shape[0], dtype=np.float32)
                result_vector = np.concatenate([base_vector, padding])
            else:
                result_vector = base_vector

            batch_vectors.append(result_vector)

            # Free memory for token vectors
            del token_vectors

        all_vectors.extend(batch_vectors)

        # Clean up to free memory
        del batch_vectors, batch_tokens
        gc.collect()

    # Assign vectors to DataFrame
    df['fused_feature'] = all_vectors

    # Verify dimension
    sample_dim = len(df['fused_feature'].iloc[0])
    print(f"Fused feature dimension before scaling: {sample_dim}")
    assert sample_dim == 128, f"Expected 128 features for fusion, got {sample_dim}"

    return df


def combine_features(df, feature_columns):
    """将融合特征与原始特征组合，确保总维度为136（基础特征8维 + 融合特征128维）。
    内存优化版本：按小批次处理，使用低精度数据类型。
    """
    # Process in very small batches to save memory
    batch_size = min(1000, len(df))
    all_combined = []

    # Force garbage collection before starting
    gc.collect()

    for i in range(0, len(df), batch_size):
        # Free memory before processing each batch
        gc.collect()

        batch = df.iloc[i:i + batch_size]

        # Get base features (ensure all columns exist)
        valid_columns = [col for col in feature_columns if col in batch.columns]

        # Convert to float32 to reduce memory usage
        base_features = batch[valid_columns].values.astype(np.float32)

        # If we have fewer than 8 base features, pad with zeros
        if base_features.shape[1] < 8:
            padding = np.zeros((base_features.shape[0], 8 - base_features.shape[1]), dtype=np.float32)
            base_features = np.concatenate([base_features, padding], axis=1)

        # Get fused features and convert to float32
        fused_features = np.stack(batch['fused_feature_scaled'].values).astype(np.float32)

        # Combine with lower precision
        combined = np.concatenate([base_features, fused_features], axis=1)

        # Store as list
        all_combined.extend(list(combined))

        # Clean up to free memory
        del base_features, fused_features, combined
        gc.collect()

    # Store combined features
    gc.collect()  # One more collection before assignment
    df['combined_features'] = all_combined

    # Verify dimension
    sample_dim = len(df['combined_features'].iloc[0])
    print(f"Combined feature dimension: {sample_dim}")
    assert sample_dim == 136, f"Expected 136 features, got {sample_dim}"

    return df


def process_fused_features(df, scaler_fused=None, batch_size=10000):
    """
    将融合特征分批处理，并使用增量标准化。
    如果 scaler_fused 为 None，则新建一个 StandardScaler，否则使用传入的 scaler 进行增量拟合。
    返回标准化后的融合特征列表。
    内存优化版本：使用更小的批次和float32数据类型。
    """
    # Verify the 'fused_feature' column exists
    if 'fused_feature' not in df.columns:
        raise ValueError("Column 'fused_feature' not found in DataFrame. Make sure to run get_fused_features() first.")

    # Create scaler if needed
    if scaler_fused is None:
        scaler_fused = StandardScaler()

    # Process in smaller chunks to save memory
    chunk_size = min(batch_size, len(df))
    all_scaled = []

    # Force garbage collection before starting
    gc.collect()

    for i in range(0, len(df), chunk_size):
        # Free memory before processing
        gc.collect()

        # Extract chunk of data
        chunk = df.iloc[i:i + chunk_size]

        # Convert list to array with lower precision
        fused_features_array = np.array(chunk['fused_feature'].tolist(), dtype=np.float32)

        # Fit or partial_fit the scaler
        if i == 0 and not hasattr(scaler_fused, 'mean_'):
            scaler_fused.fit(fused_features_array)
        else:
            scaler_fused.partial_fit(fused_features_array)

        # Transform and store with lower precision
        scaled_chunk = scaler_fused.transform(fused_features_array).astype(np.float32)
        all_scaled.append(scaled_chunk)

        # Free memory
        del fused_features_array, scaled_chunk
        gc.collect()

    # Combine all scaled chunks
    if len(all_scaled) > 1:
        # Combine chunks in smaller groups to avoid memory issues
        final_scaled = []
        group_size = 5  # Combine at most 5 chunks at a time

        for j in range(0, len(all_scaled), group_size):
            end_j = min(j + group_size, len(all_scaled))
            if end_j - j > 1:
                group = np.vstack(all_scaled[j:end_j])
            else:
                group = all_scaled[j]
            final_scaled.append(group)

            # Clean up to free memory
            for k in range(j, end_j):
                all_scaled[k] = None
            gc.collect()

        if len(final_scaled) > 1:
            fused_features_scaled = np.vstack(final_scaled)
        else:
            fused_features_scaled = final_scaled[0]
    else:
        fused_features_scaled = all_scaled[0]

    # Clean up
    del all_scaled
    gc.collect()

    return scaler_fused, fused_features_scaled


def save_pickle(obj, filename):
    """
    保存对象为pickle文件。
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    """
    从pickle文件加载对象。
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)