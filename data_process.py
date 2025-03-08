from data_utils import (
    get_midi_number,
    is_black_key,
    calculate_speed_features,
    calculate_midi_diff,
    create_word_column,
    train_word2vec,
    get_fused_features,
    combine_features,
    save_pickle,
    load_pickle,
    process_fused_features,
    create_word_column_verbose
)
from gensim.models import Word2Vec
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import torch
import gc
import psutil
import os
import warnings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def combine_npy_files_safely(file_list, output_file, axis=0):
    """
    安全地合并多个npy文件到一个大文件，使用内存映射方式避免OOM

    Args:
        file_list: 要合并的npy文件列表
        output_file: 输出的文件名
        axis: 合并的轴，默认为0
    """
    # First determine the shape and dtype by loading the first file
    print(f"Analyzing files for shape and dtype information...")
    sample_array = np.load(file_list[0])
    dtype = sample_array.dtype

    # Get shape of the entire array
    combined_shape = list(sample_array.shape)

    # Load all arrays to get the total size along the specified axis
    total_size = sample_array.shape[axis]
    for file in tqdm(file_list[1:]):
        # Just load the file to get its shape, without storing entire arrays
        arr = np.load(file)
        total_size += arr.shape[axis]

        # Verify compatible shapes for other dimensions
        for dim in range(len(arr.shape)):
            if dim != axis and arr.shape[dim] != combined_shape[dim]:
                raise ValueError(f"Arrays have incompatible shapes: {arr.shape} vs {combined_shape}")

    # Update the combined shape
    combined_shape[axis] = total_size

    print(f"Creating combined array with shape {combined_shape}, dtype {dtype}")

    # Create memory-mapped output file
    fp = np.lib.format.open_memmap(output_file, mode='w+', dtype=dtype, shape=tuple(combined_shape))

    # Write arrays to the memory-mapped file
    pos = 0
    for i, file in enumerate(tqdm(file_list)):
        if i % 10 == 0:
            gc.collect()  # Periodic GC

        # Load chunk
        chunk = np.load(file)

        # Write to the appropriate position
        if axis == 0:
            fp[pos:pos + chunk.shape[0]] = chunk
            pos += chunk.shape[0]
        else:
            # For other axes, need different slicing
            idx = [slice(None)] * len(combined_shape)
            idx[axis] = slice(pos, pos + chunk.shape[axis])
            fp[tuple(idx)] = chunk
            pos += chunk.shape[axis]

        # Force write to disk and free memory
        fp.flush()
        del chunk
        gc.collect()

    # Close the memmap file
    del fp
    gc.collect()

    print(f"Successfully combined arrays to {output_file}")
    return output_file

# Ignore specific warnings that might clutter output
warnings.filterwarnings('ignore', category=UserWarning, module='gensim')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*DataFrame.applymap.*')


# Helper function to report memory usage
def report_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    print(f"Memory usage: {memory_mb:.2f} MB")


def process_batch(df_batch, word2vec_model, feature_columns, batch_tokenized_sentences):
    """分批处理特征工程"""
    # Create a copy to avoid SettingWithCopyWarning
    df_batch = df_batch.copy()

    # Add music features
    df_batch = calculate_midi_diff(df_batch)
    df_batch = calculate_speed_features(df_batch)
    df_batch['black_key'] = df_batch['midi_number'].apply(is_black_key)
    if 'is_chord' not in df_batch.columns:
        df_batch['is_chord'] = 0
    df_batch['chord'] = df_batch['is_chord']

    # Create word column
    df_batch = create_word_column(df_batch, feature_columns)

    # Get tokenized sentences
    if batch_tokenized_sentences is None:
        batch_tokenized_sentences = df_batch['word'].apply(lambda x: x.split()).tolist()

    # Get features
    df_batch = get_fused_features(df_batch, word2vec_model, batch_tokenized_sentences)

    # Free memory
    gc.collect()

    return df_batch


def sequence_generator(X, y, seq_length, batch_size):
    """生成器，按需生成序列"""
    n_samples = len(X) - seq_length
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        X_batch = np.array([X[j:j + seq_length] for j in range(i, end)], dtype=np.float32)
        y_batch = np.array([y[j + seq_length] for j in range(i, end)], dtype=np.int64)
        yield X_batch, y_batch


def process_dataframe_in_batches(df, word2vec_model, feature_columns, batch_size=10000):
    """Process large dataframes in manageable batches"""
    num_batches = (len(df) + batch_size - 1) // batch_size
    processed_dfs = []

    print(f"Processing dataframe in {num_batches} batches")

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))

        # Get the batch
        df_batch = df.iloc[start_idx:end_idx].copy()

        # Process features for this batch
        df_batch = calculate_midi_diff(df_batch)
        df_batch = calculate_speed_features(df_batch)
        df_batch['black_key'] = df_batch['midi_number'].apply(is_black_key)
        if 'is_chord' not in df_batch.columns:
            df_batch['is_chord'] = 0
        df_batch['chord'] = df_batch['is_chord']

        # Create word column and tokenize
        df_batch = create_word_column(df_batch, feature_columns)
        batch_tokenized_sentences = df_batch['word'].apply(lambda x: x.split()).tolist()

        # Get fused features
        df_batch = get_fused_features(df_batch, word2vec_model, batch_tokenized_sentences)

        processed_dfs.append(df_batch)

        # Force garbage collection to free memory
        gc.collect()

    return pd.concat(processed_dfs, ignore_index=True)


def main(sample_mode=False, sample_size=50000):
    """
    Main function for data processing

    Args:
        sample_mode: If True, process only a subset of data for testing
        sample_size: Size of the sample to use when sample_mode is True
    """
    print("Starting data processing pipeline...")
    report_memory()

    # Load initial data
    try:
        print("Loading base data...")
        df = pd.read_pickle('df.pkl')
        print(f"DataFrame loaded with shape: {df.shape}")
        report_memory()
    except FileNotFoundError as e:
        print(f"Error loading DataFrame: {e}")
        return

    if 'train' not in df.columns:
        print("Error: 'train' column not found in df.")
        return

    # Use only a sample if in sample mode
    if sample_mode:
        print(f"SAMPLE MODE: Using only {sample_size} samples for testing")
        # Take a stratified sample
        n_train = int(sample_size * 0.8)
        n_val = sample_size - n_train

        train_idx = df[df['train'] == 1].sample(n_train, random_state=42).index
        val_idx = df[df['train'] == 0].sample(n_val, random_state=42).index

        # Create a new DataFrame with only the samples
        df = pd.concat([df.loc[train_idx], df.loc[val_idx]])

        # Reset the train column
        df.loc[train_idx, 'train'] = 1
        df.loc[val_idx, 'train'] = 0

        print(f"Sample created with {len(df)} rows")

    # Split into train and validation sets
    print("Splitting into train and validation sets...")
    df_train = df[df['train'] == 1].copy()
    df_val = df[df['train'] == 0].copy()
    print(f"Number of training samples: {len(df_train)}")
    print(f"Number of validation samples: {len(df_val)}")

    # Free memory
    del df
    gc.collect()
    report_memory()

    # Define feature columns
    feature_columns = ['pitch_encoded', 'duration_encoded', 'hand_encoded',
                       'midi_diff_processed', 'real_duration',
                       'note_density', 'black_key', 'chord']

    # Print all columns in the dataframe for debugging
    print(f"\nAll columns in df_train: {df_train.columns.tolist()}")

    # Define batch size based on available memory
    batch_size = min(50000, len(df_train) // 20)  # At most 5% of data per batch
    print(f"Using batch size: {batch_size}")

    # Process training data in batches
    print("Processing training data in batches...")

    # First, add all necessary columns
    train_batches = []
    num_train_batches = (len(df_train) + batch_size - 1) // batch_size

    for i in tqdm(range(num_train_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_train))

        batch = df_train.iloc[start_idx:end_idx].copy()

        # Calculate music features
        batch = calculate_midi_diff(batch)
        batch = calculate_speed_features(batch)
        batch['black_key'] = batch['midi_number'].apply(is_black_key)
        if 'is_chord' not in batch.columns:
            batch['is_chord'] = 0
        batch['chord'] = batch['is_chord']

        train_batches.append(batch)

        # Free memory
        gc.collect()

    # Concatenate processed batches
    df_train = pd.concat(train_batches, ignore_index=True)
    del train_batches
    gc.collect()
    report_memory()

    # Now optimize data types AFTER creating all columns
    float_cols = ['midi_diff_processed', 'real_duration', 'note_density']
    int_cols = ['pitch_encoded', 'duration_encoded', 'hand_encoded', 'black_key', 'chord']

    for col in float_cols:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype('float32')

    for col in int_cols:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype('int16')

    report_memory()

    # Process validation data similarly
    print("Processing validation data in batches...")
    val_batches = []
    num_val_batches = (len(df_val) + batch_size - 1) // batch_size

    for i in tqdm(range(num_val_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_val))

        batch = df_val.iloc[start_idx:end_idx].copy()

        # Calculate music features
        batch = calculate_midi_diff(batch)
        batch = calculate_speed_features(batch)
        batch['black_key'] = batch['midi_number'].apply(is_black_key)
        if 'is_chord' not in batch.columns:
            batch['is_chord'] = 0
        batch['chord'] = batch['is_chord']

        val_batches.append(batch)

        # Free memory
        gc.collect()

    # Concatenate processed batches
    df_val = pd.concat(val_batches, ignore_index=True)
    del val_batches
    gc.collect()

    # Optimize data types for validation data
    for col in float_cols:
        if col in df_val.columns:
            df_val[col] = df_val[col].astype('float32')

    for col in int_cols:
        if col in df_val.columns:
            df_val[col] = df_val[col].astype('int16')

    report_memory()

    # Train Word2Vec model
    print("Training Word2Vec model...")

    # Create word column for training - in batches to save memory
    print("Creating word column for Word2Vec training...")

    # First, ensure all required columns exist
    if not all(col in df_train.columns for col in feature_columns):
        missing_cols = [col for col in feature_columns if col not in df_train.columns]
        print(f"Warning: Missing columns for word creation: {missing_cols}")
        # Use only available columns
        available_cols = [col for col in feature_columns if col in df_train.columns]
        print(f"Using available columns: {available_cols}")
        feature_columns = available_cols

    # First check if 'word' column already exists
    if 'word' not in df_train.columns:
        # Create a new empty 'word' column
        df_train['word'] = ''

    # Check a sample first to see what's going on
    print("\nDEBUG: Inspecting a small sample with verbose output")
    sample_df = df_train.iloc[:100].copy()
    sample_df = create_word_column_verbose(sample_df, feature_columns)

    # Create word column in batches
    word_batch_size = min(50000, len(df_train) // 10)
    for i in tqdm(range(0, len(df_train), word_batch_size)):
        end_idx = min(i + word_batch_size, len(df_train))
        # Process the batch
        batch = df_train.iloc[i:end_idx].copy()
        batch = create_word_column(batch, feature_columns)
        # Assign only the 'word' column back to the original dataframe
        df_train.loc[df_train.index[i:end_idx], 'word'] = batch['word'].values
        gc.collect()

    # Verify the word column exists and has values
    if 'word' not in df_train.columns:
        raise ValueError("Failed to create 'word' column")

    # Check if word column has values
    sample_words = df_train['word'].head(5).tolist()
    print(f"Sample words: {sample_words}")

    # Collect sentences in batches to avoid memory issues
    print("Collecting tokenized sentences for Word2Vec...")
    tokenized_sentences_train = []
    tokenized_sentences_train = []

    for i in tqdm(range(0, len(df_train), batch_size)):
        end_idx = min(i + batch_size, len(df_train))
        batch_sentences = df_train.iloc[i:end_idx]['word'].apply(lambda x: x.split()).tolist()
        tokenized_sentences_train.extend(batch_sentences)

        # Free memory periodically
        if i > 0 and i % (batch_size * 5) == 0:
            gc.collect()

    # Train the model
    print(f"Training Word2Vec on {len(tokenized_sentences_train)} sentences...")
    word2vec_model = train_word2vec(
        tokenized_sentences_train, window=5, vector_size=64, min_count=5, workers=max(1, cpu_count() - 1)
    )
    word2vec_model.save("word2vec_cbow.model")
    print("Word2Vec model saved")

    # Free memory
    del tokenized_sentences_train
    gc.collect()
    report_memory()

    # Process fused features in batches
    print("Processing fused features for training data...")
    batches = []
    scaler_fused = StandardScaler()  # Initialize explicitly

    # Process smaller batches for better memory management
    fused_batch_size = min(10000, len(df_train) // 20)  # Limit batch size
    print(f"Using fused feature batch size: {fused_batch_size}")
    num_batches = (len(df_train) + fused_batch_size - 1) // fused_batch_size

    for i in tqdm(range(num_batches)):
        start_idx = i * fused_batch_size
        end_idx = min((i + 1) * fused_batch_size, len(df_train))

        # Process in even smaller chunks if memory is tight
        if i > 0 and i % 10 == 0:
            print(f"Processing batch {i}/{num_batches}...")
            report_memory()
            gc.collect()

        try:
            batch = df_train.iloc[start_idx:end_idx].copy()

            # Get tokenized sentences for this batch
            batch_sentences = batch['word'].apply(lambda x: x.split()).tolist()

            # Get fused features
            batch = get_fused_features(batch, word2vec_model, batch_sentences)

            # Process fused features
            scaler_fused, fused_features_scaled = process_fused_features(
                batch, scaler_fused=scaler_fused, batch_size=min(1000, len(batch))
            )

            # Store scaled features
            batch['fused_feature_scaled'] = list(fused_features_scaled)

            # Combine features
            batch = combine_features(batch, feature_columns)

            batches.append(batch)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Skip this batch and continue
            continue

        # Free memory
        gc.collect()

        # Store scaled features
        batch['fused_feature_scaled'] = list(fused_features_scaled)

        # Combine features
        batch = combine_features(batch, feature_columns)

        batches.append(batch)

        # Free memory
        gc.collect()

    # Concatenate processed batches
    df_train_processed = pd.concat(batches, ignore_index=True)

    # Free memory
    del df_train, batches
    gc.collect()
    report_memory()

    # Process validation data similarly
    print("Processing fused features for validation data...")
    batches = []

    num_batches = (len(df_val) + fused_batch_size - 1) // fused_batch_size
    for i in tqdm(range(num_batches)):
        start_idx = i * fused_batch_size
        end_idx = min((i + 1) * fused_batch_size, len(df_val))

        try:
            batch = df_val.iloc[start_idx:end_idx].copy()

            # Create word column if not exists
            if 'word' not in batch.columns:
                batch = create_word_column(batch, feature_columns)

            # Get tokenized sentences for this batch
            batch_sentences = batch['word'].apply(lambda x: x.split()).tolist()

            # Get fused features
            batch = get_fused_features(batch, word2vec_model, batch_sentences)

            # Process fused features (using already fit scaler)
            _, fused_features_scaled = process_fused_features(
                batch, scaler_fused=scaler_fused, batch_size=min(1000, len(batch))
            )

            # Store scaled features
            batch['fused_feature_scaled'] = list(fused_features_scaled)

            # Combine features
            batch = combine_features(batch, feature_columns)

            batches.append(batch)
        except Exception as e:
            print(f"Error processing validation batch {i}: {e}")
            # Skip this batch and continue
            continue

        # Free memory
        gc.collect()

    # Concatenate processed batches
    df_val_processed = pd.concat(batches, ignore_index=True)

    # Free memory
    del df_val, batches
    gc.collect()
    report_memory()

    # Extract features and target variables
    print("Extracting features and targets...")

    # Extract in batches to manage memory
    X_train = []
    y_train = []

    num_batches = (len(df_train_processed) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_train_processed))

        batch = df_train_processed.iloc[start_idx:end_idx]

        # Extract features and targets
        X_train.append(np.stack(batch['combined_features'].values))
        y_train.append(batch['fingering_encoded'].values)

    # Concatenate batches
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # Free memory
    del df_train_processed
    gc.collect()
    report_memory()

    # Similarly for validation data
    X_val = []
    y_val = []

    num_batches = (len(df_val_processed) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df_val_processed))

        batch = df_val_processed.iloc[start_idx:end_idx]

        # Extract features and targets
        X_val.append(np.stack(batch['combined_features'].values))
        y_val.append(batch['fingering_encoded'].values)

    # Concatenate batches
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)

    # Free memory
    del df_val_processed
    gc.collect()
    report_memory()

    # Standardize features
    print("Standardizing features...")

    try:
        print("Attempting to standardize using GPU...")
        # Try GPU acceleration
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device='mps')
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device='mps')

        # Calculate mean and std
        mean = torch.mean(X_train_tensor, dim=0)
        std = torch.std(X_train_tensor, dim=0)
        std[std == 0] = 1  # Avoid division by zero

        # Standardize
        X_train_scaled = ((X_train_tensor - mean) / std).cpu().numpy()
        X_val_scaled = ((X_val_tensor - mean) / std).cpu().numpy()

        # Free GPU memory
        del X_train_tensor, X_val_tensor
        torch.mps.empty_cache()

        print("GPU standardization successful")
    except Exception as e:
        print(f"GPU standardization failed: {e}")
        print("Using CPU instead...")

        # Use CPU standardization with batches
        scaler = StandardScaler()

        # Fit on training data in batches
        batch_size_scaler = min(100000, len(X_train) // 10)  # Adjust based on available memory
        print(f"Fitting scaler on {len(X_train)} samples in batches of {batch_size_scaler}...")

        for i in tqdm(range(0, len(X_train), batch_size_scaler)):
            end = min(i + batch_size_scaler, len(X_train))
            scaler.partial_fit(X_train[i:end])

        # Transform in batches
        print("Transforming training data...")
        X_train_scaled = np.zeros_like(X_train, dtype=np.float32)
        for i in tqdm(range(0, len(X_train), batch_size_scaler)):
            end = min(i + batch_size_scaler, len(X_train))
            X_train_scaled[i:end] = scaler.transform(X_train[i:end])

        print("Transforming validation data...")
        X_val_scaled = np.zeros_like(X_val, dtype=np.float32)
        for i in tqdm(range(0, len(X_val), batch_size_scaler)):
            end = min(i + batch_size_scaler, len(X_val))
            X_val_scaled[i:end] = scaler.transform(X_val[i:end])

        print("CPU standardization complete")

    # Update variables
    X_train = X_train_scaled
    X_val = X_val_scaled

    # Free memory
    del X_train_scaled, X_val_scaled
    gc.collect()
    report_memory()

    # Generate sequences
    print("Generating sequences...")
    sequence_length = 10
    seq_batch_size = min(10000, len(X_train) // 20)  # No more than 5% at once

    # Train sequences - first save as small chunks
    print("Saving training sequences in small chunks...")
    seq_counter = 0
    X_train_files = []
    y_train_files = []

    for X_batch, y_batch in tqdm(sequence_generator(X_train, y_train, sequence_length, seq_batch_size)):
        x_filename = f'X_train_seq_{seq_counter}.npy'
        y_filename = f'y_train_seq_{seq_counter}.npy'

        np.save(x_filename, X_batch)
        np.save(y_filename, y_batch)

        X_train_files.append(x_filename)
        y_train_files.append(y_filename)
        seq_counter += 1

    # Free memory
    del X_train, y_train
    gc.collect()
    report_memory()

    # Validation sequences
    print("Saving validation sequences in small chunks...")
    seq_counter = 0
    X_val_files = []
    y_val_files = []

    for X_batch, y_batch in tqdm(sequence_generator(X_val, y_val, sequence_length, seq_batch_size)):
        x_filename = f'X_val_seq_{seq_counter}.npy'
        y_filename = f'y_val_seq_{seq_counter}.npy'

        np.save(x_filename, X_batch)
        np.save(y_filename, y_batch)

        X_val_files.append(x_filename)
        y_val_files.append(y_filename)
        seq_counter += 1

    # Free memory
    del X_val, y_val
    gc.collect()
    report_memory()

    # Now combine all chunks into single files if requested
    if not args.keep_chunks:
        try:
            print("Combining all training sequence chunks into single files...")

            # Combine X_train files using memory-mapped approach
            print("Combining X_train chunks...")
            combine_npy_files_safely(X_train_files, 'X_train_combined.npy')

            # Combine y_train files
            print("Combining y_train chunks...")
            combine_npy_files_safely(y_train_files, 'y_train_combined.npy')

            # Combine X_val files
            print("Combining X_val chunks...")
            combine_npy_files_safely(X_val_files, 'X_val_combined.npy')

            # Combine y_val files
            print("Combining y_val chunks...")
            combine_npy_files_safely(y_val_files, 'y_val_combined.npy')

            print("Combination complete! You now have single files for training and validation.")

            # Optionally remove chunk files
            if args.remove_chunks:
                print("Removing chunk files...")
                for file in X_train_files + y_train_files + X_val_files + y_val_files:
                    try:
                        os.remove(file)
                    except Exception as e:
                        print(f"Error removing {file}: {e}")
                print("Chunk files removed.")

        except Exception as e:
            print(f"Error combining chunks: {e}")
            print("Individual chunk files are still available for training.")

    print("Data processing completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process piano fingering data for model training")
    parser.add_argument("--full", action="store_true", help="Process full dataset instead of sample")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of samples to use in sample mode")
    parser.add_argument("--keep-chunks", action="store_true", help="Keep data in small chunks without combining")
    parser.add_argument("--remove-chunks", action="store_true", help="Remove small chunk files after combining")

    args = parser.parse_args()

    # Use sample mode by default for testing
    main(sample_mode=not args.full, sample_size=args.sample_size)