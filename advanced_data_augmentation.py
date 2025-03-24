#!/usr/bin/env python
"""
Advanced Data Augmentation for Piano Fingering Prediction
- Multiple augmentation techniques to improve model generalization
- Class balancing through targeted augmentation
- Feature engineering for better representation
"""

import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import argparse
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_dir='.'):
    """Load the preprocessed data files"""
    print(f"Loading data from {data_dir}...")
    
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        
        # Load label encoders
        with open(os.path.join(data_dir, 'le_hand.pkl'), 'rb') as f:
            le_hand = pickle.load(f)
        with open(os.path.join(data_dir, 'le_fingering.pkl'), 'rb') as f:
            le_fingering = pickle.load(f)
            
        print(f"Data loaded successfully.")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        
        # Print class distribution
        class_counts = np.bincount(y_train, minlength=10)
        for i, count in enumerate(class_counts):
            percentage = (count / len(y_train)) * 100
            print(f"Class {i}: {count} samples ({percentage:.2f}%)")
            
        return X_train, y_train, X_val, y_val, le_hand, le_fingering
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def mirror_hand_augmentation(X, y, le_hand, le_fingering, hand_index=2):
    """
    Mirror augmentation: convert left hand to right hand and vice versa with
    corresponding fingering adjustments (1→5, 2→4, etc.)
    """
    print("Applying mirror hand augmentation...")
    
    # Determine hand encoding values
    left_hand_code = le_hand.transform(['left'])[0] if 'left' in le_hand.classes_ else None
    right_hand_code = le_hand.transform(['right'])[0] if 'right' in le_hand.classes_ else None
    
    if left_hand_code is None or right_hand_code is None:
        print("Warning: Could not find left/right hand encodings")
        return X, y
    
    # Define finger mapping for mirroring
    finger_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
    
    X_mirrored = []
    y_mirrored = []
    
    # Process each sample
    for i in range(len(X)):
        # Get the hand information from the last note in the sequence
        hand_code = X[i, -1, hand_index]
        
        # Create mirrored sample
        X_mirror = X[i].copy()
        
        # Change hand encoding for each note in the sequence
        for j in range(X_mirror.shape[0]):
            if X_mirror[j, hand_index] == left_hand_code:
                X_mirror[j, hand_index] = right_hand_code
            elif X_mirror[j, hand_index] == right_hand_code:
                X_mirror[j, hand_index] = left_hand_code
        
        # Mirror the fingering
        y_mirror = finger_map.get(y[i], y[i])
        
        X_mirrored.append(X_mirror)
        y_mirrored.append(y_mirror)
    
    X_mirrored = np.array(X_mirrored)
    y_mirrored = np.array(y_mirrored)
    
    print(f"Mirror augmentation created {len(X_mirrored)} new samples")
    
    # Combine original and mirrored data
    X_combined = np.concatenate([X, X_mirrored], axis=0)
    y_combined = np.concatenate([y, y_mirrored], axis=0)
    
    return X_combined, y_combined

def tempo_shift_augmentation(X, y, tempo_factors=[0.8, 0.9, 1.1, 1.2], 
                             real_duration_index=None, note_density_index=None,
                             target_classes=None, augment_percentage=0.5):
    """
    Tempo shift augmentation: modify tempo-related features by scaling factors.
    Target specific underrepresented classes for focused augmentation.
    
    Note: If real_duration_index or note_density_index are None, those features
    will not be modified.
    """
    print("Applying tempo shift augmentation...")
    
    # Check if we have the required feature indices
    if real_duration_index is None and note_density_index is None:
        print("Warning: Both real_duration_index and note_density_index are None, no tempo features to modify")
        print("Skipping tempo shift augmentation")
        return X, y
    
    # Check if indices are in range
    feature_count = X.shape[2]
    if (real_duration_index is not None and real_duration_index >= feature_count) or \
       (note_density_index is not None and note_density_index >= feature_count):
        print(f"Warning: Feature indices out of bounds. Data has {feature_count} features.")
        print(f"real_duration_index={real_duration_index}, note_density_index={note_density_index}")
        print("Skipping tempo shift augmentation")
        return X, y
    
    # If target classes specified, only augment those classes
    if target_classes is not None and len(target_classes) > 0:
        mask = np.isin(y, target_classes)
        indices = np.where(mask)[0]
        # Take only a percentage of samples to augment
        augment_count = int(len(indices) * augment_percentage)
        if augment_count == 0:
            augment_count = 1
        augment_indices = np.random.choice(indices, augment_count, replace=False)
    else:
        # If no specific classes, randomly select a percentage of all samples
        augment_count = int(len(X) * augment_percentage)
        augment_indices = np.random.choice(len(X), augment_count, replace=False)
    
    print(f"Selected {len(augment_indices)} samples for tempo augmentation")
    
    X_augmented = []
    y_augmented = []
    
    # Apply tempo shifts to selected samples
    for i in augment_indices:
        for factor in tempo_factors:
            X_tempo = X[i].copy()
            
            # Modify tempo-related features for each note in the sequence
            for j in range(X_tempo.shape[0]):
                # Scale real_duration (slower tempo = longer duration)
                if real_duration_index is not None:
                    X_tempo[j, real_duration_index] *= factor
                
                # Scale note_density (slower tempo = lower density)
                if note_density_index is not None:
                    X_tempo[j, note_density_index] /= factor
            
            X_augmented.append(X_tempo)
            y_augmented.append(y[i])
    
    if X_augmented:
        X_augmented = np.array(X_augmented)
        y_augmented = np.array(y_augmented)
        
        print(f"Tempo augmentation created {len(X_augmented)} new samples")
        
        # Combine original and augmented data
        X_combined = np.concatenate([X, X_augmented], axis=0)
        y_combined = np.concatenate([y, y_augmented], axis=0)
        
        return X_combined, y_combined
    else:
        print("No samples were augmented")
        return X, y

def pitch_shift_augmentation(X, y, shifts=[-2, -1, 1, 2], 
                             pitch_index=0, midi_diff_index=None, black_key_index=None,
                             target_classes=None, augment_percentage=0.3):
    """
    Pitch shift augmentation: transpose sequences up/down by semitones.
    Updates pitch and any available midi differences or black key flags.
    
    Note: If midi_diff_index or black_key_index are None, those features
    will not be modified.
    """
    print("Applying pitch shift augmentation...")
    
    # Check if pitch_index is in range
    feature_count = X.shape[2]
    if pitch_index >= feature_count:
        print(f"Warning: pitch_index={pitch_index} is out of bounds. Data has {feature_count} features.")
        print("Skipping pitch shift augmentation")
        return X, y
    
    # Check if other indices are in range
    if (midi_diff_index is not None and midi_diff_index >= feature_count) or \
       (black_key_index is not None and black_key_index >= feature_count):
        print(f"Warning: Some feature indices out of bounds. Data has {feature_count} features.")
        if midi_diff_index is not None and midi_diff_index >= feature_count:
            print(f"midi_diff_index={midi_diff_index} out of bounds, will not modify")
            midi_diff_index = None
        if black_key_index is not None and black_key_index >= feature_count:
            print(f"black_key_index={black_key_index} out of bounds, will not modify")
            black_key_index = None
    
    # If target classes specified, only augment those classes
    if target_classes is not None and len(target_classes) > 0:
        mask = np.isin(y, target_classes)
        indices = np.where(mask)[0]
        # Take only a percentage of samples to augment
        augment_count = int(len(indices) * augment_percentage)
        if augment_count == 0:
            augment_count = 1
        augment_indices = np.random.choice(indices, augment_count, replace=False)
    else:
        # If no specific classes, randomly select a percentage of all samples
        augment_count = int(len(X) * augment_percentage)
        augment_indices = np.random.choice(len(X), augment_count, replace=False)
    
    print(f"Selected {len(augment_indices)} samples for pitch shift augmentation")
    
    X_augmented = []
    y_augmented = []
    
    # Apply pitch shifts to selected samples
    for i in augment_indices:
        for shift in shifts:
            X_pitch = X[i].copy()
            
            # Boolean array to track which notes are black keys
            is_black_key = np.zeros(X_pitch.shape[0], dtype=bool)
            
            # First pass: extract pitch info and calculate new black key status
            for j in range(X_pitch.shape[0]):
                # Get original pitch value (MIDI number or encoded pitch)
                pitch = X_pitch[j, pitch_index]
                
                # Calculate new pitch
                new_pitch = pitch + shift
                
                # Store new pitch
                X_pitch[j, pitch_index] = new_pitch
                
                # Calculate new black key status (MIDI numbers: 1, 3, 6, 8, 10 mod 12 are black)
                if black_key_index is not None:
                    # Calculate new black key status
                    midi_num = int(new_pitch)
                    is_black = midi_num % 12 in [1, 3, 6, 8, 10]
                    X_pitch[j, black_key_index] = float(is_black)
            
            # Second pass: update midi differences
            if midi_diff_index is not None:
                for j in range(1, X_pitch.shape[0]):
                    # Calculate new midi difference
                    midi_diff = X_pitch[j, pitch_index] - X_pitch[j-1, pitch_index]
                    X_pitch[j, midi_diff_index] = midi_diff
            
            X_augmented.append(X_pitch)
            y_augmented.append(y[i])
    
    if X_augmented:
        X_augmented = np.array(X_augmented)
        y_augmented = np.array(y_augmented)
        
        print(f"Pitch shift augmentation created {len(X_augmented)} new samples")
        
        # Combine original and augmented data
        X_combined = np.concatenate([X, X_augmented], axis=0)
        y_combined = np.concatenate([y, y_augmented], axis=0)
        
        return X_combined, y_combined
    else:
        print("No samples were augmented")
        return X, y

def add_noise_augmentation(X, y, noise_level=0.02, exclude_indices=None, 
                           target_classes=None, augment_percentage=0.3):
    """
    Noise augmentation: add random noise to numerical features.
    Skip categorical features specified in exclude_indices.
    
    Note: If exclude_indices is None, a default value will be created
    to exclude the hand feature if we can identify it.
    """
    print("Applying noise augmentation...")
    
    # Default exclude indices if not provided
    feature_count = X.shape[2]
    if exclude_indices is None:
        # If we have 3 features and feature 2 is hand encoding, exclude it
        if feature_count == 3:
            exclude_indices = [2]  # Assuming index 2 is hand encoding 
        else:
            exclude_indices = []
    
    # Remove out of bounds indices
    exclude_indices = [idx for idx in exclude_indices if idx < feature_count]
    
    # If target classes specified, only augment those classes
    if target_classes is not None and len(target_classes) > 0:
        mask = np.isin(y, target_classes)
        indices = np.where(mask)[0]
        # Take only a percentage of samples to augment
        augment_count = int(len(indices) * augment_percentage)
        if augment_count == 0:
            augment_count = 1
        augment_indices = np.random.choice(indices, augment_count, replace=False)
    else:
        # If no specific classes, randomly select a percentage of all samples
        augment_count = int(len(X) * augment_percentage)
        augment_indices = np.random.choice(len(X), augment_count, replace=False)
    
    print(f"Selected {len(augment_indices)} samples for noise augmentation")
    
    X_augmented = []
    y_augmented = []
    
    # Apply noise to selected samples
    for i in augment_indices:
        # Convert to float32 to ensure we can add noise
        X_noise = X[i].copy().astype(np.float32)
        
        # Add noise to each note in the sequence
        for j in range(X_noise.shape[0]):
            # Get indices to add noise to (exclude specified indices)
            noise_indices = [idx for idx in range(X_noise.shape[1]) if idx not in exclude_indices]
            
            if noise_indices:
                # Add noise
                noise = np.random.normal(0, noise_level, len(noise_indices))
                for idx, n_idx in enumerate(noise_indices):
                    X_noise[j, n_idx] += noise[idx]
        
        X_augmented.append(X_noise)
        y_augmented.append(y[i])
    
    if X_augmented:
        X_augmented = np.array(X_augmented)
        y_augmented = np.array(y_augmented)
        
        print(f"Noise augmentation created {len(X_augmented)} new samples")
        
        # Combine original and augmented data
        X_combined = np.concatenate([X, X_augmented], axis=0)
        y_combined = np.concatenate([y, y_augmented], axis=0)
        
        return X_combined, y_combined
    else:
        print("No samples were augmented")
        return X, y

def balance_classes(X, y, target_counts=None, max_multiplier=5):
    """
    Class balancing through targeted augmentation.
    Either aims for equal class representation or target counts.
    """
    print("Balancing classes...")
    
    # Get current class counts
    class_counts = np.bincount(y)
    print("Initial class distribution:")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count}")
    
    # If no target counts provided, aim for balanced classes
    if target_counts is None:
        # Find the most common class
        max_count = np.max(class_counts)
        # Set target to be the maximum count, but limit growth for minority classes
        target_counts = np.minimum(max_count, class_counts * max_multiplier)
    
    X_balanced = []
    y_balanced = []
    
    # Process each class
    for class_idx in range(len(class_counts)):
        # Get indices for this class
        indices = np.where(y == class_idx)[0]
        
        if len(indices) == 0:
            continue
        
        # Add all original samples
        X_balanced.extend([X[i] for i in indices])
        y_balanced.extend([y[i] for i in indices])
        
        # Calculate how many more samples we need
        current_count = len(indices)
        target_count = target_counts[class_idx]
        
        if current_count >= target_count:
            continue
        
        # How many more samples to add
        add_count = target_count - current_count
        
        # Randomly sample with replacement to achieve target count
        add_indices = np.random.choice(indices, add_count, replace=True)
        
        X_balanced.extend([X[i] for i in add_indices])
        y_balanced.extend([y[i] for i in add_indices])
    
    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)
    
    # Check final distribution
    final_counts = np.bincount(y_balanced)
    print("Final class distribution:")
    for i, count in enumerate(final_counts):
        if i < len(class_counts):
            print(f"Class {i}: {count} (originally {class_counts[i]})")
    
    return X_balanced, y_balanced

def normalize_features(X_train, X_val):
    """Normalize features using StandardScaler"""
    print("Normalizing features...")
    
    # Reshape to 2D for scaling
    orig_shape = X_train.shape
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_val_2d = X_val.reshape(-1, X_val.shape[-1])
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    
    # Apply same transformation to validation data
    X_val_2d = scaler.transform(X_val_2d)
    
    # Reshape back to original shape
    X_train = X_train_2d.reshape(orig_shape)
    X_val = X_val.reshape(X_val.shape)
    
    return X_train, X_val, scaler

def plot_class_distribution(y_original, y_augmented, save_path='class_distribution.png'):
    """Plot original vs augmented class distribution"""
    print("Plotting class distribution...")
    
    plt.figure(figsize=(12, 6))
    
    # Count classes
    original_counts = np.bincount(y_original, minlength=10)
    augmented_counts = np.bincount(y_augmented, minlength=10)
    
    # Calculate percentages
    original_percent = 100 * original_counts / len(y_original)
    augmented_percent = 100 * augmented_counts / len(y_augmented)
    
    # Plot
    x = np.arange(10)
    width = 0.35
    
    plt.bar(x - width/2, original_percent, width, label='Original')
    plt.bar(x + width/2, augmented_percent, width, label='Augmented')
    
    plt.xlabel('Fingering Class')
    plt.ylabel('Percentage (%)')
    plt.title('Class Distribution: Original vs. Augmented')
    plt.xticks(x)
    plt.legend()
    
    # Add count labels
    for i, (orig, aug) in enumerate(zip(original_counts, augmented_counts)):
        plt.annotate(f'{orig}', xy=(i - width/2, original_percent[i] + 0.5), ha='center')
        plt.annotate(f'{aug}', xy=(i + width/2, augmented_percent[i] + 0.5), ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Class distribution plot saved to {save_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Advanced data augmentation for piano fingering prediction')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save augmented data')
    parser.add_argument('--mirror', action='store_true', help='Apply mirror hand augmentation')
    parser.add_argument('--tempo', action='store_true', help='Apply tempo shift augmentation')
    parser.add_argument('--pitch', action='store_true', help='Apply pitch shift augmentation')
    parser.add_argument('--noise', action='store_true', help='Apply noise augmentation')
    parser.add_argument('--balance', action='store_true', help='Balance classes after augmentation')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    X_train, y_train, X_val, y_val, le_hand, le_fingering = load_data(args.data_dir)
    
    # Store original data for comparison
    X_train_original = X_train.copy()
    y_train_original = y_train.copy()
    
    # Target underrepresented classes
    class_counts = np.bincount(y_train, minlength=10)
    underrepresented_classes = np.where(class_counts < np.mean(class_counts))[0]
    print(f"Underrepresented classes: {underrepresented_classes}")
    
    # Apply augmentations if specified
    if args.mirror:
        X_train, y_train = mirror_hand_augmentation(X_train, y_train, le_hand, le_fingering)
    
    if args.tempo:
        X_train, y_train = tempo_shift_augmentation(
            X_train, y_train,
            real_duration_index=None,  # Set to None since we don't have this feature
            note_density_index=None,   # Set to None since we don't have this feature
            target_classes=underrepresented_classes,
            augment_percentage=0.6
        )
    
    if args.pitch:
        X_train, y_train = pitch_shift_augmentation(
            X_train, y_train,
            pitch_index=0,             # Use index 0 for pitch
            midi_diff_index=None,      # Set to None since we don't have this feature
            black_key_index=None,      # Set to None since we don't have this feature
            target_classes=underrepresented_classes,
            augment_percentage=0.5
        )
    
    if args.noise:
        X_train, y_train = add_noise_augmentation(
            X_train, y_train, 
            exclude_indices=[2],       # Exclude index 2 (hand)
            target_classes=underrepresented_classes,
            augment_percentage=0.4
        )
    
    # Balance classes if specified
    if args.balance:
        X_train, y_train = balance_classes(X_train, y_train)
    
    # Normalize features
    X_train, X_val, scaler = normalize_features(X_train, X_val)
    
    # Plot class distribution
    plot_class_distribution(
        y_train_original, y_train, 
        save_path=os.path.join(args.output_dir, 'class_distribution.png')
    )
    
    # Save augmented data
    np.save(os.path.join(args.output_dir, 'X_train_aug.npy'), X_train)
    np.save(os.path.join(args.output_dir, 'y_train_aug.npy'), y_train)
    np.save(os.path.join(args.output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(args.output_dir, 'y_val.npy'), y_val)
    
    # Save scaler
    with open(os.path.join(args.output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Augmented data saved to {args.output_dir}")
    print(f"Original training samples: {len(X_train_original)}")
    print(f"Augmented training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

if __name__ == "__main__":
    main() 