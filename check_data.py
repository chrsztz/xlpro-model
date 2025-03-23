import numpy as np
import os

print("Checking data files...")

# Check combined files
try:
    X_train = np.load('X_train_combined.npy', mmap_mode='r')
    print(f'X_train_combined shape: {X_train.shape}')
    
    # Check if data is normalized/standardized
    mean_val = np.mean(X_train[:100])
    std_val = np.std(X_train[:100])
    print(f'  Data statistics (first 100 samples): mean={mean_val:.4f}, std={std_val:.4f}')
    if abs(mean_val) < 0.1 and abs(std_val - 1.0) < 0.5:
        print("  Data appears to be properly standardized")
    else:
        print("  Warning: Data might not be properly standardized")
        
except Exception as e:
    print(f'Error loading X_train_combined: {e}')

try:
    y_train = np.load('y_train_combined.npy', mmap_mode='r')
    print(f'y_train_combined shape: {y_train.shape}')
    
    # Count class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f'  Class distribution: {class_dist}')
    
    # Check for label imbalance
    if 10 in class_dist:  # 10 is the special "unlabeled" class
        labeled_samples = sum(counts for label, counts in class_dist.items() if label != 10)
        unlabeled_samples = class_dist.get(10, 0)
        print(f'  Labeled samples: {labeled_samples}, Unlabeled samples: {unlabeled_samples}')
        print(f'  Unlabeled percentage: {unlabeled_samples/len(y_train)*100:.2f}%')
    
except Exception as e:
    print(f'Error loading y_train_combined: {e}')

# Check sequence files
for i in range(5):  # Check first 5 sequence files
    try:
        X_train_seq = np.load(f'X_train_seq_{i}.npy', mmap_mode='r')
        y_train_seq = np.load(f'y_train_seq_{i}.npy', mmap_mode='r')
        print(f'X_train_seq_{i} shape: {X_train_seq.shape}, y_train_seq_{i} shape: {y_train_seq.shape}')
        
        # Check if data is in sequence format
        if len(X_train_seq.shape) == 3:
            print(f"  Sequence format confirmed: [samples, seq_length, features]")
            # Check a sample to see sequence relationship
            if i == 0:  # Just check the first file in detail
                sample_idx = 0
                sample = X_train_seq[sample_idx]
                print(f"  Sample sequence shape: {sample.shape}")
                # Print sequence continuity of first feature to see if it's sequential
                print(f"  First feature sequence (first 5 timesteps): {sample[:5, 0]}")
                
                # Check for sequential pattern - look for gradual changes that would indicate time series
                diffs = np.abs(np.diff(sample[:, 0]))
                avg_diff = np.mean(diffs)
                max_diff = np.max(diffs)
                print(f"  Sequential analysis - avg change between steps: {avg_diff:.4f}, max change: {max_diff:.4f}")
                
                if avg_diff < 0.5 * max_diff:
                    print("  Sequence appears to have good continuity (gradual changes between steps)")
                else:
                    print("  Warning: Sequence may not have strong continuity between timesteps")
        else:
            print(f"  WARNING: Not in expected sequence format, shape: {X_train_seq.shape}")
    except Exception as e:
        print(f'Error loading X_train_seq_{i} or y_train_seq_{i}: {e}')

# Check available memory
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"\nSystem memory: {mem.total/1e9:.1f} GB total, {mem.available/1e9:.1f} GB available")
except:
    print("\nCouldn't determine system memory")

# Recommend batch size based on available memory
print("\nRecommended settings for data_process.py:")
print("------------------------------------------")
try:
    available_gb = mem.available/1e9
    # Conservative approach - use only 70% of available memory
    usable_memory = available_gb * 0.7
    
    if usable_memory > 30:
        print(f"Your system has plenty of memory ({available_gb:.1f} GB available)")
        print("Recommended: python data_process.py --full")
    elif usable_memory > 15:
        print(f"Your system has good memory ({available_gb:.1f} GB available)")
        print("Recommended: python data_process.py --full --sample-size 500000")
    elif usable_memory > 8:
        print(f"Your system has moderate memory ({available_gb:.1f} GB available)")
        print("Recommended: python data_process.py --sample-size 200000")
    else:
        print(f"Your system has limited memory ({available_gb:.1f} GB available)")
        print("Recommended: python data_process.py --sample-size 100000 --keep-chunks")
except:
    print("Could not automatically determine recommendations.")
    print("With 36GB RAM, recommended: python data_process.py --full")