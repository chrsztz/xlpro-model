import os
import argparse

def main():
    """
    This script runs the data processing pipeline in a memory-efficient way.
    It manages the process to prevent memory issues with large datasets.
    """
    parser = argparse.ArgumentParser(description="Run the piano fingering data processing pipeline")
    parser.add_argument("--sample-size", type=int, default=200000, 
                       help="Number of samples to process (default: 200000)")
    parser.add_argument("--keep-chunks", action="store_true", 
                       help="Keep data in small chunks without combining")
    parser.add_argument("--include-ergonomic", action="store_true", 
                       help="Include ergonomic features in processing (default: True)")
    parser.add_argument("--memory-limit", type=int, default=20,
                       help="Memory limit in GB (default: 20GB)")
    args = parser.parse_args()
    
    # Calculate batch sizes based on memory limits
    memory_gb = args.memory_limit
    # Each sample is roughly 136 features x 4 bytes (float32) x 10 sequence steps = ~5.5KB
    # Add 50% overhead for processing
    sample_size_kb = 5.5 * 1.5  # KB per sample with overhead
    max_samples = int((memory_gb * 1024 * 1024) / sample_size_kb)
    
    # Limit to specified sample size or calculated max
    batch_size = min(args.sample_size, max_samples)
    
    print(f"Starting data processing with sample size: {args.sample_size}, max batch size: {batch_size}")
    
    # Ensure ergonomic features are included
    os.environ["INCLUDE_ERGONOMIC"] = "1" if args.include_ergonomic else "0"
    
    # Build command with appropriate arguments
    cmd = f"python data_process.py --sample-size {args.sample_size}"
    if args.keep_chunks:
        cmd += " --keep-chunks"
    
    # Set environment variables for fine-tuned memory management
    os.environ["BATCH_SIZE"] = str(min(50000, batch_size // 10))  # Processing batch size
    os.environ["FUSION_BATCH_SIZE"] = str(min(10000, batch_size // 50))  # Smaller batches for feature fusion
    
    # Run the data processing script
    print(f"Executing: {cmd}")
    os.system(cmd)
    
    # Verify outputs
    expected_files = [
        "X_train_combined.npy", "y_train_combined.npy",
        "X_val_combined.npy", "y_val_combined.npy"
    ]
    
    if args.keep_chunks:
        # Look for sequence chunks instead
        chunk_files = [f for f in os.listdir('.') if f.startswith('X_train_seq_') or f.startswith('y_train_seq_')]
        if chunk_files:
            print(f"Successfully created {len(chunk_files)} chunk files")
        else:
            print("Warning: No chunk files were created")
    else:
        # Check for combined files
        missing = [f for f in expected_files if not os.path.exists(f)]
        if missing:
            print(f"Warning: The following expected files are missing: {missing}")
        else:
            print("All expected output files were created successfully")

if __name__ == "__main__":
    main()