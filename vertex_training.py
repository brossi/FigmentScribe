#!/usr/bin/env python3
"""
Vertex AI training wrapper for Scribe handwriting synthesis.

This script runs inside the Vertex AI training container and:
1. Sets up Google Cloud Storage integration
2. Downloads data from GCS (if needed)
3. Runs training with specified parameters
4. Uploads results to GCS
5. Handles errors and cleanup

Environment:
    - Runs in Docker container on Vertex AI
    - Has access to GPU
    - Can read/write to mounted GCS buckets
"""

import argparse
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Import training function
from train import train


def setup_gcs_paths(args):
    """
    Handle GCS path mounting in Vertex AI.

    Vertex AI can mount GCS buckets directly to /gcs/ paths.
    This function ensures paths are accessible.
    """
    print("\n" + "="*70)
    print("Setting up Google Cloud Storage paths")
    print("="*70)

    # Check if data directory exists
    data_dir = Path(args.data_dir)
    if data_dir.exists():
        print(f"✓ Data directory: {data_dir}")

        # Verify training data file
        data_file = data_dir / "strokes_training_data.cpkl"
        if data_file.exists():
            size_mb = data_file.stat().st_size / (1024 * 1024)
            print(f"✓ Training data found: {data_file} ({size_mb:.1f} MB)")
        else:
            print(f"⚠ WARNING: Training data not found at {data_file}")
            print("  Using bundled data from container")
            args.data_dir = "./data"
    else:
        print(f"⚠ WARNING: GCS data directory not accessible: {data_dir}")
        print("  Using bundled data from container")
        args.data_dir = "./data"

    # Create local output directory (will be synced to GCS)
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Save directory: {save_dir}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Log directory: {log_dir}")

    return args


def log_system_info():
    """Log system information for debugging."""
    import tensorflow as tf

    print("\n" + "="*70)
    print("System Information")
    print("="*70)

    print(f"Python version: {sys.version.split()[0]}")
    print(f"TensorFlow version: {tf.__version__}")

    # GPU detection
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            # Try to get GPU memory info
            try:
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                print(f"    Memory: {memory_info}")
            except:
                pass
    else:
        print("⚠ WARNING: No GPUs detected - training will be SLOW")

    # Check CPU info
    import psutil
    print(f"CPU count: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    print("="*70 + "\n")


def main():
    """Main entry point for Vertex AI training."""

    parser = argparse.ArgumentParser(
        description='Vertex AI training wrapper for Scribe'
    )

    # Model params
    parser.add_argument('--rnn_size', type=int, default=400,
                       help='Size of LSTM hidden state')
    parser.add_argument('--tsteps', type=int, default=150,
                       help='Sequence length for training')
    parser.add_argument('--nmixtures', type=int, default=20,
                       help='Number of Gaussian mixtures in MDN')
    parser.add_argument('--kmixtures', type=int, default=1,
                       help='Number of Gaussian mixtures in attention')
    parser.add_argument('--alphabet', type=str,
                       default=' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#%&\'()*+,-./:;?[]',
                       help='Alphabet for character encoding')
    parser.add_argument('--tsteps_per_ascii', type=int, default=25,
                       help='Approximate pen points per character')

    # Training params
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Minibatch size')
    parser.add_argument('--nbatches', type=int, default=500,
                       help='Number of batches per epoch')
    parser.add_argument('--nepochs', type=int, default=250,
                       help='Number of epochs')
    parser.add_argument('--dropout', type=float, default=0.85,
                       help='Dropout keep probability')
    parser.add_argument('--grad_clip', type=float, default=10.0,
                       help='Gradient clipping threshold')
    parser.add_argument('--optimizer', type=str, default='rmsprop',
                       choices=['adam', 'rmsprop'],
                       help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=1.0,
                       help='Learning rate decay per epoch')
    parser.add_argument('--decay', type=float, default=0.95,
                       help='RMSprop decay parameter')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='RMSprop momentum parameter')

    # I/O params (GCS paths)
    parser.add_argument('--data_scale', type=int, default=50,
                       help='Factor to scale data by')
    parser.add_argument('--log_dir', type=str, default='./logs/',
                       help='Directory for logs')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory (can be GCS path)')
    parser.add_argument('--save_path', type=str, default='./saved/model',
                       help='Path to save checkpoints (can be GCS path)')
    parser.add_argument('--save_every', type=int, default=500,
                       help='Save checkpoint every N steps')

    args = parser.parse_args()
    args.train = True  # Training mode flag

    # Print header
    print("\n" + "="*70)
    print("Scribe Handwriting Synthesis - Vertex AI Training")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Log system info
    log_system_info()

    # Setup GCS paths
    args = setup_gcs_paths(args)

    # Print configuration
    print("\n" + "="*70)
    print("Training Configuration")
    print("="*70)
    print(f"Model: rnn_size={args.rnn_size}, nmixtures={args.nmixtures}")
    print(f"Training: epochs={args.nepochs}, batch_size={args.batch_size}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.save_path}")
    print(f"Logs: {args.log_dir}")
    print("="*70 + "\n")

    # Start training
    start_time = time.time()

    try:
        train(args)

        elapsed_time = time.time() - start_time
        print("\n" + "="*70)
        print("Training Complete! ✓")
        print("="*70)
        print(f"Total time: {elapsed_time/3600:.2f} hours")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")

        sys.exit(0)

    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        sys.exit(130)

    except Exception as e:
        elapsed_time = time.time() - start_time
        print("\n" + "="*70)
        print("Training Failed! ✗")
        print("="*70)
        print(f"Error: {e}")
        print(f"Time before failure: {elapsed_time/60:.1f} minutes")
        print("="*70 + "\n")

        import traceback
        traceback.print_exc()

        sys.exit(1)


if __name__ == '__main__':
    main()
