"""
Weight extraction script for TensorFlow 1.x checkpoints.

This script must be run in a TensorFlow 1.x environment (e.g., Phase 1 environment
with Python 3.8 + TensorFlow 1.15) to extract weights from old checkpoints.

The extracted weights can then be loaded into the TensorFlow 2.x model.

Usage:
    # In TF 1.x environment:
    python extract_weights_tf1.py saved/model.ckpt-110500 weights_tf1.npz

WARNING: This script uses TensorFlow 1.x API and will NOT work with TensorFlow 2.x
"""

import sys
import numpy as np

try:
    import tensorflow as tf
    # Disable TF 2.x behavior if accidentally running in TF 2.x
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
        tf = tf.compat.v1
        tf.disable_v2_behavior()
except ImportError:
    print("ERROR: TensorFlow not found!")
    print("This script requires TensorFlow 1.15")
    sys.exit(1)


def extract_weights(checkpoint_path, output_path):
    """
    Extract all variables from TensorFlow 1.x checkpoint.

    Args:
        checkpoint_path: Path to TF 1.x checkpoint (without .meta extension)
        output_path: Path to save numpy archive (.npz file)
    """
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Extracting weights from: {checkpoint_path}")
    print("-" * 70)

    # Load checkpoint
    with tf.Session() as sess:
        try:
            # Import meta graph and restore weights
            saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
            saver.restore(sess, checkpoint_path)
            print(f"✓ Checkpoint loaded successfully\n")
        except Exception as e:
            print(f"ERROR loading checkpoint: {e}")
            sys.exit(1)

        # Get all variables
        variables = tf.global_variables()
        weights = {}

        print(f"Found {len(variables)} variables:\n")

        for var in variables:
            try:
                var_value = sess.run(var)
                weights[var.name] = var_value
                print(f"  ✓ {var.name:50s} shape={var_value.shape}")
            except Exception as e:
                print(f"  ✗ {var.name:50s} ERROR: {e}")

        # Save as numpy archive
        try:
            np.savez_compressed(output_path, **weights)
            print(f"\n{'=' * 70}")
            print(f"SUCCESS! Saved {len(weights)} variables to: {output_path}")
            print(f"File size: {get_file_size_mb(output_path):.2f} MB")
            print(f"{'=' * 70}")
        except Exception as e:
            print(f"\nERROR saving weights: {e}")
            sys.exit(1)


def get_file_size_mb(filepath):
    """Get file size in MB"""
    import os
    return os.path.getsize(filepath) / (1024 * 1024)


def list_checkpoint_variables(checkpoint_path):
    """
    List all variables in a checkpoint without loading values.

    Useful for inspecting checkpoint structure.
    """
    try:
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()

        print(f"Variables in checkpoint: {checkpoint_path}\n")
        print(f"{'Variable Name':<50s} {'Shape':<20s}")
        print("-" * 70)

        for key in sorted(var_to_shape_map.keys()):
            print(f"{key:<50s} {str(var_to_shape_map[key]):<20s}")

        print(f"\nTotal: {len(var_to_shape_map)} variables")

    except Exception as e:
        print(f"ERROR reading checkpoint: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("TensorFlow 1.x Checkpoint Weight Extraction")
    print("=" * 70 + "\n")

    if len(sys.argv) == 2 and sys.argv[1] == '--list':
        print("Usage for listing variables:")
        print("  python extract_weights_tf1.py --list <checkpoint_path>")
        sys.exit(0)

    if len(sys.argv) == 3 and sys.argv[1] == '--list':
        list_checkpoint_variables(sys.argv[2])
        sys.exit(0)

    if len(sys.argv) != 3:
        print("Usage:")
        print("  python extract_weights_tf1.py <checkpoint_path> <output_path>")
        print("\nExample:")
        print("  python extract_weights_tf1.py saved/model.ckpt-110500 weights_tf1.npz")
        print("\nTo list variables without extracting:")
        print("  python extract_weights_tf1.py --list saved/model.ckpt-110500")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2]

    extract_weights(checkpoint_path, output_path)


if __name__ == '__main__':
    main()
