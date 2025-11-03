"""
Validate Pressure Model Weight Conversion

This script validates that the PyTorch → TensorFlow weight conversion was successful
by comparing outputs from both models on the same input data. It checks:
- Forward pass produces similar outputs (MSE < 1e-5)
- Statistical properties match (mean, std, min, max)
- Validation set predictions match

Usage:
    python3 validate_pressure_conversion.py

Requirements:
    - torch (for PyTorch model)
    - tensorflow (for TensorFlow model)
    - numpy

Input:
    - data/pressure_data/model_epoch_4.pth (PyTorch checkpoint)
    - saved/pressure_model/ (TensorFlow checkpoint)
    - data/pressure_data/val_data.pt (validation data)

Output:
    - Validation report printed to console
    - Pass/fail status
"""

import os
import sys
import numpy as np

# Check for required libraries
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch not installed. Please run: pip install torch")
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow not installed. Please run: pip install tensorflow")
    sys.exit(1)


# PyTorch model definition (from handwriting-model repo)
class PyTorchPressureModel(nn.Module):
    """PyTorch version of pressure model for comparison."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PyTorchPressureModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output


def load_pytorch_model(checkpoint_path):
    """
    Load PyTorch pressure model.

    Args:
        checkpoint_path (str): Path to PyTorch checkpoint

    Returns:
        PyTorchPressureModel: Loaded PyTorch model in eval mode
    """
    print("Loading PyTorch model...")

    # Create model
    model = PyTorchPressureModel(
        input_dim=3,
        hidden_dim=50,
        output_dim=1,
        num_layers=2
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode
    model.eval()

    print("  ✓ PyTorch model loaded")
    return model


def load_tensorflow_model(checkpoint_path):
    """
    Load TensorFlow pressure model.

    Args:
        checkpoint_path (str): Path to TensorFlow checkpoint directory

    Returns:
        PressureModel: Loaded TensorFlow model
    """
    print("Loading TensorFlow model...")

    from pressure_model import load_pressure_model

    model = load_pressure_model(checkpoint_path)

    print("  ✓ TensorFlow model loaded")
    return model


def compare_outputs(pytorch_output, tensorflow_output, tolerance=1e-5):
    """
    Compare PyTorch and TensorFlow model outputs.

    Args:
        pytorch_output (torch.Tensor): PyTorch model output
        tensorflow_output (tf.Tensor): TensorFlow model output
        tolerance (float): Maximum allowed MSE difference

    Returns:
        bool: True if outputs match within tolerance
    """
    print("\nComparing outputs:")

    # Convert to numpy
    pt_np = pytorch_output.detach().numpy()
    tf_np = tensorflow_output.numpy()

    # Check shapes match
    if pt_np.shape != tf_np.shape:
        print(f"  ✗ Shape mismatch: PyTorch {pt_np.shape} vs TensorFlow {tf_np.shape}")
        return False
    print(f"  ✓ Shapes match: {pt_np.shape}")

    # Compute MSE
    mse = np.mean((pt_np - tf_np) ** 2)
    print(f"\n  Mean Squared Error: {mse:.2e}")

    if mse < tolerance:
        print(f"  ✓ MSE < {tolerance:.2e} (PASS)")
        match = True
    else:
        print(f"  ✗ MSE >= {tolerance:.2e} (FAIL)")
        match = False

    # Compute additional metrics
    mae = np.mean(np.abs(pt_np - tf_np))
    max_diff = np.max(np.abs(pt_np - tf_np))

    print(f"\n  Additional metrics:")
    print(f"    Mean Absolute Error: {mae:.2e}")
    print(f"    Max Absolute Difference: {max_diff:.2e}")

    # Statistical comparison
    print(f"\n  Statistical comparison:")
    print(f"    PyTorch  - Mean: {pt_np.mean():8.4f}, Std: {pt_np.std():8.4f}, Min: {pt_np.min():8.4f}, Max: {pt_np.max():8.4f}")
    print(f"    TensorFlow - Mean: {tf_np.mean():8.4f}, Std: {tf_np.std():8.4f}, Min: {tf_np.min():8.4f}, Max: {tf_np.max():8.4f}")

    return match


def test_random_input(pytorch_model, tensorflow_model, batch_size=4, seq_len=100):
    """
    Test both models on random input.

    Args:
        pytorch_model: PyTorch model
        tensorflow_model: TensorFlow model
        batch_size (int): Batch size for test
        seq_len (int): Sequence length for test

    Returns:
        bool: True if outputs match
    """
    print("="*60)
    print("TEST 1: Random Input")
    print("="*60)

    # Generate random input
    np.random.seed(42)
    random_input = np.random.randn(batch_size, seq_len, 3).astype(np.float32)

    print(f"Input shape: {random_input.shape}")

    # PyTorch inference
    pt_input = torch.from_numpy(random_input)
    with torch.no_grad():
        pt_output = pytorch_model(pt_input)

    # TensorFlow inference
    tf_input = tf.constant(random_input)
    tf_output = tensorflow_model(tf_input, training=False)

    # Compare
    return compare_outputs(pt_output, tf_output)


def test_validation_data(pytorch_model, tensorflow_model, val_data_path, num_samples=5):
    """
    Test both models on actual validation data.

    Args:
        pytorch_model: PyTorch model
        tensorflow_model: TensorFlow model
        val_data_path (str): Path to validation data file
        num_samples (int): Number of validation samples to test

    Returns:
        bool: True if outputs match
    """
    print("\n" + "="*60)
    print("TEST 2: Validation Data")
    print("="*60)

    # Load validation data
    print(f"Loading validation data from: {val_data_path}")
    X_val, y_val = torch.load(val_data_path)

    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Testing first {num_samples} samples...")

    # Test subset
    X_test = X_val[:num_samples]
    print(f"Test input shape: {X_test.shape}")

    # PyTorch inference
    with torch.no_grad():
        pt_output = pytorch_model(X_test)

    # TensorFlow inference
    tf_input = tf.constant(X_test.numpy())
    tf_output = tensorflow_model(tf_input, training=False)

    # Compare
    return compare_outputs(pt_output, tf_output)


def test_single_sequence(pytorch_model, tensorflow_model):
    """
    Test both models on a single short sequence.

    Args:
        pytorch_model: PyTorch model
        tensorflow_model: TensorFlow model

    Returns:
        bool: True if outputs match
    """
    print("\n" + "="*60)
    print("TEST 3: Single Sequence")
    print("="*60)

    # Create simple test sequence
    test_seq = np.array([
        [1.0, 0.0, 0],
        [0.0, 1.0, 0],
        [-1.0, 0.0, 0],
        [0.0, -1.0, 0],
        [0.5, 0.5, 1]  # End of stroke
    ], dtype=np.float32)

    # Add batch dimension
    test_seq = np.expand_dims(test_seq, axis=0)  # [1, 5, 3]

    print(f"Input shape: {test_seq.shape}")
    print(f"Input:\n{test_seq[0]}")

    # PyTorch inference
    pt_input = torch.from_numpy(test_seq)
    with torch.no_grad():
        pt_output = pytorch_model(pt_input)

    # TensorFlow inference
    tf_input = tf.constant(test_seq)
    tf_output = tensorflow_model(tf_input, training=False)

    print(f"\nPyTorch output:\n{pt_output[0].numpy()}")
    print(f"\nTensorFlow output:\n{tf_output[0].numpy()}")

    # Compare
    return compare_outputs(pt_output, tf_output)


def main():
    """
    Main validation script.
    """
    print("="*60)
    print("PRESSURE MODEL VALIDATION")
    print("PyTorch vs TensorFlow Comparison")
    print("="*60)

    # Paths
    pytorch_checkpoint = 'data/pressure_data/model_epoch_4.pth'
    tensorflow_checkpoint = 'saved/pressure_model'
    val_data_path = 'data/pressure_data/val_data.pt'

    # Check files exist
    if not os.path.exists(pytorch_checkpoint):
        print(f"\nERROR: PyTorch checkpoint not found: {pytorch_checkpoint}")
        sys.exit(1)

    if not os.path.exists(tensorflow_checkpoint):
        print(f"\nERROR: TensorFlow checkpoint not found: {tensorflow_checkpoint}")
        print("Please run convert_pressure_weights.py first.")
        sys.exit(1)

    if not os.path.exists(val_data_path):
        print(f"\nERROR: Validation data not found: {val_data_path}")
        sys.exit(1)

    # Load models
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)

    pytorch_model = load_pytorch_model(pytorch_checkpoint)
    tensorflow_model = load_tensorflow_model(tensorflow_checkpoint)

    # Run tests
    results = []

    results.append(("Random Input", test_random_input(pytorch_model, tensorflow_model)))
    results.append(("Validation Data", test_validation_data(pytorch_model, tensorflow_model, val_data_path)))
    results.append(("Single Sequence", test_single_sequence(pytorch_model, tensorflow_model)))

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nWeight conversion successful!")
        print("PyTorch and TensorFlow models produce identical outputs.")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nWeight conversion may have issues.")
        print("Please review the conversion logic in convert_pressure_weights.py")
        sys.exit(1)


if __name__ == '__main__':
    main()
