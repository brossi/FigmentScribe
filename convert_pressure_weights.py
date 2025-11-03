"""
Convert Pressure Model Weights from PyTorch to TensorFlow

This script converts the pre-trained pressure prediction model from PyTorch format
to TensorFlow 2.15 format. It handles:
- LSTM weight transposition and reorganization
- Bias combination (PyTorch has separate biases for input and hidden)
- Checkpoint creation in TensorFlow format

Usage:
    python3 convert_pressure_weights.py

Requirements:
    - torch (for loading PyTorch checkpoint)
    - tensorflow (for creating TF checkpoint)
    - numpy

Input:
    - data/pressure_data/model_epoch_4.pth (PyTorch checkpoint)

Output:
    - saved/pressure_model/ (TensorFlow checkpoint directory)
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

from pressure_model import PressureModel


def load_pytorch_checkpoint(checkpoint_path):
    """
    Load PyTorch checkpoint and extract state dict.

    Args:
        checkpoint_path (str): Path to .pth checkpoint file

    Returns:
        dict: PyTorch state dictionary with model weights
    """
    print(f"Loading PyTorch checkpoint from: {checkpoint_path}")

    # Load checkpoint (CPU mode since we don't need GPU for conversion)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Extract state dict (checkpoint contains 'model_state_dict' key)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  Found 'model_state_dict' in checkpoint")
    else:
        state_dict = checkpoint
        print(f"  Using checkpoint directly as state dict")

    # Print loaded weights
    print(f"\nPyTorch weights loaded:")
    for key in state_dict.keys():
        shape = state_dict[key].shape
        print(f"  {key}: {tuple(shape)}")

    return state_dict


def convert_lstm_weights(pytorch_weights, layer_idx, input_dim, hidden_dim):
    """
    Convert PyTorch LSTM weights to TensorFlow format for a single layer.

    PyTorch LSTM layout:
        weight_ih_l{i}: [4*hidden, input] - Input-to-hidden weights
        weight_hh_l{i}: [4*hidden, hidden] - Hidden-to-hidden weights
        bias_ih_l{i}: [4*hidden] - Input-to-hidden bias
        bias_hh_l{i}: [4*hidden] - Hidden-to-hidden bias
        Gate order: [input, forget, cell, output]

    TensorFlow LSTM layout:
        kernel: [input, 4*hidden] - Input weights (TRANSPOSED)
        recurrent_kernel: [hidden, 4*hidden] - Recurrent weights (TRANSPOSED)
        bias: [4*hidden] - Combined bias (bias_ih + bias_hh)
        Gate order: [input, forget, cell, output] (SAME)

    Args:
        pytorch_weights (dict): PyTorch state dictionary
        layer_idx (int): Layer index (0 or 1)
        input_dim (int): Input dimension for this layer
        hidden_dim (int): Hidden dimension (50)

    Returns:
        tuple: (kernel, recurrent_kernel, bias) as numpy arrays
    """
    print(f"\nConverting LSTM layer {layer_idx}:")

    # Extract PyTorch weights
    weight_ih_key = f'lstm.weight_ih_l{layer_idx}'
    weight_hh_key = f'lstm.weight_hh_l{layer_idx}'
    bias_ih_key = f'lstm.bias_ih_l{layer_idx}'
    bias_hh_key = f'lstm.bias_hh_l{layer_idx}'

    W_ih = pytorch_weights[weight_ih_key].numpy()  # [4*hidden, input]
    W_hh = pytorch_weights[weight_hh_key].numpy()  # [4*hidden, hidden]
    b_ih = pytorch_weights[bias_ih_key].numpy()    # [4*hidden]
    b_hh = pytorch_weights[bias_hh_key].numpy()    # [4*hidden]

    print(f"  PyTorch shapes:")
    print(f"    weight_ih: {W_ih.shape}")
    print(f"    weight_hh: {W_hh.shape}")
    print(f"    bias_ih: {b_ih.shape}")
    print(f"    bias_hh: {b_hh.shape}")

    # Transpose to TensorFlow format
    # PyTorch: [4*hidden, input] → TensorFlow: [input, 4*hidden]
    kernel = W_ih.T

    # PyTorch: [4*hidden, hidden] → TensorFlow: [hidden, 4*hidden]
    recurrent_kernel = W_hh.T

    # Combine biases (PyTorch has separate biases, TensorFlow has one)
    bias = b_ih + b_hh

    print(f"  TensorFlow shapes:")
    print(f"    kernel: {kernel.shape}")
    print(f"    recurrent_kernel: {recurrent_kernel.shape}")
    print(f"    bias: {bias.shape}")

    # Verify shapes
    assert kernel.shape == (input_dim, 4 * hidden_dim), \
        f"Kernel shape mismatch: {kernel.shape} vs expected ({input_dim}, {4 * hidden_dim})"
    assert recurrent_kernel.shape == (hidden_dim, 4 * hidden_dim), \
        f"Recurrent kernel shape mismatch: {recurrent_kernel.shape} vs expected ({hidden_dim}, {4 * hidden_dim})"
    assert bias.shape == (4 * hidden_dim,), \
        f"Bias shape mismatch: {bias.shape} vs expected ({4 * hidden_dim},)"

    return kernel, recurrent_kernel, bias


def convert_fc_weights(pytorch_weights, hidden_dim, output_dim):
    """
    Convert PyTorch fully connected layer weights to TensorFlow Dense format.

    PyTorch FC layout:
        fc.weight: [output, input] - Weight matrix
        fc.bias: [output] - Bias vector

    TensorFlow Dense layout:
        kernel: [input, output] - Weight matrix (TRANSPOSED)
        bias: [output] - Bias vector (SAME)

    Args:
        pytorch_weights (dict): PyTorch state dictionary
        hidden_dim (int): Input dimension (50)
        output_dim (int): Output dimension (1)

    Returns:
        tuple: (kernel, bias) as numpy arrays
    """
    print(f"\nConverting FC layer:")

    # Extract PyTorch weights
    W_fc = pytorch_weights['fc.weight'].numpy()  # [output, input]
    b_fc = pytorch_weights['fc.bias'].numpy()    # [output]

    print(f"  PyTorch shapes:")
    print(f"    weight: {W_fc.shape}")
    print(f"    bias: {b_fc.shape}")

    # Transpose to TensorFlow format
    # PyTorch: [output, input] → TensorFlow: [input, output]
    kernel = W_fc.T
    bias = b_fc  # Bias is the same

    print(f"  TensorFlow shapes:")
    print(f"    kernel: {kernel.shape}")
    print(f"    bias: {bias.shape}")

    # Verify shapes
    assert kernel.shape == (hidden_dim, output_dim), \
        f"Kernel shape mismatch: {kernel.shape} vs expected ({hidden_dim}, {output_dim})"
    assert bias.shape == (output_dim,), \
        f"Bias shape mismatch: {bias.shape} vs expected ({output_dim},)"

    return kernel, bias


def assign_weights_to_model(model, pytorch_weights):
    """
    Assign converted PyTorch weights to TensorFlow model.

    Args:
        model (PressureModel): TensorFlow model instance
        pytorch_weights (dict): PyTorch state dictionary

    Returns:
        PressureModel: Model with assigned weights
    """
    print("\n" + "="*60)
    print("CONVERTING WEIGHTS")
    print("="*60)

    # Model configuration
    input_dim = 3
    hidden_dim = 50
    num_layers = 2
    output_dim = 1

    # Convert LSTM layers
    lstm_weights = []
    for layer_idx in range(num_layers):
        # First layer takes input_dim, subsequent layers take hidden_dim
        layer_input_dim = input_dim if layer_idx == 0 else hidden_dim

        kernel, recurrent_kernel, bias = convert_lstm_weights(
            pytorch_weights, layer_idx, layer_input_dim, hidden_dim
        )
        lstm_weights.append((kernel, recurrent_kernel, bias))

    # Convert FC layer
    fc_kernel, fc_bias = convert_fc_weights(pytorch_weights, hidden_dim, output_dim)

    # Assign weights to TensorFlow model
    print("\n" + "="*60)
    print("ASSIGNING WEIGHTS TO TENSORFLOW MODEL")
    print("="*60)

    for i, (kernel, recurrent_kernel, bias) in enumerate(lstm_weights):
        print(f"\nAssigning LSTM layer {i}:")
        lstm_layer = model.lstm_layers[i]

        # TensorFlow LSTM has 3 weights: kernel, recurrent_kernel, bias
        lstm_layer.set_weights([kernel, recurrent_kernel, bias])
        print(f"  ✓ Weights assigned")

    print(f"\nAssigning FC layer:")
    model.fc.set_weights([fc_kernel, fc_bias])
    print(f"  ✓ Weights assigned")

    print("\n" + "="*60)
    print("WEIGHT CONVERSION COMPLETE")
    print("="*60)

    return model


def save_tensorflow_checkpoint(model, output_dir):
    """
    Save TensorFlow model as checkpoint.

    Args:
        model (PressureModel): TensorFlow model with converted weights
        output_dir (str): Directory to save checkpoint

    Returns:
        str: Path to saved checkpoint
    """
    print(f"\nSaving TensorFlow checkpoint to: {output_dir}")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create checkpoint
    checkpoint = tf.train.Checkpoint(model=model)

    # Save checkpoint
    save_path = checkpoint.save(os.path.join(output_dir, 'model'))

    print(f"  ✓ Checkpoint saved: {save_path}")

    # List saved files
    print(f"\nSaved files:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        size = os.path.getsize(file_path) / 1024  # KB
        print(f"  {file} ({size:.1f} KB)")

    return save_path


def main():
    """
    Main conversion script.
    """
    print("="*60)
    print("PYTORCH → TENSORFLOW WEIGHT CONVERSION")
    print("Pressure Prediction Model")
    print("="*60)

    # Paths
    pytorch_checkpoint = 'data/pressure_data/model_epoch_4.pth'
    tensorflow_output = 'saved/pressure_model'

    # Check if PyTorch checkpoint exists
    if not os.path.exists(pytorch_checkpoint):
        print(f"\nERROR: PyTorch checkpoint not found at {pytorch_checkpoint}")
        print("Please ensure you've copied the model file:")
        print("  cp reference-repos/handwriting-model/pressure_model/saved_models/model_epoch_4.pth data/pressure_data/")
        sys.exit(1)

    # Load PyTorch weights
    pytorch_weights = load_pytorch_checkpoint(pytorch_checkpoint)

    # Create TensorFlow model
    print("\n" + "="*60)
    print("CREATING TENSORFLOW MODEL")
    print("="*60)
    model = PressureModel(input_dim=3, hidden_dim=50, num_layers=2, output_dim=1)

    # Build model (initialize weights)
    print("Building model...")
    dummy_input = tf.zeros([1, 100, 3])
    _ = model(dummy_input)
    print("  ✓ Model built")

    # Convert and assign weights
    model = assign_weights_to_model(model, pytorch_weights)

    # Save TensorFlow checkpoint
    save_path = save_tensorflow_checkpoint(model, tensorflow_output)

    print("\n" + "="*60)
    print("CONVERSION SUCCESSFUL!")
    print("="*60)
    print(f"\nTensorFlow model saved to: {tensorflow_output}")
    print("\nNext steps:")
    print("  1. Validate conversion: python3 validate_pressure_conversion.py")
    print("  2. Test inference: python3 -c 'from pressure_model import load_pressure_model; model = load_pressure_model()'")
    print("  3. Integrate with sample.py")


if __name__ == '__main__':
    main()
