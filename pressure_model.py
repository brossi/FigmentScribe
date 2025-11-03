"""
Pressure/Thickness Prediction Model for Handwriting Synthesis

This module implements a neural network that predicts pen pressure from stroke coordinates.
It adds realistic stroke width variation to generated handwriting, mimicking natural pen pressure.

Architecture:
- 2-layer LSTM (50 hidden units per layer)
- Input: Delta-encoded strokes [Δx, Δy, end_of_stroke]
- Output: Delta-encoded pressure values

Ported from handwriting-model repository (PyTorch implementation).
See: reference-repos/handwriting-model/pressure_model/
"""

import tensorflow as tf
import numpy as np
import os


class PressureModel(tf.keras.Model):
    """
    LSTM-based pressure prediction model.

    Predicts pen pressure from stroke coordinates using a 2-layer LSTM network.
    The model takes delta-encoded coordinates (Δx, Δy, eos) and outputs
    delta-encoded pressure values (Δpressure).

    Attributes:
        input_dim (int): Input dimension (default: 3 for Δx, Δy, eos)
        hidden_dim (int): LSTM hidden state dimension (default: 50)
        num_layers (int): Number of LSTM layers (default: 2)
        output_dim (int): Output dimension (default: 1 for pressure)
    """

    def __init__(self, input_dim=3, hidden_dim=50, num_layers=2, output_dim=1):
        """
        Initialize pressure prediction model.

        Args:
            input_dim (int): Input feature dimension (default: 3)
            hidden_dim (int): LSTM hidden state size (default: 50)
            num_layers (int): Number of stacked LSTM layers (default: 2)
            output_dim (int): Output dimension (default: 1)
        """
        super(PressureModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Create LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            lstm = tf.keras.layers.LSTM(
                hidden_dim,
                return_sequences=True,  # Return full sequence
                return_state=False,     # Don't need final states
                name=f'lstm_{i}'
            )
            self.lstm_layers.append(lstm)

        # Output layer (fully connected)
        self.fc = tf.keras.layers.Dense(output_dim, name='fc')

    def call(self, x, training=False):
        """
        Forward pass through the model.

        Args:
            x (tf.Tensor): Input tensor of shape [batch, timesteps, input_dim]
            training (bool): Whether in training mode (affects dropout, etc.)

        Returns:
            tf.Tensor: Predicted pressure of shape [batch, timesteps, output_dim]
        """
        # Pass through LSTM layers sequentially
        h = x
        for lstm in self.lstm_layers:
            h = lstm(h, training=training)

        # Output layer
        out = self.fc(h)

        return out

    def predict_pressure(self, strokes):
        """
        Predict absolute pressure values from delta-encoded strokes.

        This is the main inference method. It takes delta-encoded strokes,
        predicts delta-encoded pressure, and converts to absolute pressure values.

        Args:
            strokes (np.ndarray): Delta-encoded strokes of shape [n_points, 3]
                                  Columns: [Δx, Δy, end_of_stroke]

        Returns:
            np.ndarray: Absolute pressure values of shape [n_points]
                        Normalized to 0-1 range
        """
        # Convert to tensor and add batch dimension
        strokes_tensor = tf.constant(strokes, dtype=tf.float32)
        strokes_batch = tf.expand_dims(strokes_tensor, axis=0)  # [1, n_points, 3]

        # Predict delta-encoded pressure
        delta_pressure = self(strokes_batch, training=False)  # [1, n_points, 1]

        # Remove batch and output dimensions
        delta_pressure = tf.squeeze(delta_pressure).numpy()  # [n_points]

        # Convert delta-encoded to absolute pressure
        # Pressure starts at a baseline (e.g., 50) and changes with deltas
        baseline_pressure = 50.0
        absolute_pressure = np.cumsum(delta_pressure) + baseline_pressure

        # Normalize to 0-1 range for line width control
        # Clip to reasonable range to avoid extreme values
        absolute_pressure = np.clip(absolute_pressure, 10, 90)
        normalized_pressure = (absolute_pressure - 10) / 80.0

        return normalized_pressure


def load_pressure_model(checkpoint_path='saved/pressure_model'):
    """
    Load a trained pressure model from checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint directory

    Returns:
        PressureModel: Loaded model ready for inference

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    # Create model instance
    model = PressureModel(input_dim=3, hidden_dim=50, num_layers=2, output_dim=1)

    # Build model by doing a dummy forward pass
    dummy_input = tf.zeros([1, 100, 3])  # Batch of 1, 100 timesteps, 3 features
    _ = model(dummy_input)

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Pressure model checkpoint not found at {checkpoint_path}. "
            f"Please run convert_pressure_weights.py first."
        )

    # Load weights from checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

    return model


def pressure_to_line_width(pressure, base_width=2.0, scale=1.0):
    """
    Convert normalized pressure values to SVG stroke widths.

    Args:
        pressure (np.ndarray): Normalized pressure values [0, 1]
        base_width (float): Base stroke width in pixels (default: 2.0)
        scale (float): Pressure scaling factor (default: 1.0)
                       Higher values = more thickness variation

    Returns:
        np.ndarray: Stroke widths for each point

    Examples:
        >>> pressure = np.array([0.0, 0.5, 1.0])
        >>> widths = pressure_to_line_width(pressure, base_width=2.0, scale=1.0)
        >>> # Results: [1.0, 2.0, 3.0] (0.5x to 1.5x base width)
    """
    # Map pressure to width multiplier: 0.5x to 1.5x base width
    # pressure=0 → 0.5x base, pressure=0.5 → 1.0x base, pressure=1 → 1.5x base
    width_multiplier = 0.5 + pressure * scale
    widths = base_width * width_multiplier

    # Ensure minimum width (for visibility)
    widths = np.maximum(widths, 0.5)

    return widths


if __name__ == '__main__':
    """
    Test script for pressure model.

    Usage:
        python3 pressure_model.py
    """
    print("=== Pressure Model Test ===\n")

    # Create model
    print("Creating model...")
    model = PressureModel(input_dim=3, hidden_dim=50, num_layers=2, output_dim=1)

    # Test forward pass
    print("Testing forward pass...")
    dummy_strokes = tf.random.normal([2, 100, 3])  # Batch of 2, 100 timesteps
    output = model(dummy_strokes, training=False)
    print(f"Input shape: {dummy_strokes.shape}")
    print(f"Output shape: {output.shape}")

    # Test predict_pressure method
    print("\nTesting predict_pressure method...")
    test_strokes = np.random.randn(200, 3).astype(np.float32)
    pressure = model.predict_pressure(test_strokes)
    print(f"Stroke shape: {test_strokes.shape}")
    print(f"Pressure shape: {pressure.shape}")
    print(f"Pressure range: [{pressure.min():.3f}, {pressure.max():.3f}]")

    # Test pressure to line width conversion
    print("\nTesting pressure_to_line_width...")
    widths = pressure_to_line_width(pressure, base_width=2.0, scale=1.0)
    print(f"Width range: [{widths.min():.3f}, {widths.max():.3f}]")

    print("\n=== Test Complete ===")
    print(f"\nModel summary:")
    model.build(input_shape=(None, None, 3))
    model.summary()
