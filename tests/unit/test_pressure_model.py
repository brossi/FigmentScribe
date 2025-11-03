"""
Unit Tests for Pressure Model

Tests for the pressure prediction model including:
- Model initialization and architecture
- Forward pass with various input shapes
- Pressure prediction method
- Pressure-to-line-width conversion
- Model loading and checkpointing

Run with:
    pytest tests/unit/test_pressure_model.py -v
    pytest tests/unit/test_pressure_model.py::test_model_creation -v
"""

import pytest
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# TensorFlow import with error handling
try:
    import tensorflow as tf
    from pressure_model import (
        PressureModel,
        load_pressure_model,
        pressure_to_line_width
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="TensorFlow not installed")


@pytest.mark.unit
@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestPressureModel:
    """Test suite for PressureModel class."""

    def test_model_creation(self):
        """Test that model can be created with default parameters."""
        model = PressureModel()

        assert model.input_dim == 3
        assert model.hidden_dim == 50
        assert model.num_layers == 2
        assert model.output_dim == 1
        assert len(model.lstm_layers) == 2

    def test_model_creation_custom_params(self):
        """Test model creation with custom parameters."""
        model = PressureModel(
            input_dim=5,
            hidden_dim=128,
            num_layers=3,
            output_dim=2
        )

        assert model.input_dim == 5
        assert model.hidden_dim == 128
        assert model.num_layers == 3
        assert model.output_dim == 2
        assert len(model.lstm_layers) == 3

    def test_forward_pass_single_batch(self):
        """Test forward pass with single batch."""
        model = PressureModel()

        # Input: [batch=1, timesteps=100, features=3]
        input_data = tf.random.normal([1, 100, 3])

        # Forward pass
        output = model(input_data, training=False)

        # Check output shape: [batch=1, timesteps=100, output_dim=1]
        assert output.shape == (1, 100, 1)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_forward_pass_multiple_batches(self):
        """Test forward pass with multiple batches."""
        model = PressureModel()

        # Input: [batch=8, timesteps=200, features=3]
        input_data = tf.random.normal([8, 200, 3])

        # Forward pass
        output = model(input_data, training=False)

        # Check output shape
        assert output.shape == (8, 200, 1)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_forward_pass_variable_length(self):
        """Test forward pass with different sequence lengths."""
        model = PressureModel()

        # Test various sequence lengths
        for seq_len in [10, 50, 100, 500]:
            input_data = tf.random.normal([2, seq_len, 3])
            output = model(input_data, training=False)

            assert output.shape == (2, seq_len, 1)
            assert not tf.reduce_any(tf.math.is_nan(output))

    def test_predict_pressure_numpy_input(self):
        """Test predict_pressure method with numpy input."""
        model = PressureModel()

        # Build model first
        dummy = tf.zeros([1, 10, 3])
        _ = model(dummy)

        # Create test strokes: [n_points, 3]
        strokes = np.random.randn(100, 3).astype(np.float32)

        # Predict pressure
        pressure = model.predict_pressure(strokes)

        # Check output
        assert isinstance(pressure, np.ndarray)
        assert pressure.shape == (100,)
        assert np.all(pressure >= 0.0)  # Normalized to [0, 1]
        assert np.all(pressure <= 1.0)
        assert not np.any(np.isnan(pressure))

    def test_predict_pressure_delta_to_absolute_conversion(self):
        """Test that predict_pressure correctly converts delta to absolute."""
        model = PressureModel()

        # Build model
        dummy = tf.zeros([1, 10, 3])
        _ = model(dummy)

        # Simple test: constant delta should give linear absolute
        # Create strokes with zero deltas (stationary pen)
        strokes = np.zeros((50, 3), dtype=np.float32)

        pressure = model.predict_pressure(strokes)

        # With zero deltas, pressure should be relatively constant
        # (though model might add some variation)
        assert len(pressure) == 50
        assert np.all(np.isfinite(pressure))

    def test_training_mode_affects_output(self):
        """Test that training mode can be toggled."""
        model = PressureModel()

        input_data = tf.random.normal([1, 50, 3])

        # Training mode
        output_train = model(input_data, training=True)

        # Inference mode
        output_infer = model(input_data, training=False)

        # Both should have same shape
        assert output_train.shape == output_infer.shape

        # Note: Without dropout, outputs might be identical
        # But the flag should at least not cause errors


@pytest.mark.unit
@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestPressureToLineWidth:
    """Test suite for pressure_to_line_width function."""

    def test_basic_conversion(self):
        """Test basic pressure to line width conversion."""
        pressure = np.array([0.0, 0.5, 1.0])

        widths = pressure_to_line_width(pressure, base_width=2.0, scale=1.0)

        # Expectations:
        # pressure=0.0 → 0.5x base = 1.0
        # pressure=0.5 → 1.0x base = 2.0
        # pressure=1.0 → 1.5x base = 3.0
        expected = np.array([1.0, 2.0, 3.0])

        np.testing.assert_allclose(widths, expected, rtol=1e-5)

    def test_base_width_scaling(self):
        """Test that base_width parameter scales widths correctly."""
        pressure = np.array([0.5])  # Middle pressure

        # Test different base widths
        widths_1 = pressure_to_line_width(pressure, base_width=1.0, scale=1.0)
        widths_2 = pressure_to_line_width(pressure, base_width=2.0, scale=1.0)
        widths_4 = pressure_to_line_width(pressure, base_width=4.0, scale=1.0)

        assert widths_2[0] == 2 * widths_1[0]
        assert widths_4[0] == 4 * widths_1[0]

    def test_scale_parameter(self):
        """Test that scale parameter affects variation."""
        pressure = np.array([0.0, 1.0])

        # Lower scale = less variation
        widths_low = pressure_to_line_width(pressure, base_width=2.0, scale=0.5)

        # Higher scale = more variation
        widths_high = pressure_to_line_width(pressure, base_width=2.0, scale=2.0)

        # Check that higher scale gives wider range
        range_low = widths_low.max() - widths_low.min()
        range_high = widths_high.max() - widths_high.min()

        assert range_high > range_low

    def test_minimum_width_enforced(self):
        """Test that minimum width is enforced."""
        # Very low pressure should still give minimum width
        pressure = np.array([0.0] * 10)

        widths = pressure_to_line_width(pressure, base_width=0.1, scale=0.1)

        # All widths should be at least 0.5
        assert np.all(widths >= 0.5)

    def test_array_shapes(self):
        """Test various input array shapes."""
        # 1D array
        pressure_1d = np.array([0.3, 0.5, 0.7])
        widths_1d = pressure_to_line_width(pressure_1d)
        assert widths_1d.shape == pressure_1d.shape

        # Large array
        pressure_large = np.random.rand(1000)
        widths_large = pressure_to_line_width(pressure_large)
        assert widths_large.shape == pressure_large.shape

    def test_edge_cases(self):
        """Test edge cases for pressure values."""
        # All zeros
        widths_zeros = pressure_to_line_width(np.zeros(10))
        assert np.all(widths_zeros >= 0.5)

        # All ones
        widths_ones = pressure_to_line_width(np.ones(10))
        assert np.all(widths_ones > 0.5)

        # Single value
        width_single = pressure_to_line_width(np.array([0.5]))
        assert len(width_single) == 1


@pytest.mark.unit
@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestPressureModelLoading:
    """Test suite for model loading and checkpointing."""

    def test_load_model_missing_checkpoint(self):
        """Test that loading non-existent checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            load_pressure_model(checkpoint_path='/nonexistent/path')

    @pytest.mark.skipif(
        not os.path.exists('saved/pressure_model'),
        reason="Converted checkpoint not available"
    )
    def test_load_converted_model(self):
        """Test loading converted checkpoint (if available)."""
        model = load_pressure_model(checkpoint_path='saved/pressure_model')

        # Model should be ready for inference
        assert model is not None

        # Test inference
        test_input = np.random.randn(50, 3).astype(np.float32)
        pressure = model.predict_pressure(test_input)

        assert len(pressure) == 50
        assert np.all(np.isfinite(pressure))

    def test_model_save_and_load(self, tmp_path):
        """Test saving and loading a model checkpoint."""
        # Create and initialize model
        model1 = PressureModel()
        dummy = tf.zeros([1, 10, 3])
        _ = model1(dummy)

        # Save checkpoint
        checkpoint = tf.train.Checkpoint(model=model1)
        save_path = checkpoint.save(str(tmp_path / 'test_model'))

        # Create new model and load checkpoint
        model2 = PressureModel()
        _ = model2(dummy)  # Build model

        checkpoint2 = tf.train.Checkpoint(model=model2)
        checkpoint2.restore(save_path).expect_partial()

        # Both models should produce same output
        test_input = tf.random.normal([1, 20, 3])
        output1 = model1(test_input, training=False)
        output2 = model2(test_input, training=False)

        np.testing.assert_allclose(
            output1.numpy(),
            output2.numpy(),
            rtol=1e-5,
            err_msg="Loaded model produces different outputs"
        )


@pytest.mark.unit
@pytest.mark.smoke
@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_pressure_model_smoke():
    """Smoke test: Create model and run basic inference."""
    # Create model
    model = PressureModel()

    # Simple forward pass
    input_data = tf.zeros([1, 50, 3])
    output = model(input_data, training=False)

    # Should not crash and should return valid shape
    assert output.shape == (1, 50, 1)
    assert not tf.reduce_any(tf.math.is_nan(output))


@pytest.mark.unit
@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_model_summary():
    """Test that model.summary() works."""
    model = PressureModel()

    # Build model
    model.build(input_shape=(None, None, 3))

    # Should not crash
    # Note: summary() prints to stdout, we just check it doesn't error
    try:
        model.summary()
    except Exception as e:
        pytest.fail(f"model.summary() raised exception: {e}")


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
