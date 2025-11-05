"""
Shared pytest fixtures for Scribe handwriting synthesis tests.

This module contains fixtures that are automatically available to all test modules.
Fixtures provide reusable test data and setup/teardown functionality.

Usage:
    def test_example(mock_args, tiny_model):
        # Fixtures are automatically injected as function parameters
        assert mock_args.rnn_size == 10
"""

# CRITICAL: Configure matplotlib backend BEFORE any imports
# This must be done before matplotlib is imported anywhere (including sample.py)
# to prevent GUI windows from popping up during tests
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for headless test execution

import pytest
import tensorflow as tf
import numpy as np
import tempfile
import os
from pathlib import Path


# ============================================================================
# Configuration Fixtures
# ============================================================================

class MockArgs:
    """
    Mock arguments object for testing.

    Mimics the argparse.Namespace object used throughout the codebase.
    Use tiny values for fast tests.
    """
    def __init__(self, **kwargs):
        # Model parameters - tiny for fast tests
        self.rnn_size = kwargs.get('rnn_size', 10)
        self.nmixtures = kwargs.get('nmixtures', 2)
        self.kmixtures = kwargs.get('kmixtures', 1)
        self.alphabet = kwargs.get('alphabet', ' abcdefghijklmnopqrstuvwxyz')
        # CRITICAL: Must be ≤ tsteps to ensure ascii_steps = tsteps // tsteps_per_ascii ≥ 1
        # Otherwise DataLoader creates empty char_seq tensors
        self.tsteps_per_ascii = kwargs.get('tsteps_per_ascii', 5)

        # Training parameters
        self.batch_size = kwargs.get('batch_size', 4)
        self.tsteps = kwargs.get('tsteps', 10)
        self.nepochs = kwargs.get('nepochs', 1)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.dropout = kwargs.get('dropout', 1.0)  # No dropout for deterministic tests
        self.grad_clip = kwargs.get('grad_clip', 10.0)

        # Sampling parameters
        self.bias = kwargs.get('bias', 1.0)
        self.eos_threshold = kwargs.get('eos_threshold', 0.35)

        # Data parameters
        self.data_scale = kwargs.get('data_scale', 50)
        self.limit = kwargs.get('limit', 500)

        # I/O parameters
        self.data_dir = kwargs.get('data_dir', './data')
        self.save_path = kwargs.get('save_path', './saved/model')
        self.log_dir = kwargs.get('log_dir', './logs')

        # Training mode flag
        self.train = kwargs.get('train', False)


@pytest.fixture
def mock_args():
    """Provide mock arguments with tiny model size for fast tests."""
    return MockArgs()


@pytest.fixture
def mock_args_style_priming():
    """Provide mock arguments with rnn_size=400 for style priming tests."""
    return MockArgs(rnn_size=400, nmixtures=20)


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def tiny_model(mock_args):
    """
    Create a tiny HandwritingModel for fast testing.

    Uses rnn_size=10, nmixtures=2 for minimal computation.
    """
    from model import HandwritingModel
    return HandwritingModel(mock_args)


@pytest.fixture
def style_model(mock_args_style_priming):
    """
    Create a model with rnn_size=400 for style priming tests.

    Required for testing style priming functionality.
    """
    from model import HandwritingModel
    return HandwritingModel(mock_args_style_priming)


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_stroke_data():
    """
    Generate simple synthetic stroke data for testing.

    Returns:
        numpy array of shape (n_points, 3) with [Δx, Δy, end_of_stroke]
    """
    # Simple square stroke: right, down, left, up
    return np.array([
        [10.0, 0.0, 0.0],   # Right
        [0.0, 10.0, 0.0],   # Down
        [-10.0, 0.0, 0.0],  # Left
        [0.0, -10.0, 1.0],  # Up and end
    ], dtype=np.float32)


@pytest.fixture
def sample_text_simple():
    """Simple test text with only lowercase letters."""
    return "abc"


@pytest.fixture
def sample_text_full():
    """Test text with full character set (letters, numbers, punctuation)."""
    return "Hello World 123!"


@pytest.fixture
def sample_batch_inputs(mock_args, reset_random_seeds):
    """
    Generate properly formatted batch inputs for model testing.

    Returns:
        Dictionary with 'stroke_data' and 'char_seq' tensors

    Note: Uses reset_random_seeds for deterministic behavior.
    char_seq is properly one-hot encoded (not random normal distribution).
    """
    batch_size = mock_args.batch_size
    tsteps = mock_args.tsteps
    text_len = 5
    alphabet_size = len(mock_args.alphabet) + 1

    # Generate random character indices
    char_indices = tf.random.uniform(
        [batch_size, text_len],
        minval=0,
        maxval=alphabet_size,
        dtype=tf.int32
    )

    # Convert to proper one-hot encoding
    char_seq = tf.one_hot(char_indices, depth=alphabet_size)

    return {
        'stroke_data': tf.random.normal([batch_size, tsteps, 3]),
        'char_seq': char_seq,  # Properly one-hot encoded
    }


# ============================================================================
# TensorFlow Configuration Fixtures
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_tensorflow():
    """
    Configure TensorFlow for testing (runs once per test session).

    - Sets deterministic operations for reproducibility
    - Disables GPU for consistent test environment
    - Suppresses verbose logging
    """
    # Set deterministic behavior
    tf.config.experimental.enable_op_determinism()

    # Disable GPUs for testing (CPU-only for consistency)
    tf.config.set_visible_devices([], 'GPU')

    # Reduce TensorFlow logging verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)


@pytest.fixture
def reset_random_seeds():
    """
    Reset random seeds before each test for reproducibility.

    Use this fixture for tests that need deterministic behavior.
    """
    tf.random.set_seed(42)
    np.random.seed(42)


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_checkpoint_dir():
    """
    Create a temporary directory for checkpoint testing.

    Automatically cleaned up after test completes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_output_dir():
    """
    Create a temporary directory for output file testing.

    Automatically cleaned up after test completes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Data Loader Fixtures
# ============================================================================

@pytest.fixture
def mini_dataset_path(tmp_path):
    """
    Create a minimal dataset for testing DataLoader.

    Generates a small valid .cpkl file with 20 samples.
    """
    import pickle

    # Generate 20 simple samples (need 20+ for validation split)
    # DataLoader puts every 20th sample in validation set
    strokes = []
    asciis = []

    for i in range(20):
        # Generate stroke data with 50 points to handle various tsteps values
        # Default tests use tsteps=10 (needs >12 points)
        # Multiline tests use tsteps=30 (needs >32 points)
        # Using 50 points ensures compatibility with all tests
        stroke_points = []
        for j in range(49):
            # Create a simple repeating pattern
            if j % 4 == 0:
                stroke_points.append([10.0, 0.0, 0.0])
            elif j % 4 == 1:
                stroke_points.append([0.0, 10.0, 0.0])
            elif j % 4 == 2:
                stroke_points.append([-10.0, 0.0, 0.0])
            else:
                stroke_points.append([0.0, -10.0, 0.0])
        # Final point marks end of stroke
        stroke_points.append([0.0, 0.0, 1.0])

        stroke = np.array(stroke_points, dtype=np.float32)

        strokes.append(stroke)
        asciis.append(f"sample {i}")

    # Save to temporary file with the expected filename
    # DataLoader looks for "strokes_training_data.cpkl" specifically
    dataset_path = tmp_path / "strokes_training_data.cpkl"
    with open(dataset_path, 'wb') as f:
        pickle.dump([strokes, asciis], f, protocol=2)

    return str(dataset_path)


# ============================================================================
# Assertion Helpers
# ============================================================================

@pytest.fixture
def assert_tensor_properties():
    """
    Provide helper functions for asserting tensor properties.

    Returns:
        Dictionary of assertion helper functions
    """
    def assert_shape(tensor, expected_shape):
        """Assert tensor has expected shape."""
        assert tensor.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {tensor.shape}"

    def assert_no_nan(tensor):
        """Assert tensor contains no NaN values."""
        assert not tf.reduce_any(tf.math.is_nan(tensor)), \
            "Tensor contains NaN values"

    def assert_no_inf(tensor):
        """Assert tensor contains no Inf values."""
        assert not tf.reduce_any(tf.math.is_inf(tensor)), \
            "Tensor contains Inf values"

    def assert_in_range(tensor, min_val, max_val):
        """Assert all tensor values are in [min_val, max_val]."""
        assert tf.reduce_all(tensor >= min_val), \
            f"Tensor has values below {min_val}"
        assert tf.reduce_all(tensor <= max_val), \
            f"Tensor has values above {max_val}"

    def assert_sums_to_one(tensor, axis=-1, atol=1e-5):
        """Assert tensor sums to 1 along specified axis."""
        sums = tf.reduce_sum(tensor, axis=axis)
        tf.debugging.assert_near(sums, tf.ones_like(sums), atol=atol)

    return {
        'assert_shape': assert_shape,
        'assert_no_nan': assert_no_nan,
        'assert_no_inf': assert_no_inf,
        'assert_in_range': assert_in_range,
        'assert_sums_to_one': assert_sums_to_one,
    }


# ============================================================================
# Performance Monitoring Fixtures
# ============================================================================

@pytest.fixture
def performance_monitor():
    """
    Monitor test execution time and memory usage.

    Usage:
        def test_something(performance_monitor):
            with performance_monitor:
                # test code
            assert performance_monitor.elapsed < 1.0  # Max 1 second
    """
    import time

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            self.end_time = time.time()
            self.elapsed = self.end_time - self.start_time

    return PerformanceMonitor()
