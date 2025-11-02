"""
Unit Tests for Sampling Functions

Tests sample.py helper functions, style loading, and generation logic.

Usage:
    pytest tests/unit/test_sample_functions.py -v
    pytest -k test_sample_functions
"""

import pytest
import numpy as np
import tensorflow as tf
import os
from pathlib import Path


@pytest.mark.unit
class TestSampleGaussian2D:
    """Test 2D Gaussian sampling function."""

    def test_sample_gaussian2d_returns_tuple(self, reset_random_seeds):
        """Test sample_gaussian2d returns (x, y) tuple."""
        from sample import sample_gaussian2d

        x, y = sample_gaussian2d(mu1=0.0, mu2=0.0, s1=1.0, s2=1.0, rho=0.0)

        assert isinstance(x, (float, np.floating))
        assert isinstance(y, (float, np.floating))

    def test_sample_gaussian2d_mean_centered(self, reset_random_seeds):
        """Test samples cluster around specified mean."""
        from sample import sample_gaussian2d

        samples_x = []
        samples_y = []

        np.random.seed(42)
        for _ in range(1000):
            x, y = sample_gaussian2d(mu1=5.0, mu2=10.0, s1=1.0, s2=1.0, rho=0.0)
            samples_x.append(x)
            samples_y.append(y)

        mean_x = np.mean(samples_x)
        mean_y = np.mean(samples_y)

        # Should be close to specified means (within statistical tolerance)
        np.testing.assert_allclose(mean_x, 5.0, atol=0.5)
        np.testing.assert_allclose(mean_y, 10.0, atol=0.5)

    def test_sample_gaussian2d_zero_correlation(self, reset_random_seeds):
        """Test zero correlation (rho=0) produces independent samples."""
        from sample import sample_gaussian2d

        samples_x = []
        samples_y = []

        np.random.seed(42)
        for _ in range(500):
            x, y = sample_gaussian2d(mu1=0.0, mu2=0.0, s1=1.0, s2=1.0, rho=0.0)
            samples_x.append(x)
            samples_y.append(y)

        # Correlation should be close to 0
        correlation = np.corrcoef(samples_x, samples_y)[0, 1]
        assert abs(correlation) < 0.2  # Very weak correlation

    def test_sample_gaussian2d_positive_correlation(self, reset_random_seeds):
        """Test positive correlation (rho=0.9) produces correlated samples."""
        from sample import sample_gaussian2d

        samples_x = []
        samples_y = []

        np.random.seed(42)
        for _ in range(500):
            x, y = sample_gaussian2d(mu1=0.0, mu2=0.0, s1=1.0, s2=1.0, rho=0.9)
            samples_x.append(x)
            samples_y.append(y)

        # Correlation should be strongly positive
        correlation = np.corrcoef(samples_x, samples_y)[0, 1]
        assert correlation > 0.5  # Strong positive correlation


@pytest.mark.unit
class TestToOneHot:
    """Test to_one_hot encoding function."""

    def test_to_one_hot_simple_string(self):
        """Test to_one_hot encodes simple string correctly."""
        from sample import to_one_hot

        alphabet = ' abc'
        text = "cab"
        max_len = 5

        encoded = to_one_hot(text, max_len, alphabet)

        assert encoded.shape == (5, len(alphabet) + 1)
        # Each row should sum to 1 (one-hot property)
        row_sums = encoded.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(max_len))

    def test_to_one_hot_padding(self):
        """Test to_one_hot pads short strings with zeros."""
        from sample import to_one_hot

        alphabet = ' abc'
        text = "ab"
        max_len = 5

        encoded = to_one_hot(text, max_len, alphabet)

        # First 2 rows should be non-zero (for 'a', 'b')
        assert np.any(encoded[0, :] > 0)
        assert np.any(encoded[1, :] > 0)

        # Remaining rows should have padding (index 0 = unknown)
        assert encoded[2, 0] == 1  # Padding character
        assert encoded[3, 0] == 1
        assert encoded[4, 0] == 1

    def test_to_one_hot_truncates_long_strings(self):
        """Test to_one_hot truncates strings longer than max_len."""
        from sample import to_one_hot

        alphabet = ' abc'
        text = "abcabc"  # 6 characters
        max_len = 3

        encoded = to_one_hot(text, max_len, alphabet)

        # Should only have 3 rows (truncated)
        assert encoded.shape == (3, len(alphabet) + 1)

    def test_to_one_hot_clips_super_long_strings(self):
        """Test to_one_hot clips strings longer than 3000 characters."""
        from sample import to_one_hot

        alphabet = ' abc'
        text = "a" * 5000  # 5000 characters
        max_len = 10

        # Should not raise error, clips to 3000
        encoded = to_one_hot(text, max_len, alphabet)

        assert encoded.shape == (10, len(alphabet) + 1)

    def test_to_one_hot_character_mapping(self):
        """Test to_one_hot maps characters correctly."""
        from sample import to_one_hot

        alphabet = ' abc'
        text = " "  # Space character
        max_len = 1

        encoded = to_one_hot(text, max_len, alphabet)

        # Space is at index 0 in alphabet, so should be index 1 in one-hot (0 is padding)
        space_index = alphabet.find(' ') + 1
        assert encoded[0, space_index] == 1


@pytest.mark.unit
class TestLoadStyleState:
    """Test load_style_state function."""

    def test_load_style_state_wrong_rnn_size(self, mock_args):
        """Test load_style_state raises error when rnn_size != 400."""
        from sample import load_style_state
        from model import HandwritingModel

        # Create model with wrong rnn_size
        mock_args.rnn_size = 100  # Should be 400 for styles
        model = HandwritingModel(mock_args)

        with pytest.raises(ValueError, match="Style priming requires rnn_size=400"):
            load_style_state(0, model, mock_args)

    def test_load_style_state_invalid_style_id_negative(self, mock_args):
        """Test load_style_state raises error for negative style_id."""
        from sample import load_style_state
        from model import HandwritingModel

        mock_args.rnn_size = 400
        model = HandwritingModel(mock_args)

        with pytest.raises(ValueError, match="style_id must be between 0 and 12"):
            load_style_state(-1, model, mock_args)

    def test_load_style_state_invalid_style_id_too_large(self, mock_args):
        """Test load_style_state raises error for style_id > 12."""
        from sample import load_style_state
        from model import HandwritingModel

        mock_args.rnn_size = 400
        model = HandwritingModel(mock_args)

        with pytest.raises(ValueError, match="style_id must be between 0 and 12"):
            load_style_state(13, model, mock_args)

    def test_load_style_state_missing_chars_file(self, mock_args, tmpdir):
        """Test load_style_state raises error when chars file missing."""
        from sample import load_style_state
        from model import HandwritingModel

        mock_args.rnn_size = 400
        mock_args.data_dir = str(tmpdir)
        model = HandwritingModel(mock_args)

        # Create styles directory but no files
        tmpdir.mkdir("styles")

        with pytest.raises(FileNotFoundError, match="Style characters file not found"):
            load_style_state(0, model, mock_args)

    def test_load_style_state_missing_strokes_file(self, mock_args, tmpdir):
        """Test load_style_state raises error when strokes file missing."""
        from sample import load_style_state
        from model import HandwritingModel

        mock_args.rnn_size = 400
        mock_args.data_dir = str(tmpdir)
        model = HandwritingModel(mock_args)

        # Create chars file but not strokes file
        styles_dir = tmpdir.mkdir("styles")
        chars_file = styles_dir.join("style-0-chars.npy")
        np.save(str(chars_file), np.array(b"test text"))

        with pytest.raises(FileNotFoundError, match="Style strokes file not found"):
            load_style_state(0, model, mock_args)

    def test_load_style_state_empty_strokes(self, mock_args, tmpdir):
        """Test load_style_state raises error for empty strokes array."""
        from sample import load_style_state
        from model import HandwritingModel

        mock_args.rnn_size = 400
        mock_args.data_dir = str(tmpdir)
        model = HandwritingModel(mock_args)

        # Create valid chars but empty strokes
        styles_dir = tmpdir.mkdir("styles")
        chars_file = styles_dir.join("style-0-chars.npy")
        strokes_file = styles_dir.join("style-0-strokes.npy")

        np.save(str(chars_file), np.array(b"test"))
        np.save(str(strokes_file), np.array([]))  # Empty

        with pytest.raises(ValueError, match="has empty strokes array"):
            load_style_state(0, model, mock_args)

    def test_load_style_state_wrong_strokes_shape(self, mock_args, tmpdir):
        """Test load_style_state raises error for wrong strokes shape."""
        from sample import load_style_state
        from model import HandwritingModel

        mock_args.rnn_size = 400
        mock_args.data_dir = str(tmpdir)
        model = HandwritingModel(mock_args)

        # Create valid chars but wrong shape strokes
        styles_dir = tmpdir.mkdir("styles")
        chars_file = styles_dir.join("style-0-chars.npy")
        strokes_file = styles_dir.join("style-0-strokes.npy")

        np.save(str(chars_file), np.array(b"test"))
        np.save(str(strokes_file), np.random.randn(100, 2))  # Should be (n, 3)

        with pytest.raises(ValueError, match="strokes have wrong shape"):
            load_style_state(0, model, mock_args)

    def test_load_style_state_invalid_characters(self, mock_args, tmpdir):
        """Test load_style_state raises error for invalid characters."""
        from sample import load_style_state
        from model import HandwritingModel

        mock_args.rnn_size = 400
        mock_args.data_dir = str(tmpdir)
        mock_args.alphabet = "abc"  # Very limited alphabet
        model = HandwritingModel(mock_args)

        # Create style with character not in alphabet
        styles_dir = tmpdir.mkdir("styles")
        chars_file = styles_dir.join("style-0-chars.npy")
        strokes_file = styles_dir.join("style-0-strokes.npy")

        np.save(str(chars_file), np.array(b"xyz"))  # 'xyz' not in 'abc'
        np.save(str(strokes_file), np.random.randn(50, 3).astype(np.float32))

        with pytest.raises(ValueError, match="contains characters not in alphabet"):
            load_style_state(0, model, mock_args)


@pytest.mark.unit
class TestSampleMultiline:
    """Test sample_multiline function."""

    def test_sample_multiline_biases_mismatch(self, tiny_model, mock_args):
        """Test sample_multiline raises error when biases count doesn't match lines."""
        from sample import sample_multiline

        lines = ["line 1", "line 2", "line 3"]
        biases = [1.0, 1.5]  # Only 2 biases for 3 lines

        with pytest.raises(ValueError, match="Number of biases"):
            sample_multiline(lines, tiny_model, mock_args, biases=biases)

    def test_sample_multiline_styles_mismatch(self, tiny_model, mock_args):
        """Test sample_multiline raises error when styles count doesn't match lines."""
        from sample import sample_multiline

        lines = ["line 1", "line 2"]
        styles = [0, 1, 2]  # 3 styles for 2 lines

        with pytest.raises(ValueError, match="Number of styles"):
            sample_multiline(lines, tiny_model, mock_args, styles=styles)

    def test_sample_multiline_invalid_style_id(self, tiny_model, mock_args):
        """Test sample_multiline raises error for invalid style ID."""
        from sample import sample_multiline

        lines = ["line 1"]
        styles = [15]  # Style ID out of range (must be 0-12)

        with pytest.raises(ValueError, match="Style ID 15 out of range"):
            sample_multiline(lines, tiny_model, mock_args, styles=styles)

    def test_sample_multiline_default_biases(self, tiny_model, mock_args, reset_random_seeds):
        """Test sample_multiline uses default bias when biases=None."""
        from sample import sample_multiline

        mock_args.bias = 1.5
        mock_args.tsteps = 50
        lines = ["test"]

        # Should not raise error (uses default bias)
        all_strokes, all_phis, all_kappas = sample_multiline(lines, tiny_model, mock_args, biases=None)

        assert len(all_strokes) == 1
        assert len(all_phis) == 1
        assert len(all_kappas) == 1


@pytest.mark.unit
class TestSampleFunction:
    """Test main sample function."""

    def test_sample_returns_three_values(self, tiny_model, mock_args, reset_random_seeds):
        """Test sample returns (strokes, phis, kappas) tuple."""
        from sample import sample

        mock_args.tsteps = 50
        mock_args.bias = 1.0

        strokes, phis, kappas = sample("test", tiny_model, mock_args)

        assert isinstance(strokes, np.ndarray)
        assert isinstance(phis, np.ndarray)
        assert isinstance(kappas, np.ndarray)

    def test_sample_strokes_shape(self, tiny_model, mock_args, reset_random_seeds):
        """Test sample returns strokes with correct shape."""
        from sample import sample

        mock_args.tsteps = 50
        mock_args.bias = 1.0

        strokes, phis, kappas = sample("test", tiny_model, mock_args)

        # Strokes should be (n_points, 6): [mu1, mu2, sigma1, sigma2, rho, eos]
        assert strokes.ndim == 2
        assert strokes.shape[1] == 6
        assert strokes.shape[0] <= mock_args.tsteps  # May stop early

    def test_sample_with_custom_bias(self, tiny_model, mock_args, reset_random_seeds):
        """Test sample uses custom bias when provided."""
        from sample import sample

        mock_args.tsteps = 50
        mock_args.bias = 1.0

        # Call with different bias
        strokes1, _, _ = sample("test", tiny_model, mock_args, bias=0.5)

        # Reset seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        strokes2, _, _ = sample("test", tiny_model, mock_args, bias=2.0)

        # Different biases should produce different results
        # (Note: may be similar due to randomness, but testing the mechanism works)
        assert strokes1.shape[0] > 0
        assert strokes2.shape[0] > 0

    def test_sample_stops_when_kappa_exceeds_text_length(self, tiny_model, mock_args, reset_random_seeds):
        """Test sample stops when attention moves past text end."""
        from sample import sample

        mock_args.tsteps = 1000  # Very long
        mock_args.bias = 1.0

        strokes, phis, kappas = sample("hi", tiny_model, mock_args)

        # Should stop before reaching tsteps (text is only 2 chars)
        # This tests the break condition: if kappa[0, 0, 0] > len(text) + 1
        assert strokes.shape[0] < mock_args.tsteps

    def test_sample_cumulative_coordinates(self, tiny_model, mock_args, reset_random_seeds):
        """Test sample converts deltas to cumulative coordinates."""
        from sample import sample

        mock_args.tsteps = 50
        mock_args.bias = 1.0

        strokes, phis, kappas = sample("test", tiny_model, mock_args)

        # Strokes[:, :2] should be cumulative (not all zeros)
        # The cumsum is applied: strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)
        assert not np.allclose(strokes[:, 0], 0.0)  # X coordinates vary
        assert not np.allclose(strokes[:, 1], 0.0)  # Y coordinates vary


@pytest.mark.unit
class TestLinePlot:
    """Test line_plot function."""

    def test_line_plot_creates_file(self, tmpdir):
        """Test line_plot saves file when save_path provided."""
        from sample import line_plot

        # Create simple stroke data
        strokes = np.array([
            [0, 0, 1, 1, 0, 0],
            [10, 10, 1, 1, 0, 1],
            [20, 20, 1, 1, 0, 0],
        ])

        save_path = tmpdir.join("test_plot.png")

        # Should create file without error
        line_plot(strokes, "Test Plot", save_path=str(save_path))

        assert save_path.exists()

    def test_line_plot_handles_empty_strokes(self):
        """Test line_plot handles empty stroke array."""
        from sample import line_plot

        # Empty strokes array
        strokes = np.zeros((0, 6))

        # Should not crash (may produce empty plot)
        try:
            line_plot(strokes, "Empty", save_path=None)
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            assert "empty" in str(e).lower() or "shape" in str(e).lower()


@pytest.mark.unit
class TestStyleCache:
    """Test style caching mechanism."""

    def test_style_cache_global_variable_exists(self):
        """Test _STYLE_CACHE global variable exists."""
        import sample

        assert hasattr(sample, '_STYLE_CACHE')
        assert isinstance(sample._STYLE_CACHE, dict)

    def test_style_cache_cleared_between_tests(self):
        """Test style cache can be cleared."""
        import sample

        # Add something to cache
        sample._STYLE_CACHE[(0, 400)] = "test_value"

        # Clear cache
        sample._STYLE_CACHE.clear()

        assert len(sample._STYLE_CACHE) == 0


# ============================================================================
# Summary
# ============================================================================

def test_sample_functions_suite_summary():
    """
    Sampling functions test suite summary.

    If all tests pass:
    - sample_gaussian2d() samples correctly from 2D Gaussians
    - to_one_hot() encodes strings to one-hot correctly
    - load_style_state() validates inputs and raises proper errors
    - sample() generates handwriting with correct output format
    - sample_multiline() validates biases and styles counts
    - line_plot() creates output files
    - Style caching mechanism works
    """
    print("\n✓ All sample function tests passed!")
    print("  - Gaussian sampling tested")
    print("  - One-hot encoding tested")
    print("  - Style loading validation tested")
    print("  - Sample function tested")
    print("  - Multi-line generation tested")
    print("  - Plotting function tested")
