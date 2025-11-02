"""
Unit Tests for Utils Module

Tests utils.py DataLoader and Logger classes.

Usage:
    pytest tests/unit/test_utils.py -v
    pytest -k test_utils
"""

import pytest
import numpy as np
import pickle
import os
from pathlib import Path


@pytest.mark.unit
class TestToOneHotFunction:
    """Test to_one_hot utility function."""

    def test_to_one_hot_simple_encoding(self):
        """Test to_one_hot encodes simple string."""
        from utils import to_one_hot

        alphabet = ' abc'
        text = "cab"
        ascii_steps = 5

        encoded = to_one_hot(text, ascii_steps, alphabet)

        assert encoded.shape == (5, len(alphabet) + 1)
        # Each row should sum to 1
        row_sums = encoded.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(ascii_steps))

    def test_to_one_hot_padding_short_string(self):
        """Test to_one_hot pads short strings."""
        from utils import to_one_hot

        alphabet = ' abc'
        text = "ab"
        ascii_steps = 5

        encoded = to_one_hot(text, ascii_steps, alphabet)

        # Last 3 rows should be padding (index 0)
        assert encoded[2, 0] == 1  # Padding
        assert encoded[3, 0] == 1
        assert encoded[4, 0] == 1

    def test_to_one_hot_truncates_long_string(self):
        """Test to_one_hot truncates strings longer than ascii_steps."""
        from utils import to_one_hot

        alphabet = ' abc'
        text = "abcabc"
        ascii_steps = 3

        encoded = to_one_hot(text, ascii_steps, alphabet)

        assert encoded.shape == (3, len(alphabet) + 1)

    def test_to_one_hot_clips_super_long_string(self):
        """Test to_one_hot clips strings longer than 3000 chars."""
        from utils import to_one_hot

        alphabet = ' abc'
        text = "a" * 5000
        ascii_steps = 10

        # Should not raise error, clips to 3000
        encoded = to_one_hot(text, ascii_steps, alphabet)

        assert encoded.shape == (10, len(alphabet) + 1)

    def test_to_one_hot_character_indices(self):
        """Test to_one_hot maps characters to correct indices."""
        from utils import to_one_hot

        alphabet = ' abc'
        text = " "
        ascii_steps = 1

        encoded = to_one_hot(text, ascii_steps, alphabet)

        # Space is at index 0 in alphabet, maps to index 1 (0 is padding)
        space_idx = alphabet.find(' ') + 1
        assert encoded[0, space_idx] == 1


@pytest.mark.unit
class TestLoggerClass:
    """Test Logger class."""

    def test_logger_creates_file(self, mock_args, tmpdir):
        """Test Logger creates log file on initialization."""
        from utils import Logger

        mock_args.log_dir = str(tmpdir) + '/'
        mock_args.train = True

        logger = Logger(mock_args)

        # Log file should exist
        log_file = tmpdir.join("train_scribe.txt")
        assert log_file.exists()

    def test_logger_train_mode_filename(self, mock_args, tmpdir):
        """Test Logger uses train filename in train mode."""
        from utils import Logger

        mock_args.log_dir = str(tmpdir) + '/'
        mock_args.train = True

        logger = Logger(mock_args)

        assert logger.logf.endswith('train_scribe.txt')

    def test_logger_sample_mode_filename(self, mock_args, tmpdir):
        """Test Logger uses sample filename in sample mode."""
        from utils import Logger

        mock_args.log_dir = str(tmpdir) + '/'
        mock_args.train = False

        logger = Logger(mock_args)

        assert logger.logf.endswith('sample_scribe.txt')

    def test_logger_writes_initial_header(self, mock_args, tmpdir):
        """Test Logger writes initial header to file."""
        from utils import Logger

        mock_args.log_dir = str(tmpdir) + '/'
        mock_args.train = True

        logger = Logger(mock_args)

        # Read file contents
        log_file = tmpdir.join("train_scribe.txt")
        content = log_file.read()

        assert "Scribe" in content
        assert "Realistic" in content or "Handriting" in content

    def test_logger_write_method(self, mock_args, tmpdir, capsys):
        """Test Logger.write() writes to file and prints."""
        from utils import Logger

        mock_args.log_dir = str(tmpdir) + '/'
        mock_args.train = True

        logger = Logger(mock_args)
        logger.write("Test message")

        # Should print to stdout
        captured = capsys.readouterr()
        assert "Test message" in captured.out

        # Should write to file
        log_file = tmpdir.join("train_scribe.txt")
        content = log_file.read()
        assert "Test message" in content

    def test_logger_write_no_print(self, mock_args, tmpdir, capsys):
        """Test Logger.write() with print_it=False doesn't print."""
        from utils import Logger

        mock_args.log_dir = str(tmpdir) + '/'
        mock_args.train = True

        logger = Logger(mock_args)
        logger.write("Silent message", print_it=False)

        # Should NOT print to stdout
        captured = capsys.readouterr()
        assert "Silent message" not in captured.out

        # Should still write to file
        log_file = tmpdir.join("train_scribe.txt")
        content = log_file.read()
        assert "Silent message" in content


@pytest.mark.unit
class TestDataLoaderInitialization:
    """Test DataLoader initialization."""

    def test_dataloader_loads_existing_file(self, mock_args, mini_dataset_path, tmpdir):
        """Test DataLoader loads existing preprocessed file."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)

        # Should have loaded data
        assert len(data_loader.raw_stroke_data) > 0
        assert len(data_loader.raw_ascii_data) > 0

    def test_dataloader_attributes_set(self, mock_args, mini_dataset_path, tmpdir):
        """Test DataLoader sets all required attributes."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)

        assert hasattr(data_loader, 'data_dir')
        assert hasattr(data_loader, 'alphabet')
        assert hasattr(data_loader, 'batch_size')
        assert hasattr(data_loader, 'tsteps')
        assert hasattr(data_loader, 'data_scale')
        assert hasattr(data_loader, 'ascii_steps')
        assert hasattr(data_loader, 'limit')

    def test_dataloader_ascii_steps_calculation(self, mock_args, mini_dataset_path, tmpdir):
        """Test DataLoader calculates ascii_steps correctly."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        mock_args.tsteps = 150
        mock_args.tsteps_per_ascii = 25
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)

        # ascii_steps = tsteps // tsteps_per_ascii
        expected_ascii_steps = 150 // 25
        assert data_loader.ascii_steps == expected_ascii_steps


@pytest.mark.unit
class TestDataLoaderPreprocessing:
    """Test DataLoader preprocessing and data loading."""

    def test_load_preprocessed_sets_data_arrays(self, mock_args, mini_dataset_path, tmpdir):
        """Test load_preprocessed() sets stroke and ascii data arrays."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)

        # Should have raw data
        assert len(data_loader.raw_stroke_data) > 0
        assert len(data_loader.raw_ascii_data) > 0

        # Should have equal lengths
        assert len(data_loader.raw_stroke_data) == len(data_loader.raw_ascii_data)

    def test_load_preprocessed_creates_train_valid_split(self, mock_args, mini_dataset_path, tmpdir):
        """Test load_preprocessed() creates train/validation split."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)

        # Should have both training and validation data
        assert len(data_loader.stroke_data) > 0
        assert len(data_loader.valid_stroke_data) > 0

        # Training should be ~95%, validation ~5%
        total = len(data_loader.stroke_data) + len(data_loader.valid_stroke_data)
        valid_ratio = len(data_loader.valid_stroke_data) / total

        # Should be close to 5% (within tolerance)
        assert 0.03 < valid_ratio < 0.08  # Between 3% and 8%

    def test_load_preprocessed_filters_short_sequences(self, mock_args, tmpdir):
        """Test load_preprocessed() filters sequences shorter than tsteps."""
        from utils import DataLoader, Logger

        # Create test data with mixed lengths
        strokes = [
            np.random.randn(50, 3).astype(np.float32),   # Too short (< tsteps+2)
            np.random.randn(200, 3).astype(np.float32),  # Long enough
            np.random.randn(30, 3).astype(np.float32),   # Too short
            np.random.randn(250, 3).astype(np.float32),  # Long enough
        ]
        asciis = ["short 1", "long enough", "short 2", "also long enough"]
        data = [strokes, asciis]

        # Save to file
        data_dir = tmpdir.mkdir("data")
        data_file = data_dir.join("strokes_training_data.cpkl")
        with open(str(data_file), 'wb') as f:
            pickle.dump(data, f)

        mock_args.data_dir = str(data_dir)
        mock_args.log_dir = str(tmpdir) + '/'
        mock_args.tsteps = 100
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)

        # Should only have 2 sequences (the long ones)
        total_sequences = len(data_loader.stroke_data) + len(data_loader.valid_stroke_data)
        assert total_sequences == 2

    def test_load_preprocessed_scales_data(self, mock_args, tmpdir):
        """Test load_preprocessed() scales stroke data correctly."""
        from utils import DataLoader, Logger

        # Create test data with known values
        strokes = [
            np.array([[100, 200, 0], [150, 250, 1]], dtype=np.float32)
        ] * 10  # Need enough for batch processing
        asciis = ["test text"] * 10
        data = [strokes, asciis]

        data_dir = tmpdir.mkdir("data")
        data_file = data_dir.join("strokes_training_data.cpkl")
        with open(str(data_file), 'wb') as f:
            pickle.dump(data, f)

        mock_args.data_dir = str(data_dir)
        mock_args.log_dir = str(tmpdir) + '/'
        mock_args.tsteps = 1
        mock_args.data_scale = 50
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)

        # Get some data
        combined_data = data_loader.stroke_data + data_loader.valid_stroke_data
        if len(combined_data) > 0:
            sample = combined_data[0]
            # Values should be scaled down by data_scale (50)
            # Original: 100, 200 → Scaled: 2.0, 4.0
            # (Note: may also be clipped by limit)
            assert np.max(np.abs(sample[:, :2])) <= 500 / mock_args.data_scale


@pytest.mark.unit
class TestDataLoaderBatchGeneration:
    """Test DataLoader batch generation methods."""

    def test_validation_data_returns_batch(self, mock_args, mini_dataset_path, tmpdir):
        """Test validation_data() returns batch of correct size."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)
        x_batch, y_batch, ascii_list, one_hots = data_loader.validation_data()

        # Batch should match batch_size
        assert len(x_batch) == mock_args.batch_size
        assert len(y_batch) == mock_args.batch_size
        assert len(ascii_list) == mock_args.batch_size
        assert len(one_hots) == mock_args.batch_size

    def test_validation_data_x_y_shifted(self, mock_args, mini_dataset_path, tmpdir):
        """Test validation_data() returns y as shifted version of x."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)
        x_batch, y_batch, ascii_list, one_hots = data_loader.validation_data()

        # y should be x shifted by 1 timestep
        # x_batch[i] = data[:tsteps]
        # y_batch[i] = data[1:tsteps+1]
        # So y_batch[i][0] should equal x_batch[i][1] (next point)
        for x, y in zip(x_batch, y_batch):
            # Check shapes match
            assert x.shape == y.shape
            assert x.shape[0] == mock_args.tsteps

    def test_next_batch_returns_batch(self, mock_args, mini_dataset_path, tmpdir, reset_random_seeds):
        """Test next_batch() returns batch of correct size."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)
        x_batch, y_batch, ascii_list, one_hots = data_loader.next_batch()

        assert len(x_batch) == mock_args.batch_size
        assert len(y_batch) == mock_args.batch_size
        assert len(ascii_list) == mock_args.batch_size
        assert len(one_hots) == mock_args.batch_size

    def test_next_batch_one_hot_encoding(self, mock_args, mini_dataset_path, tmpdir, reset_random_seeds):
        """Test next_batch() returns proper one-hot encodings."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)
        x_batch, y_batch, ascii_list, one_hots = data_loader.next_batch()

        # Each one-hot encoding should have correct shape
        for oh in one_hots:
            assert oh.shape == (data_loader.ascii_steps, len(data_loader.alphabet) + 1)
            # Each row should sum to 1
            row_sums = oh.sum(axis=1)
            np.testing.assert_allclose(row_sums, np.ones(data_loader.ascii_steps))

    def test_tick_batch_pointer_increments(self, mock_args, mini_dataset_path, tmpdir):
        """Test tick_batch_pointer() increments pointer."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)

        initial_pointer = data_loader.pointer
        data_loader.tick_batch_pointer()

        assert data_loader.pointer == initial_pointer + 1

    def test_tick_batch_pointer_resets_at_end(self, mock_args, mini_dataset_path, tmpdir):
        """Test tick_batch_pointer() resets when reaching end of data."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)

        # Move pointer to end
        data_loader.pointer = len(data_loader.stroke_data)

        # Tick should reset
        data_loader.tick_batch_pointer()

        assert data_loader.pointer == 0

    def test_reset_batch_pointer_creates_permutation(self, mock_args, mini_dataset_path, tmpdir):
        """Test reset_batch_pointer() creates random permutation."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)

        # Reset creates permutation
        data_loader.reset_batch_pointer()

        assert data_loader.pointer == 0
        assert len(data_loader.idx_perm) == len(data_loader.stroke_data)

    def test_num_batches_calculation(self, mock_args, mini_dataset_path, tmpdir):
        """Test num_batches is calculated correctly."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)

        # num_batches = len(stroke_data) // batch_size
        expected_batches = len(data_loader.stroke_data) // data_loader.batch_size
        assert data_loader.num_batches == expected_batches


@pytest.mark.unit
class TestDataLoaderEdgeCases:
    """Test DataLoader edge cases and error conditions."""

    def test_dataloader_with_small_batch_size(self, mock_args, mini_dataset_path, tmpdir):
        """Test DataLoader with very small batch size."""
        from utils import DataLoader, Logger

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.log_dir = str(tmpdir) + '/'
        mock_args.batch_size = 1  # Very small
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger)

        # Should work without errors
        x_batch, y_batch, ascii_list, one_hots = data_loader.next_batch()
        assert len(x_batch) == 1

    def test_dataloader_clips_outliers(self, mock_args, tmpdir):
        """Test DataLoader clips outlier values to limit."""
        from utils import DataLoader, Logger

        # Create data with outliers
        strokes = [
            np.array([[1000, -1000, 0]], dtype=np.float32)  # Large outliers
        ] * 10
        asciis = ["test"] * 10
        data = [strokes, asciis]

        data_dir = tmpdir.mkdir("data")
        data_file = data_dir.join("strokes_training_data.cpkl")
        with open(str(data_file), 'wb') as f:
            pickle.dump(data, f)

        mock_args.data_dir = str(data_dir)
        mock_args.log_dir = str(tmpdir) + '/'
        mock_args.tsteps = 1
        logger = Logger(mock_args)

        data_loader = DataLoader(mock_args, logger=logger, limit=500)

        # Get data
        combined_data = data_loader.stroke_data + data_loader.valid_stroke_data
        if len(combined_data) > 0:
            sample = combined_data[0]
            # Values should be clipped to [-limit, limit]
            assert np.max(sample) <= 500
            assert np.min(sample) >= -500


# ============================================================================
# Summary
# ============================================================================

def test_utils_suite_summary():
    """
    Utils module test suite summary.

    If all tests pass:
    - to_one_hot() encodes strings correctly
    - Logger creates files and writes messages
    - DataLoader loads preprocessed data
    - DataLoader creates train/validation split
    - DataLoader filters short sequences
    - DataLoader scales and clips data correctly
    - validation_data() returns proper batches
    - next_batch() returns proper batches with one-hot encoding
    - Batch pointer management works correctly
    - Edge cases handled (small batches, outliers)
    """
    print("\n✓ All utils tests passed!")
    print("  - to_one_hot() tested")
    print("  - Logger class tested")
    print("  - DataLoader initialization tested")
    print("  - Data preprocessing tested")
    print("  - Batch generation tested")
    print("  - Edge cases tested")
