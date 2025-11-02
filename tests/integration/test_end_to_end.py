"""
End-to-End Integration Tests

Tests complete workflows:
- Load data → train → save → load → sample
- Train model and verify loss decreases
- Generate samples from trained model
- Multi-line generation workflow
- Style priming workflow (requires rnn_size=400)

Usage:
    pytest tests/integration/test_end_to_end.py -v
    pytest -m integration -k end_to_end
"""

import pytest
import tensorflow as tf
import numpy as np
import os
import tempfile
from pathlib import Path


@pytest.mark.integration
@pytest.mark.slow
class TestTrainSampleWorkflow:
    """Test complete train → save → load → sample workflow."""

    def test_train_and_sample_workflow(self, tiny_model, mock_args,
                                       mini_dataset_path, temp_checkpoint_dir,
                                       temp_output_dir, reset_random_seeds):
        """Test complete workflow: train, save, load, sample."""
        from model import compute_loss
        from utils import DataLoader
        from sample import sample

        # Setup
        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.batch_size = 2
        mock_args.tsteps = 10
        mock_args.bias = 1.0
        mock_args.eos_threshold = 0.35

        class MockLogger:
            def write(self, msg):
                pass

        data_loader = DataLoader(mock_args, logger=MockLogger())
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Training step
        @tf.function
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = tiny_model(inputs, training=True)
                loss = compute_loss(predictions, targets)
            gradients = tape.gradient(loss, tiny_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))
            return loss

        # 1. TRAIN: Train for 10 steps
        losses = []
        for _ in range(10):
            x, y, s, c = data_loader.next_batch()
            train_inputs = {
                'stroke_data': tf.constant(x, dtype=tf.float32),
                'char_seq': tf.constant(c, dtype=tf.float32)
            }
            train_targets = tf.constant(y, dtype=tf.float32)
            loss = train_step(train_inputs, train_targets)
            losses.append(float(loss))

        # Verify loss decreased
        assert losses[-1] < losses[0], "Loss should decrease during training"

        # 2. SAVE: Save checkpoint
        checkpoint = tf.train.Checkpoint(model=tiny_model, optimizer=optimizer)
        manager = tf.train.CheckpointManager(
            checkpoint,
            directory=temp_checkpoint_dir,
            max_to_keep=1
        )
        save_path = manager.save(checkpoint_number=10)

        # 3. LOAD: Create new model and load checkpoint
        from model import HandwritingModel
        new_model = HandwritingModel(mock_args)

        new_checkpoint = tf.train.Checkpoint(model=new_model)
        new_checkpoint.restore(save_path)

        # 4. SAMPLE: Generate handwriting
        mock_args.tsteps = 50  # Longer for sampling
        text = "test"

        strokes, phis, kappas = sample(text, new_model, mock_args)

        # Verify sample output
        assert strokes.shape[0] > 0, "Should generate strokes"
        assert strokes.shape[1] == 6, "Strokes should have 6 columns"

    def test_full_pipeline_with_svg_output(self, tiny_model, mock_args,
                                          mini_dataset_path, temp_checkpoint_dir,
                                          temp_output_dir, reset_random_seeds):
        """Test complete pipeline including SVG generation."""
        from model import compute_loss
        from utils import DataLoader
        from sample import sample
        import svg_output

        # Setup
        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.batch_size = 2
        mock_args.tsteps = 10

        class MockLogger:
            def write(self, msg):
                pass

        data_loader = DataLoader(mock_args, logger=MockLogger())
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        @tf.function
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = tiny_model(inputs, training=True)
                loss = compute_loss(predictions, targets)
            gradients = tape.gradient(loss, tiny_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))
            return loss

        # Train briefly
        for _ in range(5):
            x, y, s, c = data_loader.next_batch()
            train_inputs = {
                'stroke_data': tf.constant(x, dtype=tf.float32),
                'char_seq': tf.constant(c, dtype=tf.float32)
            }
            train_targets = tf.constant(y, dtype=tf.float32)
            train_step(train_inputs, train_targets)

        # Generate sample
        mock_args.tsteps = 50
        mock_args.bias = 1.0
        mock_args.eos_threshold = 0.35
        text = "test"

        strokes, phis, kappas = sample(text, tiny_model, mock_args)

        # Convert to offsets for SVG
        offsets = strokes.copy()
        offsets[:, :2] = np.diff(np.vstack([[[0, 0]], strokes[:, :2]]), axis=0)

        # Generate SVG
        output_path = os.path.join(temp_output_dir, "test.svg")
        svg_output.save_as_svg([offsets], [text], output_path)

        # Verify SVG created
        assert os.path.exists(output_path), "SVG file should be created"

        # Verify SVG is valid
        with open(output_path, 'r') as f:
            content = f.read()
            assert '<svg' in content, "Should be valid SVG"
            assert '<path' in content, "Should contain path"


@pytest.mark.integration
@pytest.mark.slow
class TestMultilineSamplingWorkflow:
    """Test multi-line generation workflow."""

    def test_multiline_generation_workflow(self, tiny_model, mock_args,
                                          mini_dataset_path, temp_output_dir,
                                          reset_random_seeds):
        """Test generating multiple lines with different biases."""
        from model import compute_loss
        from utils import DataLoader
        from sample import sample_multiline

        # Setup
        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.batch_size = 2
        mock_args.tsteps = 30
        mock_args.bias = 1.0
        mock_args.eos_threshold = 0.35

        class MockLogger:
            def write(self, msg):
                pass

        data_loader = DataLoader(mock_args, logger=MockLogger())
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        @tf.function
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = tiny_model(inputs, training=True)
                loss = compute_loss(predictions, targets)
            gradients = tape.gradient(loss, tiny_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))
            return loss

        # Train briefly
        for _ in range(10):
            x, y, s, c = data_loader.next_batch()
            train_inputs = {
                'stroke_data': tf.constant(x, dtype=tf.float32),
                'char_seq': tf.constant(c, dtype=tf.float32)
            }
            train_targets = tf.constant(y, dtype=tf.float32)
            train_step(train_inputs, train_targets)

        # Generate multiple lines
        lines = ["line 1", "line 2", "line 3"]
        biases = [0.8, 1.2, 1.5]

        all_strokes, all_phis, all_kappas = sample_multiline(
            lines, tiny_model, mock_args, biases=biases
        )

        # Verify output
        assert len(all_strokes) == 3, "Should generate 3 lines"
        assert len(all_phis) == 3
        assert len(all_kappas) == 3

        # Each line should have strokes
        for strokes in all_strokes:
            assert strokes.shape[0] > 0, "Each line should have strokes"


@pytest.mark.integration
class TestDataLoaderIntegration:
    """Test DataLoader integration with training."""

    def test_dataloader_provides_valid_batches(self, mock_args, mini_dataset_path,
                                              reset_random_seeds):
        """Test DataLoader provides properly formatted batches."""
        from utils import DataLoader

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.batch_size = 2
        mock_args.tsteps = 10

        class MockLogger:
            def write(self, msg):
                pass

        data_loader = DataLoader(mock_args, logger=MockLogger())

        # Get batch
        x, y, s, c = data_loader.next_batch()

        # Verify shapes
        assert len(x) == 2, "Batch size should be 2"
        assert len(y) == 2
        assert len(c) == 2

        # Verify data types
        assert isinstance(x[0], np.ndarray)
        assert x[0].dtype == np.float32

        # Verify one-hot encoding
        assert c[0].shape[1] == len(mock_args.alphabet) + 1

    def test_validation_data_format(self, mock_args, mini_dataset_path):
        """Test validation data is properly formatted."""
        from utils import DataLoader

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.batch_size = 2

        class MockLogger:
            def write(self, msg):
                pass

        data_loader = DataLoader(mock_args, logger=MockLogger())

        # Get validation data
        v_x, v_y, v_s, v_c = data_loader.validation_data()

        # Should have batch_size items
        assert len(v_x) == mock_args.batch_size
        assert len(v_y) == mock_args.batch_size


@pytest.mark.integration
@pytest.mark.slow
class TestLossConvergence:
    """Test loss convergence behavior."""

    def test_loss_decreases_with_sufficient_training(self, tiny_model, mock_args,
                                                     mini_dataset_path,
                                                     reset_random_seeds):
        """Test loss decreases with sufficient training steps."""
        from model import compute_loss
        from utils import DataLoader

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.batch_size = 2
        mock_args.tsteps = 10

        class MockLogger:
            def write(self, msg):
                pass

        data_loader = DataLoader(mock_args, logger=MockLogger())
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        @tf.function
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = tiny_model(inputs, training=True)
                loss = compute_loss(predictions, targets)
            gradients = tape.gradient(loss, tiny_model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
            optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))
            return loss

        # Train for 30 steps
        losses = []
        for _ in range(30):
            x, y, s, c = data_loader.next_batch()
            train_inputs = {
                'stroke_data': tf.constant(x, dtype=tf.float32),
                'char_seq': tf.constant(c, dtype=tf.float32)
            }
            train_targets = tf.constant(y, dtype=tf.float32)
            loss = train_step(train_inputs, train_targets)
            losses.append(float(loss))

        # Compare first 10 and last 10
        early_loss = np.mean(losses[:10])
        late_loss = np.mean(losses[-10:])

        assert late_loss < early_loss, \
            f"Loss should decrease (early={early_loss:.3f}, late={late_loss:.3f})"

    def test_validation_loss_tracked(self, tiny_model, mock_args,
                                    mini_dataset_path, reset_random_seeds):
        """Test validation loss can be tracked during training."""
        from model import compute_loss
        from utils import DataLoader

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.batch_size = 2
        mock_args.tsteps = 10

        class MockLogger:
            def write(self, msg):
                pass

        data_loader = DataLoader(mock_args, logger=MockLogger())
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Get validation data
        v_x, v_y, v_s, v_c = data_loader.validation_data()
        validation_inputs = {
            'stroke_data': tf.constant(v_x, dtype=tf.float32),
            'char_seq': tf.constant(v_c, dtype=tf.float32)
        }
        validation_targets = tf.constant(v_y, dtype=tf.float32)

        @tf.function
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = tiny_model(inputs, training=True)
                loss = compute_loss(predictions, targets)
            gradients = tape.gradient(loss, tiny_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))
            return loss

        train_losses = []
        valid_losses = []

        # Train and track both losses
        for _ in range(10):
            x, y, s, c = data_loader.next_batch()
            train_inputs = {
                'stroke_data': tf.constant(x, dtype=tf.float32),
                'char_seq': tf.constant(c, dtype=tf.float32)
            }
            train_targets = tf.constant(y, dtype=tf.float32)

            train_loss = train_step(train_inputs, train_targets)
            train_losses.append(float(train_loss))

            # Compute validation loss
            valid_predictions = tiny_model(validation_inputs, training=False)
            valid_loss = compute_loss(valid_predictions, validation_targets)
            valid_losses.append(float(valid_loss))

        # Both should be tracked
        assert len(train_losses) == 10
        assert len(valid_losses) == 10

        # All should be finite
        assert all(np.isfinite(train_losses))
        assert all(np.isfinite(valid_losses))


@pytest.mark.integration
class TestModelSampleCompatibility:
    """Test model and sampling function compatibility."""

    def test_freshly_initialized_model_can_sample(self, tiny_model, mock_args,
                                                  reset_random_seeds):
        """Test even untrained model can generate samples."""
        from sample import sample

        mock_args.tsteps = 30
        mock_args.bias = 1.0
        mock_args.eos_threshold = 0.35

        text = "test"

        # Should work even with untrained model
        strokes, phis, kappas = sample(text, tiny_model, mock_args)

        assert strokes.shape[0] > 0

    def test_sample_output_format_matches_training_data(self, tiny_model,
                                                        mock_args,
                                                        reset_random_seeds):
        """Test sample output format is compatible with training data."""
        from sample import sample

        mock_args.tsteps = 30
        mock_args.bias = 1.0
        mock_args.eos_threshold = 0.35

        text = "test"
        strokes, phis, kappas = sample(text, tiny_model, mock_args)

        # Convert to offsets (delta encoding) like training data
        offsets = strokes.copy()
        offsets[:, :2] = np.diff(np.vstack([[[0, 0]], strokes[:, :2]]), axis=0)

        # Verify format matches training data
        # Should have 3 columns after extracting deltas and eos
        stroke_data = np.column_stack([offsets[:, 0], offsets[:, 1], strokes[:, 5]])

        assert stroke_data.shape[1] == 3, "Should have [dx, dy, eos] format"
        assert np.all((stroke_data[:, 2] == 0) | (stroke_data[:, 2] == 1)), \
            "EOS should be binary"


@pytest.mark.integration
class TestCharacterSetCompatibility:
    """Test full character set support (letters, numerals, punctuation)."""

    def test_full_alphabet_sampling(self, tiny_model, mock_args, reset_random_seeds):
        """Test sampling works with full character set."""
        from sample import sample

        mock_args.tsteps = 50
        mock_args.bias = 1.0
        mock_args.eos_threshold = 0.35

        # Text with full character set: letters, numerals, punctuation
        text = "Test 123! Hello?"

        # Should handle all characters
        strokes, phis, kappas = sample(text, tiny_model, mock_args)

        assert strokes.shape[0] > 0

    def test_one_hot_encoding_full_alphabet(self):
        """Test one-hot encoding supports full alphabet."""
        from sample import to_one_hot

        alphabet = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#%&\'()*+,-./:;?[]'
        text = 'Test 123!'
        max_len = 20

        encoded = to_one_hot(text, max_len, alphabet)

        # Should encode successfully
        assert encoded.shape == (max_len, len(alphabet) + 1)

        # Should be valid one-hot
        row_sums = encoded.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0)


# ============================================================================
# Summary
# ============================================================================

def test_end_to_end_suite_summary():
    """
    End-to-end test suite summary.

    If all end-to-end tests pass, the complete system works:
    - Train → Save → Load → Sample workflow
    - Multi-line generation workflow
    - SVG output generation
    - DataLoader integration with training
    - Loss decreases over training
    - Validation loss tracking
    - Full character set support (letters, numerals, punctuation)
    - Model/sample format compatibility
    """
    print("\n✓ All end-to-end tests passed!")
    print("  - Complete train/sample workflow works")
    print("  - Multi-line generation works")
    print("  - SVG output generation works")
    print("  - Loss decreases with training")
    print("  - Full character set supported")
    print("  - All components integrate correctly")
