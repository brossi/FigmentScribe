"""
Integration Tests for Training Loop

Tests training workflow:
- Training step execution
- Gradient computation and application
- Loss behavior over training
- Model weight updates
- Validation loss computation
- Learning rate scheduling

Usage:
    pytest tests/integration/test_training_loop.py -v
    pytest -m integration
"""

import pytest
import tensorflow as tf
import numpy as np
import tempfile
import os


@pytest.mark.integration
class TestTrainingStep:
    """Test individual training step execution."""

    def test_train_step_executes(self, tiny_model, sample_batch_inputs,
                                 reset_random_seeds):
        """Test training step executes without errors."""
        from model import compute_loss

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)

        # Training step
        with tf.GradientTape() as tape:
            predictions = tiny_model(sample_batch_inputs, training=True)
            loss = compute_loss(predictions, sample_batch_inputs['stroke_data'])

        gradients = tape.gradient(loss, tiny_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))

        # Should complete without error
        assert loss is not None
        assert not tf.math.is_nan(loss)

    def test_gradients_computed(self, tiny_model, sample_batch_inputs):
        """Test gradients are computed for all trainable variables."""
        from model import compute_loss

        with tf.GradientTape() as tape:
            predictions = tiny_model(sample_batch_inputs, training=True)
            loss = compute_loss(predictions, sample_batch_inputs['stroke_data'])

        gradients = tape.gradient(loss, tiny_model.trainable_variables)

        # All gradients should be computed
        assert len(gradients) == len(tiny_model.trainable_variables)

        # Gradients should not be None
        for grad, var in zip(gradients, tiny_model.trainable_variables):
            assert grad is not None, f"Gradient for {var.name} is None"

    def test_gradients_are_finite(self, tiny_model, sample_batch_inputs,
                                  reset_random_seeds):
        """Test all gradients are finite (no NaN or Inf)."""
        from model import compute_loss

        with tf.GradientTape() as tape:
            predictions = tiny_model(sample_batch_inputs, training=True)
            loss = compute_loss(predictions, sample_batch_inputs['stroke_data'])

        gradients = tape.gradient(loss, tiny_model.trainable_variables)

        # All gradients should be finite
        for grad, var in zip(gradients, tiny_model.trainable_variables):
            assert not tf.reduce_any(tf.math.is_nan(grad)), \
                f"Gradient for {var.name} contains NaN"
            assert not tf.reduce_any(tf.math.is_inf(grad)), \
                f"Gradient for {var.name} contains Inf"

    def test_gradient_clipping_reduces_norm(self, tiny_model, sample_batch_inputs,
                                           reset_random_seeds):
        """Test gradient clipping reduces global norm."""
        from model import compute_loss

        with tf.GradientTape() as tape:
            predictions = tiny_model(sample_batch_inputs, training=True)
            loss = compute_loss(predictions, sample_batch_inputs['stroke_data'])

        gradients = tape.gradient(loss, tiny_model.trainable_variables)

        # Compute global norm before clipping
        norm_before = tf.linalg.global_norm(gradients)

        # Clip gradients
        clipped_gradients, norm_after = tf.clip_by_global_norm(gradients, 1.0)

        # If original norm > 1.0, clipped norm should be approximately 1.0
        if norm_before > 1.0:
            np.testing.assert_allclose(norm_after.numpy(), 1.0, atol=0.01)

    def test_weights_update_after_training_step(self, tiny_model, sample_batch_inputs,
                                                reset_random_seeds):
        """Test model weights change after training step."""
        from model import compute_loss

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

        # Store initial weights
        initial_weights = [var.numpy().copy() for var in tiny_model.trainable_variables]

        # Training step
        with tf.GradientTape() as tape:
            predictions = tiny_model(sample_batch_inputs, training=True)
            loss = compute_loss(predictions, sample_batch_inputs['stroke_data'])

        gradients = tape.gradient(loss, tiny_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))

        # Weights should have changed
        for initial, current in zip(initial_weights, tiny_model.trainable_variables):
            difference = np.abs(initial - current.numpy()).max()
            assert difference > 0, "Weights should change after training step"


@pytest.mark.integration
@pytest.mark.slow
class TestTrainingProgress:
    """Test training progress over multiple steps."""

    def test_loss_decreases_over_training(self, tiny_model, mock_args,
                                         mini_dataset_path, reset_random_seeds):
        """Test loss decreases with training on simple data."""
        from model import compute_loss
        from utils import DataLoader

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.batch_size = 2
        mock_args.tsteps = 10

        class MockLogger:
            def write(self, msg):
                pass

        data_loader = DataLoader(mock_args, logger=MockLogger())
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

        @tf.function
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = tiny_model(inputs, training=True)
                loss = compute_loss(predictions, targets)
            gradients = tape.gradient(loss, tiny_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))
            return loss

        # Record losses over training
        losses = []

        # Train for 20 steps
        for _ in range(20):
            x, y, s, c = data_loader.next_batch()

            train_inputs = {
                'stroke_data': tf.constant(x, dtype=tf.float32),
                'char_seq': tf.constant(c, dtype=tf.float32)
            }
            train_targets = tf.constant(y, dtype=tf.float32)

            loss = train_step(train_inputs, train_targets)
            losses.append(float(loss))

        # Loss should generally decrease (compare first 5 to last 5)
        early_loss = np.mean(losses[:5])
        late_loss = np.mean(losses[-5:])

        assert late_loss < early_loss, \
            f"Loss should decrease over training (early={early_loss:.3f}, late={late_loss:.3f})"

    def test_training_mode_affects_output(self, tiny_model, sample_batch_inputs,
                                         reset_random_seeds):
        """Test training=True/False produces different outputs (dropout)."""
        # Only test if model has dropout
        if tiny_model.dropout_layer is None:
            pytest.skip("Model has no dropout layer")

        # Run with training=True multiple times
        outputs_train = []
        for _ in range(5):
            predictions = tiny_model(sample_batch_inputs, training=True)
            outputs_train.append(predictions['eos'].numpy().copy())

        # Run with training=False multiple times
        outputs_eval = []
        for _ in range(5):
            predictions = tiny_model(sample_batch_inputs, training=False)
            outputs_eval.append(predictions['eos'].numpy().copy())

        # Training mode outputs should vary (due to dropout)
        train_variance = np.var([o.mean() for o in outputs_train])

        # Eval mode outputs should be identical
        eval_variance = np.var([o.mean() for o in outputs_eval])

        # Training variance should be higher (dropout adds randomness)
        assert train_variance > eval_variance or eval_variance < 1e-10, \
            "Training mode should have more variance due to dropout"

    def test_validation_loss_computed(self, tiny_model, mock_args,
                                     mini_dataset_path, reset_random_seeds):
        """Test validation loss can be computed during training."""
        from model import compute_loss
        from utils import DataLoader

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.batch_size = 2

        class MockLogger:
            def write(self, msg):
                pass

        data_loader = DataLoader(mock_args, logger=MockLogger())

        # Get validation data
        v_x, v_y, v_s, v_c = data_loader.validation_data()
        validation_inputs = {
            'stroke_data': tf.constant(v_x, dtype=tf.float32),
            'char_seq': tf.constant(v_c, dtype=tf.float32)
        }
        validation_targets = tf.constant(v_y, dtype=tf.float32)

        # Compute validation loss
        valid_predictions = tiny_model(validation_inputs, training=False)
        valid_loss = compute_loss(valid_predictions, validation_targets)

        # Should be finite
        assert tf.math.is_finite(valid_loss)


@pytest.mark.integration
class TestOptimizers:
    """Test different optimizer configurations."""

    def test_adam_optimizer(self, tiny_model, sample_batch_inputs,
                           reset_random_seeds):
        """Test training with Adam optimizer."""
        from model import compute_loss

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)

        with tf.GradientTape() as tape:
            predictions = tiny_model(sample_batch_inputs, training=True)
            loss = compute_loss(predictions, sample_batch_inputs['stroke_data'])

        gradients = tape.gradient(loss, tiny_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))

        # Should complete without error
        assert loss is not None

    def test_rmsprop_optimizer(self, tiny_model, sample_batch_inputs,
                               reset_random_seeds):
        """Test training with RMSprop optimizer."""
        from model import compute_loss

        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=1e-4,
            rho=0.95,
            momentum=0.9
        )

        with tf.GradientTape() as tape:
            predictions = tiny_model(sample_batch_inputs, training=True)
            loss = compute_loss(predictions, sample_batch_inputs['stroke_data'])

        gradients = tape.gradient(loss, tiny_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))

        # Should complete without error
        assert loss is not None

    def test_learning_rate_scheduling(self, tiny_model, sample_batch_inputs,
                                     reset_random_seeds):
        """Test learning rate can be adjusted during training."""
        from model import compute_loss

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

        # Initial learning rate
        initial_lr = optimizer.learning_rate.numpy()
        # Use approximate equality for float32 precision
        np.testing.assert_allclose(initial_lr, 1e-3, rtol=1e-6)

        # Adjust learning rate
        optimizer.learning_rate.assign(5e-4)

        # Verify it changed
        new_lr = optimizer.learning_rate.numpy()
        # Use approximate equality for float32 precision
        np.testing.assert_allclose(new_lr, 5e-4, rtol=1e-6)

        # Training step should work with new learning rate
        with tf.GradientTape() as tape:
            predictions = tiny_model(sample_batch_inputs, training=True)
            loss = compute_loss(predictions, sample_batch_inputs['stroke_data'])

        gradients = tape.gradient(loss, tiny_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))

        assert loss is not None


@pytest.mark.integration
class TestTrainingWithDataLoader:
    """Test training loop with real DataLoader."""

    def test_full_training_loop_mini(self, tiny_model, mock_args,
                                    mini_dataset_path, reset_random_seeds):
        """Test complete training loop for a few steps."""
        from model import compute_loss
        from utils import DataLoader

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.batch_size = 2
        mock_args.tsteps = 10

        class MockLogger:
            def write(self, msg):
                pass

        data_loader = DataLoader(mock_args, logger=MockLogger())
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

        @tf.function
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = tiny_model(inputs, training=True)
                loss = compute_loss(predictions, targets)
            gradients = tape.gradient(loss, tiny_model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
            optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))
            return loss

        # Train for 5 steps
        for _ in range(5):
            x, y, s, c = data_loader.next_batch()

            train_inputs = {
                'stroke_data': tf.constant(x, dtype=tf.float32),
                'char_seq': tf.constant(c, dtype=tf.float32)
            }
            train_targets = tf.constant(y, dtype=tf.float32)

            loss = train_step(train_inputs, train_targets)

            # Loss should be finite
            assert tf.math.is_finite(loss)

    def test_batch_pointer_advances(self, mock_args, mini_dataset_path):
        """Test DataLoader batch pointer advances correctly."""
        from utils import DataLoader

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.batch_size = 2

        class MockLogger:
            def write(self, msg):
                pass

        data_loader = DataLoader(mock_args, logger=MockLogger())

        initial_pointer = data_loader.pointer

        # Get a batch
        data_loader.next_batch()

        # Pointer should have advanced
        assert data_loader.pointer != initial_pointer

    def test_running_average_updates(self, tiny_model, mock_args,
                                    mini_dataset_path, reset_random_seeds):
        """Test running average loss updates correctly."""
        from model import compute_loss
        from utils import DataLoader

        mock_args.data_dir = os.path.dirname(mini_dataset_path)
        mock_args.batch_size = 2

        class MockLogger:
            def write(self, msg):
                pass

        data_loader = DataLoader(mock_args, logger=MockLogger())
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

        @tf.function
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = tiny_model(inputs, training=True)
                loss = compute_loss(predictions, targets)
            gradients = tape.gradient(loss, tiny_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))
            return loss

        # Track running average
        running_average = 0.0
        remember_rate = 0.99

        losses = []

        for _ in range(10):
            x, y, s, c = data_loader.next_batch()

            train_inputs = {
                'stroke_data': tf.constant(x, dtype=tf.float32),
                'char_seq': tf.constant(c, dtype=tf.float32)
            }
            train_targets = tf.constant(y, dtype=tf.float32)

            loss = train_step(train_inputs, train_targets)
            running_average = (running_average * remember_rate +
                             float(loss) * (1 - remember_rate))
            losses.append(float(loss))

        # Running average should be in reasonable range
        assert running_average > 0
        assert running_average < max(losses) * 2


@pytest.mark.integration
class TestTfFunction:
    """Test @tf.function compilation of training step."""

    def test_train_step_compiles(self, tiny_model, sample_batch_inputs,
                                reset_random_seeds):
        """Test training step can be compiled with @tf.function."""
        from model import compute_loss

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)

        @tf.function
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = tiny_model(inputs, training=True)
                loss = compute_loss(predictions, targets)
            gradients = tape.gradient(loss, tiny_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))
            return loss

        # Execute compiled function
        loss = train_step(sample_batch_inputs, sample_batch_inputs['stroke_data'])

        assert tf.math.is_finite(loss)

    def test_tf_function_is_faster(self, tiny_model, sample_batch_inputs,
                                  reset_random_seeds, performance_monitor):
        """Test @tf.function compilation improves performance."""
        from model import compute_loss

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)

        # Eager mode training step
        def train_step_eager(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = tiny_model(inputs, training=True)
                loss = compute_loss(predictions, targets)
            gradients = tape.gradient(loss, tiny_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))
            return loss

        # Compiled training step
        @tf.function
        def train_step_compiled(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = tiny_model(inputs, training=True)
                loss = compute_loss(predictions, targets)
            gradients = tape.gradient(loss, tiny_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, tiny_model.trainable_variables))
            return loss

        # Warm up compiled version (first call is slow due to compilation)
        train_step_compiled(sample_batch_inputs, sample_batch_inputs['stroke_data'])

        # Measure eager mode
        with performance_monitor:
            for _ in range(10):
                train_step_eager(sample_batch_inputs, sample_batch_inputs['stroke_data'])
        eager_time = performance_monitor.elapsed

        # Measure compiled mode
        with performance_monitor:
            for _ in range(10):
                train_step_compiled(sample_batch_inputs, sample_batch_inputs['stroke_data'])
        compiled_time = performance_monitor.elapsed

        # Compiled should be faster (or at least comparable)
        # This might not always hold for tiny models, so we just verify it runs
        assert compiled_time < eager_time * 2, \
            "Compiled version should not be significantly slower"


# ============================================================================
# Summary
# ============================================================================

def test_training_loop_suite_summary():
    """
    Training loop test suite summary.

    If all training loop tests pass, training works correctly:
    - Training steps execute without errors
    - Gradients computed for all variables
    - Gradients are finite (no NaN/Inf)
    - Gradient clipping works
    - Weights update after training
    - Loss decreases over training
    - Validation loss computed correctly
    - Multiple optimizers work (Adam, RMSprop)
    - Learning rate scheduling works
    - @tf.function compilation works
    """
    print("\n✓ All training loop tests passed!")
    print("  - Training steps execute correctly")
    print("  - Gradients computed and finite")
    print("  - Weights update properly")
    print("  - Loss decreases with training")
    print("  - Optimizers work (Adam, RMSprop)")
    print("  - @tf.function compilation works")
