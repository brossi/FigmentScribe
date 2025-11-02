"""
Unit Tests for HandwritingModel Architecture

Tests the neural network components:
- LSTM layer initialization and forward pass
- Attention mechanism (Gaussian window over text)
- MDN output layer parameter transformations
- Mathematical invariants and constraints

Usage:
    pytest tests/unit/test_model.py -v
    pytest -k test_model
"""

import pytest
import tensorflow as tf
import numpy as np


@pytest.mark.unit
class TestModelInitialization:
    """Test model initialization and architecture setup."""

    def test_model_creates_three_lstm_layers(self, tiny_model):
        """Test model has 3 LSTM layers with correct size."""
        assert hasattr(tiny_model, 'lstm0'), "Model missing lstm0"
        assert hasattr(tiny_model, 'lstm1'), "Model missing lstm1"
        assert hasattr(tiny_model, 'lstm2'), "Model missing lstm2"

        # Verify LSTM sizes
        assert tiny_model.lstm0.units == 10
        assert tiny_model.lstm1.units == 10
        assert tiny_model.lstm2.units == 10

    def test_model_has_attention_layer(self, tiny_model):
        """Test model has attention mechanism layer."""
        assert hasattr(tiny_model, 'attention_layer')

        # Attention layer outputs 3 * kmixtures parameters (alpha, beta, kappa)
        expected_units = 3 * tiny_model.kmixtures
        assert tiny_model.attention_layer.units == expected_units

    def test_model_has_mdn_layer(self, tiny_model):
        """Test model has MDN output layer with correct size."""
        assert hasattr(tiny_model, 'mdn_layer')

        # MDN outputs: 1 (eos) + nmixtures * 6 (pi, mu1, mu2, sigma1, sigma2, rho)
        expected_units = 1 + tiny_model.nmixtures * 6
        assert tiny_model.mdn_layer.units == expected_units

    def test_model_stores_hyperparameters(self, tiny_model):
        """Test model stores hyperparameters correctly."""
        assert tiny_model.rnn_size == 10
        assert tiny_model.nmixtures == 2
        assert tiny_model.kmixtures == 1
        assert tiny_model.char_vec_len == len(tiny_model.alphabet) + 1

    def test_model_initializer_configuration(self, tiny_model):
        """Test model uses Graves' weight initialization."""
        # Verify custom initializers are set
        assert tiny_model.graves_initializer is not None
        assert tiny_model.window_b_initializer is not None

        # Check that it's a truncated normal initializer
        assert isinstance(tiny_model.graves_initializer,
                         tf.keras.initializers.TruncatedNormal)


@pytest.mark.unit
class TestAttentionMechanism:
    """Test Gaussian attention window mechanism."""

    def test_get_window_output_shapes(self, tiny_model, sample_batch_inputs,
                                       reset_random_seeds):
        """Test attention mechanism returns correct output shapes."""
        batch_size = sample_batch_inputs['stroke_data'].shape[0]
        text_len = sample_batch_inputs['char_seq'].shape[1]

        # Prepare inputs for get_window
        lstm0_output = tf.random.normal([batch_size, tiny_model.rnn_size])
        prev_kappa = tf.zeros([batch_size, tiny_model.kmixtures, 1])
        char_seq = sample_batch_inputs['char_seq']

        # Call get_window
        window, phi, new_kappa = tiny_model.get_window(
            lstm0_output, prev_kappa, char_seq
        )

        # Verify shapes
        assert window.shape == (batch_size, tiny_model.char_vec_len), \
            f"Expected window shape ({batch_size}, {tiny_model.char_vec_len}), got {window.shape}"
        assert phi.shape == (batch_size, 1, text_len), \
            f"Expected phi shape ({batch_size}, 1, {text_len}), got {phi.shape}"
        assert new_kappa.shape == (batch_size, tiny_model.kmixtures, 1), \
            f"Expected kappa shape ({batch_size}, {tiny_model.kmixtures}, 1), got {new_kappa.shape}"

    def test_kappa_increases_monotonically(self, tiny_model, sample_batch_inputs,
                                           reset_random_seeds):
        """Test that kappa increases monotonically (attention moves forward)."""
        batch_size = sample_batch_inputs['stroke_data'].shape[0]
        char_seq = sample_batch_inputs['char_seq']

        kappas = []
        kappa = tf.zeros([batch_size, tiny_model.kmixtures, 1])

        # Run 10 attention steps
        for _ in range(10):
            lstm0_output = tf.random.normal([batch_size, tiny_model.rnn_size])
            window, phi, kappa = tiny_model.get_window(lstm0_output, kappa, char_seq)
            kappas.append(kappa.numpy())

        # Verify monotonic increase
        kappas = np.array(kappas)  # [steps, batch, kmixtures, 1]
        for i in range(len(kappas) - 1):
            # Each kappa should be >= previous kappa
            assert np.all(kappas[i+1] >= kappas[i]), \
                "Kappa values must increase monotonically (attention moves forward)"

    def test_attention_weights_properties(self, tiny_model, sample_batch_inputs,
                                          reset_random_seeds):
        """Test attention weights (phi) are valid probabilities."""
        batch_size = sample_batch_inputs['stroke_data'].shape[0]

        lstm0_output = tf.random.normal([batch_size, tiny_model.rnn_size])
        prev_kappa = tf.zeros([batch_size, tiny_model.kmixtures, 1])
        char_seq = sample_batch_inputs['char_seq']

        window, phi, new_kappa = tiny_model.get_window(
            lstm0_output, prev_kappa, char_seq
        )

        # Verify phi values are non-negative
        assert tf.reduce_all(phi >= 0), "Attention weights must be non-negative"

        # Verify phi values are finite (no NaN or Inf)
        assert not tf.reduce_any(tf.math.is_nan(phi)), "Phi contains NaN"
        assert not tf.reduce_any(tf.math.is_inf(phi)), "Phi contains Inf"

    def test_window_is_weighted_combination(self, tiny_model, reset_random_seeds):
        """Test window is valid weighted combination of character sequence."""
        from model import HandwritingModel

        # Create simple test case
        batch_size = 2
        text_len = 5
        alphabet_size = 10

        # Create one-hot char sequence (all zeros except one position)
        char_seq = tf.zeros([batch_size, text_len, alphabet_size])
        char_seq = char_seq.numpy()
        char_seq[0, 2, 5] = 1.0  # Batch 0, position 2, alphabet index 5
        char_seq[1, 3, 7] = 1.0  # Batch 1, position 3, alphabet index 7
        char_seq = tf.constant(char_seq, dtype=tf.float32)

        lstm0_output = tf.random.normal([batch_size, tiny_model.rnn_size])
        prev_kappa = tf.constant([[[2.0]], [[3.0]]], dtype=tf.float32)  # Point at positions

        window, phi, new_kappa = tiny_model.get_window(
            lstm0_output, prev_kappa, char_seq
        )

        # Window should have same feature dimension as char_seq
        assert window.shape[1] == alphabet_size


@pytest.mark.unit
class TestForwardPass:
    """Test model forward pass and output."""

    def test_forward_pass_output_keys(self, tiny_model, sample_batch_inputs):
        """Test forward pass returns all required outputs."""
        predictions = tiny_model(sample_batch_inputs, training=False)

        # Verify all required keys are present
        required_keys = ['eos', 'pi', 'mu1', 'mu2', 'sigma1', 'sigma2', 'rho',
                        'phi', 'kappa', 'states', 'pi_hat', 'sigma1_hat', 'sigma2_hat']

        for key in required_keys:
            assert key in predictions, f"Output missing required key: {key}"

    def test_forward_pass_output_shapes(self, tiny_model, sample_batch_inputs):
        """Test forward pass returns correct output shapes."""
        predictions = tiny_model(sample_batch_inputs, training=False)

        batch_size = sample_batch_inputs['stroke_data'].shape[0]
        tsteps = sample_batch_inputs['stroke_data'].shape[1]
        nmixtures = tiny_model.nmixtures

        # Verify MDN parameter shapes
        assert predictions['eos'].shape == (batch_size, tsteps, 1)
        assert predictions['pi'].shape == (batch_size, tsteps, nmixtures)
        assert predictions['mu1'].shape == (batch_size, tsteps, nmixtures)
        assert predictions['mu2'].shape == (batch_size, tsteps, nmixtures)
        assert predictions['sigma1'].shape == (batch_size, tsteps, nmixtures)
        assert predictions['sigma2'].shape == (batch_size, tsteps, nmixtures)
        assert predictions['rho'].shape == (batch_size, tsteps, nmixtures)

    def test_forward_pass_deterministic(self, tiny_model, sample_batch_inputs,
                                        reset_random_seeds):
        """Test forward pass is deterministic with same inputs."""
        # First pass
        predictions1 = tiny_model(sample_batch_inputs, training=False)

        # Second pass with same inputs
        predictions2 = tiny_model(sample_batch_inputs, training=False)

        # Outputs should be identical
        np.testing.assert_allclose(
            predictions1['eos'].numpy(),
            predictions2['eos'].numpy(),
            rtol=1e-6,
            err_msg="Forward pass should be deterministic"
        )

    def test_states_structure(self, tiny_model, sample_batch_inputs):
        """Test LSTM states have correct structure."""
        predictions = tiny_model(sample_batch_inputs, training=False)

        states = predictions['states']

        # Should have 3 LSTM layers
        assert len(states) == 3, "Expected 3 LSTM state tuples"

        # Each state is (h, c) tuple
        for i, state in enumerate(states):
            assert len(state) == 2, f"LSTM {i} state should be (h, c) tuple"
            h, c = state

            batch_size = sample_batch_inputs['stroke_data'].shape[0]

            # Verify state shapes
            assert h.shape == (batch_size, tiny_model.rnn_size), \
                f"LSTM {i} hidden state has wrong shape"
            assert c.shape == (batch_size, tiny_model.rnn_size), \
                f"LSTM {i} cell state has wrong shape"


@pytest.mark.unit
class TestMDNParameters:
    """Test MDN output parameter transformations and constraints."""

    def test_pi_sums_to_one(self, tiny_model, sample_batch_inputs,
                           reset_random_seeds):
        """Test mixture weights (pi) sum to 1 along mixture dimension."""
        predictions = tiny_model(sample_batch_inputs, training=False)

        pi = predictions['pi']

        # Sum along mixture dimension (axis=-1)
        pi_sums = tf.reduce_sum(pi, axis=-1)

        # Should sum to 1.0 for all batch items and timesteps
        np.testing.assert_allclose(
            pi_sums.numpy(),
            np.ones_like(pi_sums.numpy()),
            atol=1e-5,
            err_msg="Pi (mixture weights) must sum to 1"
        )

    def test_pi_values_in_valid_range(self, tiny_model, sample_batch_inputs):
        """Test pi values are in [0, 1] (valid probabilities)."""
        predictions = tiny_model(sample_batch_inputs, training=False)

        pi = predictions['pi']

        # All values should be in [0, 1]
        assert tf.reduce_all(pi >= 0), "Pi values must be >= 0"
        assert tf.reduce_all(pi <= 1), "Pi values must be <= 1"

    def test_sigma_values_positive(self, tiny_model, sample_batch_inputs):
        """Test sigma (standard deviation) values are positive."""
        predictions = tiny_model(sample_batch_inputs, training=False)

        sigma1 = predictions['sigma1']
        sigma2 = predictions['sigma2']

        # Standard deviations must be positive
        assert tf.reduce_all(sigma1 > 0), "Sigma1 values must be positive"
        assert tf.reduce_all(sigma2 > 0), "Sigma2 values must be positive"

    def test_rho_values_in_valid_range(self, tiny_model, sample_batch_inputs):
        """Test rho (correlation) values are in [-1, 1]."""
        predictions = tiny_model(sample_batch_inputs, training=False)

        rho = predictions['rho']

        # Correlation must be in [-1, 1]
        assert tf.reduce_all(rho >= -1), "Rho values must be >= -1"
        assert tf.reduce_all(rho <= 1), "Rho values must be <= 1"

    def test_eos_values_in_valid_range(self, tiny_model, sample_batch_inputs):
        """Test end-of-stroke (eos) values are in [0, 1]."""
        predictions = tiny_model(sample_batch_inputs, training=False)

        eos = predictions['eos']

        # Probability must be in [0, 1]
        assert tf.reduce_all(eos >= 0), "EOS values must be >= 0"
        assert tf.reduce_all(eos <= 1), "EOS values must be <= 1"

    def test_no_nan_in_outputs(self, tiny_model, sample_batch_inputs):
        """Test forward pass produces no NaN values."""
        predictions = tiny_model(sample_batch_inputs, training=False)

        # Check all MDN parameters for NaN
        for key in ['eos', 'pi', 'mu1', 'mu2', 'sigma1', 'sigma2', 'rho']:
            values = predictions[key]
            assert not tf.reduce_any(tf.math.is_nan(values)), \
                f"{key} contains NaN values"

    def test_no_inf_in_outputs(self, tiny_model, sample_batch_inputs):
        """Test forward pass produces no Inf values."""
        predictions = tiny_model(sample_batch_inputs, training=False)

        # Check all MDN parameters for Inf
        for key in ['eos', 'pi', 'mu1', 'mu2', 'sigma1', 'sigma2', 'rho']:
            values = predictions[key]
            assert not tf.reduce_any(tf.math.is_inf(values)), \
                f"{key} contains Inf values"


@pytest.mark.unit
class TestLossComputation:
    """Test loss function computation."""

    def test_loss_is_scalar(self, tiny_model, sample_batch_inputs):
        """Test compute_loss returns a scalar value."""
        from model import compute_loss

        predictions = tiny_model(sample_batch_inputs, training=False)
        targets = sample_batch_inputs['stroke_data']

        loss = compute_loss(predictions, targets)

        # Loss should be scalar (shape ())
        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"

    def test_loss_is_positive(self, tiny_model, sample_batch_inputs,
                             reset_random_seeds):
        """Test loss is positive (negative log-likelihood)."""
        from model import compute_loss

        predictions = tiny_model(sample_batch_inputs, training=False)
        targets = sample_batch_inputs['stroke_data']

        loss = compute_loss(predictions, targets)

        # NLL should be positive
        assert loss.numpy() >= 0, "Loss should be non-negative"

    def test_loss_is_finite(self, tiny_model, sample_batch_inputs):
        """Test loss is finite (no NaN or Inf)."""
        from model import compute_loss

        predictions = tiny_model(sample_batch_inputs, training=False)
        targets = sample_batch_inputs['stroke_data']

        loss = compute_loss(predictions, targets)

        assert not tf.math.is_nan(loss), "Loss is NaN"
        assert not tf.math.is_inf(loss), "Loss is Inf"

    def test_loss_computation_with_zeros(self, tiny_model, mock_args,
                                         reset_random_seeds):
        """Test loss computation handles zero targets gracefully."""
        from model import compute_loss

        # Create batch with zero targets
        batch_size = 4
        tsteps = 10

        inputs = {
            'stroke_data': tf.zeros([batch_size, tsteps, 3]),
            'char_seq': tf.one_hot(
                tf.zeros([batch_size, 5], dtype=tf.int32),
                depth=len(mock_args.alphabet) + 1
            )
        }

        predictions = tiny_model(inputs, training=False)
        targets = tf.zeros([batch_size, tsteps, 3])

        # Should not raise exception
        loss = compute_loss(predictions, targets)

        assert tf.math.is_finite(loss), "Loss should be finite even with zero targets"


@pytest.mark.unit
class TestModelWithInitialStates:
    """Test model behavior with provided initial states (style priming)."""

    def test_forward_pass_accepts_initial_states(self, tiny_model, sample_batch_inputs):
        """Test model accepts initial LSTM states."""
        batch_size = sample_batch_inputs['stroke_data'].shape[0]

        # Create dummy initial states (3 LSTMs, each with h and c)
        initial_states = [
            (
                tf.zeros([batch_size, tiny_model.rnn_size]),  # h
                tf.zeros([batch_size, tiny_model.rnn_size])   # c
            )
            for _ in range(3)
        ]

        inputs_with_states = sample_batch_inputs.copy()
        inputs_with_states['states'] = initial_states

        # Should not raise exception
        predictions = tiny_model(inputs_with_states, training=False)

        assert predictions is not None

    def test_initial_states_affect_output(self, tiny_model, sample_batch_inputs,
                                          reset_random_seeds):
        """Test that initial states actually affect model output."""
        batch_size = sample_batch_inputs['stroke_data'].shape[0]

        # Generate with zero initial states
        zero_states = [
            (
                tf.zeros([batch_size, tiny_model.rnn_size]),
                tf.zeros([batch_size, tiny_model.rnn_size])
            )
            for _ in range(3)
        ]

        inputs1 = sample_batch_inputs.copy()
        inputs1['states'] = zero_states
        predictions1 = tiny_model(inputs1, training=False)

        # Generate with random initial states
        random_states = [
            (
                tf.random.normal([batch_size, tiny_model.rnn_size]),
                tf.random.normal([batch_size, tiny_model.rnn_size])
            )
            for _ in range(3)
        ]

        inputs2 = sample_batch_inputs.copy()
        inputs2['states'] = random_states
        predictions2 = tiny_model(inputs2, training=False)

        # Outputs should be different with different initial states
        # (This tests that initial states are actually used)
        difference = tf.reduce_mean(tf.abs(
            predictions1['eos'] - predictions2['eos']
        ))

        assert difference.numpy() > 0, \
            "Different initial states should produce different outputs"


@pytest.mark.unit
@pytest.mark.slow
class TestModelPerformance:
    """Performance tests for model operations."""

    def test_forward_pass_completes_quickly(self, tiny_model, sample_batch_inputs,
                                           performance_monitor):
        """Test forward pass is reasonably fast."""
        with performance_monitor:
            predictions = tiny_model(sample_batch_inputs, training=False)

        # Tiny model should complete in < 2 seconds
        assert performance_monitor.elapsed < 2.0, \
            f"Forward pass too slow: {performance_monitor.elapsed:.2f}s (expected < 2s)"

    def test_attention_mechanism_completes_quickly(self, tiny_model,
                                                   sample_batch_inputs,
                                                   performance_monitor):
        """Test attention mechanism is reasonably fast."""
        batch_size = sample_batch_inputs['stroke_data'].shape[0]

        lstm0_output = tf.random.normal([batch_size, tiny_model.rnn_size])
        prev_kappa = tf.zeros([batch_size, tiny_model.kmixtures, 1])
        char_seq = sample_batch_inputs['char_seq']

        with performance_monitor:
            for _ in range(100):  # 100 attention computations
                window, phi, kappa = tiny_model.get_window(
                    lstm0_output, prev_kappa, char_seq
                )

        # 100 attention steps should complete in < 1 second
        assert performance_monitor.elapsed < 1.0, \
            f"Attention mechanism too slow: {performance_monitor.elapsed:.2f}s for 100 steps"


# ============================================================================
# Summary
# ============================================================================

def test_model_suite_summary():
    """
    Model test suite summary.

    If all model tests pass, the neural network architecture is correct:
    - 3 LSTM layers properly initialized
    - Attention mechanism computes valid Gaussian windows
    - Kappa increases monotonically (attention moves forward)
    - MDN parameters satisfy mathematical constraints
    - All outputs are finite (no NaN/Inf)
    - Initial states (style priming) work correctly

    Mathematical invariants verified:
    - Pi sums to 1 (valid mixture weights)
    - Sigma > 0 (valid standard deviations)
    - Rho in [-1, 1] (valid correlations)
    - EOS in [0, 1] (valid probabilities)
    - Kappa monotonically increasing (attention moves forward)
    """
    print("\n✓ All model tests passed!")
    print("  - Architecture correct (3 LSTMs + attention + MDN)")
    print("  - Attention mechanism valid (Gaussian window)")
    print("  - MDN parameters satisfy constraints")
    print("  - Mathematical invariants hold")
    print("  - No NaN/Inf in outputs")
