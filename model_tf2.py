"""
TensorFlow 2.x implementation of Alex Graves' handwriting synthesis model.

This is a complete rewrite of model.py using:
- Keras Model API instead of low-level TensorFlow
- Eager execution instead of session-based execution
- tf.keras.layers.LSTM instead of tf.contrib.rnn.*

Architecture:
- 3 LSTM layers with custom attention mechanism
- Gaussian Mixture Density Network (MDN) output layer
- Attention mechanism with Gaussian window over input text
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


class HandwritingModel(keras.Model):
    """
    TensorFlow 2.x implementation of Alex Graves' handwriting synthesis model.
    """

    def __init__(self, args):
        super(HandwritingModel, self).__init__()

        # Store hyperparameters
        self.rnn_size = args.rnn_size
        self.nmixtures = args.nmixtures
        self.kmixtures = args.kmixtures
        self.alphabet = args.alphabet
        self.char_vec_len = len(self.alphabet) + 1
        self.tsteps_per_ascii = args.tsteps_per_ascii

        # Initialize weights with Graves' initialization
        self.graves_initializer = tf.keras.initializers.TruncatedNormal(
            mean=0.0, stddev=0.075
        )
        self.window_b_initializer = tf.keras.initializers.TruncatedNormal(
            mean=-3.0, stddev=0.25
        )

        # Build LSTM layers
        self.lstm0 = keras.layers.LSTM(
            args.rnn_size,
            return_sequences=True,
            return_state=True,
            kernel_initializer=self.graves_initializer,
            name='lstm0'
        )
        self.lstm1 = keras.layers.LSTM(
            args.rnn_size,
            return_sequences=True,
            return_state=True,
            kernel_initializer=self.graves_initializer,
            name='lstm1'
        )
        self.lstm2 = keras.layers.LSTM(
            args.rnn_size,
            return_sequences=True,
            return_state=True,
            kernel_initializer=self.graves_initializer,
            name='lstm2'
        )

        # Dropout layers (applied during training only)
        if hasattr(args, 'dropout') and args.dropout < 1.0:
            self.dropout_layer = keras.layers.Dropout(1.0 - args.dropout)
        else:
            self.dropout_layer = None

        # Attention mechanism layer
        self.attention_layer = keras.layers.Dense(
            3 * args.kmixtures,
            kernel_initializer=self.graves_initializer,
            bias_initializer=self.window_b_initializer,
            name='attention_window'
        )

        # MDN output layer
        n_out = 1 + args.nmixtures * 6
        self.mdn_layer = keras.layers.Dense(
            n_out,
            kernel_initializer=self.graves_initializer,
            name='mdn_output'
        )

    def get_window(self, lstm0_output, prev_kappa, char_seq):
        """
        Compute attention window over character sequence.

        Args:
            lstm0_output: Output from first LSTM layer [batch, features]
            prev_kappa: Previous kappa values [batch, kmixtures, 1]
            char_seq: One-hot encoded character sequence [batch, seq_len, alphabet_size]

        Returns:
            window: Weighted sum over character sequence [batch, alphabet_size]
            phi: Attention weights [batch, 1, seq_len]
            new_kappa: Updated kappa values [batch, kmixtures, 1]
        """
        # Get attention parameters
        abk_hats = self.attention_layer(lstm0_output)
        abk = tf.exp(tf.reshape(abk_hats, [-1, 3 * self.kmixtures, 1]))

        alpha, beta, kappa_offset = tf.split(abk, 3, axis=1)
        new_kappa = prev_kappa + kappa_offset

        # Compute phi (attention weights)
        ascii_steps = tf.shape(char_seq)[1]
        u = tf.cast(tf.range(ascii_steps), tf.float32)
        u = tf.reshape(u, [1, 1, -1])  # [1, 1, seq_len]

        kappa_term = tf.square(new_kappa - u)  # [batch, kmixtures, seq_len]
        exp_term = -beta * kappa_term
        phi_k = alpha * tf.exp(exp_term)
        phi = tf.reduce_sum(phi_k, axis=1, keepdims=True)  # [batch, 1, seq_len]

        # Compute window
        window = tf.matmul(phi, char_seq)  # [batch, 1, alphabet_size]
        window = tf.squeeze(window, axis=1)  # [batch, alphabet_size]

        return window, phi, new_kappa

    def call(self, inputs, training=False):
        """
        Forward pass through the model.

        Args:
            inputs: Dictionary with keys:
                - 'stroke_data': [batch, timesteps, 3] stroke inputs
                - 'char_seq': [batch, max_text_len, alphabet_size] one-hot text
                - 'kappa': [batch, kmixtures, 1] initial kappa (optional)
                - 'states': List of tuples of LSTM states (optional, for sampling)
            training: Boolean, whether in training mode

        Returns:
            Dictionary with MDN parameters and attention info
        """
        stroke_data = inputs['stroke_data']
        char_seq = inputs['char_seq']

        # Initialize kappa if not provided
        if 'kappa' in inputs:
            kappa = inputs['kappa']
        else:
            batch_size = tf.shape(stroke_data)[0]
            kappa = tf.zeros([batch_size, self.kmixtures, 1])

        # Get initial states for LSTMs if provided
        initial_states = inputs.get('states', None)

        # Process through first LSTM
        if initial_states is not None and len(initial_states) > 0:
            lstm0_out, h0, c0 = self.lstm0(
                stroke_data,
                initial_state=initial_states[0],
                training=training
            )
        else:
            lstm0_out, h0, c0 = self.lstm0(stroke_data, training=training)

        if self.dropout_layer and training:
            lstm0_out = self.dropout_layer(lstm0_out, training=training)

        # Process each timestep through attention mechanism
        timesteps = tf.shape(stroke_data)[1]

        # Compute windows for all timesteps
        windows = []
        phis = []
        kappas = [kappa]

        # Note: This loop is necessary because attention mechanism depends on previous kappa
        for t in range(timesteps):
            lstm0_t = lstm0_out[:, t, :]
            window, phi, kappa = self.get_window(lstm0_t, kappas[-1], char_seq)
            windows.append(window)
            phis.append(phi)
            kappas.append(kappa)

        windows = tf.stack(windows, axis=1)  # [batch, timesteps, alphabet_size]
        phis = tf.stack(phis, axis=1)  # [batch, timesteps, 1, seq_len]

        # Concatenate LSTM output with window and input
        lstm0_augmented = tf.concat([lstm0_out, windows, stroke_data], axis=-1)

        # Second LSTM
        if initial_states is not None and len(initial_states) > 1:
            lstm1_out, h1, c1 = self.lstm1(
                lstm0_augmented,
                initial_state=initial_states[1],
                training=training
            )
        else:
            lstm1_out, h1, c1 = self.lstm1(lstm0_augmented, training=training)

        if self.dropout_layer and training:
            lstm1_out = self.dropout_layer(lstm1_out, training=training)

        # Third LSTM
        if initial_states is not None and len(initial_states) > 2:
            lstm2_out, h2, c2 = self.lstm2(
                lstm1_out,
                initial_state=initial_states[2],
                training=training
            )
        else:
            lstm2_out, h2, c2 = self.lstm2(lstm1_out, training=training)

        if self.dropout_layer and training:
            lstm2_out = self.dropout_layer(lstm2_out, training=training)

        # MDN output layer
        mdn_params = self.mdn_layer(lstm2_out)

        # Parse MDN parameters
        eos_hat = mdn_params[..., 0:1]
        mdn_rest = mdn_params[..., 1:]
        pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat = tf.split(
            mdn_rest, 6, axis=-1
        )

        # Transform parameters
        eos = tf.sigmoid(-eos_hat)
        pi = tf.nn.softmax(pi_hat, axis=-1)
        sigma1 = tf.exp(sigma1_hat)
        sigma2 = tf.exp(sigma2_hat)
        rho = tf.tanh(rho_hat)

        return {
            'eos': eos,
            'pi': pi,
            'mu1': mu1,
            'mu2': mu2,
            'sigma1': sigma1,
            'sigma2': sigma2,
            'rho': rho,
            'phi': phis,
            'kappa': kappas[-1],
            'states': [(h0, c0), (h1, c1), (h2, c2)],
            # Raw parameters for bias adjustment during sampling
            'pi_hat': pi_hat,
            'sigma1_hat': sigma1_hat,
            'sigma2_hat': sigma2_hat
        }


def compute_loss(predictions, targets):
    """
    Compute loss for handwriting synthesis.

    Args:
        predictions: Output dictionary from model
        targets: [batch, timesteps, 3] target stroke data

    Returns:
        Total loss value
    """
    # Split target data
    x1_data = targets[..., 0:1]
    x2_data = targets[..., 1:2]
    eos_data = targets[..., 2:3]

    # Get predictions
    pi = predictions['pi']
    mu1 = predictions['mu1']
    mu2 = predictions['mu2']
    sigma1 = predictions['sigma1']
    sigma2 = predictions['sigma2']
    rho = predictions['rho']
    eos = predictions['eos']

    # Compute 2D Gaussian
    x_mu1 = x1_data - mu1
    x_mu2 = x2_data - mu2

    Z = tf.square(x_mu1 / sigma1) + \
        tf.square(x_mu2 / sigma2) - \
        2 * rho * x_mu1 * x_mu2 / (sigma1 * sigma2)

    rho_square_term = 1 - tf.square(rho)
    power_e = tf.exp(-Z / (2 * rho_square_term))
    regularize_term = 2 * np.pi * sigma1 * sigma2 * tf.sqrt(rho_square_term)
    gaussian = power_e / regularize_term

    # Mixture loss
    term1 = pi * gaussian
    term1 = tf.reduce_sum(term1, axis=-1, keepdims=True)
    term1 = -tf.math.log(tf.maximum(term1, 1e-20))

    # End-of-stroke loss (binary cross-entropy)
    term2 = eos * eos_data + (1 - eos) * (1 - eos_data)
    term2 = -tf.math.log(tf.maximum(term2, 1e-20))

    # Total loss
    return tf.reduce_mean(term1 + term2)
