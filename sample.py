"""
Sampling script for TensorFlow 2.x handwriting synthesis model.

This replaces the session-based sampling in run.py with eager execution.

Usage:
    python sample_tf2.py --text "Hello World" --bias 1.5
    python sample_tf2.py  # Uses default test strings
"""

import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
import os

from model import HandwritingModel
from utils import Logger
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def sample_gaussian2d(mu1, mu2, s1, s2, rho):
    """Sample from a 2D Gaussian distribution"""
    mean = [mu1, mu2]
    cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def to_one_hot(s, max_len, alphabet):
    """Convert string to one-hot encoding"""
    s = s[:3000] if len(s) > 3000 else s  # Clip super-long strings
    seq = [alphabet.find(char) + 1 for char in s]

    if len(seq) >= max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [0] * (max_len - len(seq))

    one_hot = np.zeros((max_len, len(alphabet) + 1))
    one_hot[np.arange(max_len), seq] = 1
    return one_hot


def sample(text, model, args):
    """
    Generate handwriting from text.

    Args:
        text: Input string to generate handwriting for
        model: HandwritingModel instance
        args: Arguments object with hyperparameters

    Returns:
        strokes: Generated stroke sequence [n_points, 6]
        phis: Attention weights over time
        kappas: Attention position over time
    """
    # Prepare inputs
    char_seq = to_one_hot(text, len(text), args.alphabet)
    char_seq = np.expand_dims(char_seq, 0)  # [1, text_len, alphabet_size]

    # Initial stroke (pen at origin, lifted)
    prev_x = np.array([[[0, 0, 1]]], dtype=np.float32)

    # Initialize states
    kappa = np.zeros((1, args.kmixtures, 1), dtype=np.float32)
    states = None

    # Storage for generated strokes
    strokes = []
    phis = []
    kappas = []

    # Generation loop
    for i in range(args.tsteps):
        # Prepare inputs
        model_inputs = {
            'stroke_data': tf.constant(prev_x, dtype=tf.float32),
            'char_seq': tf.constant(char_seq, dtype=tf.float32),
            'kappa': tf.constant(kappa, dtype=tf.float32)
        }

        if states is not None:
            model_inputs['states'] = states

        # Forward pass
        predictions = model(model_inputs, training=False)

        # Apply bias to sigma (controls randomness)
        sigma1 = np.exp(predictions['sigma1_hat'].numpy() - args.bias)
        sigma2 = np.exp(predictions['sigma2_hat'].numpy() - args.bias)

        # Apply bias to pi
        pi_hat = predictions['pi_hat'].numpy() * (1 + args.bias)
        pi = np.exp(pi_hat[0, 0]) / np.sum(np.exp(pi_hat[0, 0]))

        # Sample from mixture
        idx = np.random.choice(len(pi), p=pi)

        mu1 = predictions['mu1'].numpy()[0, 0, idx]
        mu2 = predictions['mu2'].numpy()[0, 0, idx]
        rho = predictions['rho'].numpy()[0, 0, idx]
        eos = predictions['eos'].numpy()[0, 0, 0]

        # Sample point
        x1, x2 = sample_gaussian2d(mu1, mu2, sigma1[0, 0, idx],
                                   sigma2[0, 0, idx], rho)
        eos = 1 if eos > args.eos_threshold else 0

        # Store results
        strokes.append([mu1, mu2, sigma1[0, 0, idx], sigma2[0, 0, idx], rho, eos])
        phis.append(predictions['phi'][0].numpy()[0, 0])
        kappas.append(predictions['kappa'].numpy()[0])

        # Update for next iteration
        prev_x[0, 0] = np.array([x1, x2, eos], dtype=np.float32)
        kappa = predictions['kappa'].numpy()
        states = predictions['states']

        # Check if finished (attention has read past end of text)
        if kappa[0, 0, 0] > len(text) + 1:
            break

    strokes = np.vstack(strokes)
    phis = np.vstack(phis)
    kappas = np.vstack(kappas)

    # Convert from deltas to absolute positions
    strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)

    return strokes, phis, kappas


def line_plot(strokes, title, figsize=(20, 2), save_path=None):
    """
    Plot handwriting strokes.

    Args:
        strokes: Stroke array [n_points, ...]
        title: Plot title
        figsize: Figure size tuple
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=figsize)
    eos_preds = np.where(strokes[:, -1] == 1)
    eos_preds = [0] + list(eos_preds[0]) + [-1]

    for i in range(len(eos_preds) - 1):
        start = eos_preds[i] + 1
        stop = eos_preds[i + 1]
        plt.plot(strokes[start:stop, 0], strokes[start:stop, 1],
                'b-', linewidth=2.0)

    plt.title(title, fontsize=20)
    plt.gca().invert_yaxis()
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Parse arguments and generate samples"""
    parser = argparse.ArgumentParser(
        description='Generate handwriting samples (TensorFlow 2.x)'
    )

    # Model params (must match training)
    parser.add_argument('--rnn_size', type=int, default=100,
                       help='Size of LSTM hidden state')
    parser.add_argument('--tsteps', type=int, default=700,
                       help='Maximum length of generated sequence')
    parser.add_argument('--nmixtures', type=int, default=8,
                       help='Number of Gaussian mixtures in MDN')
    parser.add_argument('--kmixtures', type=int, default=1,
                       help='Number of Gaussian mixtures in attention')
    parser.add_argument('--alphabet', type=str,
                       default=' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
                       help='Alphabet for character encoding')
    parser.add_argument('--tsteps_per_ascii', type=int, default=25,
                       help='Approximate pen points per character')

    # Sampling params
    parser.add_argument('--text', type=str, default='',
                       help='Text to generate (empty for default test strings)')
    parser.add_argument('--bias', type=float, default=1.0,
                       help='Bias for sampling (higher = neater, lower = messier)')
    parser.add_argument('--eos_threshold', type=float, default=0.35,
                       help='Threshold for end-of-stroke probability')
    parser.add_argument('--style', type=int, default=-1,
                       help='Style to use (-1 for random)')

    # I/O params
    parser.add_argument('--save_path', type=str, default='saved_tf2/model',
                       help='Path to saved model checkpoint')
    parser.add_argument('--log_dir', type=str, default='./logs/',
                       help='Directory for logs and figures')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')

    args = parser.parse_args()
    args.train = False  # Sampling mode flag

    # Print configuration
    print("\n" + "=" * 70)
    print("TensorFlow 2.x Handwriting Synthesis - Sampling")
    print("=" * 70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"Bias: {args.bias} (higher = neater, lower = messier)")
    print("=" * 70 + "\n")

    # Setup logger
    logger = Logger(args)
    logger.write("\nSAMPLING MODE (TensorFlow 2.x)...")

    # Test strings
    if args.text == '':
        strings = [
            'call me ishmael some years ago',
            'A project by Sam Greydanus',
            'You know nothing Jon Snow',
            'The quick brown fox jumps'
        ]
    else:
        strings = [args.text]

    # Build model
    logger.write("Building model...")
    model = HandwritingModel(args)

    # Load checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_dir = Path(args.save_path).parent

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=str(checkpoint_dir),
        max_to_keep=5
    )

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        logger.write(f"Loaded model: {checkpoint_manager.latest_checkpoint}")

        # Create output directory
        output_dir = Path(args.log_dir) / 'figures'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate samples
        for text in strings:
            logger.write(f"\nGenerating: '{text}'")
            strokes, phis, kappas = sample(text, model, args)

            # Save figure
            safe_filename = text[:20].replace(' ', '_').replace('/', '_')
            save_path = output_dir / f"sample-{safe_filename}.png"

            line_plot(strokes, f'"{text}"',
                     figsize=(max(10, len(text) // 2), 2),
                     save_path=str(save_path))
            logger.write(f"Saved to {save_path}")
    else:
        logger.write("ERROR: No saved model found!")
        logger.write(f"Expected checkpoint in: {checkpoint_dir}")
        logger.write("\nPlease train the model first using:")
        logger.write("  python train_tf2.py")


if __name__ == '__main__':
    main()
