"""
Sampling script for TensorFlow 2.x handwriting synthesis model.

This replaces the session-based sampling in run.py with eager execution.

Usage:
    python sample.py --text "Hello World" --bias 1.5
    python sample.py  # Uses default test strings
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
import svg_output


def sample_gaussian2d(mu1, mu2, s1, s2, rho):
    """Sample from a 2D Gaussian distribution"""
    mean = [mu1, mu2]
    cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


# Global cache for loaded styles to avoid redundant computation
# Key: (style_id, rnn_size) -> Value: LSTM states
_STYLE_CACHE = {}


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


def load_style_state(style_id, model, args, max_prime_len=500):
    """
    Load style priming data and extract LSTM states.

    Style priming works by running a sample of handwriting through the model
    to get the final LSTM hidden states, then using those states as the initial
    states for generating new text. This causes the model to continue in the
    same handwriting style.

    IMPORTANT: Styles require rnn_size=400. This function will fail if the model
    was trained with a different rnn_size.

    Args:
        style_id: Style index (0-12)
        model: HandwritingModel instance
        args: Arguments object with hyperparameters
        max_prime_len: Maximum number of strokes to use for priming (default: 500)

    Returns:
        initial_states: LSTM states tuple for priming generation

    Raises:
        ValueError: If model rnn_size is not 400
        FileNotFoundError: If style files not found
        ValueError: If style contains characters not in alphabet
    """
    import os

    # Check cache first (keyed by style_id and rnn_size)
    cache_key = (style_id, model.rnn_size)
    if cache_key in _STYLE_CACHE:
        print(f"    Style {style_id} loaded from cache")
        return _STYLE_CACHE[cache_key]

    # Validate model size - styles are trained with rnn_size=400
    if model.rnn_size != 400:
        raise ValueError(
            f"Style priming requires rnn_size=400, but model has rnn_size={model.rnn_size}. "
            f"Please train or load a model with --rnn_size 400 to use style priming."
        )

    # Validate style_id range
    if not (0 <= style_id <= 12):
        raise ValueError(f"style_id must be between 0 and 12, got {style_id}")

    # Load style data files
    style_dir = os.path.join(args.data_dir, 'styles')
    chars_file = os.path.join(style_dir, f'style-{style_id}-chars.npy')
    strokes_file = os.path.join(style_dir, f'style-{style_id}-strokes.npy')

    if not os.path.exists(chars_file):
        raise FileNotFoundError(
            f"Style characters file not found: {chars_file}\n"
            f"Ensure style files are present in {style_dir}"
        )
    if not os.path.exists(strokes_file):
        raise FileNotFoundError(
            f"Style strokes file not found: {strokes_file}\n"
            f"Ensure style files are present in {style_dir}"
        )

    # Load and validate style data
    try:
        style_chars = np.load(chars_file, allow_pickle=True)
        style_strokes = np.load(strokes_file)
    except Exception as e:
        raise IOError(f"Error loading style {style_id} files: {e}")

    # Decode chars (stored as bytes in numpy scalar)
    try:
        if isinstance(style_chars, np.ndarray) and style_chars.dtype.kind == 'S':
            style_text = style_chars.item().decode('utf-8')
        else:
            style_text = str(style_chars)
    except Exception as e:
        raise ValueError(f"Error decoding style {style_id} characters: {e}")

    # Validate strokes data
    if len(style_strokes) == 0:
        raise ValueError(f"Style {style_id} has empty strokes array")
    if style_strokes.shape[1] != 3:
        raise ValueError(
            f"Style {style_id} strokes have wrong shape: {style_strokes.shape}. "
            f"Expected (n_points, 3)"
        )

    # Validate all characters are in alphabet
    valid_chars = set(args.alphabet)
    invalid_chars = set(style_text) - valid_chars
    if invalid_chars:
        raise ValueError(
            f"Style {style_id} contains characters not in alphabet: {invalid_chars}\n"
            f"Style text: '{style_text}'\n"
            f"Valid alphabet: '{args.alphabet}'"
        )

    print(f"    Loading style {style_id}: '{style_text}' ({len(style_strokes)} points)")

    # Prepare one-hot encoding for style text
    char_seq = to_one_hot(style_text, len(style_text), args.alphabet)
    char_seq = np.expand_dims(char_seq, 0)  # [1, text_len, alphabet_size]

    # Initial stroke (pen at origin, lifted)
    prev_x = np.array([[[0, 0, 1]]], dtype=np.float32)

    # Initialize states
    kappa = np.zeros((1, args.kmixtures, 1), dtype=np.float32)
    states = None

    # Run style strokes through model to get final states
    # We don't need the output, just the final LSTM states
    actual_prime_len = min(len(style_strokes), max_prime_len)

    for i in range(actual_prime_len):
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

        # Update for next iteration (use actual style stroke, not sampled)
        if i < len(style_strokes):
            prev_x[0, 0] = style_strokes[i].astype(np.float32)

        kappa = predictions['kappa'].numpy()
        states = predictions['states']

    print(f"    Style {style_id} primed with {actual_prime_len} points")

    # Cache the result
    _STYLE_CACHE[cache_key] = states

    return states


def sample(text, model, args, bias=None, initial_states=None):
    """
    Generate handwriting from text.

    Args:
        text: Input string to generate handwriting for
        model: HandwritingModel instance
        args: Arguments object with hyperparameters
        bias: Sampling bias (overrides args.bias if provided)
        initial_states: Initial LSTM states for style priming (optional)
                       Use load_style_state() to get states from a style file

    Returns:
        strokes: Generated stroke sequence [n_points, 6]
        phis: Attention weights over time
        kappas: Attention position over time
    """
    # Use provided bias or default to args.bias
    if bias is None:
        bias = args.bias
    # Prepare inputs
    char_seq = to_one_hot(text, len(text), args.alphabet)
    char_seq = np.expand_dims(char_seq, 0)  # [1, text_len, alphabet_size]

    # Initial stroke (pen at origin, lifted)
    prev_x = np.array([[[0, 0, 1]]], dtype=np.float32)

    # Initialize states (use style-primed states if provided)
    kappa = np.zeros((1, args.kmixtures, 1), dtype=np.float32)
    states = initial_states  # Will be None for non-primed generation

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
        sigma1 = np.exp(predictions['sigma1_hat'].numpy() - bias)
        sigma2 = np.exp(predictions['sigma2_hat'].numpy() - bias)

        # Apply bias to pi
        pi_hat = predictions['pi_hat'].numpy() * (1 + bias)
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


def sample_multiline(lines, model, args, biases=None, styles=None):
    """
    Generate handwriting for multiple lines of text.

    Args:
        lines: List of text strings (one per line)
        model: HandwritingModel instance
        args: Arguments object with hyperparameters
        biases: List of bias values (one per line, optional)
        styles: List of style IDs (0-12, one per line, optional)

    Returns:
        all_strokes: List of stroke arrays, one per line
        all_phis: List of attention weights, one per line
        all_kappas: List of attention positions, one per line
    """
    # Default biases if not provided
    if biases is None:
        biases = [args.bias] * len(lines)

    # Ensure biases list matches lines length
    if len(biases) != len(lines):
        raise ValueError(f"Number of biases ({len(biases)}) must match number of lines ({len(lines)})")

    # Validate styles if provided
    if styles is not None:
        if len(styles) != len(lines):
            raise ValueError(f"Number of styles ({len(styles)}) must match number of lines ({len(lines)})")
        for style_id in styles:
            if not (0 <= style_id <= 12):
                raise ValueError(f"Style ID {style_id} out of range (must be 0-12)")

    all_strokes = []
    all_phis = []
    all_kappas = []

    for i, (line, bias) in enumerate(zip(lines, biases)):
        # Load style states if style specified for this line
        initial_states = None
        style_info = ""
        if styles is not None:
            style_id = styles[i]
            initial_states = load_style_state(style_id, model, args)
            style_info = f", style={style_id}"

        print(f"  Generating line {i+1}/{len(lines)}: '{line}' (bias={bias:.2f}{style_info})")

        # Generate line with specific bias and optional style
        strokes, phis, kappas = sample(line, model, args, bias=bias, initial_states=initial_states)

        all_strokes.append(strokes)
        all_phis.append(phis)
        all_kappas.append(kappas)

    return all_strokes, all_phis, all_kappas


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
                       default=' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#%&\'()*+,-./:;?[]',
                       help='Alphabet for character encoding (83 chars: letters, numerals, punctuation from IAM dataset)')
    parser.add_argument('--tsteps_per_ascii', type=int, default=25,
                       help='Approximate pen points per character')

    # Sampling params
    parser.add_argument('--text', type=str, default='',
                       help='Text to generate (empty for default test strings)')
    parser.add_argument('--lines', nargs='+', type=str,
                       help='Multiple lines of text (overrides --text)')
    parser.add_argument('--bias', type=float, default=1.0,
                       help='Bias for sampling (higher = neater, lower = messier)')
    parser.add_argument('--biases', nargs='+', type=float,
                       help='Per-line bias values (one per line, overrides --bias)')
    parser.add_argument('--styles', nargs='+', type=int,
                       help='Style IDs (0-12) for each line (overrides --style)')
    parser.add_argument('--eos_threshold', type=float, default=0.35,
                       help='Threshold for end-of-stroke probability')
    parser.add_argument('--style', type=int, default=-1,
                       help='Style to use (-1 for random)')

    # I/O params
    parser.add_argument('--save_path', type=str, default='saved/model',
                       help='Path to saved model checkpoint')
    parser.add_argument('--log_dir', type=str, default='./logs/',
                       help='Directory for logs and figures')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--format', type=str, choices=['png', 'svg'], default='png',
                       help='Output format: png (matplotlib) or svg (pen plotter)')

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

    # Determine text input (priority: --lines > --text > default)
    if args.lines:
        # Multi-line mode
        strings = args.lines
        is_multiline = True
        biases = args.biases if args.biases else [args.bias] * len(strings)
        styles = args.styles if args.styles else None

        # Validate biases length
        if len(biases) != len(strings):
            raise ValueError(f"Number of biases ({len(biases)}) must match number of lines ({len(strings)})")

        # Validate styles if provided
        if styles is not None and len(styles) != len(strings):
            raise ValueError(f"Number of styles ({len(styles)}) must match number of lines ({len(strings)})")

    elif args.text != '':
        # Single line from --text
        strings = [args.text]
        is_multiline = False
        biases = [args.bias]
    else:
        # Default test strings (single-line mode for backwards compatibility)
        strings = [
            'call me ishmael some years ago',
            'A project by Sam Greydanus',
            'You know nothing Jon Snow',
            'The quick brown fox jumps'
        ]
        is_multiline = False
        biases = [args.bias] * len(strings)

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
        if is_multiline:
            # Multi-line generation
            logger.write(f"\nGenerating {len(strings)} lines:")
            for i, line in enumerate(strings):
                style_info = f", style={styles[i]}" if styles else ""
                logger.write(f"  Line {i+1}: '{line}' (bias={biases[i]:.2f}{style_info})")

            all_strokes, all_phis, all_kappas = sample_multiline(strings, model, args, biases=biases, styles=styles)

            # Save output
            if args.format == 'svg':
                # SVG output for pen plotter
                safe_filename = "_".join(strings[0][:15].split()).replace('/', '_')
                save_path = output_dir / f"multiline-{safe_filename}.svg"

                # Convert strokes to offsets (SVG expects deltas, not cumulative)
                all_offsets = []
                for strokes in all_strokes:
                    offsets = strokes.copy()
                    offsets[:, :2] = np.diff(np.vstack([[[0, 0]], strokes[:, :2]]), axis=0)
                    all_offsets.append(offsets)

                svg_output.save_as_svg(all_offsets, strings, str(save_path))
                logger.write(f"\nSaved to {save_path}")
            else:
                # PNG output (save each line separately for now)
                for i, (text, strokes) in enumerate(zip(strings, all_strokes)):
                    safe_filename = text[:20].replace(' ', '_').replace('/', '_')
                    save_path = output_dir / f"multiline-{i+1:02d}-{safe_filename}.png"

                    line_plot(strokes, f'Line {i+1}: "{text}"',
                             figsize=(max(10, len(text) // 2), 2),
                             save_path=str(save_path))
                    logger.write(f"  Saved line {i+1} to {save_path}")
        else:
            # Single-line generation (original behavior)
            for text in strings:
                logger.write(f"\nGenerating: '{text}'")
                strokes, phis, kappas = sample(text, model, args)

                # Save figure
                safe_filename = text[:20].replace(' ', '_').replace('/', '_')
                file_ext = 'svg' if args.format == 'svg' else 'png'
                save_path = output_dir / f"sample-{safe_filename}.{file_ext}"

                if args.format == 'svg':
                    # Convert to offsets for SVG
                    offsets = strokes.copy()
                    offsets[:, :2] = np.diff(np.vstack([[[0, 0]], strokes[:, :2]]), axis=0)
                    svg_output.save_as_svg([offsets], [text], str(save_path))
                else:
                    # PNG via matplotlib
                    line_plot(strokes, f'"{text}"',
                             figsize=(max(10, len(text) // 2), 2),
                             save_path=str(save_path))

                logger.write(f"Saved to {save_path}")
    else:
        logger.write("ERROR: No saved model found!")
        logger.write(f"Expected checkpoint in: {checkpoint_dir}")
        logger.write("\nPlease train the model first using:")
        logger.write("  python train.py")


if __name__ == '__main__':
    main()
