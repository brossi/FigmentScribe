"""
Training script for TensorFlow 2.x handwriting synthesis model.

This replaces the session-based training in run.py with eager execution
and GradientTape-based training.

Usage:
    python train.py --rnn_size 400 --nmixtures 20 --nepochs 250
"""

import tensorflow as tf
from tensorflow import keras
import argparse
import time
import os
from pathlib import Path

from model import HandwritingModel, compute_loss
from utils import DataLoader, Logger


def train(args):
    """Main training loop for TensorFlow 2.x"""

    # Setup logging
    logger = Logger(args)
    logger.write("\nTRAINING MODE (TensorFlow 2.x)...")
    logger.write(f"{args}\n")

    # Load data
    logger.write("Loading data...")
    data_loader = DataLoader(args, logger=logger)

    # Build model
    logger.write("Building model...")
    model = HandwritingModel(args)

    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    elif args.optimizer == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(
            learning_rate=args.learning_rate,
            rho=args.decay,
            momentum=args.momentum
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Setup checkpointing
    checkpoint_dir = Path(args.save_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=str(checkpoint_dir),
        max_to_keep=5
    )

    # Attempt to restore from checkpoint
    global_step = 0
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        # Try to extract step number from checkpoint name
        try:
            global_step = int(checkpoint_manager.latest_checkpoint.split('-')[-1])
        except:
            global_step = 0
        logger.write(f"Restored from {checkpoint_manager.latest_checkpoint}")
        logger.write(f"Resuming from step {global_step}")
    else:
        logger.write("Starting new training session")

    # Get validation data
    v_x, v_y, v_s, v_c = data_loader.validation_data()
    validation_inputs = {
        'stroke_data': tf.constant(v_x, dtype=tf.float32),
        'char_seq': tf.constant(v_c, dtype=tf.float32)
    }
    validation_targets = tf.constant(v_y, dtype=tf.float32)

    # Training loop
    logger.write("Training...")
    running_average = 0.0
    remember_rate = 0.99

    @tf.function
    def train_step(inputs, targets):
        """Single training step with gradient computation"""
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = compute_loss(predictions, targets)

        gradients = tape.gradient(loss, model.trainable_variables)

        # Gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, args.grad_clip)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Calculate starting epoch based on global_step
    start_epoch = global_step // args.nbatches

    for epoch in range(start_epoch, args.nepochs):
        # Learning rate decay
        lr = args.learning_rate * (args.lr_decay ** epoch)
        optimizer.learning_rate.assign(lr)
        logger.write(f"\nEpoch {epoch + 1}/{args.nepochs}, learning rate: {lr:.6f}")

        for batch in range(args.nbatches):
            i = epoch * args.nbatches + batch

            # Skip batches if resuming from checkpoint
            if i < global_step:
                continue

            # Save checkpoint
            if i % args.save_every == 0 and i > 0:
                checkpoint_manager.save(checkpoint_number=i)
                logger.write(f'Step {i}: SAVED MODEL')

            # Get batch
            start_time = time.time()
            x, y, s, c = data_loader.next_batch()

            # Prepare inputs
            train_inputs = {
                'stroke_data': tf.constant(x, dtype=tf.float32),
                'char_seq': tf.constant(c, dtype=tf.float32)
            }
            train_targets = tf.constant(y, dtype=tf.float32)

            # Train step
            train_loss = train_step(train_inputs, train_targets)

            # Validation loss
            valid_predictions = model(validation_inputs, training=False)
            valid_loss = compute_loss(valid_predictions, validation_targets)

            # Update running average
            running_average = (running_average * remember_rate +
                             float(train_loss) * (1 - remember_rate))

            end_time = time.time()

            # Log progress
            if i % 10 == 0:
                logger.write(
                    f"{i}/{args.nepochs * args.nbatches}, "
                    f"loss = {train_loss:.3f}, "
                    f"regloss = {running_average:.5f}, "
                    f"valid_loss = {valid_loss:.3f}, "
                    f"time = {end_time - start_time:.3f}s"
                )

    # Final save
    final_step = args.nepochs * args.nbatches
    checkpoint_manager.save(checkpoint_number=final_step)
    logger.write(f"\nTraining complete! Final model saved at step {final_step}")


def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(
        description='Train handwriting synthesis model (TensorFlow 2.x)'
    )

    # Model params
    parser.add_argument('--rnn_size', type=int, default=100,
                       help='Size of LSTM hidden state')
    parser.add_argument('--tsteps', type=int, default=150,
                       help='Sequence length for training')
    parser.add_argument('--nmixtures', type=int, default=8,
                       help='Number of Gaussian mixtures in MDN')
    parser.add_argument('--kmixtures', type=int, default=1,
                       help='Number of Gaussian mixtures in attention')
    parser.add_argument('--alphabet', type=str,
                       default=' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#%&\'()*+,-./:;?[]',
                       help='Alphabet for character encoding (83 chars: letters, numerals, punctuation from IAM dataset)')
    parser.add_argument('--tsteps_per_ascii', type=int, default=25,
                       help='Approximate pen points per character')

    # Training params
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Minibatch size')
    parser.add_argument('--nbatches', type=int, default=500,
                       help='Number of batches per epoch')
    parser.add_argument('--nepochs', type=int, default=250,
                       help='Number of epochs')
    parser.add_argument('--dropout', type=float, default=0.85,
                       help='Dropout keep probability')
    parser.add_argument('--grad_clip', type=float, default=10.0,
                       help='Gradient clipping threshold')
    parser.add_argument('--optimizer', type=str, default='rmsprop',
                       choices=['adam', 'rmsprop'],
                       help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=1.0,
                       help='Learning rate decay per epoch')
    parser.add_argument('--decay', type=float, default=0.95,
                       help='RMSprop decay parameter')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='RMSprop momentum parameter')

    # I/O params
    parser.add_argument('--data_scale', type=int, default=50,
                       help='Factor to scale data by')
    parser.add_argument('--log_dir', type=str, default='./logs/',
                       help='Directory for logs')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--save_path', type=str, default='saved/model',
                       help='Path to save checkpoints')
    parser.add_argument('--save_every', type=int, default=500,
                       help='Save checkpoint every N steps')

    args = parser.parse_args()
    args.train = True  # Training mode flag

    # Print configuration
    print("\n" + "=" * 70)
    print("TensorFlow 2.x Handwriting Synthesis - Training")
    print("=" * 70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("=" * 70 + "\n")

    # Start training
    train(args)


if __name__ == '__main__':
    main()
