"""
Scribe Web UI - Flask Application

A simple web interface for generating multi-line handwritten text using Scribe.

Features:
- Multi-line text input
- 13 pre-trained handwriting styles
- Bias control for neatness/messiness
- SVG and PNG output formats
- Download generated handwriting

Run with:
    cd scribe_web
    python app.py

Then visit: http://localhost:5000
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import Scribe modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, send_file
import uuid

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'scribe-web-dev-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# Output directory for generated files
OUTPUTS_DIR = Path(__file__).parent / 'outputs'
OUTPUTS_DIR.mkdir(exist_ok=True)

# Global model instance (loaded once at startup)
model = None
model_args = None


def load_scribe_model():
    """
    Load Scribe handwriting model once at startup.

    This caches the model in memory to avoid reloading on every request.
    Loading takes ~2-3 seconds, so we do it once when the app starts.
    """
    global model, model_args

    print("Loading Scribe model...")

    try:
        # Import Scribe modules
        from model import HandwritingModel

        # Load model configuration
        # NOTE: These should match your trained model's parameters
        # Default Scribe settings:
        model_args = {
            'rnn_size': 400,      # Use 400 for style priming support
            'nmixtures': 20,       # Number of Gaussian mixtures
            'kmixtures': 1,        # Number of attention heads
            'alphabet': ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#%&\'()*+,-./:;?[]',
        }

        # Create model instance
        model = HandwritingModel(
            rnn_size=model_args['rnn_size'],
            nmixtures=model_args['nmixtures'],
            kmixtures=model_args['kmixtures'],
            alphabet=model_args['alphabet']
        )

        # Load checkpoint
        checkpoint_path = Path(__file__).parent.parent / 'saved' / 'model'

        if not checkpoint_path.exists():
            print(f"WARNING: Model checkpoint not found at {checkpoint_path}")
            print("Continuing without loading weights (model will produce random output)")
        else:
            # Build model by doing dummy forward pass
            import tensorflow as tf
            dummy_input_data = tf.zeros([1, 1, 3])
            dummy_char_seq = tf.zeros([1, 200, len(model_args['alphabet']) + 1])
            dummy_init_kappa = tf.zeros([1, model_args['kmixtures'], 1])
            _ = model(dummy_input_data, dummy_char_seq, dummy_init_kappa, training=False)

            # Load weights from checkpoint
            checkpoint = tf.train.Checkpoint(net=model)
            status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
            print(f"✓ Model loaded from {checkpoint_path}")

        print("✓ Scribe model ready!")

    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("The web UI will start but generation will fail.")
        model = None


@app.before_first_request
def initialize():
    """Initialize application before first request."""
    load_scribe_model()


# Import routes after app initialization
from routes import *


if __name__ == '__main__':
    # Run Flask development server
    print("="*60)
    print("Scribe Web UI")
    print("="*60)
    print()
    print("Starting server...")
    print("Visit: http://localhost:5000")
    print()
    print("Press Ctrl+C to stop")
    print("="*60)

    app.run(
        host='0.0.0.0',  # Allow external connections (use 127.0.0.1 for local only)
        port=5000,
        debug=True,       # Enable debug mode for development
        use_reloader=False  # Disable reloader to avoid double model loading
    )
