# TensorFlow 2.x Implementation - Quick Start

This document explains how to use the newly created TensorFlow 2.x implementation of the Scribe handwriting synthesis model.

## ✅ Migration Status

- **Phase 0**: Data Verification ✅ COMPLETE
- **Phase 1**: Python 3 Compatibility ✅ COMPLETE
- **Phase 2**: TensorFlow 2.x Migration ✅ COMPLETE

## 📁 New Files Created

The following TensorFlow 2.x files have been created:

- **`model_tf2.py`** - Keras-based model implementation
- **`train_tf2.py`** - Training script with eager execution
- **`sample_tf2.py`** - Sampling/generation script
- **`extract_weights_tf1.py`** - Utility to extract weights from TF 1.x checkpoints
- **`requirements-tf2.txt`** - Python 3.11 + TensorFlow 2.15 dependencies

## 🚀 Getting Started

### 1. Install Dependencies

```bash
# Create a new virtual environment (recommended)
python3.11 -m venv venv-tf2
source venv-tf2/bin/activate  # On Windows: venv-tf2\Scripts\activate

# Install TensorFlow 2.x and dependencies
pip install -r requirements-tf2.txt

# For Apple Silicon Macs:
pip install tensorflow-macos==2.15.0 tensorflow-metal==1.1.0
pip install numpy==1.26.4 matplotlib==3.8.3
```

### 2. Verify Installation

```bash
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

Expected output:
```
TensorFlow version: 2.15.0
GPU available: True  # or False if no GPU
```

### 3. Verify Data

```bash
python3 verify_data.py
```

Should show: `SUCCESS! All checks passed!` with 11,916 training samples.

## 🎯 Usage

### Training

Train a new model from scratch:

```bash
# Quick training (100 hidden units, 8 mixtures)
python3 train_tf2.py

# High-quality training (400 hidden units, 20 mixtures)
python3 train_tf2.py --rnn_size 400 --nmixtures 20 --nepochs 250

# Custom configuration
python3 train_tf2.py \
    --rnn_size 400 \
    --nmixtures 20 \
    --batch_size 32 \
    --nepochs 250 \
    --learning_rate 1e-4 \
    --save_path saved_tf2/model
```

**Training parameters:**
- `--rnn_size`: LSTM hidden state size (100, 400, or 900)
- `--nmixtures`: Number of Gaussian mixtures (8 or 20 recommended)
- `--batch_size`: Minibatch size (32 typical)
- `--nepochs`: Number of epochs (250 typical)
- `--learning_rate`: Initial learning rate (1e-4 typical)
- `--save_path`: Where to save checkpoints

**Training time estimates:**
- CPU: ~2-3 days for 250 epochs (rnn_size=100)
- GPU: ~12-18 hours for 250 epochs (rnn_size=100)
- Apple Silicon (M1/M2/M3): ~18-24 hours

### Sampling (Generation)

Generate handwriting from text:

```bash
# Generate with default test strings
python3 sample_tf2.py

# Generate specific text
python3 sample_tf2.py --text "Hello World"

# Control randomness (bias)
python3 sample_tf2.py --text "Neat handwriting" --bias 2.0  # Neater
python3 sample_tf2.py --text "Messy handwriting" --bias 0.5  # Messier

# Custom configuration
python3 sample_tf2.py \
    --text "Your text here" \
    --bias 1.5 \
    --tsteps 700 \
    --save_path saved_tf2/model
```

**Sampling parameters:**
- `--text`: Text to generate (empty for default test strings)
- `--bias`: Randomness control (0.5=messy, 1.0=balanced, 2.0=neat)
- `--tsteps`: Maximum generation length (700 typical)
- `--eos_threshold`: End-of-stroke threshold (0.35 typical)

**Output:** PNG images saved to `logs/figures/`

## 📊 Model Architecture

The TensorFlow 2.x implementation preserves the original architecture:

- **3 LSTM layers** (100-900 hidden units each)
- **Attention mechanism** (Gaussian window over text)
- **Mixture Density Network** (8-20 Gaussian mixtures)
- **End-of-stroke prediction** (binary classifier)

### Key Changes from TF 1.x:

| TensorFlow 1.x | TensorFlow 2.x |
|----------------|----------------|
| `tf.Session()` | Eager execution |
| `tf.placeholder()` | Function arguments |
| `tf.contrib.rnn.LSTMCell` | `keras.layers.LSTM` |
| `tf.contrib.legacy_seq2seq.*` | Custom RNN loops |
| `tf.train.Saver` | `tf.train.Checkpoint` |
| `sess.run()` | Direct calls |

## 🔄 Migrating from TF 1.x Checkpoints

If you have a trained TF 1.x model, you can extract and convert the weights:

### Step 1: Extract Weights (TF 1.x environment)

First, in a Python 3.8 + TensorFlow 1.15 environment:

```bash
# Install TF 1.15 (in separate environment)
python3.8 -m venv venv-tf1
source venv-tf1/bin/activate
pip install tensorflow==1.15.5

# Extract weights
python3 extract_weights_tf1.py saved/model.ckpt-110500 weights_tf1.npz
```

### Step 2: Load Weights (TF 2.x environment)

Then, in your TF 2.x environment, you'll need to manually map and load the weights. See `model_tf2.py` for the `load_tf1_weights()` function template.

**Note:** Due to architectural differences, manual weight mapping may be required. Retraining from scratch is recommended for best results.

## 📁 File Structure

```
scribe/
├── model_tf2.py           # TF 2.x model (Keras API)
├── train_tf2.py           # TF 2.x training script
├── sample_tf2.py          # TF 2.x sampling script
├── extract_weights_tf1.py # TF 1.x weight extraction
├── requirements-tf2.txt   # TF 2.x dependencies
│
├── model.py               # Original TF 1.x model (legacy)
├── run.py                 # Original TF 1.x scripts (legacy)
├── utils.py               # Data loader (Python 3 compatible)
├── sample.py              # Original sampling utils (legacy)
│
├── data/
│   ├── strokes_training_data.cpkl  # 11,916 samples ✅
│   └── styles.p                     # 5 style vectors
│
├── saved_tf2/             # TF 2.x checkpoints (created during training)
├── logs/                  # Training logs and generated figures
│
└── docs/
    ├── MIGRATION_GUIDE.md          # Complete migration documentation
    ├── DATA_VERIFICATION_REPORT.md # Data verification results
    └── README.md                   # Documentation index
```

## 🐛 Troubleshooting

### TensorFlow not found
```bash
pip install tensorflow==2.15.0
# or for Apple Silicon:
pip install tensorflow-macos==2.15.0 tensorflow-metal==1.1.0
```

### No saved model found
Train a model first:
```bash
python3 train_tf2.py --nepochs 10  # Quick test
```

### Out of memory during training
Reduce batch size:
```bash
python3 train_tf2.py --batch_size 16  # Default is 32
```

### Generated handwriting is gibberish
- Train for more epochs (250+ recommended)
- Try adjusting bias parameter during sampling
- Ensure model has converged (check training loss)

### Data file not found
Run data verification:
```bash
python3 verify_data.py
```

## 📖 Additional Resources

- **Original Paper**: [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) by Alex Graves (2013)
- **Blog Post**: [Handwriting Generation with RNNs](https://greydanus.github.io/2016/08/21/handwriting/)
- **Migration Guide**: `docs/MIGRATION_GUIDE.md`
- **CLAUDE.md**: Complete codebase documentation

## ⚡ Quick Commands Summary

```bash
# Setup
pip install -r requirements-tf2.txt
python3 verify_data.py

# Train
python3 train_tf2.py --rnn_size 400 --nmixtures 20

# Generate
python3 sample_tf2.py --text "Hello World" --bias 1.5

# Extract old weights (TF 1.x environment)
python3 extract_weights_tf1.py saved/model.ckpt-110500 weights_tf1.npz
```

## 🎉 Success!

You now have a fully functional TensorFlow 2.x implementation of the handwriting synthesis model! The codebase is Python 3.11 compatible and uses modern TensorFlow 2.15+ APIs.

**Next steps:**
1. Train a model: `python3 train_tf2.py`
2. Generate samples: `python3 sample_tf2.py --text "Your text"`
3. Experiment with different hyperparameters and biases

Happy handwriting synthesis! ✍️
