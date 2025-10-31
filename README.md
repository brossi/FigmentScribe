# Scribe: Realistic Handwriting Synthesis with Neural Networks

Generate realistic handwriting from text using deep learning. This implementation uses a 3-layer LSTM with attention mechanism and Mixture Density Network (MDN) to synthesize handwriting in diverse styles.

Based on Alex Graves' 2013 paper "[Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)".

> **Original implementation by [Sam Greydanus](https://github.com/greydanus)** • [Blog post](https://greydanus.github.io/2016/08/21/handwriting/)

---

## ✨ Features

- **Generate handwriting from any text** - Convert ASCII text to realistic handwriting
- **Style control** - Adjust randomness/neatness with bias parameter
- **Attention mechanism** - Model learns to focus on one character at a time
- **Diverse outputs** - Generate cursive, print, messy, or neat handwriting
- **Pre-trained model ready** - Start generating immediately with included data
- **Pure Python** - Modern Python 3.11 + TensorFlow 2.15 implementation

---

## 📸 Sample Outputs

**"A project by Sam Greydanus"**

![Sample output 1](static/author.png?raw=true)

**"You know nothing Jon Snow" (print style)**

![Sample output 2](static/jon_print.png?raw=true)

**"You know nothing Jon Snow" (cursive style)**

![Sample output 3](static/jon_cursive.png?raw=true)

### Controlling Randomness with Bias

**Lowering the bias (neat, bias=2.0)**

![Sample output 4](static/bias-1.png?raw=true)

**Medium bias (balanced, bias=1.0)**

![Sample output 5](static/bias-0.75.png?raw=true)

**Higher randomness (messy, bias=0.5)**

![Sample output 6](static/bias-0.5.png?raw=true)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scribe.git
cd scribe

# Install dependencies (Python 3.11+)
pip install -r requirements.txt

# Verify installation
python3 verify_data.py
```

### 2. Generate Handwriting (Using Pre-trained Data)

```bash
# Generate with default test strings
python3 sample.py

# Generate your custom text
python3 sample.py --text "Hello World"

# Control the style (0.5=messy, 1.0=balanced, 2.0=neat)
python3 sample.py --text "Neat handwriting" --bias 2.0
```

**Output**: Images saved to `logs/figures/`

### 3. Train Your Own Model (Optional)

```bash
# Quick training (smaller model)
python3 train.py --rnn_size 100 --nmixtures 8 --nepochs 50

# High-quality training (recommended)
python3 train.py --rnn_size 400 --nmixtures 20 --nepochs 250
```

**Training time**: ~12-24 hours on GPU for full training

---

## 📋 Requirements

- **Python**: 3.11 or higher
- **TensorFlow**: 2.15+
- **NumPy**: 1.26+
- **Matplotlib**: 3.8+

**Full dependencies**: See `requirements.txt`

### Installation Options

**Standard (CPU/GPU)**:
```bash
pip install tensorflow==2.15.0 numpy==1.26.4 matplotlib==3.8.3
```

**Apple Silicon (M1/M2/M3)**:
```bash
pip install tensorflow-macos==2.15.0 tensorflow-metal==1.1.0
pip install numpy==1.26.4 matplotlib==3.8.3
```

---

## 💻 Usage

### Command-Line Interface

#### Sampling (Generation)

```bash
# Basic generation
python3 sample.py --text "Your text here"

# Advanced options
python3 sample.py \
    --text "Custom handwriting" \
    --bias 1.5 \                    # Randomness (0.5-2.0)
    --tsteps 700 \                  # Maximum length
    --rnn_size 400 \                # Model size (must match trained model)
    --nmixtures 20                  # Mixture count (must match trained model)
```

#### Training

```bash
# Basic training
python3 train.py

# Advanced training
python3 train.py \
    --rnn_size 400 \                # LSTM hidden size (100/400/900)
    --nmixtures 20 \                # Gaussian mixtures (8/20)
    --batch_size 32 \               # Batch size
    --nepochs 250 \                 # Number of epochs
    --learning_rate 0.0001 \        # Learning rate
    --save_path saved/model     # Checkpoint directory
```

### Python API

```python
import tensorflow as tf
from model import HandwritingModel
from sample import sample, to_one_hot

# Configure model
class Args:
    rnn_size = 400
    nmixtures = 20
    kmixtures = 1
    alphabet = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    tsteps_per_ascii = 25
    tsteps = 700
    bias = 1.5

args = Args()

# Load model
model = HandwritingModel(args)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore("saved/checkpoint")

# Generate handwriting
text = "Hello World"
strokes, phis, kappas = sample(text, model, args)

# strokes contains the generated handwriting coordinates
# Use matplotlib to visualize
```

---

## 🏗️ Architecture

### Model Overview

The model implements a sequence-to-sequence architecture with three key components:

#### 1. LSTM Layers (3 layers)
- Processes input stroke sequences
- Maintains hidden state for temporal dependencies
- Size: 100-900 hidden units per layer

#### 2. Attention Mechanism
- Gaussian window over input text
- Learns to focus on one character at a time
- Enables variable-length text input

#### 3. Mixture Density Network (MDN)
- Outputs mixture of Gaussians (8-20 components)
- Models uncertainty in handwriting generation
- Predicts 2D stroke coordinates and end-of-stroke

### Architecture Diagram

**Single timestep:**

![Model at one timestep](static/model_rolled.png?raw=true)

**Unrolled in time:**

![Model unrolled](static/model_unrolled.png?raw=true)

**Attention mechanism:**

![Attention mechanism](static/diag_window.png?raw=true)

---

## 📊 Dataset

The model is trained on the **IAM Handwriting Database**, containing 11,916 handwriting samples with corresponding text transcriptions.

**Data format**: Vector strokes (Δx, Δy, end-of-stroke) with text labels

**Included in repository**: Pre-processed training data (`data/strokes_training_data.cpkl`)

**Size**: 44 MB (11,916 samples)

No additional download required! The repository includes all necessary data.

---

## 📁 Project Structure

```
scribe/
├── model.py                  # Neural network architecture
├── train.py                  # Training script
├── sample.py                 # Sampling/generation script
├── utils.py                  # Data loader and utilities
├── verify_data.py            # Data verification tool
│
├── dataloader.ipynb          # Jupyter notebook: data exploration
├── sample.ipynb              # Jupyter notebook: model demonstration
│
├── legacy_tf1/               # Legacy TensorFlow 1.x implementation (archived)
│   ├── model.py              # Original TF 1.x model
│   ├── run.py                # Original TF 1.x training/sampling
│   ├── sample.py             # Original TF 1.x utilities
│   ├── extract_weights_tf1.py # Checkpoint weight extraction utility
│   └── README.md             # Legacy usage instructions
│
├── requirements.txt          # Python 3.11 + TensorFlow 2.15 dependencies
├── README.md                 # This file
├── CLAUDE.md                 # Detailed codebase documentation
│
├── data/
│   ├── strokes_training_data.cpkl  # Training data (11,916 samples)
│   └── styles.p                     # Style vectors (5 styles)
│
├── saved/                # Model checkpoints (created during training)
├── logs/                     # Training logs and generated figures
│
├── docs/                     # Additional documentation
│   ├── MIGRATION_GUIDE.md    # Complete migration documentation
│   ├── AUDIT_SUMMARY.md      # Code audit results
│   ├── README.md             # Documentation index
│   └── archive/              # Historical documentation
│       ├── MIGRATION_EVALUATION.md
│       └── DATA_VERIFICATION_REPORT.md
│
└── static/                   # Sample images and diagrams
```

---

## 🎯 Hyperparameter Guide

### Model Size

| `rnn_size` | Quality | Training Time | Memory | Use Case |
|------------|---------|---------------|--------|----------|
| 100 | Good | ~12 hours | ~2 GB | Quick experiments |
| 400 | Excellent | ~24 hours | ~4 GB | **Recommended** |
| 900 | Best | ~48 hours | ~8 GB | Maximum quality |

### Mixture Components

| `nmixtures` | Quality | Training Time | Use Case |
|-------------|---------|---------------|----------|
| 8 | Good | Faster | Quick training |
| 20 | Excellent | Standard | **Recommended** |

### Sampling Bias

| Bias Value | Style | Description |
|------------|-------|-------------|
| 0.5 | Messy | High randomness, diverse strokes |
| 1.0 | Balanced | Natural handwriting variation |
| 1.5 | Neat | Cleaner, more consistent |
| 2.0 | Very Neat | Minimal variation, precise |

---

## 📓 Jupyter Notebooks

Interactive notebooks with equations and explanations:

- **[dataloader.ipynb](dataloader.ipynb)** - Explore the IAM dataset structure
- **[sample.ipynb](sample.ipynb)** - Step-by-step generation walkthrough

Launch with:
```bash
jupyter notebook
```

---

## 🔧 Troubleshooting

### TensorFlow not found
```bash
pip install tensorflow==2.15.0
# or for Apple Silicon:
pip install tensorflow-macos==2.15.0 tensorflow-metal==1.1.0
```

### Data verification fails
```bash
python3 verify_data.py
```
Should show: `SUCCESS! All checks passed!`

### Out of memory during training
```bash
# Reduce batch size
python3 train.py --batch_size 16

# Or reduce model size
python3 train.py --rnn_size 100
```

### Generated handwriting looks wrong
- Ensure model is trained (250+ epochs recommended)
- Try different bias values: `--bias 0.5` to `--bias 2.0`
- Check that model size parameters match trained checkpoint

### Import errors
```bash
# Verify Python version
python3 --version  # Should be 3.11+

# Verify TensorFlow
python3 -c "import tensorflow as tf; print(tf.__version__)"  # Should be 2.15+
```

---

## 🎓 How It Works

### 1. Input Processing
- Text is converted to one-hot encoding
- Previous strokes feed into LSTM layers

### 2. Attention Mechanism
The model uses a Gaussian attention window that:
- Focuses on one character at a time
- Moves left-to-right through text
- Learns alignment between text and strokes

**Attention parameters:**
- α (alpha): Importance weights
- β (beta): Window sharpness
- κ (kappa): Window position

### 3. LSTM Processing
Three stacked LSTM layers:
- **Layer 1**: Integrates strokes with attention window
- **Layer 2**: Learns mid-level patterns
- **Layer 3**: Models high-level sequence structure

### 4. MDN Output
Mixture Density Network outputs:
- **End-of-stroke probability** (pen lift)
- **Mixture weights** (π): Which Gaussian to sample from
- **Means** (μ₁, μ₂): Expected stroke position
- **Variances** (σ₁, σ₂): Uncertainty in position
- **Correlation** (ρ): Relationship between x and y

### 5. Sampling
During generation:
1. Sample mixture component based on π
2. Sample (Δx, Δy) from 2D Gaussian
3. Decide pen lift based on end-of-stroke probability
4. Repeat until text is complete

---

## 🧪 Examples

### Generate Multiple Samples

```bash
# Generate several variations
for i in {1..5}; do
    python3 sample.py --text "Hello World" --bias $(echo "scale=1; $i/2" | bc)
done
```

### Batch Generation

```python
# generate_batch.py
from sample import sample
import matplotlib.pyplot as plt

texts = [
    "Hello World",
    "Machine Learning",
    "Neural Networks",
    "Deep Learning"
]

for text in texts:
    strokes, _, _ = sample(text, model, args)
    # Save or display strokes
```

### Training from Scratch

```bash
# Full training pipeline
python3 verify_data.py                    # Verify data
python3 train.py \
    --rnn_size 400 \
    --nmixtures 20 \
    --nepochs 250 \
    --save_path saved/model           # Train model
python3 sample.py --text "Test"       # Generate samples
```

---

## 🔄 Advanced Topics

### Migrating from TensorFlow 1.x Checkpoints

If you have a previously trained TensorFlow 1.x model checkpoint, you can extract and migrate the weights to TensorFlow 2.x.

**Step 1: Extract weights from TF 1.x checkpoint**

This requires a separate Python 3.8 + TensorFlow 1.15 environment:

```bash
# Create TF 1.x environment
python3.8 -m venv venv-tf1
source venv-tf1/bin/activate  # On Windows: venv-tf1\Scripts\activate

# Install TensorFlow 1.15
pip install tensorflow==1.15.5

# Extract weights
python3 legacy_tf1/extract_weights_tf1.py saved/model.ckpt-110500 weights_tf1.npz
```

**Step 2: Load weights in TF 2.x**

The extracted weights can then be loaded into the TF 2.x model. Note that due to architectural differences between TF 1.x session-based execution and TF 2.x eager execution, manual weight mapping may be required.

**Recommendation**: For best results, we recommend training from scratch with the TF 2.x implementation (`train.py`), which typically converges in 12-24 hours on a GPU.

---

## 📚 References

### Paper
- **Alex Graves** (2013). "[Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)". arXiv:1308.0850

### Dataset
- **IAM Handwriting Database**: [http://www.fki.inf.unibe.ch/databases/iam-handwriting-database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)

### Related Work
- [Handwriting Generation with RNNs](https://greydanus.github.io/2016/08/21/handwriting/) - Blog post by Sam Greydanus
- [Synthesis of Handwriting](http://www.cs.toronto.edu/~graves/handwriting.html) - Alex Graves' demo

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional training data sources
- Web interface for generation
- Real-time stroke visualization
- Style transfer between handwriting samples
- Export to SVG/font formats

---

## 📄 License

This project is provided as-is for research and educational purposes.

**Original implementation**: Sam Greydanus ([GitHub](https://github.com/greydanus))

**Dataset**: IAM Handwriting Database (requires separate license for commercial use)

---

## 🙏 Acknowledgments

- **Alex Graves** for the groundbreaking research
- **Sam Greydanus** for the original implementation and blog post
- **IAM Handwriting Database** contributors
- **TensorFlow team** for the deep learning framework

---

## 📞 Support

- **Documentation**: See `CLAUDE.md` for detailed codebase documentation
- **Issues**: Report bugs or request features via GitHub issues
- **Questions**: Check existing issues or create a new one

---

## ⭐ Quick Reference

```bash
# Verify installation
python3 verify_data.py

# Generate handwriting
python3 sample.py --text "Your text" --bias 1.5

# Train model
python3 train.py --rnn_size 400 --nmixtures 20 --nepochs 250

# View notebooks
jupyter notebook dataloader.ipynb
```

**Ready to generate handwriting? Start with:** `python3 sample.py` 🎨✍️

---

<div align="center">

**Made with 🧠 and TensorFlow**

[Original Blog Post](https://greydanus.github.io/2016/08/21/handwriting/) • [Research Paper](https://arxiv.org/abs/1308.0850) • [Report Issues](https://github.com/yourusername/scribe/issues)

</div>
