# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**Scribe** is a handwriting synthesis neural network that generates realistic handwriting from text input. It implements Alex Graves' 2013 paper "Generating Sequences With Recurrent Neural Networks" using a 3-layer LSTM with attention mechanism and Mixture Density Network (MDN) output.

**Current State:** Python 2.7 + TensorFlow 1.0 (original), with complete migration plan to Python 3.11 + TensorFlow 2.15

**Data:** 11,916 preprocessed IAM handwriting samples (vector strokes) in `data/strokes_training_data.cpkl` - **already verified and Python 3 compatible**

---

## Commands

### Data Verification
```bash
# ALWAYS run this first before any work
python3 verify_data.py
# Expected: "SUCCESS! All checks passed!" with 11,916 samples
```

### Training (Python 2.7 - Original)
```bash
# Basic training
python run.py --train

# Custom configuration
python run.py --train \
    --rnn_size 400 \
    --nmixtures 20 \
    --tsteps 300 \
    --batch_size 32 \
    --nepochs 250 \
    --learning_rate 1e-4 \
    --save_path saved/model.ckpt
```

### Sampling (Python 2.7 - Original)
```bash
# Generate handwriting
mkdir -p logs/figures
python run.py --sample --tsteps 700 --text "your text here"

# With style conditioning
python run.py --sample --style 2 --bias 1.5 --text "styled text"

# Output: logs/figures/*.png
```

### Migration (See docs/MIGRATION_GUIDE.md)
```bash
# Phase 0: Data verification (COMPLETE)
python3 verify_data.py

# Phase 1: Python 3.8 + TF 1.15 (4-6 hours)
# See docs/MIGRATION_GUIDE.md Phase 1 for detailed steps

# Phase 2: Python 3.11 + TF 2.15 (2-3 days)
# See docs/MIGRATION_GUIDE.md Phase 2 for TF 2.x implementation
```

### Jupyter Notebooks
```bash
jupyter notebook
# Open dataloader.ipynb for data exploration
# Open sample.ipynb for generation walkthrough with equations
```

---

## Architecture Overview

### High-Level Flow
```
TEXT → One-Hot → Attention Window
                       ↓
STROKE → LSTM1 ← [Window + Stroke]
           ↓
         LSTM2 (augmented)
           ↓
         LSTM3
           ↓
       Dense Layer
           ↓
    MDN (8 Gaussians)
           ↓
    Sample 2D Point
           ↓
  GENERATED STROKE
```

### Key Components

**model.py (222 lines)** - Neural network architecture
- Lines 41-50: Three LSTM cells (100-400 hidden units each)
- Lines 64-116: **Attention mechanism** (Gaussian window over text)
- Lines 119-122: LSTM 2 & 3 with augmented inputs
- Lines 124-179: **MDN output layer** (1 + nmixtures×6 parameters)
- Lines 147-182: Loss function (mixture likelihood + binary cross-entropy)

**run.py (153 lines)** - Training/sampling orchestration
- Lines 58-111: Training loop with session-based execution
- Lines 112-149: Sampling loop with state management

**utils.py (227 lines)** - Data loading and preprocessing
- Lines 10-202: DataLoader class (IAM dataset handling)
- Lines 203-214: One-hot encoding for text

**sample.py (160 lines)** - Sampling utilities and visualization
- Lines 39-94: Generation loop with bias control
- Lines 13-37: Style priming (experimental, unreliable)
- Lines 96-159: Visualization functions (attention, heatmap, line plot)

---

## Data Format (CRITICAL)

### Input Format: `data/strokes_training_data.cpkl`
```python
[strokes, asciis]  # Pickle file

# strokes: List of numpy arrays, shape (n_points, 3)
#   Column 0: Δx (pen displacement in x direction)
#   Column 1: Δy (pen displacement in y direction)
#   Column 2: end_of_stroke flag (0 or 1)

# asciis: List of strings (text transcriptions)

# Example:
strokes[0].shape = (568, 3)
asciis[0] = "A MOVE to stop Mr . Gaitskell"
```

**Key Points:**
- **Delta encoding:** Strokes store displacements, not absolute positions
- **Normalization:** Divided by scale factor (50), clipped to [-500, 500]
- **Python 3 loading:** Use `pickle.load(f, encoding='latin1')`
- **Train/val split:** 95% train, 5% validation (every 20th sample)

### Training Tensors
```python
input_data: [batch_size, tsteps, 3]          # Current strokes
target_data: [batch_size, tsteps, 3]         # Next strokes (shifted)
char_seq: [batch_size, ascii_steps, 54]      # One-hot text
init_kappa: [batch_size, kmixtures, 1]       # Attention position
```

### MDN Output Parameters
```python
eos: [batch, 1]              # End-of-stroke probability
pi: [batch, nmixtures]       # Mixture weights (sum to 1)
mu1, mu2: [batch, nmixtures] # Gaussian means (x, y)
sigma1, sigma2: [batch, nmixtures]  # Standard deviations
rho: [batch, nmixtures]      # Correlation [-1, 1]
```

---

## Critical Implementation Details

### Attention Mechanism (model.py:64-116)

**Purpose:** Soft window over character sequence that moves left-to-right

**Mathematics:**
```python
# 1. Compute attention parameters from LSTM1 output
(α̂, β̂, κ̂) = Linear(h1)

# 2. Transform to valid ranges
α = exp(α̂)              # Importance (positive)
β = exp(β̂)              # Sharpness (positive)
κ = κ_prev + exp(κ̂)     # Position (monotonic increasing)

# 3. Compute Gaussian attention weights
φ(t,u) = Σ_k α_k exp(-β_k (κ_k - u)²)

# 4. Apply window to character sequence
w_t = Σ_u φ(t,u) c_u
```

**Key:** `κ` must monotonically increase (ensured by adding positive exp(κ̂) to previous value)

### MDN Sampling (sample.py:39-94)

**Training:** Compute loss over full mixture distribution
```python
loss = -log(Σ_j π_j N(x | μ_j, Σ_j))
```

**Inference:** Stochastic sampling
```python
# 1. Select mixture component
j ~ Categorical(π)

# 2. Sample from 2D Gaussian
(x, y) ~ N(μ_j, Σ_j)

# 3. Apply bias to control randomness
σ = exp(σ̂ - bias)  # Higher bias → lower variance → neater
```

**Bias Parameter:**
- `< 1.0`: Messy, random handwriting
- `= 1.0`: Balanced (default)
- `> 1.0`: Neat, conservative handwriting

### Coordinate System

**Storage (delta encoding):**
```python
stroke[i, 0] = x_i - x_{i-1}  # Δx
stroke[i, 1] = y_i - y_{i-1}  # Δy
```

**Rendering (cumulative sum):**
```python
absolute_coords = np.cumsum(strokes[:, :2], axis=0)
```

**Why deltas?** Translation invariance, smaller numerical range, easier to learn

---

## Migration Status

### Phase 0: Data Verification ✅ COMPLETE
- **11,916 training samples** verified in `strokes_training_data.cpkl`
- **Python 3 compatible** (loads with `encoding='latin1'`)
- **IAM dataset download NOT needed** - preprocessed data is sufficient
- **IAM_TrOCR-dataset removed** - was incompatible (raster images vs vector strokes)

### Phase 1: Python 3 Compatibility (4-6 hours)
**Target:** Python 3.8 + TensorFlow 1.15

**Required changes:**
- `cPickle` → `pickle` (utils.py:5, sample.py:3)
- `print` statements → `print()` functions (utils.py:224)
- `xrange()` → `range()` (utils.py:185, sample.py:28)
- Integer division `/` → `//` where needed (model.py:37, run.py:78)
- Identity comparison `is` → `==` for non-singletons (sample.py:16, run.py:89, 109)
- Binary file mode `'r'` → `'rb'` for pickle (sample.py:18)

**See:** `docs/MIGRATION_GUIDE.md` Phase 1 for line-by-line instructions

### Phase 2: TensorFlow 2.x Migration (2-3 days)
**Target:** Python 3.11 + TensorFlow 2.15

**Major changes required:**
- Session-based execution → Eager execution
- `tf.placeholder()` / `feed_dict` → Function arguments
- `tf.contrib.rnn.*` → `tf.keras.layers.LSTM`
- `tf.contrib.legacy_seq2seq.rnn_decoder` → Custom RNN loop or `@tf.function`
- `tf.global_variables()` → `model.variables`
- `tf.train.Saver` → `tf.train.Checkpoint`

**Complete TF 2.x implementation provided in migration guide**
- `model_tf2.py`: Keras Model subclass (600+ lines)
- `train_tf2.py`: Training script with GradientTape
- `sample_tf2.py`: Sampling script for eager execution

**See:** `docs/MIGRATION_GUIDE.md` Phase 2 for complete implementation

---

## Working with This Codebase

### Before Any Changes
1. **ALWAYS verify data first:** `python3 verify_data.py`
2. **Check migration status:** Current code is Python 2.7 + TF 1.0
3. **Read migration guide:** `docs/MIGRATION_GUIDE.md` for context

### Understanding the Model
1. **Start with architecture:** Read model.py lines 40-180
2. **Attention mechanism:** Lines 64-116 (most complex part)
3. **MDN output:** Lines 124-179 (second most complex)
4. **Data flow:** Follow from run.py → model.py → sample.py

### Modifying Hyperparameters

**Model size:**
```python
rnn_size = 100   # Fast (85% quality)
rnn_size = 400   # Balanced (100% quality)
rnn_size = 900   # Slow (105% quality)

nmixtures = 8    # Fast
nmixtures = 20   # Quality (recommended)
```

**Training:**
```python
batch_size = 32  # GPU memory dependent
tsteps = 150     # Sequence length (150-300)
learning_rate = 1e-4  # Standard
dropout = 0.85   # Keep probability
```

**Sampling:**
```python
bias = 1.0       # Randomness (0.5=messy, 2.0=neat)
tsteps = 700     # Length (longer = more complete)
```

### Common Issues

**NaN in loss:**
- MDN numerical instability
- Check: `log(x + 1e-20)` used for stability
- Verify sigma values are positive

**Gibberish output:**
- Check checkpoint loaded correctly
- Verify text characters are in alphabet
- Try increasing bias (1.5-2.0)

**Out of memory:**
- Reduce batch_size (try 16)
- Reduce tsteps (try 100)
- Reduce rnn_size (try 100)

**Data not found:**
- Run `python3 verify_data.py`
- Check `data/strokes_training_data.cpkl` exists (44 MB)

---

## Key Architecture Decisions

### Why 3 LSTM Layers?
- Layer 1: Low-level stroke features + attention integration
- Layer 2: Mid-level pattern recognition
- Layer 3: High-level sequence modeling
- Empirically optimal per Graves' paper

### Why Mixture Density Network?
- Handwriting is inherently uncertain (multiple valid trajectories)
- Single Gaussian cannot capture multimodal distributions
- MDN models uncertainty via mixture of Gaussians

### Why Attention Mechanism?
- Variable-length text input
- Model needs to know which character to draw
- Soft alignment learned during training

### Why Delta Encoding?
- Translation invariance (handwriting position doesn't matter)
- Smaller numerical range (easier optimization)
- Local patterns easier to learn than absolute positions

---

## TensorFlow 1.x Patterns (Current Code)

### Session-Based Execution
```python
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Training
loss, _ = sess.run([model.cost, model.train_op], feed_dict)

# Sampling
outputs = sess.run(fetch_list, feed_dict)
```

### Placeholder Pattern
```python
input_data = tf.placeholder(dtype=tf.float32, shape=[None, tsteps, 3])
target_data = tf.placeholder(dtype=tf.float32, shape=[None, tsteps, 3])

feed = {
    model.input_data: x_batch,
    model.target_data: y_batch,
    # ... all other placeholders
}
```

### RNN Decoder Pattern
```python
# Build sequence of inputs
inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(input_data, tsteps, 1)]

# Process with decoder
outputs, states = tf.contrib.legacy_seq2seq.rnn_decoder(
    inputs, initial_state, cell, scope='cell0'
)
```

**Note:** These patterns are deprecated in TF 2.x - see migration guide for TF 2.x equivalents

---

## File Organization

```
scribe/
├── model.py              # Network architecture (TF 1.x)
├── run.py               # Training/sampling orchestration
├── utils.py             # Data loading (IAM format)
├── sample.py            # Sampling utilities & visualization
├── verify_data.py       # Data integrity checker
├── dataloader.ipynb     # Data exploration notebook
├── sample.ipynb         # Sampling walkthrough notebook
├── data/
│   ├── strokes_training_data.cpkl  # 11,916 samples (44 MB)
│   └── styles.p                     # 5 style vectors (134 KB)
├── docs/
│   ├── MIGRATION_GUIDE.md          # Complete migration plan ⭐
│   ├── MIGRATION_EVALUATION.md     # Technical review
│   ├── DATA_VERIFICATION_REPORT.md # Phase 0 results
│   └── README.md                   # Documentation index
├── static/              # Sample output images
└── README.md           # Project overview
```

---

## Default Configuration

### Model Hyperparameters
- `rnn_size`: 100 (or 400 for quality)
- `nmixtures`: 8 (or 20 for quality)
- `kmixtures`: 1 (attention heads)
- `alphabet`: ` abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`

### Training Hyperparameters
- `batch_size`: 32
- `tsteps`: 150 (training), 700 (sampling)
- `nepochs`: 250
- `learning_rate`: 1e-4
- `optimizer`: RMSprop (decay=0.95, momentum=0.9)
- `dropout`: 0.85 keep probability
- `grad_clip`: 10.0

### Data Hyperparameters
- `data_scale`: 50 (normalization divisor)
- `tsteps_per_ascii`: 25 (pen points per character)
- `limit`: 500 (clip outliers to [-500, 500])

---

## Documentation

**Primary references:**
- `docs/MIGRATION_GUIDE.md` - Complete Python 3 + TF 2 migration plan
- `docs/MIGRATION_EVALUATION.md` - Technical evaluation and risk assessment
- `docs/DATA_VERIFICATION_REPORT.md` - Data format and verification results

**Original paper:**
- Alex Graves, "Generating Sequences With Recurrent Neural Networks" (2013)
- arXiv: https://arxiv.org/abs/1308.0850

**Blog post:**
- https://greydanus.github.io/2016/08/21/handwriting/

---

## Important Notes for Future Claude Instances

1. **This is Python 2.7 + TensorFlow 1.0 code** - migration in progress to Python 3.11 + TF 2.15
2. **Data is ready** - 11,916 samples verified, no IAM download needed
3. **Attention mechanism (model.py:64-116) is the most complex component** - understand it first
4. **MDN (model.py:124-179) is second most complex** - focus on parameter transformations
5. **TF 1.x patterns are deprecated** - refer to migration guide for TF 2.x equivalents
6. **Style priming is unreliable** - noted limitation in original implementation
7. **Delta encoding is critical** - all strokes are displacements, not absolute positions
8. **Migration guide is comprehensive** - contains complete TF 2.x implementation

When helping with migration, **always reference `docs/MIGRATION_GUIDE.md`** for the detailed implementation plan and TF 2.x code examples.
