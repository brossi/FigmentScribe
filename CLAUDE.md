# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**Scribe** is a handwriting synthesis neural network that generates realistic handwriting from text input. It implements Alex Graves' 2013 paper "Generating Sequences With Recurrent Neural Networks" using a 3-layer LSTM with attention mechanism and Mixture Density Network (MDN) output.

**Current State:** Python 3.11 + TensorFlow 2.15 (PRIMARY) ✅ Migration Complete
**Legacy:** Python 3 + TensorFlow 1.x (archived in `legacy_tf1/`)

**Data:** 11,916 preprocessed IAM handwriting samples (vector strokes) in `data/strokes_training_data.cpkl` - **verified and ready**

---

## Commands

### Data Verification
```bash
# ALWAYS run this first before any work
python3 verify_data.py
# Expected: "SUCCESS! All checks passed!" with 11,916 samples
```

### Training (Python 3.11 + TensorFlow 2.15 - PRIMARY)
```bash
# Basic training (rnn_size=100, no style priming support)
python3 train.py

# High-quality training with style priming support (REQUIRED for --styles)
# NOTE: Style priming REQUIRES rnn_size=400 (styles trained with this size)
python3 train.py --rnn_size 400 --nmixtures 20 --nepochs 250

# Custom configuration
python3 train.py \
    --rnn_size 400 \
    --nmixtures 20 \
    --batch_size 32 \
    --nepochs 250 \
    --learning_rate 1e-4 \
    --save_path saved/model
```

### Sampling (Python 3.11 + TensorFlow 2.15 - PRIMARY)
```bash
# Single-line generation
python3 sample.py --text "Hello World"

# Control style with bias (higher = neater, lower = messier)
python3 sample.py --text "Neat handwriting" --bias 2.0
python3 sample.py --text "Messy handwriting" --bias 0.5

# Full character set: letters, numerals, punctuation
python3 sample.py --text "Meet me at 3:00pm on July 15th, 1985!"

# Generate default test strings
python3 sample.py

# Output: logs/figures/*.png

# Multi-line generation for documents (NEW!)
python3 sample.py \
    --lines "First line of text" \
            "Second line of text" \
            "Third line of text" \
    --format svg

# Per-line bias control for fictional characters
python3 sample.py \
    --lines "Character A writes neatly" \
            "Character B writes messily" \
            "Character C writes normally" \
    --biases 1.8 0.6 1.2 \
    --format svg

# Style priming for distinct handwriting (13 styles: 0-12)
# REQUIREMENT: Model must be trained with --rnn_size 400 for style priming
python3 sample.py \
    --lines "This uses style 0" \
            "This uses style 5" \
            "This uses style 12" \
    --styles 0 5 12 \
    --format svg

# Combined: different styles AND different neatness
python3 sample.py \
    --lines "Character A: neat style 3" \
            "Character B: messy style 7" \
            "Character C: normal style 11" \
    --styles 3 7 11 \
    --biases 1.8 0.6 1.2 \
    --format svg

# SVG output optimized for pen plotters and gcode conversion
python3 sample.py --text "For pen plotter" --format svg

# Output SVG: logs/figures/*.svg (ready for vpype, svg2gcode, etc.)
```

### Legacy TensorFlow 1.x (ARCHIVED)
```bash
# Original TF 1.x implementation (requires TensorFlow 1.15)
# Files located in legacy_tf1/ directory
# See legacy_tf1/README.md for usage instructions
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

### Key Components (TensorFlow 2.x - PRIMARY)

**model.py** - Neural network architecture (Keras API)
- `HandwritingModel` class: Keras Model subclass
- LSTM layers: Three stacked layers using `tf.keras.layers.LSTM`
- **Attention mechanism**: Gaussian window over text (custom implementation)
- **MDN output layer**: Mixture Density Network (1 + nmixtures×6 parameters)
- Eager execution compatible, no sessions required

**train.py** - Training script
- `@tf.function` decorated training loop for performance
- `tf.GradientTape` for gradient computation
- `tf.train.Checkpoint` for model saving
- DataLoader from `utils.py` for IAM dataset

**sample.py** - Sampling/generation script
- Eager execution sampling loop
- Multi-line text generation support
- Bias control for randomness (0.5=messy, 2.0=neat)
- Per-line bias control for character-specific handwriting
- **Style priming**: 13 pre-trained handwriting styles (0-12, requires rnn_size=400)
- `load_style_state()`: Loads style examples and extracts LSTM states
  - Validates model has rnn_size=400 (fails early with clear error)
  - Caches loaded styles to avoid redundant computation
  - Comprehensive error handling for missing files, invalid characters, etc.
- `sample()`: Accepts `initial_states` for style-primed generation
- `sample_multiline()`: Per-line style and bias control
- Generates PNG or SVG output in `logs/figures/`

**svg_output.py** - SVG generation module (for pen plotters)
- Ported from sjvasquez/handwriting-synthesis
- `offsets_to_coords()`: Convert delta encoding to absolute coordinates
- `denoise()`: Savitzky-Golay smoothing filter
- `align()`: Rotation correction for handwriting slant
- `save_as_svg()`: Generate plotter-ready SVG files
- Optimized for gcode conversion (clean paths, M/L commands)

**utils.py** - Data loading and preprocessing (shared)
- DataLoader class: IAM dataset handling (11,916 samples)
- One-hot encoding for text
- Python 3.11 compatible, used by both TF 1.x and TF 2.x

**verify_data.py** - Data integrity verification
- Validates preprocessed training data
- Checks for 11,916 samples in correct format

**character_profiles.py** - Character handwriting profile templates
- Pre-built character personalities with bias settings
- BIAS_GUIDE for standardized neatness levels
- Helper functions for multi-character document generation
- Example usage for fictional character handwriting

### Legacy Components (TensorFlow 1.x - ARCHIVED)

**legacy_tf1/model.py** - Original TF 1.x model (session-based)
**legacy_tf1/run.py** - Original TF 1.x training/sampling
**legacy_tf1/sample.py** - Original TF 1.x sampling utilities
**See `legacy_tf1/README.md` for details**

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

### Style Files Format: `data/styles/style-{0-12}-{chars,strokes}.npy`
```python
# Each style has two files (13 styles total: 0-12)
# Example: style-0-chars.npy, style-0-strokes.npy

# chars file: Scalar numpy array containing bytes string
chars = np.load('style-0-chars.npy', allow_pickle=True)
text = chars.item().decode('utf-8')  # e.g., "thought that vengeance"

# strokes file: Array of shape (n_points, 3)
strokes = np.load('style-0-strokes.npy')
strokes.shape  # e.g., (631, 3) - [Δx, Δy, eos]
strokes.dtype  # float32

# Style priming process:
# 1. Load style chars and strokes
# 2. Run strokes through model to get final LSTM states
# 3. Use those states as initial_states for new text generation
# 4. New text continues in the same handwriting style
```

**Key Points:**
- **13 pre-trained styles** ported from sjvasquez/handwriting-synthesis
- **Chars file**: Scalar array with bytes string (decode with `.item().decode('utf-8')`)
- **Strokes file**: Delta-encoded handwriting sample
- **Style priming**: Extracts LSTM hidden states to maintain consistent handwriting
- **⚠️ CRITICAL**: Styles require rnn_size=400 (NOT compatible with default rnn_size=100)
  - `load_style_state()` validates model size and fails with clear error if mismatch
  - Must train model with `--rnn_size 400` to use style priming

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

### ✅ ALL PHASES COMPLETE (0-4) - PRODUCTION READY

### Phase 0: Data Verification ✅ COMPLETE
- **11,916 training samples** verified in `strokes_training_data.cpkl`
- **Python 3 compatible** (loads with `encoding='latin1'`)
- **IAM dataset download NOT needed** - preprocessed data is sufficient

### Phase 1: Python 3 Compatibility ✅ COMPLETE
- All Python 2.7 syntax converted to Python 3.11
- Legacy TF 1.x files archived in `legacy_tf1/`
- `cPickle` → `pickle`, `xrange()` → `range()`, etc.
- **See:** `legacy_tf1/` for Python 3 compatible TF 1.x files

### Phase 2: TensorFlow 2.x Migration ✅ COMPLETE
**Primary implementation:** Python 3.11 + TensorFlow 2.15

**Production files:**
- `model.py`: Keras Model subclass with eager execution
- `train.py`: Training with `@tf.function` and GradientTape
- `sample.py`: Eager execution sampling
- All using modern TF 2.15 APIs (no sessions, no placeholders)

**Changes implemented:**
- Session-based → Eager execution
- `tf.placeholder()` → Function arguments
- `tf.contrib.rnn.*` → `tf.keras.layers.LSTM`
- `tf.train.Saver` → `tf.train.Checkpoint`

**See:** `docs/MIGRATION_GUIDE.md` for complete migration documentation

### Phase 3: Multi-line SVG Features ✅ COMPLETE
**sjvasquez handwriting-synthesis port for pen plotter output**

**Features implemented:**
- `svg_output.py`: SVG generation module (ported from sjvasquez)
  - `offsets_to_coords()`: Delta to absolute coordinate conversion
  - `denoise()`: Savitzky-Golay smoothing (window=7, poly=3)
  - `align()`: Rotation correction using linear regression
  - `save_as_svg()`: Multi-line SVG generation for pen plotters
- Multi-line text generation in `sample.py`
  - `sample_multiline()`: Process multiple lines with independent parameters
  - `--lines` argument: Multiple text strings
  - `--biases` argument: Per-line bias control
  - `--format svg`: SVG output for gcode conversion
- `character_profiles.py`: Character handwriting template system

**Commits:**
- c5a1fa3: Add multi-line SVG handwriting generation for pen plotter
- fc5bc68: Add documentation and character profile templates
- 991cf87: Fix documentation gaps in CLAUDE.md

### Phase 4: Style Priming System ✅ COMPLETE
**13 handwriting styles for character-specific generation**

**⚠️ CRITICAL REQUIREMENT: Model MUST be trained with `--rnn_size 400` to use style priming**

**Features implemented:**
- **Expanded alphabet** from 54 to 83 characters
  - Added numerals (0-9) and punctuation (!"#%&'()*+,-./:;?[])
  - Enables realistic fictional character letters with dates, emphasis, quotations
  - Matches full IAM dataset character set
- 13 pre-trained styles ported from sjvasquez (styles 0-12)
  - 26 `.npy` files in `data/styles/` (chars + strokes for each style)
- `load_style_state()` function in `sample.py`
  - Loads style sample text and strokes
  - Runs through model to extract LSTM hidden states
  - Returns initial states for style-primed generation
  - **Validates model has rnn_size=400** (fails with clear error otherwise)
  - **Caches loaded styles** to avoid redundant computation
- `sample()` updated to accept `initial_states` parameter
- `sample_multiline()` updated for per-line style control
- `--styles` CLI argument: List of style IDs (0-12) for each line
- Comprehensive error handling and validation

**Usage examples:**
```bash
# First, ensure you have a model trained with rnn_size=400:
python3 train.py --rnn_size 400 --nmixtures 20 --nepochs 250

# Then use styles:
python3 sample.py --lines "Text in style 5" --styles 5 --format svg

# Multiple styles (fictional characters)
python3 sample.py \
    --lines "Character A" "Character B" "Character C" \
    --styles 3 7 11 \
    --biases 1.8 0.6 1.2 \
    --format svg
```

**Why rnn_size=400 is required:**
- Style files contain LSTM states trained with rnn_size=400
- State shape must match: `[3 layers, 2 states, batch, 400 units]`
- Using different sizes causes TensorFlow shape mismatch errors
- Function validates size and provides clear error message if mismatch detected

---

## Working with This Codebase

### Before Any Changes
1. **ALWAYS verify data first:** `python3 verify_data.py`
2. **Use current files:** `model.py`, `train.py`, `sample.py` are the production implementation
3. **Legacy reference:** TF 1.x files are in `legacy_tf1/` for reference only

### Understanding the Model
1. **Start with architecture:** Read `model.py` (Keras Model implementation)
2. **Attention mechanism:** Custom Gaussian window over text (most complex part)
3. **MDN output:** Mixture Density Network layer (second most complex)
4. **Data flow:** `train.py` → `model.py` → `sample.py`
5. **Legacy reference:** See `legacy_tf1/model.py` for original TF 1.x implementation

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
├── model.py              # Neural network architecture (Keras API)
├── train.py              # Training script
├── sample.py             # Sampling/generation script
├── svg_output.py         # SVG generation for pen plotters (NEW)
├── utils.py              # Data loading (IAM format)
├── verify_data.py        # Data integrity checker
├── character_profiles.py # Character handwriting templates (NEW)
│
├── dataloader.ipynb      # Data exploration notebook
├── sample.ipynb          # Sampling walkthrough notebook
│
├── legacy_tf1/           # ← TensorFlow 1.x implementation (ARCHIVED)
│   ├── model.py          #    Original TF 1.x model
│   ├── run.py            #    Original TF 1.x training/sampling
│   ├── sample.py         #    Original TF 1.x sampling utilities
│   ├── extract_weights_tf1.py # Weight extraction utility
│   └── README.md         #    Legacy usage instructions
│
├── data/
│   ├── strokes_training_data.cpkl  # 11,916 samples (44 MB)
│   ├── styles.p                     # 5 legacy style vectors (134 KB, unused)
│   └── styles/                      # 13 handwriting styles (NEW)
│       ├── style-0-chars.npy        # Style 0 text sample
│       ├── style-0-strokes.npy      # Style 0 stroke data
│       ├── ...                      # Styles 1-11
│       ├── style-12-chars.npy       # Style 12 text sample
│       └── style-12-strokes.npy     # Style 12 stroke data
│
├── docs/
│   ├── MIGRATION_GUIDE.md    # Complete migration documentation ⭐
│   ├── AUDIT_SUMMARY.md      # Code audit results
│   ├── README.md             # Documentation index
│   └── archive/              # ← Historical documentation
│       ├── MIGRATION_EVALUATION.md
│       └── DATA_VERIFICATION_REPORT.md
│
├── saved/                # Model checkpoints (created during training)
├── logs/                 # Training logs and generated figures
│   └── figures/          # PNG and SVG output (generated)
├── static/               # Sample output images
├── requirements.txt      # Python 3.11 + TensorFlow 2.15 + scipy + svgwrite
├── README.md             # Project overview
└── CLAUDE.md             # This file
```

---

## Default Configuration

### Model Hyperparameters
- `rnn_size`: 100 (or 400 for quality)
- `nmixtures`: 8 (or 20 for quality)
- `kmixtures`: 1 (attention heads)
- `alphabet`: ` abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#%&'()*+,-./:;?[]` (83 chars: letters, numerals, punctuation)

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

1. **✅ ALL PHASES COMPLETE (0-4)** - Python 3.11 + TensorFlow 2.15 with multi-line SVG and style priming
2. **Use current files** - `model.py`, `train.py`, `sample.py` are the production implementation
3. **Legacy files archived** - TF 1.x implementation in `legacy_tf1/` for reference only
4. **Data is ready** - 11,916 samples verified, no IAM download needed
5. **Full character set** - 83 chars including numerals and punctuation (letters, 0-9, !"#%&'()*+,-./:;?[])
6. **13 handwriting styles available** - Use `--styles 0-12` for character-specific generation (REQUIRES rnn_size=400)
7. **Attention mechanism** is the most complex component - see `model.py`
8. **MDN output** is second most complex - focus on parameter transformations
9. **Delta encoding is critical** - all strokes are displacements, not absolute positions
10. **Style priming implemented** - `load_style_state()` extracts LSTM states from style samples
11. **SVG output for pen plotters** - `svg_output.py` generates gcode-ready files
12. **Multi-line generation** - `sample_multiline()` with per-line bias and style control
13. **Eager execution** - Uses no sessions, no placeholders, direct function calls
14. **Migration history** - see `docs/MIGRATION_GUIDE.md` and `docs/AUDIT_SUMMARY.md`

**For new work:** Use the current implementation in root directory. Legacy TF 1.x files are for historical reference only.

**For fictional character handwriting:**
1. Use `character_profiles.py` as template
2. Combine `--styles` (handwriting appearance) with `--biases` (neatness level)
3. Output SVG for pen plotter with `--format svg`
