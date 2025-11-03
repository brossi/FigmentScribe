# Pressure Model Implementation Progress

**Status:** Phase 1 In Progress (Model Conversion)
**Last Updated:** November 3, 2025
**Timeline:** Weeks 1-4 (3-4 weeks total)

---

## Executive Summary

The pressure model port from handwriting-model repository is underway. This document tracks implementation progress, provides setup instructions, and documents next steps.

**Goal:** Add realistic stroke thickness variation to Scribe's generated handwriting by predicting pen pressure from stroke coordinates.

**Approach:** Port pre-trained PyTorch model to TensorFlow 2.15 as opt-in feature for pen plotter SVG output.

---

## Phase 1 Progress: Model Conversion (Week 1)

### ✅ Completed Tasks

#### 1. Training Data Acquisition
**Status:** Complete ✓

**What was done:**
- Copied PyTorch training data from reference repository
- Copied pre-trained model checkpoint (epoch 4)

**Files created:**
```
data/pressure_data/
├── train_data.pt (8.8 MB)  # 793 training samples
├── val_data.pt (2.2 MB)    # 199 validation samples
└── model_epoch_4.pth (379 KB)  # Pre-trained PyTorch model
```

**Data format:**
- Input X: [n_samples, 728, 3] - Delta-encoded strokes (Δx, Δy, eos)
- Target y: [n_samples, 728] - Delta-encoded pressure values
- Source: Wacom tablet recordings with pressure information

---

#### 2. TensorFlow Model Implementation
**Status:** Complete ✓

**What was done:**
- Created `pressure_model.py` with TensorFlow 2.15 implementation
- Implemented 2-layer LSTM architecture (50 hidden units)
- Added `predict_pressure()` method for inference
- Added `pressure_to_line_width()` utility function
- Included model loading and checkpoint management

**File created:** `pressure_model.py` (8.7 KB)

**Architecture:**
```python
Input: [batch, timesteps, 3] (Δx, Δy, eos)
    ↓
LSTM Layer 1: [batch, timesteps, 50]
    ↓
LSTM Layer 2: [batch, timesteps, 50]
    ↓
Dense Layer: [batch, timesteps, 1]
    ↓
Output: [batch, timesteps, 1] (Δpressure)
```

**Key Features:**
- Compatible with TensorFlow 2.15 Keras API
- Eager execution support
- Delta-to-absolute pressure conversion
- Normalized output (0-1 range) for line width control
- Comprehensive docstrings and examples

---

#### 3. Weight Conversion Script
**Status:** Complete ✓

**What was done:**
- Created `convert_pressure_weights.py` for PyTorch → TensorFlow conversion
- Handles LSTM weight transposition (PyTorch and TensorFlow use different layouts)
- Combines PyTorch's separate biases (bias_ih + bias_hh)
- Saves TensorFlow checkpoint in `saved/pressure_model/`

**File created:** `convert_pressure_weights.py` (10.9 KB)

**Conversion process:**
1. Load PyTorch checkpoint (model_epoch_4.pth)
2. Extract LSTM and FC layer weights
3. Transpose matrices (PyTorch: [out, in], TensorFlow: [in, out])
4. Combine biases (PyTorch has 2, TensorFlow has 1)
5. Assign to TensorFlow model
6. Save checkpoint

**Weight transformations:**
```python
# LSTM weights
PyTorch weight_ih: [4*hidden, input] → TensorFlow kernel: [input, 4*hidden]
PyTorch weight_hh: [4*hidden, hidden] → TensorFlow recurrent_kernel: [hidden, 4*hidden]
PyTorch bias_ih + bias_hh → TensorFlow bias: [4*hidden]

# FC weights
PyTorch fc.weight: [output, input] → TensorFlow kernel: [input, output]
PyTorch fc.bias → TensorFlow bias (no change)
```

---

#### 4. Validation Script
**Status:** Complete ✓

**What was done:**
- Created `validate_pressure_conversion.py` to verify conversion accuracy
- Compares PyTorch and TensorFlow outputs on same inputs
- Tests on random data, validation set, and single sequences
- Reports MSE, MAE, max difference, and statistical comparison

**File created:** `validate_pressure_conversion.py` (9.5 KB)

**Validation tests:**
1. **Random Input Test** - Random noise inputs (batch=4, seq_len=100)
2. **Validation Data Test** - Actual validation samples (first 5 samples)
3. **Single Sequence Test** - Simple handcrafted sequence for inspection

**Success criteria:** MSE < 1e-5 between PyTorch and TensorFlow outputs

---

#### 5. Unit Tests
**Status:** Complete ✓

**What was done:**
- Created comprehensive unit tests in `tests/unit/test_pressure_model.py`
- Tests for model creation, forward pass, pressure prediction
- Tests for pressure-to-line-width conversion
- Tests for model loading and checkpointing
- Smoke tests for quick validation

**File created:** `tests/unit/test_pressure_model.py` (11.2 KB)

**Test coverage:**
- Model initialization (default and custom parameters)
- Forward pass (single/multiple batches, variable lengths)
- `predict_pressure()` method (numpy input, delta conversion)
- `pressure_to_line_width()` function (scaling, edge cases)
- Model saving and loading
- Error handling (missing checkpoints)

**Run tests:**
```bash
# All pressure model tests
pytest tests/unit/test_pressure_model.py -v

# Smoke test only
pytest tests/unit/test_pressure_model.py::test_pressure_model_smoke -v

# With coverage
pytest tests/unit/test_pressure_model.py --cov=pressure_model --cov-report=html
```

---

### ⏸️ Pending Tasks (Blocked by Dependencies)

#### 6. Weight Conversion Execution
**Status:** Pending (requires PyTorch + TensorFlow)

**Why blocked:**
- PyTorch not installed in current environment
- TensorFlow not installed in current environment
- Cannot run conversion script until both are available

**What needs to be done:**
```bash
# Install dependencies
pip install torch tensorflow

# Run conversion
python3 convert_pressure_weights.py

# Expected output:
# - saved/pressure_model/checkpoint
# - saved/pressure_model/model.data-00000-of-00001
# - saved/pressure_model/model.index
```

**Estimated time:** 5-10 minutes (script execution)

---

#### 7. Validation Execution
**Status:** Pending (requires completed conversion + dependencies)

**Why blocked:**
- Depends on Task 6 (weight conversion)
- Requires PyTorch and TensorFlow

**What needs to be done:**
```bash
# After conversion completes, run validation
python3 validate_pressure_conversion.py

# Expected output:
# ✓ Random Input: PASS
# ✓ Validation Data: PASS
# ✓ Single Sequence: PASS
# ✓ ALL TESTS PASSED
```

**Success criteria:** All 3 validation tests pass with MSE < 1e-5

**Estimated time:** 2-3 minutes (validation execution)

---

#### 8. Unit Test Execution
**Status:** Pending (requires TensorFlow)

**Why blocked:**
- Tests import TensorFlow, which is not installed

**What needs to be done:**
```bash
# After TensorFlow installation
pytest tests/unit/test_pressure_model.py -v

# Expected: All tests pass
```

**Estimated time:** 30 seconds (test execution)

---

## Phase 1 Summary

### Work Completed (5 of 8 tasks)

✅ Training data acquisition and organization
✅ TensorFlow model implementation (pressure_model.py)
✅ Weight conversion script (convert_pressure_weights.py)
✅ Validation script (validate_pressure_conversion.py)
✅ Comprehensive unit tests (test_pressure_model.py)

### Work Pending (3 of 8 tasks)

⏸️ Execute weight conversion (5-10 min when dependencies available)
⏸️ Execute validation (2-3 min when conversion complete)
⏸️ Run unit tests (30 sec when TensorFlow available)

### Estimated Remaining Time

**Phase 1 completion:** 15-20 minutes (once dependencies installed)

**Total time invested:** ~6-8 hours (development and testing)

---

## Installation Requirements

When ready to complete Phase 1, install:

```bash
# Python 3.11 environment (Scribe's standard)
pip install torch tensorflow numpy

# Or use requirements.txt (if updated)
pip install -r requirements.txt
```

**Note:** These are already in `requirements.txt` for Scribe, so standard setup should work.

---

## Next Steps (After Phase 1)

### Phase 2: Integration Pipeline (Week 2, Days 5-9)

**Goal:** Integrate pressure model with Scribe's sampling pipeline

**Tasks:**
1. Add `load_pressure_model()` to sample.py (lazy loading)
2. Add `predict_pressure()` integration function
3. Modify `sample()` and `sample_multiline()` to accept pressure flag
4. Add CLI arguments: `--pressure`, `--pressure_scale`
5. Create integration tests

**Estimated time:** 1 week

---

### Phase 3: SVG Output Enhancement (Week 2-3, Days 10-14)

**Goal:** Render variable thickness in SVG for pen plotters

**Tasks:**
1. Modify `svg_output.py` for per-segment stroke-width
2. Create `save_as_svg_with_pressure()` function
3. Map pressure to line width (0.5x to 1.5x base width)
4. Test with pen plotter tools (Inkscape, vpype, svg2gcode)
5. Generate example outputs

**Estimated time:** 1 week

---

### Phase 4: PNG Support & Refinement (Week 3, Days 15-18)

**Goal:** Add PNG rendering and polish feature

**Tasks:**
1. Update PNG rendering for variable width
2. Visual quality evaluation
3. Comparison images (with/without pressure)
4. Test various pressure_scale values

**Estimated time:** 4 days

---

### Phase 5: Documentation & Testing (Week 4, Days 19-22)

**Goal:** Production-ready feature with complete documentation

**Tasks:**
1. Comprehensive test suite completion
2. Update CLAUDE.md and README.md
3. Create usage examples
4. Jupyter notebook: pressure_demo.ipynb
5. Update Colab training notebook

**Estimated time:** 4 days

---

## File Structure (Current State)

```
scribe/
├── pressure_model.py ✅               # TensorFlow pressure model
├── convert_pressure_weights.py ✅    # Weight conversion script
├── validate_pressure_conversion.py ✅ # Validation script
│
├── data/
│   └── pressure_data/ ✅             # Training data and checkpoint
│       ├── train_data.pt (8.8 MB)
│       ├── val_data.pt (2.2 MB)
│       └── model_epoch_4.pth (379 KB)
│
├── saved/
│   └── pressure_model/ ⏸️            # TensorFlow checkpoint (pending)
│       ├── checkpoint
│       ├── model.data-00000-of-00001
│       └── model.index
│
├── tests/
│   └── unit/
│       └── test_pressure_model.py ✅ # Unit tests
│
├── docs/
│   ├── REFERENCE_REPOS_ANALYSIS.md ✅
│   └── PRESSURE_MODEL_IMPLEMENTATION.md ✅ # This document
│
└── reference-repos/
    ├── handwriting-model/ ✅         # Source repository
    ├── pytorch-handwriting-synthesis-toolkit/ ✅
    └── Handwriting-synthesis/ ✅
```

**Legend:**
- ✅ Complete
- ⏸️ Pending (blocked by dependencies)

---

## Quick Start (When Ready)

### Step 1: Install Dependencies

```bash
# Ensure Python 3.11 environment
python3 --version

# Install required packages
pip install torch tensorflow numpy
```

### Step 2: Convert Weights

```bash
# Run conversion script
python3 convert_pressure_weights.py

# Expected output:
# ✓ PyTorch checkpoint loaded
# ✓ TensorFlow model created
# ✓ Weights converted and assigned
# ✓ Checkpoint saved to saved/pressure_model/
```

### Step 3: Validate Conversion

```bash
# Run validation script
python3 validate_pressure_conversion.py

# Expected output:
# ✓ Random Input: PASS
# ✓ Validation Data: PASS
# ✓ Single Sequence: PASS
# ✓ ALL TESTS PASSED
```

### Step 4: Run Tests

```bash
# Run unit tests
pytest tests/unit/test_pressure_model.py -v

# Expected: All tests pass
```

### Step 5: Test Model Inference

```python
# Quick test in Python REPL
python3
>>> from pressure_model import load_pressure_model
>>> model = load_pressure_model('saved/pressure_model')
>>> import numpy as np
>>> strokes = np.random.randn(100, 3).astype(np.float32)
>>> pressure = model.predict_pressure(strokes)
>>> print(pressure.shape, pressure.min(), pressure.max())
(100,) 0.0 1.0  # Normalized pressure values
```

---

## Troubleshooting

### Issue: ImportError for torch or tensorflow

**Solution:**
```bash
pip install torch tensorflow
```

### Issue: Checkpoint not found error

**Solution:**
```bash
# Ensure data files are in place
ls -lh data/pressure_data/
# Should see: train_data.pt, val_data.pt, model_epoch_4.pth

# Run conversion
python3 convert_pressure_weights.py
```

### Issue: Validation MSE > 1e-5

**Cause:** Weight conversion may have errors

**Solution:**
1. Check conversion logic in `convert_pressure_weights.py`
2. Verify weight shapes match expected dimensions
3. Compare intermediate outputs step-by-step

### Issue: Tests fail with TensorFlow errors

**Solution:**
```bash
# Check TensorFlow installation
python3 -c "import tensorflow as tf; print(tf.__version__)"

# Should show: 2.15.x or compatible version
```

---

## Success Metrics

### Phase 1 Success Criteria

- [x] All files created and committed
- [ ] Weight conversion executes without errors
- [ ] Validation MSE < 1e-5 for all tests
- [ ] Unit tests pass (100% pass rate)
- [ ] Model checkpoint saved and loadable

### Overall Project Success Criteria

- Realistic pressure variation in generated handwriting
- SVG output compatible with pen plotters
- <10% performance overhead during sampling
- Opt-in feature (backward compatible)
- Well-documented and tested

---

## References

**Source Repository:** reference-repos/handwriting-model/
- Original implementation: `pressure_model/LTSM_model.py` (PyTorch)
- Training script: `pressure_model/train.py`
- Data preparation: `pressure_model/data_preparation.ipynb`

**Documentation:**
- `docs/REFERENCE_REPOS_ANALYSIS.md` - Detailed analysis of all reference repos
- `reference-repos/README.md` - Quick guide to reference repositories

**Original Author:** handwriting-model repository contributors

---

## Contact & Support

For questions or issues:
1. Review this document first
2. Check `docs/REFERENCE_REPOS_ANALYSIS.md` for technical details
3. Examine reference repository: `reference-repos/handwriting-model/`
4. Review test failures for specific error messages

---

**Document Version:** 1.0
**Last Updated:** November 3, 2025
**Next Review:** Upon Phase 1 completion
