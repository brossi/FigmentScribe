# Pressure Model Implementation - Next Steps

**Quick Reference Guide**
**Status:** Phase 1 partially complete, ready for execution when dependencies available

---

## What's Been Completed ✅

1. ✅ **Training data** copied from reference repository (11 MB)
2. ✅ **TensorFlow model** implemented (`pressure_model.py`)
3. ✅ **Weight conversion script** created (`convert_pressure_weights.py`)
4. ✅ **Validation script** created (`validate_pressure_conversion.py`)
5. ✅ **Unit tests** written (`tests/unit/test_pressure_model.py`)
6. ✅ **Documentation** created (`docs/PRESSURE_MODEL_IMPLEMENTATION.md`)

**Time invested:** ~6-8 hours of development

---

## What's Next (15-20 minutes when ready)

### Prerequisites

Install dependencies:
```bash
pip install torch tensorflow
```

### Step 1: Convert Weights (5-10 min)

```bash
python3 convert_pressure_weights.py
```

**Expected output:**
```
========================================================
PYTORCH → TENSORFLOW WEIGHT CONVERSION
Pressure Prediction Model
========================================================
Loading PyTorch checkpoint from: data/pressure_data/model_epoch_4.pth
  ✓ Found 'model_state_dict' in checkpoint

Converting LSTM layer 0:
  ✓ Weights converted

Converting LSTM layer 1:
  ✓ Weights converted

Converting FC layer:
  ✓ Weights converted

Saving TensorFlow checkpoint to: saved/pressure_model
  ✓ Checkpoint saved

CONVERSION SUCCESSFUL!
```

### Step 2: Validate Conversion (2-3 min)

```bash
python3 validate_pressure_conversion.py
```

**Expected output:**
```
========================================================
PRESSURE MODEL VALIDATION
PyTorch vs TensorFlow Comparison
========================================================

TEST 1: Random Input
  ✓ Shapes match
  ✓ MSE < 1e-5 (PASS)

TEST 2: Validation Data
  ✓ Shapes match
  ✓ MSE < 1e-5 (PASS)

TEST 3: Single Sequence
  ✓ Shapes match
  ✓ MSE < 1e-5 (PASS)

========================================================
✓ ALL TESTS PASSED
========================================================
```

### Step 3: Run Unit Tests (30 sec)

```bash
pytest tests/unit/test_pressure_model.py -v
```

**Expected output:**
```
tests/unit/test_pressure_model.py::test_model_creation PASSED
tests/unit/test_pressure_model.py::test_forward_pass_single_batch PASSED
tests/unit/test_pressure_model.py::test_predict_pressure_numpy_input PASSED
...
================ X passed in Y.YYs ================
```

### Step 4: Quick Smoke Test

```python
# Test in Python REPL
python3
>>> from pressure_model import load_pressure_model
>>> model = load_pressure_model('saved/pressure_model')
>>> import numpy as np
>>> strokes = np.random.randn(100, 3).astype(np.float32)
>>> pressure = model.predict_pressure(strokes)
>>> print(f"Pressure shape: {pressure.shape}, Range: [{pressure.min():.2f}, {pressure.max():.2f}]")
Pressure shape: (100,), Range: [0.00, 1.00]
```

---

## After Phase 1 Completion

Once Steps 1-4 pass, you're ready for Phase 2!

### Phase 2: Integration with sample.py (Week 2)

**Tasks:**
1. Add `load_pressure_model()` to sample.py
2. Integrate `predict_pressure()` in sampling loop
3. Add CLI flags: `--pressure`, `--pressure_scale`
4. Test end-to-end pressure prediction during sampling

**Goal:** Generate handwriting with pressure data

### Phase 3: SVG Variable Thickness (Week 3)

**Tasks:**
1. Modify `svg_output.py` for per-segment stroke-width
2. Map pressure to line width
3. Test with pen plotter tools

**Goal:** SVG files with realistic variable thickness

### Phase 4: Testing & Documentation (Week 4)

**Tasks:**
1. Final testing and refinement
2. Update main documentation
3. Create examples and demo notebook

**Goal:** Production-ready feature

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch
```

### "ModuleNotFoundError: No module named 'tensorflow'"
```bash
pip install tensorflow
```

### "Checkpoint not found"
Make sure you ran Step 1 (convert_pressure_weights.py) first.

### "Validation MSE too high"
This indicates the weight conversion may have issues. Check:
1. PyTorch checkpoint is correct file (model_epoch_4.pth)
2. Conversion script completed without errors
3. Review weight shapes in conversion output

---

## Files Created

| File | Size | Purpose |
|------|------|---------|
| `pressure_model.py` | 8.7 KB | TensorFlow model implementation |
| `convert_pressure_weights.py` | 10.9 KB | PyTorch → TF converter |
| `validate_pressure_conversion.py` | 9.5 KB | Validation script |
| `tests/unit/test_pressure_model.py` | 11.2 KB | Unit tests |
| `docs/PRESSURE_MODEL_IMPLEMENTATION.md` | 15.1 KB | Full documentation |
| `data/pressure_data/` | 11 MB | Training data & checkpoint |

---

## Quick Commands Cheatsheet

```bash
# Complete Phase 1
python3 convert_pressure_weights.py
python3 validate_pressure_conversion.py
pytest tests/unit/test_pressure_model.py -v

# Test model loading
python3 -c "from pressure_model import load_pressure_model; m = load_pressure_model(); print('✓ Model loaded')"

# Run test script in pressure_model.py
python3 pressure_model.py

# Check checkpoint files
ls -lh saved/pressure_model/
```

---

## Documentation

**Detailed docs:** `docs/PRESSURE_MODEL_IMPLEMENTATION.md`
**Reference analysis:** `docs/REFERENCE_REPOS_ANALYSIS.md`
**Source code:** `reference-repos/handwriting-model/pressure_model/`

---

**Ready to proceed?** Run Step 1 above!

**Questions?** Review `docs/PRESSURE_MODEL_IMPLEMENTATION.md` for full details.

**Last Updated:** November 3, 2025
