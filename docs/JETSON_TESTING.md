# Running Tests on NVIDIA Jetson

Complete guide for setting up and running the Scribe test suite on NVIDIA Jetson hardware.

---

## Overview

This guide covers:
- Setting up the Scribe project on NVIDIA Jetson
- Installing dependencies (TensorFlow + testing packages)
- Running the comprehensive test suite
- Viewing coverage reports
- Troubleshooting common issues

**Expected time:** 15-20 minutes for complete setup + testing

**Tested on:**
- Jetson Nano, Jetson Xavier NX, Jetson AGX Xavier, Jetson Orin
- JetPack 4.6+ (Python 3.6+) and JetPack 5.x+ (Python 3.8+)

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] **NVIDIA Jetson board** (any model: Nano, Xavier NX, AGX, Orin, etc.)
- [ ] **JetPack installed** with Python 3.8+ (JetPack 5.x recommended)
  - Check: `python3 --version` (should show 3.8 or higher)
- [ ] **Git installed**
  - Check: `git --version`
  - Install: `sudo apt-get install git`
- [ ] **Internet connection** (for pip packages)
- [ ] **~500 MB free disk space** (includes data, tests, coverage reports)
- [ ] **Repository access** (GitHub credentials if private repo)

---

## Step 1: Clone Repository

### 1.1 Clone from Git

```bash
# Navigate to desired directory
cd ~

# Clone repository
git clone <your-repo-url> scribe

# Navigate to project
cd scribe
```

### 1.2 Verify File Structure

```bash
# Check required files exist
ls -la

# Expected output should include:
# - model.py, train.py, sample.py, utils.py
# - data/ directory with strokes_training_data.cpkl
# - tests/ directory with conftest.py
# - requirements.txt, requirements-test.txt
```

**Time:** ~1-2 minutes (depending on network speed)

---

## Step 2: Install Dependencies

### 2.1 Verify Python Environment

```bash
# Check Python version (need 3.8+)
python3 --version

# Check pip is installed
pip3 --version

# Upgrade pip if needed
python3 -m pip install --upgrade pip
```

### 2.2 Install TensorFlow for Jetson

**⚠️ IMPORTANT:** Use NVIDIA's official TensorFlow build for Jetson, not PyPI version.

**For JetPack 5.x (Python 3.8+):**

```bash
# Install dependencies
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

# Install TensorFlow (NVIDIA pre-built wheel)
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 tensorflow==2.12.0+nv23.06
```

**For JetPack 4.6.x (Python 3.6):**

```bash
# Install dependencies
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

# Install TensorFlow 2.7 (compatible with JetPack 4.6)
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==2.7.0+nv22.1
```

**Alternative (if above fails):**

```bash
# Use NVIDIA's forums download links
# See: https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson/
```

### 2.3 Install Core Requirements

```bash
# Install project dependencies
pip3 install -r requirements.txt

# Expected packages:
# - numpy==1.26.4 (or compatible version for Jetson)
# - matplotlib==3.8.3
# - scipy==1.12.0
# - svgwrite==1.4.3
```

### 2.4 Install Testing Requirements

```bash
# Install test dependencies
pip3 install -r requirements-test.txt

# Expected packages:
# - pytest==7.4.3
# - pytest-cov==4.1.0
# - pytest-xdist==3.5.0 (for parallel execution)
# - pytest-mock==3.12.0
# - hypothesis==6.92.1 (property-based testing)
```

### 2.5 Verify Installation

```bash
# Test TensorFlow import
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"

# Expected output:
# TensorFlow: 2.12.0 (or 2.7.0 for JetPack 4.6)
# GPU available: True

# Test pytest
pytest --version

# Expected output:
# pytest 7.4.3
```

**Time:** ~5-10 minutes (depending on network and Jetson model)

---

## Step 3: Verify Data Files

Before running tests, verify the training data is intact.

```bash
# Run data verification script
python3 verify_data.py

# Expected output:
# ✅ Found 11,916 stroke samples
# ✅ Found 11,916 text labels (matches stroke count)
# ✅ strokes_training_data.cpkl is VALID and ready for migration!
# 🎉 SUCCESS! All checks passed!
```

**If verification fails:**
- Ensure `data/strokes_training_data.cpkl` exists (44 MB file)
- Re-clone repository if file is missing or corrupted

**Time:** ~10 seconds

---

## Step 4: Run Tests

### 4.1 Quick Smoke Tests (Recommended First)

```bash
# Run fast sanity checks (< 30 seconds)
pytest -m smoke -v

# Expected output:
# tests/unit/test_smoke.py::TestImports::test_tensorflow_imports PASSED
# tests/unit/test_smoke.py::TestImports::test_project_imports PASSED
# tests/unit/test_smoke.py::TestModelInstantiation::test_create_tiny_model PASSED
# ...
# ============ 20 passed in 15.23s ============
```

**What smoke tests check:**
- All modules import successfully
- TensorFlow is properly installed
- Model can be instantiated
- Data files exist
- Basic functionality works

**Time:** < 30 seconds

### 4.2 Unit Tests (Comprehensive)

```bash
# Run all unit tests
pytest tests/unit -v

# Expected output:
# tests/unit/test_model.py::TestModelInitialization::test_model_creates_three_lstm_layers PASSED
# tests/unit/test_data_loader.py::TestDataLoader::test_dataloader_loads_existing_file PASSED
# ...
# ============ 250+ passed in 2-5 minutes ============
```

**What unit tests check:**
- Model architecture (LSTM layers, MDN output, attention mechanism)
- Data loading (DataLoader, one-hot encoding, batching)
- Sampling functions (Gaussian sampling, style loading, generation)
- SVG output (coordinate conversion, denoising, file generation)
- Training logic (optimizers, learning rate decay, checkpointing)
- Utilities (Logger, to_one_hot, data preprocessing)

**Time:** ~2-5 minutes (varies by Jetson model)

### 4.3 Integration Tests (Optional)

```bash
# Run integration tests (multi-component workflows)
pytest tests/integration -v

# Expected output:
# tests/integration/test_training_loop.py::TestTrainingStep::test_single_training_step PASSED
# tests/integration/test_checkpointing.py::TestCheckpointSave::test_checkpoint_creates_files PASSED
# ...
# ============ 60+ passed in 3-5 minutes ============
```

**What integration tests check:**
- Complete training loop (model + optimizer + data loader)
- Checkpoint save/restore workflows
- End-to-end generation pipeline
- Multi-line document generation

**Time:** ~3-5 minutes

### 4.4 Full Test Suite with Coverage

```bash
# Run all tests with coverage report
pytest --cov=. --cov-report=html --cov-report=term-missing -v

# Expected output:
# ...tests running...
# ----------- coverage: platform linux, python 3.8.10-final-0 -----------
# Name                           Stmts   Miss  Cover   Missing
# ------------------------------------------------------------
# model.py                         287     57    80%   124-135, 245-256
# utils.py                         134     13    90%   45-48, 112-115
# sample.py                        234     28    88%   67-72, 189-195
# ...
# ------------------------------------------------------------
# TOTAL                           1542    154    90%
#
# ============ 315+ passed in 5-10 minutes ============
```

**Time:** ~5-10 minutes

### 4.5 Parallel Execution (Faster)

```bash
# Run tests in parallel (use all CPU cores)
pytest -n auto -v

# Expected speedup: 2-4x faster on multi-core Jetson models
```

**Time:** ~2-3 minutes for full suite (vs 5-10 minutes sequential)

---

## Step 5: View Coverage Report

### 5.1 Generate HTML Report (if not already generated)

```bash
# Generate coverage report
pytest --cov=. --cov-report=html

# Creates htmlcov/ directory
```

### 5.2 Open in Browser

```bash
# On Jetson with desktop environment:
firefox htmlcov/index.html

# or
chromium-browser htmlcov/index.html

# On headless Jetson (copy to desktop):
# Use scp to copy htmlcov/ folder to your desktop machine
scp -r htmlcov/ user@desktop:~/scribe-coverage/
```

### 5.3 Interpret Coverage Metrics

**Coverage report shows:**
- **Overall coverage:** Should be ~90%+
- **Per-file coverage:** Most files at 80-90%
- **Missing lines:** Click on files to see uncovered lines (highlighted in red)

**Coverage targets:**
- **90%+ overall** - Excellent ✅
- **80-90% per file** - Good ✅
- **< 80% per file** - Needs improvement ⚠️

**Expected coverage by file:**
- `model.py`: ~80% (attention mechanism is complex)
- `utils.py`: ~90% (well-tested data loading)
- `sample.py`: ~88% (comprehensive sampling tests)
- `train.py`: ~85% (training logic tested)
- `svg_output.py`: ~85% (SVG generation tested)
- `verify_data.py`: ~90% (validation logic tested)
- `character_profiles.py`: ~85% (profile templates tested)

**Time:** ~1 minute to view

---

## Troubleshooting

### Issue: TensorFlow Import Fails

**Symptom:**
```
ImportError: libcublas.so.11: cannot open shared object file
```

**Solution:**
```bash
# Ensure CUDA libraries are in path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Add to ~/.bashrc to make permanent
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

### Issue: numpy Version Conflict

**Symptom:**
```
ERROR: numpy 1.26.4 has requirement python>=3.9, but you have python 3.8
```

**Solution (for Python 3.8):**
```bash
# Install compatible numpy version
pip3 install 'numpy<1.24'

# Update requirements.txt locally if needed
```

---

### Issue: Out of Memory During Tests

**Symptom:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution:**
```bash
# Tests use tiny models, but Jetson Nano (2GB RAM) may struggle

# Option 1: Enable swap (Jetson Nano)
sudo systemctl disable nvzramconfig
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo bash -c 'echo "/swapfile swap swap defaults 0 0" >> /etc/fstab'

# Option 2: Run tests sequentially (not parallel)
pytest -v  # Instead of pytest -n auto

# Option 3: Run only smoke tests
pytest -m smoke -v
```

---

### Issue: Tests Fail with CUDA Errors

**Symptom:**
```
CUDA_ERROR_OUT_OF_MEMORY
```

**Solution:**
```bash
# Tests are configured for CPU-only, but may detect GPU

# Force CPU-only testing
export CUDA_VISIBLE_DEVICES=""
pytest -v

# Or disable GPU in test configuration (already done in conftest.py)
# Tests use configure_tensorflow fixture which limits GPU memory
```

---

### Issue: pytest Not Found

**Symptom:**
```
bash: pytest: command not found
```

**Solution:**
```bash
# Ensure pip3 install location is in PATH
which pytest  # Check if it exists

# If not in PATH, use python3 -m pytest instead
python3 -m pytest -v

# Or add to PATH
export PATH=$PATH:~/.local/bin
echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
```

---

### Issue: Permission Errors

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: 'htmlcov/index.html'
```

**Solution:**
```bash
# Fix permissions
chmod -R 755 htmlcov/

# Or run with appropriate user
sudo chown -R $USER:$USER htmlcov/
```

---

### Issue: Slow Test Execution

**Performance tips:**

```bash
# 1. Run only fast tests
pytest -m "not slow" -v

# 2. Use parallel execution
pytest -n auto -v

# 3. Skip integration tests
pytest tests/unit -v

# 4. Run smoke tests only (< 30 seconds)
pytest -m smoke -v

# 5. Disable coverage (faster)
pytest -v --no-cov
```

**Expected performance by Jetson model:**
- **Jetson Nano:** 5-10 minutes (full suite)
- **Jetson Xavier NX:** 3-5 minutes
- **Jetson AGX Xavier:** 2-3 minutes
- **Jetson Orin:** 1-2 minutes

---

## Jetson-Specific Notes

### GPU vs CPU Testing

**Tests are configured for CPU-only** to ensure deterministic behavior:
- `conftest.py` fixture `configure_tensorflow` disables GPU for tests
- This prevents flaky tests due to non-deterministic GPU operations
- GPU is available for training (use `train.py` after tests pass)

**To test GPU functionality separately:**
```python
import tensorflow as tf
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
```

---

### Memory Optimization

**Jetson Nano (2GB RAM):**
- Enable swap (see troubleshooting above)
- Run tests sequentially: `pytest -v` (no `-n auto`)
- Run smoke tests only: `pytest -m smoke -v`

**Jetson Xavier NX / AGX (8GB+ RAM):**
- Full test suite runs comfortably
- Parallel execution recommended: `pytest -n auto -v`

**Jetson Orin (16GB+ RAM):**
- No limitations
- Full parallel execution

---

### TensorFlow Versions

| JetPack | Python | TensorFlow | Notes |
|---------|--------|------------|-------|
| 4.6     | 3.6    | 2.7.0      | Legacy, still supported |
| 5.0     | 3.8    | 2.11.0     | Stable |
| 5.1     | 3.8    | 2.12.0     | Recommended |
| 6.0     | 3.10   | 2.13.0     | Latest |

**⚠️ Important:** Always use NVIDIA's pre-built wheels, not PyPI `pip install tensorflow`

---

### Power Mode

For faster test execution, set maximum performance mode:

```bash
# Check current power mode
sudo nvpmodel -q

# Set to max performance (model-dependent)
# Jetson Nano: MODE_10W
sudo nvpmodel -m 0

# Jetson Xavier NX: MODE_15W or MODE_20W
sudo nvpmodel -m 2

# Jetson AGX / Orin: MAXN
sudo nvpmodel -m 0

# Verify
sudo jetson_clocks --show
```

---

## Next Steps

### After Tests Pass

1. **Run training on Jetson** (optional)
   ```bash
   # Quick training test (10 epochs)
   python3 train.py --nepochs 10 --save_every 100
   ```

2. **Generate samples**
   ```bash
   # Requires trained model checkpoint
   python3 sample.py --text "Hello from Jetson!"
   ```

3. **Set up CI/CD** (optional)
   - Configure GitHub Actions or Jenkins
   - Automate test runs on Jetson
   - See `.github/workflows/` for examples

4. **Monitor system resources**
   ```bash
   # Install jtop for Jetson monitoring
   sudo pip3 install jetson-stats
   sudo jtop
   ```

---

## Quick Reference

### Essential Commands

```bash
# Setup (one-time)
git clone <repo> scribe && cd scribe
pip3 install -r requirements.txt -r requirements-test.txt

# Quick validation (< 30 seconds)
pytest -m smoke -v

# Full test suite
pytest --cov=. --cov-report=html -v

# View coverage
firefox htmlcov/index.html

# Parallel execution (faster)
pytest -n auto -v
```

### File Locations

- **Test files:** `tests/unit/`, `tests/integration/`
- **Test config:** `pytest.ini`, `tests/conftest.py`
- **Coverage report:** `htmlcov/index.html`
- **Training data:** `data/strokes_training_data.cpkl` (44 MB)
- **Style data:** `data/styles/*.npy` (26 files, 13 styles)

### Support

- **Project README:** `README.md`
- **Testing guidelines:** `CLAUDE.md` (Testing Best Practices section)
- **Migration guide:** `docs/MIGRATION_GUIDE.md`
- **Colab training:** `docs/COLAB_SETUP.md`
- **GitHub issues:** Report issues at repository issue tracker

---

## Summary Checklist

Use this checklist to verify your Jetson testing setup:

- [ ] Repository cloned successfully
- [ ] Python 3.8+ verified
- [ ] TensorFlow installed (NVIDIA build, not PyPI)
- [ ] Core requirements installed (`requirements.txt`)
- [ ] Test requirements installed (`requirements-test.txt`)
- [ ] Data files verified (`python3 verify_data.py`)
- [ ] Smoke tests pass (`pytest -m smoke -v`)
- [ ] Unit tests pass (`pytest tests/unit -v`)
- [ ] Coverage report generated (`pytest --cov=.`)
- [ ] Coverage ≥90% overall
- [ ] HTML coverage report viewable

**If all checkboxes complete:** ✅ Your Jetson is ready for development and deployment!

---

**Document version:** 1.0
**Last updated:** 2024
**Tested on:** Jetson Nano, Xavier NX, AGX Xavier, Orin
**Maintained by:** Scribe Project Team
