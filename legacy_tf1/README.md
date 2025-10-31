# Legacy TensorFlow 1.x Implementation

This directory contains the **original TensorFlow 1.x implementation** of the Scribe handwriting synthesis model.

## ⚠️ Status: LEGACY - Reference Only

These files are **Python 3 compatible** but require **TensorFlow 1.15** to run. They are kept for:
- Historical reference
- Comparison with TensorFlow 2.x implementation
- Educational purposes (understanding migration from TF 1.x → TF 2.x)

## Files in This Directory

- **model.py** - Original TensorFlow 1.x model architecture using:
  - `tf.placeholder()` for inputs
  - `tf.Session()` for execution
  - `tf.contrib.rnn.LSTMCell` for LSTM layers
  - `tf.contrib.legacy_seq2seq.rnn_decoder` for sequence processing

- **run.py** - Original training and sampling orchestration script:
  - Session-based training loop
  - Uses `sess.run()` for execution
  - Handles both training and sampling modes

- **sample.py** - Original sampling utilities:
  - Session-based generation
  - Style priming (experimental)
  - Visualization functions

## 🚫 Do NOT Use for New Projects

**For current usage, use the TensorFlow 2.x implementation in the parent directory:**

```bash
# Modern TensorFlow 2.x commands (USE THESE)
python3 train.py --rnn_size 400 --nmixtures 20
python3 sample.py --text "Hello World" --bias 1.5
```

## 📚 If You Need to Run These Files

**Requirements:**
- Python 3.8+ (NOT Python 2.7)
- TensorFlow 1.15.x (NOT TensorFlow 2.x)

**Setup (separate environment):**
```bash
# Create TF 1.x environment
python3.8 -m venv venv-tf1
source venv-tf1/bin/activate

# Install TensorFlow 1.15
pip install tensorflow==1.15.5

# Run legacy code
python run.py --train
python run.py --sample --text "test"
```

## 🔄 Migration Notes

These files were part of Phase 1 migration (Python 2.7 → Python 3.11):
- ✅ All Python 2.7 syntax converted to Python 3
- ✅ `cPickle` → `pickle`
- ✅ `xrange()` → `range()`
- ✅ Print statements → `print()` functions
- ⚠️ Still uses TensorFlow 1.x API (session-based execution)

**For complete migration details, see:** `../docs/MIGRATION_GUIDE.md`

## 📖 Key Architectural Differences vs TF 2.x

| TensorFlow 1.x (This Directory) | TensorFlow 2.x (Parent Directory) |
|--------------------------------|-----------------------------------|
| `tf.Session()` | Eager execution (no sessions) |
| `tf.placeholder()` + `feed_dict` | Direct function arguments |
| `tf.contrib.rnn.*` | `tf.keras.layers.LSTM` |
| `sess.run()` | Direct calls to model |
| `tf.train.Saver` | `tf.train.Checkpoint` |
| `tf.global_variables()` | `model.variables` |

## ✅ Recommended Path

**Use the TensorFlow 2.x implementation** in the parent directory:
- Modern TensorFlow 2.15+ API
- Python 3.11 compatible
- Eager execution (easier debugging)
- Better performance
- Active maintenance

---

**Last Updated:** 2025-10-31
**Status:** Archived for reference only
**Replacement:** See `../model.py`, `../train.py`, `../sample.py`
