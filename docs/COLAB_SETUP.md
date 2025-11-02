# Google Colab Pro Training Setup

Complete guide for training Scribe handwriting synthesis models on Google Colab Pro with GPU acceleration.

---

## Prerequisites

### Required
- **Google Colab Pro subscription** ($10/month)
  - Provides access to faster GPUs (T4/V100/A100)
  - Higher resource limits and priority access
  - 100 compute units/month (training uses ~10-20 CU)
- **Google Drive** with ~15 GB free space
  - Training data: 44 MB
  - Checkpoints: ~5-10 GB during training
  - Logs and samples: ~100 MB

### Optional but Recommended
- **Stable internet connection** (for 3-6 hour sessions)
- **Chrome browser** (best Colab compatibility)

---

## Quick Start

### 1. Prepare Your Google Drive

**Upload the entire `scribe` directory to your Google Drive root:**

```
Google Drive/
└── scribe/
    ├── train.py
    ├── model.py
    ├── sample.py
    ├── utils.py
    ├── verify_data.py
    ├── data/
    │   ├── strokes_training_data.cpkl  (44 MB - REQUIRED)
    │   └── styles/                      (26 .npy files)
    └── COLAB_TRAINING.ipynb
```

**Critical files to verify:**
- `data/strokes_training_data.cpkl` - 11,916 training samples (44 MB)
- `data/styles/*.npy` - 26 style files for character handwriting

### 2. Open the Notebook in Colab

**Option A: Direct upload**
1. Go to https://colab.research.google.com
2. File → Upload notebook
3. Select `COLAB_TRAINING.ipynb` from your Google Drive

**Option B: From Google Drive**
1. Navigate to the `scribe` folder in Google Drive
2. Double-click `COLAB_TRAINING.ipynb`
3. Select "Open with Google Colaboratory"

### 3. Enable GPU Runtime

**CRITICAL: Must enable GPU before training**

1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Set **Runtime shape** to **High-RAM** (optional, helps with large batches)
4. Click **Save**

**Verify GPU is enabled:**
- You should see "GPU" in the top-right corner
- Cell 2 will confirm GPU availability

### 4. Run the Training Cells

**Execute cells in order:**

1. **Cell 1:** Install TensorFlow 2.15 (~2 minutes)
2. **Cell 2:** Verify installation and GPU (~10 seconds)
3. **Cell 3:** Mount Google Drive (requires authorization)
4. **Cell 4:** Verify files are present (~5 seconds)
5. **Cell 5:** Verify training data (~10 seconds)
6. **Cell 6:** **START TRAINING** (3-6 hours)

**During training:**
- Do NOT close the browser tab
- Colab will auto-disconnect after 90 minutes of idle time
- If disconnected, re-run Cell 6 to resume from last checkpoint

---

## Training Configuration

### Recommended Settings (High Quality + Style Support)

```bash
python train.py \
    --rnn_size 400       # REQUIRED for style priming
    --nmixtures 20       # High quality output
    --nepochs 250        # Full training
    --batch_size 32      # Optimal for T4/V100
    --learning_rate 1e-4 # Standard
    --save_every 250     # Checkpoint every 250 steps
```

**Training time estimates:**
- **T4 GPU:** ~6 hours (standard Colab Pro)
- **V100 GPU:** ~3-4 hours (priority access)
- **A100 GPU:** ~2-3 hours (occasional availability)

### Alternative: Fast Training (Lower Quality)

For testing or prototyping:

```bash
python train.py \
    --rnn_size 100       # ⚠️ Cannot use style priming
    --nmixtures 8        # Lower quality
    --nepochs 100        # Faster convergence
    --batch_size 32
```

**Training time:** ~1.5 hours on T4

---

## Session Management

### Auto-Resume from Checkpoints

The training script **automatically resumes** from the latest checkpoint if interrupted.

**How it works:**
1. Checkpoints saved every 250 steps to `saved/checkpoint-XXXX`
2. On restart, script detects latest checkpoint
3. Training continues from that step

**To resume after disconnection:**
1. Reconnect to Colab (Runtime → Reconnect)
2. Re-run setup cells (1-5)
3. Re-run Cell 6 (training cell)
4. Script will print: `Restored from checkpoint-XXXX`

### Monitoring Training Progress

**Option 1: Live output in Cell 6**
- Loss values printed every 10 steps
- Checkpoints saved every 250 steps

**Option 2: Log files (Cell 7)**
```bash
tail -f /content/drive/MyDrive/scribe/logs/*.log
```

**What to look for:**
- `loss` should decrease over time (expect ~40 → ~5)
- `valid_loss` should track training loss
- `regloss` (running average) smooths out fluctuations

**Example output:**
```
Step 1000/125000, loss = 12.345, regloss = 12.340, valid_loss = 12.350, time = 0.120s
Step 1010/125000, loss = 12.210, regloss = 12.325, valid_loss = 12.305, time = 0.118s
```

### Session Limits

**Colab Pro limits:**
- **Maximum session:** 24 hours
- **Idle timeout:** 90 minutes (browser must stay open)
- **Background execution:** Not supported (keep tab active)

**Recommendations:**
- Train during time you can monitor (~4-6 hours)
- Use caffeine/stay-awake browser extensions
- Plan for potential reconnections

---

## After Training

### 1. Verify Model Quality

**Generate test samples (Cell 8):**

```bash
python sample.py \
    --text "The quick brown fox jumps over 1234567890" \
    --bias 1.0 \
    --format svg
```

**Quality indicators:**
- Text should be legible
- Strokes should be smooth (not jagged)
- Character spacing should be natural

**If quality is poor:**
- Check training loss (should be < 10 by epoch 250)
- Verify full training completed (125,000 steps)
- Consider training longer (--nepochs 300)

### 2. Test Style Priming

**Generate multi-character document:**

```bash
python sample.py \
    --lines "Report due by 09:00 on March 15th, 1985." \
            "Can't believe it - met him at 3:00pm!" \
            "Got your message (thanks!) - call me." \
    --biases 1.8 0.6 1.2 \
    --styles 0 3 7 \
    --format svg
```

**Each line uses a different handwriting style:**
- Style 0: Neat, precise (bias=1.8)
- Style 3: Messy, emotional (bias=0.6)
- Style 7: Balanced, natural (bias=1.2)

### 3. Download Results

**Option A: Direct download (Cell 9)**
- Uncomment the download code in Cell 9
- Downloads latest sample to your local machine

**Option B: Access via Google Drive**
- Open Google Drive in browser
- Navigate to `scribe/logs/figures/`
- Download SVG files directly

**Option C: Sync entire folder**
- Install Google Drive for Desktop
- Sync the `scribe` folder to local machine
- Access all checkpoints and samples locally

---

## Troubleshooting

### GPU Not Available

**Symptoms:**
- Cell 2 prints "WARNING: No GPU detected"
- Training is extremely slow (CPU-only)

**Solutions:**
1. Runtime → Change runtime type → GPU
2. Disconnect and reconnect (Runtime → Disconnect and delete runtime)
3. Check Colab Pro subscription is active
4. Try again later (GPU availability varies)

### Out of Memory (OOM) Errors

**Symptoms:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
1. **Reduce batch size:**
   ```bash
   --batch_size 16  # Instead of 32
   ```

2. **Enable memory growth (Cell 2 does this automatically):**
   ```python
   tf.config.experimental.set_memory_growth(gpu, True)
   ```

3. **Use smaller model:**
   ```bash
   --rnn_size 100   # Instead of 400
   --nmixtures 8    # Instead of 20
   ```

### Session Disconnected During Training

**Symptoms:**
- Browser tab shows "Reconnecting..."
- Training output stops updating

**Solutions:**
1. Click "Reconnect" in Colab interface
2. Re-run setup cells (1-5)
3. Re-run training cell (6)
4. Script will auto-resume from latest checkpoint

**Prevention:**
- Keep browser tab active (don't minimize)
- Disable computer sleep mode
- Use browser extension to prevent idle timeout

### File Not Found Errors

**Symptoms:**
```
FileNotFoundError: data/strokes_training_data.cpkl not found
```

**Solutions:**
1. Verify Google Drive is mounted (Cell 3)
2. Check file exists in Drive: `scribe/data/strokes_training_data.cpkl`
3. Ensure correct path in Cell 3 (`PROJECT_DIR`)
4. Re-upload scribe folder if files are missing

### TensorFlow Version Mismatch

**Symptoms:**
```
AttributeError: module 'tensorflow' has no attribute 'X'
```

**Solutions:**
1. Verify Cell 2 shows "TensorFlow version: 2.15.0"
2. Re-run Cell 1 to reinstall TensorFlow 2.15
3. Restart runtime (Runtime → Restart runtime)
4. Check for conflicting package installations

### Training Loss Not Decreasing

**Symptoms:**
- Loss stays high (>20) after 10,000 steps
- Loss becomes NaN (not a number)

**Solutions:**
1. **Check data integrity:**
   ```bash
   python verify_data.py
   ```

2. **Reduce learning rate:**
   ```bash
   --learning_rate 1e-5  # Instead of 1e-4
   ```

3. **Increase gradient clipping:**
   ```bash
   --grad_clip 5.0  # Instead of 10.0
   ```

4. **Restart training from scratch:**
   - Delete `saved/` directory
   - Re-run Cell 6

### Style Priming Not Working

**Symptoms:**
```
ValueError: Style priming requires rnn_size=400
```

**Solutions:**
1. **Train with correct model size:**
   ```bash
   --rnn_size 400  # REQUIRED for styles
   ```

2. **Verify style files exist:**
   ```bash
   ls data/styles/
   # Should show 26 .npy files (style-0-chars.npy through style-12-strokes.npy)
   ```

3. **Use valid style IDs (0-12):**
   ```bash
   --styles 0 3 7  # Valid
   --styles 15     # Invalid (only 0-12 available)
   ```

---

## Cost Estimation

### Compute Units (CU) Usage

**Colab Pro provides 100 CU/month with $10 subscription**

**CU consumption rates (approximate):**
- **T4 GPU:** ~3 CU/hour
- **V100 GPU:** ~7 CU/hour
- **A100 GPU:** ~15 CU/hour

**Training cost estimates:**

| Configuration | GPU | Time | CU Used | Cost Beyond $10 |
|--------------|-----|------|---------|-----------------|
| Full training (rnn_size=400, nepochs=250) | T4 | 6 hours | 18 CU | ~$0.80 |
| Full training | V100 | 3 hours | 21 CU | ~$1.10 |
| Full training | A100 | 2 hours | 30 CU | ~$3.00 |
| Fast training (rnn_size=100, nepochs=100) | T4 | 1.5 hours | 4.5 CU | $0 |

**Note:** These are estimates. Actual CU usage depends on:
- GPU availability (Colab assigns GPUs automatically)
- Session idle time
- Number of training restarts

### Recommendations

1. **Minimize costs:**
   - Train during off-peak hours (better GPU availability)
   - Keep browser active (avoid reconnections)
   - Use fast training for testing, full training for production

2. **Monitor CU usage:**
   - Check usage: Colab menu → Resources → View resources
   - 100 CU allows ~2-3 full training runs per month

3. **Alternative: Colab Pay As You Go**
   - No monthly subscription
   - Pay per CU consumed (~$0.10/CU)
   - Better for occasional training

---

## Best Practices

### 1. Development Workflow

**Recommended iteration cycle:**

1. **Local testing (no Colab):**
   - Test code changes locally with `--nepochs 1`
   - Verify data loading works
   - Check for syntax errors

2. **Fast training (Colab - 1.5 hours):**
   - Use rnn_size=100, nepochs=100
   - Verify training runs without errors
   - Check checkpoint saving works

3. **Full training (Colab - 6 hours):**
   - Use rnn_size=400, nepochs=250
   - Monitor first 30 minutes for issues
   - Let complete training run overnight if needed

### 2. Checkpoint Management

**Keep checkpoints organized:**

```
scribe/saved/
├── checkpoint              # Latest checkpoint pointer
├── checkpoint-25000.index
├── checkpoint-25000.data-00000-of-00001
├── checkpoint-50000.index
├── checkpoint-50000.data-00000-of-00001
└── ...
```

**Cleanup old checkpoints:**
- Training automatically keeps last 5 checkpoints
- Manually delete old checkpoints to save Drive space
- Keep final checkpoint for production use

### 3. Experiment Tracking

**Document training runs:**

Create a training log in `training_log.md`:

```markdown
## Training Run 2025-01-15

**Configuration:**
- rnn_size: 400
- nmixtures: 20
- nepochs: 250
- GPU: T4
- Time: 5.5 hours

**Results:**
- Final loss: 6.234
- Valid loss: 6.189
- Sample quality: Good
- Notes: Style priming works well
```

---

## Advanced Configuration

### Custom Alphabet

**To train with custom character set:**

Edit `train.py` line 183:

```python
parser.add_argument('--alphabet', type=str,
    default=' abcdefghijklmnopqrstuvwxyz',  # Lowercase only
    help='Custom alphabet')
```

**Note:** Must use same alphabet for training and sampling

### Multi-GPU Training

**Colab Pro does not support multi-GPU**, but you can:

1. Use `tf.distribute.MirroredStrategy` for local multi-GPU
2. Consider Google Cloud AI Platform for production training
3. Use TPUs with Colab Pro (requires code modifications)

### Custom Data

**To train on your own handwriting:**

1. Convert to IAM format (see `utils.py` DataLoader)
2. Upload to `data/custom_training_data.cpkl`
3. Modify `train.py` to use custom data:
   ```bash
   --data_dir data/custom_training_data.cpkl
   ```

---

## Additional Resources

**Official Documentation:**
- Scribe README: `/README.md`
- Migration Guide: `/docs/MIGRATION_GUIDE.md`
- Claude Instructions: `/CLAUDE.md`

**Original Research:**
- Alex Graves (2013): "Generating Sequences With Recurrent Neural Networks"
- arXiv: https://arxiv.org/abs/1308.0850

**Google Colab:**
- Official FAQ: https://research.google.com/colaboratory/faq.html
- Pricing: https://colab.research.google.com/signup

**Community:**
- Report issues: Create GitHub issue in your repository
- Share results: Consider contributing trained models back

---

## FAQ

**Q: Can I use free Colab instead of Colab Pro?**

A: Technically yes, but not recommended:
- Free tier has strict session limits (~2 hours)
- GPU availability is limited
- No guaranteed GPU access
- Training will require multiple reconnections

**Q: Can I train on my local machine instead?**

A: Yes, if you have a CUDA-capable GPU:
```bash
# Install dependencies
pip install -r requirements.txt

# Train locally
python train.py --rnn_size 400 --nmixtures 20 --nepochs 250
```

**Q: How do I convert SVG to gcode for my pen plotter?**

A: Several options:
- Inkscape with gcode extension
- Universal Gcode Sender
- Online converters (search "SVG to gcode")
- Custom scripts using svg.path library

**Q: Can I use the trained model commercially?**

A: Depends on:
- **IAM dataset license** (academic/research use only)
- Your intended use case
- Check original licensing terms

**Q: How can I improve sample quality?**

A:
1. Train longer (--nepochs 300+)
2. Increase model size (--rnn_size 900)
3. Adjust bias parameter (1.5-2.0 for neater)
4. Use style priming (--styles)

**Q: Training completed but samples look wrong?**

A: Verify:
1. Same alphabet used for training and sampling
2. Correct checkpoint loaded (check --save_path)
3. Model size matches (rnn_size=400)
4. All characters in text are in alphabet

---

**Last Updated:** 2025-11-01
**Scribe Version:** Python 3.11 + TensorFlow 2.15
**Colab Compatibility:** Tested with Colab Pro (2025)
