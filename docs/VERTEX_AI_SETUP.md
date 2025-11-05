# Google Cloud Vertex AI Training Setup

Complete guide for training Scribe handwriting synthesis on Google Cloud Vertex AI with GPU acceleration.

**Why Vertex AI?**
- ✅ **Command-line submission** - No manual notebook interaction
- ✅ **Cheaper than Colab Pro** - T4 GPU at $0.79/hr vs Colab's $1.67/hr
- ✅ **More control** - Choose exact GPU type (T4, V100, A100)
- ✅ **Programmatic monitoring** - Track progress from CLI
- ✅ **Auto-resume** - Jobs continue after disconnection

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Quick Start](#quick-start)
4. [Submitting Training Jobs](#submitting-training-jobs)
5. [Monitoring Jobs](#monitoring-jobs)
6. [Downloading Results](#downloading-results)
7. [Cost Estimates](#cost-estimates)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **gcloud CLI** (Google Cloud command-line tool)
   ```bash
   # macOS
   brew install google-cloud-sdk

   # Verify installation
   gcloud --version
   ```

2. **Docker** (for building training containers)
   ```bash
   # macOS - download from https://www.docker.com/products/docker-desktop
   # Or with Homebrew:
   brew install --cask docker

   # Verify installation
   docker --version
   ```

3. **Python 3.11+** (for submission scripts)
   ```bash
   python3 --version  # Should show 3.11 or higher
   ```

### Google Cloud Account

1. **Google Cloud account** with billing enabled
   - New users get $300 free credit
   - Sign up at: https://console.cloud.google.com

2. **Enable billing** on your GCP project
   - Required for Vertex AI and GPU access
   - Follow: https://cloud.google.com/billing/docs/how-to/modify-project

---

## Initial Setup

### Step 1: Create GCP Project

```bash
# Set your project ID (must be globally unique)
export PROJECT_ID="my-scribe-project"

# Create project
gcloud projects create $PROJECT_ID --name="Scribe Training"

# Set as default project
gcloud config set project $PROJECT_ID
```

**Or use existing project:**
```bash
# List your projects
gcloud projects list

# Set existing project as default
gcloud config set project YOUR_EXISTING_PROJECT_ID
```

### Step 2: Enable Required APIs

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable Container Registry (for Docker images)
gcloud services enable containerregistry.googleapis.com

# Enable Cloud Storage (for data and results)
gcloud services enable storage.googleapis.com

# Verify APIs are enabled
gcloud services list --enabled | grep -E "(aiplatform|container|storage)"
```

**Expected output:**
```
aiplatform.googleapis.com       Vertex AI API
containerregistry.googleapis.com Container Registry API
storage.googleapis.com          Cloud Storage JSON API
```

### Step 3: Authenticate

```bash
# Login with your Google account
gcloud auth login

# Set up Application Default Credentials (for Python scripts)
gcloud auth application-default login

# Configure Docker for Google Container Registry
gcloud auth configure-docker
```

### Step 4: Create GCS Bucket

Create a Google Cloud Storage bucket to store training data and results:

```bash
# Choose a globally unique bucket name
export BUCKET_NAME="my-scribe-training"

# Create bucket (choose region close to you)
gsutil mb -l us-central1 gs://$BUCKET_NAME

# Verify bucket created
gsutil ls gs://$BUCKET_NAME
```

**Region recommendations:**
- `us-central1` - Iowa, USA (cheapest, most GPU availability)
- `us-west1` - Oregon, USA
- `europe-west4` - Netherlands
- `asia-east1` - Taiwan

### Step 5: Upload Training Data

```bash
# Upload preprocessed IAM dataset to GCS
gsutil -m cp -r data/strokes_training_data.cpkl gs://$BUCKET_NAME/data/

# Verify upload
gsutil ls -lh gs://$BUCKET_NAME/data/
```

**Expected output:**
```
44.1 MB  gs://my-scribe-training/data/strokes_training_data.cpkl
```

---

## Quick Start

**Once setup is complete, training is a single command:**

```bash
# Submit production training job (250 epochs, V100 GPU)
python3 submit_training.py \
    --rnn_size 400 \
    --nmixtures 20 \
    --nepochs 250 \
    --gpu-type v100 \
    --bucket my-scribe-training

# Estimated cost: ~$8
# Estimated time: ~2.5 hours
```

That's it! The script will:
1. ✅ Build Docker container
2. ✅ Push to Google Container Registry
3. ✅ Submit job to Vertex AI
4. ✅ Provide monitoring commands

---

## Submitting Training Jobs

### Basic Submission

```bash
python3 submit_training.py --bucket YOUR_BUCKET_NAME --gpu-type t4
```

### Common Configurations

**1. Quick Test (10 epochs, T4 GPU, ~15 minutes, ~$0.20)**
```bash
python3 submit_training.py \
    --rnn_size 100 \
    --nepochs 10 \
    --gpu-type t4 \
    --bucket my-scribe-training
```

**2. Development (30 epochs, T4 GPU, ~45 minutes, ~$0.60)**
```bash
python3 submit_training.py \
    --rnn_size 400 \
    --nmixtures 20 \
    --nepochs 30 \
    --gpu-type t4 \
    --bucket my-scribe-training
```

**3. Production (250 epochs, V100 GPU, ~2.5 hours, ~$8)**
```bash
python3 submit_training.py \
    --rnn_size 400 \
    --nmixtures 20 \
    --nepochs 250 \
    --gpu-type v100 \
    --bucket my-scribe-training
```

**4. High Quality (250 epochs, A100 GPU, ~1.8 hours, ~$12)**
```bash
python3 submit_training.py \
    --rnn_size 400 \
    --nmixtures 20 \
    --nepochs 250 \
    --gpu-type a100 \
    --bucket my-scribe-training
```

### All Parameters

```bash
python3 submit_training.py \
    --rnn_size 400 \           # LSTM hidden state size (100, 400, 900)
    --nmixtures 20 \           # Gaussian mixtures in MDN (8, 20)
    --nepochs 250 \            # Training epochs
    --batch_size 32 \          # Batch size (16, 32, 64)
    --learning_rate 1e-4 \     # Initial learning rate
    --save_every 500 \         # Checkpoint interval (steps)
    --gpu-type v100 \          # GPU type (t4, v100, a100)
    --region us-central1 \     # GCP region
    --bucket my-bucket \       # GCS bucket name (REQUIRED)
    --yes                      # Skip confirmation prompt
```

### Skip Docker Rebuild

If you've already built the container and just want to resubmit:

```bash
python3 submit_training.py \
    --bucket my-bucket \
    --skip-build \
    --yes
```

---

## Monitoring Jobs

### List All Jobs

```bash
python3 monitor_training.py
```

**Example output:**
```
Recent Vertex AI Training Jobs
======================================================================
Name                              State           Created
----------------------------------------------------------------------
scribe_training_20250105_143022  ▶ JOB_STATE_RUNNING       2025-01-05
scribe_training_20250105_120533  ✓ JOB_STATE_SUCCEEDED     2025-01-05
scribe_training_20250104_091244  ✗ JOB_STATE_FAILED        2025-01-04
```

### Check Job Status

```bash
python3 monitor_training.py scribe_training_20250105_143022
```

**Example output:**
```
Job Status
======================================================================
Name:         scribe_training_20250105_143022
State:        JOB_STATE_RUNNING
Created:      2025-01-05T14:30:22.482Z
Last Updated: 2025-01-05T16:15:10.234Z

Machine Type: n1-standard-8
GPU:          NVIDIA_TESLA_V100
======================================================================
```

### Follow Logs in Real-Time

```bash
python3 monitor_training.py scribe_training_20250105_143022 --follow
```

This streams training logs to your terminal in real-time.

### Watch Job Status (Auto-Refresh)

```bash
python3 monitor_training.py scribe_training_20250105_143022 --watch
```

Updates every 30 seconds until job completes.

### View in Web Console

```bash
# Opens Vertex AI console in browser
open "https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$(gcloud config get-value project)"
```

---

## Downloading Results

### Download from Bucket

```bash
python3 download_results.py --bucket my-scribe-training
```

This downloads:
- Model checkpoints → `./saved/`
- Training logs → `./logs/`

### Download Model Only

```bash
python3 download_results.py --bucket my-scribe-training --model-only
```

### Download to Specific Directory

```bash
python3 download_results.py \
    --bucket my-scribe-training \
    --output ./models/run1
```

### Download from Specific GCS Path

```bash
python3 download_results.py gs://my-bucket/saved/ckpt-125000*
```

### Verify Downloaded Model

```bash
ls -lh saved/

# Expected files:
# checkpoint                  (167 B)
# ckpt-125000.data-00000-of-00001  (2.8 MB)
# ckpt-125000.index           (3.2 KB)
```

### Generate Samples

```bash
# After downloading, generate handwriting samples
python3 sample.py --text "Trained on Vertex AI!"
```

---

## Cost Estimates

### GPU Pricing (us-central1)

| GPU Type | Compute | GPU Cost | Total/Hour | Use Case |
|----------|---------|----------|------------|----------|
| **T4** | $0.35 | $0.44 | **$0.79** | Testing, development |
| **V100** | $0.74 | $2.48 | **$3.22** | Production (recommended) |
| **A100** | Included | Included | **$6.50** | High-performance |

### Training Cost Estimates

| Configuration | GPU | Time | Cost | Quality |
|--------------|-----|------|------|---------|
| Quick test (10 epochs) | T4 | 15 min | $0.20 | Poor (wavy lines) |
| Development (30 epochs) | T4 | 45 min | $0.60 | Basic (rough letters) |
| Development (30 epochs) | V100 | 30 min | $1.60 | Basic (rough letters) |
| **Production (250 epochs)** | **T4** | **6.2 hrs** | **$4.90** | **Excellent** |
| **Production (250 epochs)** | **V100** | **4.2 hrs** | **$13.50** | **Excellent** |
| Production (250 epochs) | A100 | 2.9 hrs | $18.85 | Excellent |

**Recommendation:** Use **T4 for production training** - cheapest and good enough quality. V100/A100 only if you need results faster.

### Storage Costs

- **Cloud Storage:** $0.020 per GB-month
- **Container Registry:** $0.026 per GB-month

Typical usage:
- Training data: 44 MB = ~$0.001/month
- Model checkpoints: 3 MB per checkpoint = ~$0.06/month
- Docker images: 500 MB = ~$0.013/month
- **Total storage:** < $0.10/month

### Free Tier

- **$300 credit** for new GCP accounts (valid 90 days)
- Enough for ~60 production training runs
- Or ~1,500 quick test runs

---

## Troubleshooting

### Issue 1: "gcloud: command not found"

**Solution:**
```bash
# Install gcloud CLI
brew install google-cloud-sdk

# Add to PATH if needed
echo 'source "$(brew --prefix)/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.bash.inc"' >> ~/.zshrc
source ~/.zshrc
```

### Issue 2: "Docker daemon is not running"

**Solution:**
```bash
# Start Docker Desktop application
open -a Docker

# Wait for Docker to start, then verify
docker ps
```

### Issue 3: "Permission denied while trying to connect to Docker daemon"

**Solution:**
```bash
# Add your user to docker group (Linux)
sudo usermod -aG docker $USER

# Or run with sudo (not recommended)
sudo python3 submit_training.py ...
```

### Issue 4: "Quota exceeded for resource 'GPU'"

**Error:**
```
Quota 'NVIDIA_T4_GPUS' exceeded. Limit: 0.0
```

**Cause:** GPU quota not enabled for your project.

**Solution:**
1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. Filter: "Vertex AI API" + "NVIDIA T4 GPUs"
3. Request quota increase (usually approved instantly for new accounts)

### Issue 5: "Container build failed"

**Error:**
```
ERROR: failed to solve: process "/bin/sh -c pip install ..." did not complete successfully
```

**Solution:**
```bash
# Clean Docker build cache
docker system prune -a

# Retry build
python3 submit_training.py --bucket my-bucket
```

### Issue 6: "Training data not found in container"

**Error:**
```
FileNotFoundError: data/strokes_training_data.cpkl not found
```

**Solution:**
```bash
# Verify data file exists locally
ls -lh data/strokes_training_data.cpkl

# If missing, run data verification
python3 verify_data.py

# Upload to GCS
gsutil cp data/strokes_training_data.cpkl gs://my-bucket/data/
```

### Issue 7: "Job failed with OOM error"

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Cause:** Model too large for GPU memory.

**Solution:**
```bash
# Reduce batch size
python3 submit_training.py --batch_size 16 --bucket my-bucket

# Or reduce model size
python3 submit_training.py --rnn_size 100 --bucket my-bucket

# Or use larger GPU
python3 submit_training.py --gpu-type v100 --bucket my-bucket
```

### Issue 8: "Cannot download results - access denied"

**Error:**
```
AccessDeniedException: 403 Insufficient Permission
```

**Solution:**
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login

# Verify bucket access
gsutil ls gs://my-bucket/

# Grant yourself storage admin role if needed
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="user:YOUR_EMAIL@gmail.com" \
    --role="roles/storage.admin"
```

---

## Workflow Summary

**Complete training workflow:**

```bash
# 1. One-time setup (5-10 minutes)
gcloud auth login
gcloud config set project my-project
gsutil mb -l us-central1 gs://my-scribe-bucket
gsutil cp data/strokes_training_data.cpkl gs://my-scribe-bucket/data/

# 2. Submit training job (1 command)
python3 submit_training.py \
    --rnn_size 400 \
    --nepochs 250 \
    --gpu-type t4 \
    --bucket my-scribe-bucket

# 3. Monitor progress
python3 monitor_training.py scribe_training_TIMESTAMP --follow

# 4. Download results (when complete)
python3 download_results.py --bucket my-scribe-bucket

# 5. Generate samples
python3 sample.py --text "Hello World!"
```

---

## Comparison: Vertex AI vs Colab vs M1 Mac

| Aspect | Vertex AI | Colab Pro | M1 Mac (8GB) |
|--------|-----------|-----------|--------------|
| **Setup** | 10 min initial | Instant | 10 min initial |
| **Submission** | 1 command | Manual "Run All" | 1 command |
| **Cost (250 epochs)** | **$4.90** (T4) | ~$10 | Free |
| **Training time** | **4-6 hours** | 3-6 hours | 9-12 hours |
| **GPU** | T4/V100/A100 (your choice) | T4/V100 (random) | Metal GPU |
| **Auto-resume** | ✅ Yes | ❌ No | ✅ Yes |
| **Monitoring** | CLI + Web | Web only | Local |
| **Recommended for** | **Production** | Quick experiments | Development |

**Best practice:**
1. **M1 Mac** - Development, testing, smoke tests
2. **Vertex AI** - Production training (cheaper, more control)
3. **Colab** - Quick experiments if you don't want to set up Vertex AI

---

## Next Steps

**After training completes:**

1. **Download model**
   ```bash
   python3 download_results.py --bucket my-bucket
   ```

2. **Generate samples**
   ```bash
   python3 sample.py --text "Vertex AI training complete!"
   ```

3. **Multi-line documents**
   ```bash
   python3 sample.py \
       --lines "First line" "Second line" "Third line" \
       --format svg
   ```

4. **Style priming** (if trained with rnn_size=400)
   ```bash
   python3 sample.py \
       --lines "Character A" "Character B" \
       --styles 3 7 \
       --format svg
   ```

---

## Additional Resources

- **Vertex AI Documentation:** https://cloud.google.com/vertex-ai/docs
- **Pricing Calculator:** https://cloud.google.com/products/calculator
- **GPU Quotas:** https://console.cloud.google.com/iam-admin/quotas
- **Project README:** ../README.md
- **M1 Setup Guide:** M1_SETUP.md
- **Project Documentation:** ../CLAUDE.md

---

**Last updated:** 2025-01-05
**Compatible with:** TensorFlow 2.15, Python 3.11+, Vertex AI
