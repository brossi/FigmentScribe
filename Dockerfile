# Dockerfile for Scribe training on Google Cloud Vertex AI
# Builds a container with TensorFlow 2.15 and all training dependencies

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
# Note: Use standard tensorflow (not tensorflow-macos) for cloud training
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        tensorflow==2.15.0 \
        numpy==1.26.4 \
        matplotlib==3.8.3 \
        scipy==1.12.0 \
        google-cloud-storage==2.14.0

# Copy training code
COPY model.py .
COPY utils.py .
COPY train.py .
COPY vertex_training.py .

# Copy data directory structure (data will be mounted from GCS)
COPY data/strokes_training_data.cpkl data/

# Set permissions
RUN chmod +x vertex_training.py

# Use vertex_training.py as entrypoint
ENTRYPOINT ["python3", "vertex_training.py"]

# Default arguments (can be overridden)
CMD ["--rnn_size", "400", "--nmixtures", "20", "--nepochs", "250"]
