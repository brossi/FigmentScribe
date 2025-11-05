#!/usr/bin/env python3
"""
Submit training job to Google Cloud Vertex AI.

Usage:
    python3 submit_training.py --rnn_size 400 --nepochs 250
    python3 submit_training.py --rnn_size 400 --nepochs 250 --gpu-type v100
    python3 submit_training.py --config my_training_config.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def check_gcloud_installed():
    """Verify gcloud CLI is installed and configured."""
    try:
        result = subprocess.run(
            ['gcloud', '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ gcloud CLI found")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ ERROR: gcloud CLI not found")
        print("\nInstall with:")
        print("  brew install google-cloud-sdk")
        print("  gcloud auth login")
        print("  gcloud config set project YOUR_PROJECT_ID")
        return False


def check_project_configured():
    """Verify GCP project is configured."""
    try:
        result = subprocess.run(
            ['gcloud', 'config', 'get-value', 'project'],
            capture_output=True,
            text=True,
            check=True
        )
        project_id = result.stdout.strip()
        if project_id and project_id != '(unset)':
            print(f"✓ GCP Project: {project_id}")
            return project_id
        else:
            print("✗ ERROR: No GCP project configured")
            print("\nConfigure with:")
            print("  gcloud config set project YOUR_PROJECT_ID")
            return None
    except subprocess.CalledProcessError:
        print("✗ ERROR: Cannot determine GCP project")
        return None


def check_vertex_ai_enabled(project_id):
    """Check if Vertex AI API is enabled."""
    try:
        result = subprocess.run(
            ['gcloud', 'services', 'list', '--enabled',
             '--filter=name:aiplatform.googleapis.com', '--format=value(name)'],
            capture_output=True,
            text=True,
            check=True
        )
        if 'aiplatform.googleapis.com' in result.stdout:
            print("✓ Vertex AI API enabled")
            return True
        else:
            print("⚠ WARNING: Vertex AI API not enabled")
            print("\nEnable with:")
            print("  gcloud services enable aiplatform.googleapis.com")
            return False
    except subprocess.CalledProcessError:
        print("⚠ WARNING: Cannot verify Vertex AI API status")
        return False


def build_and_push_container(project_id, tag='latest'):
    """Build Docker container and push to Google Container Registry."""
    print("\n" + "="*70)
    print("Building Docker Container")
    print("="*70)

    image_uri = f"gcr.io/{project_id}/scribe-training:{tag}"

    # Build container
    print(f"\nBuilding container image: {image_uri}")
    build_cmd = [
        'docker', 'build',
        '-t', image_uri,
        '-f', 'Dockerfile',
        '.'
    ]

    try:
        subprocess.run(build_cmd, check=True)
        print("✓ Container built successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Container build failed: {e}")
        return None

    # Configure Docker for GCR
    print("\nConfiguring Docker authentication for GCR...")
    try:
        subprocess.run(
            ['gcloud', 'auth', 'configure-docker', '--quiet'],
            check=True
        )
        print("✓ Docker configured for GCR")
    except subprocess.CalledProcessError as e:
        print(f"⚠ WARNING: Docker configuration may have failed: {e}")

    # Push to GCR
    print(f"\nPushing container to GCR...")
    push_cmd = ['docker', 'push', image_uri]

    try:
        subprocess.run(push_cmd, check=True)
        print("✓ Container pushed to GCR")
        return image_uri
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Container push failed: {e}")
        return None


def submit_vertex_ai_job(project_id, image_uri, args):
    """Submit training job to Vertex AI."""
    print("\n" + "="*70)
    print("Submitting Vertex AI Training Job")
    print("="*70)

    # Generate unique job name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"scribe_training_{timestamp}"

    # Determine region
    region = args.region or 'us-central1'

    # Determine machine type and GPU
    gpu_type_map = {
        't4': ('n1-standard-4', 'NVIDIA_TESLA_T4', 1),
        'v100': ('n1-standard-8', 'NVIDIA_TESLA_V100', 1),
        'a100': ('a2-highgpu-1g', 'NVIDIA_TESLA_A100', 1),
    }

    gpu_type = args.gpu_type.lower()
    if gpu_type not in gpu_type_map:
        print(f"✗ ERROR: Unknown GPU type '{gpu_type}'")
        print(f"  Valid options: {list(gpu_type_map.keys())}")
        return None

    machine_type, accelerator_type, accelerator_count = gpu_type_map[gpu_type]

    # Build training arguments
    training_args = [
        f"--rnn_size={args.rnn_size}",
        f"--nmixtures={args.nmixtures}",
        f"--nepochs={args.nepochs}",
        f"--batch_size={args.batch_size}",
        f"--learning_rate={args.learning_rate}",
        f"--save_every={args.save_every}",
        f"--data_dir=/gcs/{args.bucket}/data",
        f"--save_path=/gcs/{args.bucket}/saved/model",
        f"--log_dir=/gcs/{args.bucket}/logs/",
    ]

    # Construct gcloud command
    submit_cmd = [
        'gcloud', 'ai', 'custom-jobs', 'create',
        f'--region={region}',
        f'--display-name={job_name}',
        f'--worker-pool-spec=machine-type={machine_type},replica-count=1,'
        f'accelerator-type={accelerator_type},accelerator-count={accelerator_count},'
        f'container-image-uri={image_uri}',
        f'--args={",".join(training_args)}',
    ]

    # Print job details
    print(f"\nJob Configuration:")
    print(f"  Name:          {job_name}")
    print(f"  Region:        {region}")
    print(f"  Machine Type:  {machine_type}")
    print(f"  GPU:           {accelerator_type} x{accelerator_count}")
    print(f"  RNN Size:      {args.rnn_size}")
    print(f"  Epochs:        {args.nepochs}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"\nEstimated cost: ${estimate_cost(gpu_type, args.nepochs):.2f}")
    print(f"Estimated time: {estimate_time(gpu_type, args.nepochs):.1f} hours")

    # Confirm submission
    if not args.yes:
        response = input("\nSubmit job? [y/N]: ")
        if response.lower() != 'y':
            print("Job submission cancelled")
            return None

    print("\nSubmitting job to Vertex AI...")
    try:
        result = subprocess.run(
            submit_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ Job submitted successfully!")
        print(f"\nJob Name: {job_name}")
        print(f"\nMonitor with:")
        print(f"  python3 monitor_training.py {job_name}")
        print(f"\nOr view in console:")
        print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}")

        return job_name

    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Job submission failed")
        print(f"\nOutput: {e.stdout}")
        print(f"Error: {e.stderr}")
        return None


def estimate_cost(gpu_type, epochs):
    """Estimate training cost."""
    # Cost per hour (compute + GPU)
    costs = {
        't4': 0.79,      # n1-standard-4 + T4
        'v100': 3.22,    # n1-standard-8 + V100
        'a100': 6.50,    # a2-highgpu-1g (includes A100)
    }

    # Estimate hours (rough approximation)
    hours_per_epoch = {
        't4': 0.015,     # ~1 minute per epoch
        'v100': 0.010,   # ~40 seconds per epoch
        'a100': 0.007,   # ~25 seconds per epoch
    }

    cost_per_hour = costs[gpu_type]
    hours = epochs * hours_per_epoch[gpu_type]

    return cost_per_hour * hours


def estimate_time(gpu_type, epochs):
    """Estimate training time in hours."""
    hours_per_epoch = {
        't4': 0.015,
        'v100': 0.010,
        'a100': 0.007,
    }
    return epochs * hours_per_epoch[gpu_type]


def main():
    parser = argparse.ArgumentParser(
        description='Submit Scribe training job to Google Cloud Vertex AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (10 epochs, T4 GPU, ~$0.12)
  python3 submit_training.py --rnn_size 100 --nepochs 10 --gpu-type t4

  # Production training (250 epochs, V100 GPU, ~$8)
  python3 submit_training.py --rnn_size 400 --nepochs 250 --gpu-type v100

  # High quality (250 epochs, A100 GPU, ~$12)
  python3 submit_training.py --rnn_size 400 --nmixtures 20 --nepochs 250 --gpu-type a100

GPU Types:
  t4    - NVIDIA Tesla T4    ($0.79/hr) - Good for testing/development
  v100  - NVIDIA Tesla V100  ($3.22/hr) - Balanced performance (recommended)
  a100  - NVIDIA Tesla A100  ($6.50/hr) - Fastest training
        """
    )

    # Training parameters
    parser.add_argument('--rnn_size', type=int, default=400,
                       help='Size of LSTM hidden state (default: 400)')
    parser.add_argument('--nmixtures', type=int, default=20,
                       help='Number of Gaussian mixtures (default: 20)')
    parser.add_argument('--nepochs', type=int, default=250,
                       help='Number of training epochs (default: 250)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--save_every', type=int, default=500,
                       help='Save checkpoint every N steps (default: 500)')

    # Cloud parameters
    parser.add_argument('--gpu-type', type=str, default='t4',
                       choices=['t4', 'v100', 'a100'],
                       help='GPU type to use (default: t4)')
    parser.add_argument('--region', type=str, default='us-central1',
                       help='GCP region (default: us-central1)')
    parser.add_argument('--bucket', type=str, required=True,
                       help='GCS bucket name for data and results (required)')

    # Workflow control
    parser.add_argument('--skip-build', action='store_true',
                       help='Skip Docker build (use existing image)')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')

    args = parser.parse_args()

    print("="*70)
    print("Scribe - Vertex AI Training Submission")
    print("="*70)

    # Verification checks
    print("\nVerifying prerequisites...")

    if not check_gcloud_installed():
        sys.exit(1)

    project_id = check_project_configured()
    if not project_id:
        sys.exit(1)

    check_vertex_ai_enabled(project_id)

    # Build and push container (unless skipped)
    if args.skip_build:
        print("\n⚠ Skipping Docker build (using existing image)")
        image_uri = f"gcr.io/{project_id}/scribe-training:latest"
    else:
        image_uri = build_and_push_container(project_id)
        if not image_uri:
            sys.exit(1)

    # Submit job
    job_name = submit_vertex_ai_job(project_id, image_uri, args)
    if not job_name:
        sys.exit(1)

    print("\n" + "="*70)
    print("Job Submitted Successfully! ✓")
    print("="*70)


if __name__ == '__main__':
    main()
