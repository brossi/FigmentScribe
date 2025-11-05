#!/usr/bin/env python3
"""
Download trained model and logs from Google Cloud Storage.

Usage:
    python3 download_results.py gs://bucket/saved/  # Download from GCS path
    python3 download_results.py JOB_NAME             # Download by job name (auto-detect GCS path)
    python3 download_results.py --bucket my-bucket   # Download from specific bucket
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_project_id():
    """Get configured GCP project ID."""
    try:
        result = subprocess.run(
            ['gcloud', 'config', 'get-value', 'project'],
            capture_output=True,
            text=True,
            check=True
        )
        project_id = result.stdout.strip()
        return project_id if project_id != '(unset)' else None
    except subprocess.CalledProcessError:
        return None


def check_gsutil_installed():
    """Verify gsutil is installed."""
    try:
        subprocess.run(
            ['gsutil', '--version'],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ ERROR: gsutil not found")
        print("\nInstall with: brew install google-cloud-sdk")
        return False


def download_from_gcs(gcs_path, local_path, recursive=True):
    """Download files from GCS to local directory."""
    local_dir = Path(local_path)
    local_dir.mkdir(parents=True, exist_ok=True)

    cmd = ['gsutil', '-m', 'cp']
    if recursive:
        cmd.append('-r')

    cmd.extend([gcs_path, str(local_dir)])

    print(f"\nDownloading from: {gcs_path}")
    print(f"              to: {local_dir}")

    try:
        result = subprocess.run(cmd, check=True)
        print("✓ Download complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Download failed")
        print(f"  {e}")
        return False


def list_gcs_contents(gcs_path):
    """List contents of GCS path."""
    cmd = ['gsutil', 'ls', '-lh', gcs_path]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Failed to list GCS contents")
        print(f"  {e.stderr}")
        return None


def download_model(bucket, local_dir='./saved'):
    """Download model checkpoints from GCS bucket."""
    print("\n" + "="*70)
    print("Downloading Model Checkpoints")
    print("="*70)

    gcs_path = f"gs://{bucket}/saved/*"

    # List contents first
    print(f"\nChecking GCS path: gs://{bucket}/saved/")
    contents = list_gcs_contents(f"gs://{bucket}/saved/")

    if not contents:
        print("⚠ WARNING: No files found at GCS path")
        return False

    print("\nFound files:")
    print(contents)

    # Download
    return download_from_gcs(gcs_path, local_dir, recursive=False)


def download_logs(bucket, local_dir='./logs'):
    """Download training logs from GCS bucket."""
    print("\n" + "="*70)
    print("Downloading Training Logs")
    print("="*70)

    gcs_path = f"gs://{bucket}/logs/*"

    # Check if logs exist
    contents = list_gcs_contents(f"gs://{bucket}/logs/")

    if not contents:
        print("⚠ No logs found (this is okay if training just started)")
        return True

    print("\nFound files:")
    print(contents)

    # Download
    return download_from_gcs(gcs_path, local_dir, recursive=False)


def download_all(bucket, output_dir='.'):
    """Download all training outputs from GCS bucket."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"Downloading All Training Outputs from: {bucket}")
    print("="*70)

    success = True

    # Download model checkpoints
    saved_dir = output_path / 'saved'
    if not download_model(bucket, saved_dir):
        success = False

    # Download logs
    logs_dir = output_path / 'logs'
    if not download_logs(bucket, logs_dir):
        # Logs are optional, don't fail
        pass

    return success


def main():
    parser = argparse.ArgumentParser(
        description='Download trained models from Google Cloud Storage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from specific GCS bucket
  python3 download_results.py --bucket my-scribe-bucket

  # Download from specific GCS path
  python3 download_results.py gs://my-bucket/saved/

  # Download to specific local directory
  python3 download_results.py --bucket my-bucket --output ./models/run1

  # Download only model (not logs)
  python3 download_results.py --bucket my-bucket --model-only
        """
    )

    parser.add_argument('gcs_path', type=str, nargs='?',
                       help='GCS path to download from (e.g., gs://bucket/saved/)')
    parser.add_argument('--bucket', type=str,
                       help='GCS bucket name (alternative to gcs_path)')
    parser.add_argument('--output', type=str, default='.',
                       help='Local output directory (default: current directory)')
    parser.add_argument('--model-only', action='store_true',
                       help='Download only model checkpoints (not logs)')
    parser.add_argument('--logs-only', action='store_true',
                       help='Download only logs (not model)')

    args = parser.parse_args()

    # Check gsutil is installed
    if not check_gsutil_installed():
        sys.exit(1)

    # Check project is configured
    project_id = get_project_id()
    if not project_id:
        print("✗ ERROR: No GCP project configured")
        print("\nConfigure with: gcloud config set project YOUR_PROJECT_ID")
        sys.exit(1)

    # Determine what to download
    if args.gcs_path:
        # Direct GCS path provided
        if not args.gcs_path.startswith('gs://'):
            print(f"✗ ERROR: Invalid GCS path: {args.gcs_path}")
            print("  GCS paths must start with gs://")
            sys.exit(1)

        success = download_from_gcs(args.gcs_path, args.output)

    elif args.bucket:
        # Bucket name provided
        bucket = args.bucket

        if args.model_only:
            success = download_model(bucket, Path(args.output) / 'saved')
        elif args.logs_only:
            success = download_logs(bucket, Path(args.output) / 'logs')
        else:
            success = download_all(bucket, args.output)

    else:
        print("✗ ERROR: Must provide either gcs_path or --bucket")
        print("\nExamples:")
        print("  python3 download_results.py gs://my-bucket/saved/")
        print("  python3 download_results.py --bucket my-bucket")
        sys.exit(1)

    if success:
        print("\n" + "="*70)
        print("Download Complete! ✓")
        print("="*70)

        # Show next steps
        saved_dir = Path(args.output) / 'saved'
        if saved_dir.exists():
            checkpoints = list(saved_dir.glob('ckpt-*'))
            if checkpoints:
                print(f"\nModel checkpoints saved to: {saved_dir}")
                print(f"Found {len(checkpoints)//3} checkpoints")  # Each checkpoint = 3 files

                print("\nGenerate samples with:")
                print(f"  python3 sample.py --text 'Hello World!'")
        else:
            print(f"\nFiles downloaded to: {args.output}")

    else:
        print("\n⚠ Download completed with errors")
        sys.exit(1)


if __name__ == '__main__':
    main()
