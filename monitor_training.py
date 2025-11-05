#!/usr/bin/env python3
"""
Monitor Vertex AI training job progress.

Usage:
    python3 monitor_training.py                    # List all jobs
    python3 monitor_training.py JOB_NAME           # Monitor specific job
    python3 monitor_training.py JOB_NAME --follow  # Follow logs in real-time
"""

import argparse
import subprocess
import sys
import time
import json
from datetime import datetime


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


def list_jobs(region='us-central1', limit=10):
    """List recent Vertex AI training jobs."""
    print("="*70)
    print("Recent Vertex AI Training Jobs")
    print("="*70)

    cmd = [
        'gcloud', 'ai', 'custom-jobs', 'list',
        f'--region={region}',
        f'--limit={limit}',
        '--format=json'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        jobs = json.loads(result.stdout)

        if not jobs:
            print("\nNo jobs found.")
            return

        print(f"\n{'Name':<35} {'State':<15} {'Created':<20}")
        print("-"*70)

        for job in jobs:
            name = job['displayName']
            state = job['state']
            create_time = job['createTime'].split('T')[0]

            # Color code by state
            if state == 'JOB_STATE_SUCCEEDED':
                state_display = f"✓ {state}"
            elif state == 'JOB_STATE_FAILED':
                state_display = f"✗ {state}"
            elif state == 'JOB_STATE_RUNNING':
                state_display = f"▶ {state}"
            else:
                state_display = state

            print(f"{name:<35} {state_display:<15} {create_time:<20}")

        print("\nMonitor specific job: python3 monitor_training.py JOB_NAME")

    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Failed to list jobs")
        print(f"  {e.stderr}")
        sys.exit(1)


def get_job_status(job_name, region='us-central1'):
    """Get detailed status of a specific job."""
    cmd = [
        'gcloud', 'ai', 'custom-jobs', 'describe',
        job_name,
        f'--region={region}',
        '--format=json'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Failed to get job status")
        print(f"  {e.stderr}")
        return None


def display_job_status(job_info):
    """Display formatted job status."""
    print("\n" + "="*70)
    print("Job Status")
    print("="*70)

    name = job_info['displayName']
    state = job_info['state']
    create_time = job_info['createTime']
    update_time = job_info.get('updateTime', 'N/A')

    print(f"\nName:         {name}")
    print(f"State:        {state}")
    print(f"Created:      {create_time}")
    print(f"Last Updated: {update_time}")

    # Worker pool info
    if 'jobSpec' in job_info and 'workerPoolSpecs' in job_info['jobSpec']:
        worker_pool = job_info['jobSpec']['workerPoolSpecs'][0]
        machine_type = worker_pool.get('machineSpec', {}).get('machineType', 'Unknown')
        accelerator = worker_pool.get('machineSpec', {}).get('acceleratorType', 'None')

        print(f"\nMachine Type: {machine_type}")
        print(f"GPU:          {accelerator}")

    # Error info if failed
    if state == 'JOB_STATE_FAILED' and 'error' in job_info:
        error = job_info['error']
        print(f"\n⚠ ERROR:")
        print(f"  Code: {error.get('code', 'Unknown')}")
        print(f"  Message: {error.get('message', 'No message')}")

    print("="*70)


def follow_logs(job_name, region='us-central1'):
    """Follow job logs in real-time."""
    print(f"\nFollowing logs for job: {job_name}")
    print("Press Ctrl+C to stop\n")

    cmd = [
        'gcloud', 'ai', 'custom-jobs', 'stream-logs',
        job_name,
        f'--region={region}'
    ]

    try:
        # Stream logs to stdout in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end='')

        process.wait()

    except KeyboardInterrupt:
        print("\n\n⚠ Stopped following logs")
        process.terminate()
        sys.exit(0)

    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Failed to stream logs")
        print(f"  {e}")
        sys.exit(1)


def watch_job(job_name, region='us-central1', interval=30):
    """Watch job status and update periodically."""
    print(f"\nWatching job: {job_name}")
    print(f"Refresh interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            job_info = get_job_status(job_name, region)
            if not job_info:
                break

            # Clear screen
            print("\033[2J\033[H", end='')

            display_job_status(job_info)

            state = job_info['state']

            # Check if job is complete
            if state in ['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED']:
                print(f"\n✓ Job completed with state: {state}")

                if state == 'JOB_STATE_SUCCEEDED':
                    print("\nDownload results with:")
                    print(f"  python3 download_results.py {job_name}")

                break

            # Wait before next refresh
            print(f"\nRefreshing in {interval} seconds... (Ctrl+C to stop)")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n⚠ Stopped watching job")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Monitor Vertex AI training jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all recent jobs
  python3 monitor_training.py

  # Check status of specific job
  python3 monitor_training.py scribe_training_20250105_143022

  # Follow logs in real-time
  python3 monitor_training.py scribe_training_20250105_143022 --follow

  # Watch job status (updates every 30 seconds)
  python3 monitor_training.py scribe_training_20250105_143022 --watch
        """
    )

    parser.add_argument('job_name', type=str, nargs='?',
                       help='Job name to monitor (omit to list all jobs)')
    parser.add_argument('--region', type=str, default='us-central1',
                       help='GCP region (default: us-central1)')
    parser.add_argument('--follow', '-f', action='store_true',
                       help='Follow logs in real-time')
    parser.add_argument('--watch', '-w', action='store_true',
                       help='Watch job status (refresh periodically)')
    parser.add_argument('--interval', type=int, default=30,
                       help='Watch interval in seconds (default: 30)')

    args = parser.parse_args()

    # Check gcloud is installed
    try:
        subprocess.run(['gcloud', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ ERROR: gcloud CLI not found")
        print("\nInstall with: brew install google-cloud-sdk")
        sys.exit(1)

    # Get project ID
    project_id = get_project_id()
    if not project_id:
        print("✗ ERROR: No GCP project configured")
        print("\nConfigure with: gcloud config set project YOUR_PROJECT_ID")
        sys.exit(1)

    # No job name provided - list all jobs
    if not args.job_name:
        list_jobs(args.region)
        sys.exit(0)

    # Follow logs
    if args.follow:
        follow_logs(args.job_name, args.region)
        sys.exit(0)

    # Watch job status
    if args.watch:
        watch_job(args.job_name, args.region, args.interval)
        sys.exit(0)

    # Show job status once
    job_info = get_job_status(args.job_name, args.region)
    if job_info:
        display_job_status(job_info)

        state = job_info['state']
        if state == 'JOB_STATE_SUCCEEDED':
            print("\n✓ Training completed successfully!")
            print("\nNext steps:")
            print(f"  1. Download results: python3 download_results.py {args.job_name}")
            print(f"  2. Generate samples: python3 sample.py --text 'Hello World'")

        elif state == 'JOB_STATE_RUNNING':
            print(f"\n▶ Job is currently running")
            print("\nMonitor logs: python3 monitor_training.py {} --follow".format(args.job_name))
            print("Watch status: python3 monitor_training.py {} --watch".format(args.job_name))


if __name__ == '__main__':
    main()
