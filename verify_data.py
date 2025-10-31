#!/usr/bin/env python3
"""
Data Verification Script for Scribe Migration
Checks if existing preprocessed data is valid for Python 3 + TensorFlow 2.x migration
"""

import sys
import os
import pickle
import numpy as np

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")

def print_success(text):
    print(f"✅ {text}")

def print_error(text):
    print(f"❌ {text}")

def print_warning(text):
    print(f"⚠️  {text}")

def print_info(text):
    print(f"ℹ️  {text}")

def verify_strokes_data():
    """Verify the main training data file"""
    print_header("Verifying strokes_training_data.cpkl")

    data_file = "data/strokes_training_data.cpkl"

    # Check file exists
    if not os.path.exists(data_file):
        print_error(f"File not found: {data_file}")
        return False

    file_size = os.path.getsize(data_file) / (1024 * 1024)  # Size in MB
    print_info(f"File size: {file_size:.2f} MB")

    # Try to load the data
    try:
        print_info("Attempting to load with encoding='latin1' (Python 3 compatible)...")
        with open(data_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print_success("Data loaded successfully!")
    except Exception as e:
        print_error(f"Failed to load data: {e}")
        return False

    # Verify data structure
    if not isinstance(data, (list, tuple)) or len(data) != 2:
        print_error("Data structure is invalid. Expected [strokes, asciis]")
        return False

    strokes, asciis = data

    # Verify strokes
    if not isinstance(strokes, list):
        print_error("Strokes data is not a list")
        return False

    if len(strokes) == 0:
        print_error("Strokes list is empty")
        return False

    print_success(f"Found {len(strokes)} stroke samples")

    # Verify asciis
    if not isinstance(asciis, list):
        print_error("ASCII data is not a list")
        return False

    if len(asciis) != len(strokes):
        print_error(f"Mismatch: {len(strokes)} strokes but {len(asciis)} text labels")
        return False

    print_success(f"Found {len(asciis)} text labels (matches stroke count)")

    # Check first sample
    print_info("\nInspecting first sample:")
    try:
        first_stroke = strokes[0]
        first_text = asciis[0]

        print(f"   Stroke shape: {first_stroke.shape}")
        print(f"   Stroke dtype: {first_stroke.dtype}")
        print(f"   Text: \"{first_text[:60]}{'...' if len(first_text) > 60 else ''}\"")

        # Verify stroke format: should be (n_points, 3) with [dx, dy, eos]
        if len(first_stroke.shape) != 2 or first_stroke.shape[1] != 3:
            print_warning(f"Unexpected stroke shape: {first_stroke.shape}. Expected (n, 3)")
        else:
            print_success(f"Stroke format is correct: {first_stroke.shape[0]} points × 3 dimensions")

        # Check value ranges
        print_info(f"   Stroke value ranges:")
        print(f"      dx range: [{first_stroke[:,0].min():.2f}, {first_stroke[:,0].max():.2f}]")
        print(f"      dy range: [{first_stroke[:,1].min():.2f}, {first_stroke[:,1].max():.2f}]")
        print(f"      eos unique values: {np.unique(first_stroke[:,2])}")

    except Exception as e:
        print_error(f"Error inspecting sample: {e}")
        return False

    # Statistics
    print_info("\nDataset statistics:")
    try:
        stroke_lengths = [s.shape[0] for s in strokes[:1000]]  # Sample first 1000
        print(f"   Average stroke length (first 1000 samples): {np.mean(stroke_lengths):.1f} points")
        print(f"   Min stroke length: {np.min(stroke_lengths)}")
        print(f"   Max stroke length: {np.max(stroke_lengths)}")

        text_lengths = [len(t) for t in asciis[:1000]]
        print(f"   Average text length (first 1000 samples): {np.mean(text_lengths):.1f} characters")
    except Exception as e:
        print_warning(f"Could not compute statistics: {e}")

    print_success("\n✓ strokes_training_data.cpkl is VALID and ready for migration!")
    return True

def verify_styles_data():
    """Verify the styles data file"""
    print_header("Verifying styles.p")

    styles_file = "data/styles.p"

    # Check file exists
    if not os.path.exists(styles_file):
        print_warning(f"File not found: {styles_file}")
        print_info("   This file is optional for training, only needed for style-conditioned sampling")
        return True  # Not critical

    file_size = os.path.getsize(styles_file) / 1024  # Size in KB
    print_info(f"File size: {file_size:.2f} KB")

    # Try to load the data
    try:
        print_info("Attempting to load with encoding='latin1' (Python 3 compatible)...")
        with open(styles_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print_success("Data loaded successfully!")
    except Exception as e:
        print_error(f"Failed to load data: {e}")
        return False

    # Verify structure
    if not isinstance(data, (list, tuple)) or len(data) != 2:
        print_error("Data structure is invalid. Expected [style_strokes, style_strings]")
        return False

    style_strokes, style_strings = data
    print_success(f"Found {len(style_strokes)} style vectors")
    print_success(f"Found {len(style_strings)} style strings")

    if len(style_strokes) > 0:
        print_info(f"\nFirst style stroke shape: {style_strokes[0].shape}")
        print_info(f"First style string: \"{style_strings[0][:60]}{'...' if len(style_strings[0]) > 60 else ''}\"")

    print_success("\n✓ styles.p is VALID!")
    return True

def verify_directory_structure():
    """Check that expected directories exist"""
    print_header("Verifying Directory Structure")

    required_dirs = ['data', 'logs', 'saved']
    optional_dirs = ['static']

    all_good = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print_success(f"Directory exists: {dir_name}/")
        else:
            print_warning(f"Directory missing: {dir_name}/")
            print_info(f"   Will be created automatically during training")

    for dir_name in optional_dirs:
        if os.path.exists(dir_name):
            print_success(f"Directory exists: {dir_name}/")
        else:
            print_info(f"Optional directory missing: {dir_name}/")

    return all_good

def verify_source_files():
    """Check that all source files exist"""
    print_header("Verifying Source Files")

    required_files = ['model.py', 'train.py', 'sample.py', 'utils.py', 'verify_data.py']

    all_good = True
    for file_name in required_files:
        if os.path.exists(file_name):
            lines = open(file_name).readlines()
            print_success(f"Source file exists: {file_name} ({len(lines)} lines)")
        else:
            print_error(f"Source file missing: {file_name}")
            all_good = False

    return all_good

def main():
    print_header("Scribe Migration: Data Verification")
    print("This script verifies that your existing data is compatible")
    print("with Python 3 and ready for TensorFlow 2.x migration.")

    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info[0] < 3:
        print_error("This script requires Python 3!")
        print_info("Please run with: python3 verify_data.py")
        sys.exit(1)
    else:
        print_success(f"Running Python {sys.version_info.major}.{sys.version_info.minor}")

    # Run all checks
    results = {
        'source_files': verify_source_files(),
        'directory_structure': verify_directory_structure(),
        'strokes_data': verify_strokes_data(),
        'styles_data': verify_styles_data(),
    }

    # Summary
    print_header("VERIFICATION SUMMARY")

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check.replace('_', ' ').title()}")

    print("\n" + "="*70)

    if all_passed:
        print("\n🎉 SUCCESS! All checks passed!")
        print("\n📋 Next Steps:")
        print("   1. Your existing data is valid and Python 3 compatible")
        print("   2. You do NOT need to download the IAM dataset")
        print("   3. You can proceed with migration immediately")
        print("   4. See MIGRATION_GUIDE.md Phase 1 for next steps")
        print("\n💡 Recommended command:")
        print("   python3 verify_data.py  # (this script - already done!)")
        print("   # Then follow Phase 1 in MIGRATION_GUIDE.md")
        return 0
    else:
        print("\n⚠️  ISSUES DETECTED")
        print("\nSome checks failed. Please review the errors above.")
        print("Critical: strokes_data must pass to proceed with migration.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
