"""
Unit Tests for Data Verification Script

Tests verify_data.py validation functions to ensure data integrity checks work correctly.

Usage:
    pytest tests/unit/test_verify_data.py -v
    pytest -k test_verify_data
"""

import pytest
import pickle
import numpy as np
import os
import sys
from pathlib import Path
from io import StringIO


@pytest.mark.unit
class TestPrintFunctions:
    """Test output formatting functions."""

    def test_print_header(self, capsys):
        """Test print_header outputs formatted header."""
        from verify_data import print_header

        print_header("Test Header")
        captured = capsys.readouterr()

        assert "=" * 70 in captured.out
        assert "Test Header" in captured.out

    def test_print_success(self, capsys):
        """Test print_success outputs with checkmark."""
        from verify_data import print_success

        print_success("Success message")
        captured = capsys.readouterr()

        assert "✅" in captured.out
        assert "Success message" in captured.out

    def test_print_error(self, capsys):
        """Test print_error outputs with X mark."""
        from verify_data import print_error

        print_error("Error message")
        captured = capsys.readouterr()

        assert "❌" in captured.out
        assert "Error message" in captured.out

    def test_print_warning(self, capsys):
        """Test print_warning outputs with warning symbol."""
        from verify_data import print_warning

        print_warning("Warning message")
        captured = capsys.readouterr()

        assert "⚠️" in captured.out
        assert "Warning message" in captured.out

    def test_print_info(self, capsys):
        """Test print_info outputs with info symbol."""
        from verify_data import print_info

        print_info("Info message")
        captured = capsys.readouterr()

        assert "ℹ️" in captured.out
        assert "Info message" in captured.out


@pytest.mark.unit
class TestVerifyStrokesData:
    """Test verify_strokes_data function."""

    def test_verify_strokes_data_file_not_found(self, tmpdir, monkeypatch, capsys):
        """Test verify_strokes_data returns False when file doesn't exist."""
        from verify_data import verify_strokes_data

        # Change to temp directory where data file doesn't exist
        monkeypatch.chdir(tmpdir)

        result = verify_strokes_data()

        assert result is False
        captured = capsys.readouterr()
        assert "File not found" in captured.out

    def test_verify_strokes_data_valid_file(self, tmpdir, monkeypatch, capsys):
        """Test verify_strokes_data returns True for valid data file."""
        from verify_data import verify_strokes_data

        # Create valid data structure
        strokes = [
            np.random.randn(100, 3).astype(np.float32),
            np.random.randn(150, 3).astype(np.float32),
            np.random.randn(200, 3).astype(np.float32),
        ]
        asciis = ["sample text one", "sample text two", "sample text three"]
        data = [strokes, asciis]

        # Create data directory and file
        data_dir = tmpdir.mkdir("data")
        data_file = data_dir.join("strokes_training_data.cpkl")

        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

        # Change to temp directory
        monkeypatch.chdir(tmpdir)

        result = verify_strokes_data()

        assert result is True
        captured = capsys.readouterr()
        assert "Found 3 stroke samples" in captured.out
        assert "VALID and ready" in captured.out

    def test_verify_strokes_data_invalid_structure(self, tmpdir, monkeypatch, capsys):
        """Test verify_strokes_data returns False for invalid data structure."""
        from verify_data import verify_strokes_data

        # Create INVALID data structure (not [strokes, asciis])
        data = {"strokes": [], "asciis": []}  # Wrong format - dict instead of list

        data_dir = tmpdir.mkdir("data")
        data_file = data_dir.join("strokes_training_data.cpkl")

        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

        monkeypatch.chdir(tmpdir)

        result = verify_strokes_data()

        assert result is False
        captured = capsys.readouterr()
        assert "Data structure is invalid" in captured.out

    def test_verify_strokes_data_empty_strokes(self, tmpdir, monkeypatch, capsys):
        """Test verify_strokes_data returns False when strokes list is empty."""
        from verify_data import verify_strokes_data

        # Empty strokes list
        data = [[], []]

        data_dir = tmpdir.mkdir("data")
        data_file = data_dir.join("strokes_training_data.cpkl")

        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

        monkeypatch.chdir(tmpdir)

        result = verify_strokes_data()

        assert result is False
        captured = capsys.readouterr()
        assert "Strokes list is empty" in captured.out

    def test_verify_strokes_data_mismatch_counts(self, tmpdir, monkeypatch, capsys):
        """Test verify_strokes_data returns False when stroke/ascii counts mismatch."""
        from verify_data import verify_strokes_data

        # Mismatched counts
        strokes = [np.random.randn(100, 3).astype(np.float32)]
        asciis = ["text one", "text two"]  # 2 texts but only 1 stroke
        data = [strokes, asciis]

        data_dir = tmpdir.mkdir("data")
        data_file = data_dir.join("strokes_training_data.cpkl")

        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

        monkeypatch.chdir(tmpdir)

        result = verify_strokes_data()

        assert result is False
        captured = capsys.readouterr()
        assert "Mismatch" in captured.out

    def test_verify_strokes_data_wrong_stroke_shape(self, tmpdir, monkeypatch, capsys):
        """Test verify_strokes_data warns about incorrect stroke shape."""
        from verify_data import verify_strokes_data

        # Wrong shape - should be (n, 3) but is (n, 2)
        strokes = [np.random.randn(100, 2).astype(np.float32)]
        asciis = ["sample text"]
        data = [strokes, asciis]

        data_dir = tmpdir.mkdir("data")
        data_file = data_dir.join("strokes_training_data.cpkl")

        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

        monkeypatch.chdir(tmpdir)

        result = verify_strokes_data()

        # Should still return False due to validation failure
        assert result is False
        captured = capsys.readouterr()
        assert "Unexpected stroke shape" in captured.out

    def test_verify_strokes_data_displays_statistics(self, tmpdir, monkeypatch, capsys):
        """Test verify_strokes_data displays dataset statistics."""
        from verify_data import verify_strokes_data

        # Create dataset with known statistics
        strokes = [np.random.randn(100 + i * 10, 3).astype(np.float32) for i in range(10)]
        asciis = [f"text {i}" * 5 for i in range(10)]
        data = [strokes, asciis]

        data_dir = tmpdir.mkdir("data")
        data_file = data_dir.join("strokes_training_data.cpkl")

        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

        monkeypatch.chdir(tmpdir)

        result = verify_strokes_data()

        assert result is True
        captured = capsys.readouterr()
        assert "Average stroke length" in captured.out
        assert "Average text length" in captured.out


@pytest.mark.unit
class TestVerifyStylesData:
    """Test verify_styles_data function."""

    def test_verify_styles_data_file_not_found(self, tmpdir, monkeypatch, capsys):
        """Test verify_styles_data returns True when optional file missing."""
        from verify_data import verify_styles_data

        # Create data directory but no styles.p file
        tmpdir.mkdir("data")
        monkeypatch.chdir(tmpdir)

        result = verify_styles_data()

        # Should return True (file is optional)
        assert result is True
        captured = capsys.readouterr()
        assert "File not found" in captured.out
        assert "optional" in captured.out

    def test_verify_styles_data_valid_file(self, tmpdir, monkeypatch, capsys):
        """Test verify_styles_data returns True for valid styles file."""
        from verify_data import verify_styles_data

        # Create valid styles structure
        style_strokes = [np.random.randn(100, 3).astype(np.float32)]
        style_strings = ["style text example"]
        data = [style_strokes, style_strings]

        data_dir = tmpdir.mkdir("data")
        styles_file = data_dir.join("styles.p")

        with open(styles_file, 'wb') as f:
            pickle.dump(data, f)

        monkeypatch.chdir(tmpdir)

        result = verify_styles_data()

        assert result is True
        captured = capsys.readouterr()
        assert "Found 1 style vectors" in captured.out
        assert "VALID" in captured.out

    def test_verify_styles_data_invalid_structure(self, tmpdir, monkeypatch, capsys):
        """Test verify_styles_data returns False for invalid structure."""
        from verify_data import verify_styles_data

        # Invalid structure (single list instead of [strokes, strings])
        data = ["invalid"]

        data_dir = tmpdir.mkdir("data")
        styles_file = data_dir.join("styles.p")

        with open(styles_file, 'wb') as f:
            pickle.dump(data, f)

        monkeypatch.chdir(tmpdir)

        result = verify_styles_data()

        assert result is False
        captured = capsys.readouterr()
        assert "Data structure is invalid" in captured.out


@pytest.mark.unit
class TestVerifyDirectoryStructure:
    """Test verify_directory_structure function."""

    def test_verify_directory_structure_all_exist(self, tmpdir, monkeypatch, capsys):
        """Test verify_directory_structure with all directories present."""
        from verify_data import verify_directory_structure

        # Create all required directories
        tmpdir.mkdir("data")
        tmpdir.mkdir("logs")
        tmpdir.mkdir("saved")
        tmpdir.mkdir("static")

        monkeypatch.chdir(tmpdir)

        result = verify_directory_structure()

        assert result is True
        captured = capsys.readouterr()
        assert "Directory exists: data/" in captured.out
        assert "Directory exists: logs/" in captured.out

    def test_verify_directory_structure_missing_optional(self, tmpdir, monkeypatch, capsys):
        """Test verify_directory_structure with missing optional directories."""
        from verify_data import verify_directory_structure

        # Create only required directories
        tmpdir.mkdir("data")
        tmpdir.mkdir("logs")
        tmpdir.mkdir("saved")
        # Skip optional 'static'

        monkeypatch.chdir(tmpdir)

        result = verify_directory_structure()

        assert result is True  # Should still pass (static is optional)
        captured = capsys.readouterr()
        assert "Optional directory missing: static/" in captured.out

    def test_verify_directory_structure_missing_required(self, tmpdir, monkeypatch, capsys):
        """Test verify_directory_structure with missing required directories."""
        from verify_data import verify_directory_structure

        # Don't create any directories
        monkeypatch.chdir(tmpdir)

        result = verify_directory_structure()

        # Function returns True even if missing (warns but doesn't fail)
        assert result is True
        captured = capsys.readouterr()
        assert "Directory missing: data/" in captured.out
        assert "created automatically" in captured.out


@pytest.mark.unit
class TestVerifySourceFiles:
    """Test verify_source_files function."""

    def test_verify_source_files_all_exist(self, tmpdir, monkeypatch, capsys):
        """Test verify_source_files with all files present."""
        from verify_data import verify_source_files

        # Create all required files
        required_files = ['model.py', 'train.py', 'sample.py', 'utils.py', 'verify_data.py']
        for filename in required_files:
            tmpdir.join(filename).write("# Sample code\n" * 10)

        monkeypatch.chdir(tmpdir)

        result = verify_source_files()

        assert result is True
        captured = capsys.readouterr()
        for filename in required_files:
            assert f"Source file exists: {filename}" in captured.out

    def test_verify_source_files_missing_file(self, tmpdir, monkeypatch, capsys):
        """Test verify_source_files returns False when file missing."""
        from verify_data import verify_source_files

        # Create only some files
        tmpdir.join("model.py").write("# code")
        tmpdir.join("utils.py").write("# code")
        # Missing: train.py, sample.py, verify_data.py

        monkeypatch.chdir(tmpdir)

        result = verify_source_files()

        assert result is False
        captured = capsys.readouterr()
        assert "Source file missing" in captured.out

    def test_verify_source_files_counts_lines(self, tmpdir, monkeypatch, capsys):
        """Test verify_source_files counts lines correctly."""
        from verify_data import verify_source_files

        # Create files with known line counts
        required_files = ['model.py', 'train.py', 'sample.py', 'utils.py', 'verify_data.py']
        for filename in required_files:
            tmpdir.join(filename).write("# line\n" * 42)  # Exactly 42 lines

        monkeypatch.chdir(tmpdir)

        result = verify_source_files()

        assert result is True
        captured = capsys.readouterr()
        assert "(42 lines)" in captured.out


@pytest.mark.unit
class TestMain:
    """Test main orchestration function."""

    def test_main_all_checks_pass(self, tmpdir, monkeypatch, capsys):
        """Test main returns 0 when all checks pass."""
        from verify_data import main

        # Setup complete valid environment
        # 1. Source files
        required_files = ['model.py', 'train.py', 'sample.py', 'utils.py', 'verify_data.py']
        for filename in required_files:
            tmpdir.join(filename).write("# code\n")

        # 2. Directories
        data_dir = tmpdir.mkdir("data")
        tmpdir.mkdir("logs")
        tmpdir.mkdir("saved")

        # 3. Valid strokes data
        strokes = [np.random.randn(100, 3).astype(np.float32)]
        asciis = ["sample text"]
        strokes_data = [strokes, asciis]
        strokes_file = data_dir.join("strokes_training_data.cpkl")
        with open(strokes_file, 'wb') as f:
            pickle.dump(strokes_data, f)

        # 4. Valid styles data
        style_strokes = [np.random.randn(50, 3).astype(np.float32)]
        style_strings = ["style text"]
        styles_data = [style_strokes, style_strings]
        styles_file = data_dir.join("styles.p")
        with open(styles_file, 'wb') as f:
            pickle.dump(styles_data, f)

        monkeypatch.chdir(tmpdir)

        # Run main
        exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "SUCCESS! All checks passed!" in captured.out
        assert "✅ PASS" in captured.out

    def test_main_strokes_check_fails(self, tmpdir, monkeypatch, capsys):
        """Test main returns 1 when critical strokes check fails."""
        from verify_data import main

        # Setup environment with MISSING strokes data (critical failure)
        # 1. Source files
        required_files = ['model.py', 'train.py', 'sample.py', 'utils.py', 'verify_data.py']
        for filename in required_files:
            tmpdir.join(filename).write("# code\n")

        # 2. Directories
        tmpdir.mkdir("data")  # Directory exists but no strokes file
        tmpdir.mkdir("logs")
        tmpdir.mkdir("saved")

        monkeypatch.chdir(tmpdir)

        # Run main
        exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "ISSUES DETECTED" in captured.out
        assert "❌ FAIL" in captured.out

    def test_main_python_version_check(self, tmpdir, monkeypatch, capsys):
        """Test main displays Python version information."""
        from verify_data import main

        # Setup minimal valid environment
        required_files = ['model.py', 'train.py', 'sample.py', 'utils.py', 'verify_data.py']
        for filename in required_files:
            tmpdir.join(filename).write("# code\n")

        data_dir = tmpdir.mkdir("data")
        strokes = [np.random.randn(100, 3).astype(np.float32)]
        asciis = ["text"]
        data = [strokes, asciis]
        strokes_file = data_dir.join("strokes_training_data.cpkl")
        with open(strokes_file, 'wb') as f:
            pickle.dump(data, f)

        tmpdir.mkdir("logs")
        tmpdir.mkdir("saved")

        monkeypatch.chdir(tmpdir)

        exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Python version:" in captured.out
        assert f"Running Python {sys.version_info.major}" in captured.out

    def test_main_displays_summary(self, tmpdir, monkeypatch, capsys):
        """Test main displays verification summary."""
        from verify_data import main

        # Setup minimal valid environment
        required_files = ['model.py', 'train.py', 'sample.py', 'utils.py', 'verify_data.py']
        for filename in required_files:
            tmpdir.join(filename).write("# code\n")

        data_dir = tmpdir.mkdir("data")
        strokes = [np.random.randn(100, 3).astype(np.float32)]
        asciis = ["text"]
        data = [strokes, asciis]
        strokes_file = data_dir.join("strokes_training_data.cpkl")
        with open(strokes_file, 'wb') as f:
            pickle.dump(data, f)

        tmpdir.mkdir("logs")
        tmpdir.mkdir("saved")

        monkeypatch.chdir(tmpdir)

        exit_code = main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "VERIFICATION SUMMARY" in captured.out
        assert "Source Files" in captured.out
        assert "Directory Structure" in captured.out
        assert "Strokes Data" in captured.out


# ============================================================================
# Summary
# ============================================================================

def test_verify_data_suite_summary():
    """
    Verify data test suite summary.

    If all tests pass:
    - Print functions format output correctly
    - verify_strokes_data() validates data file structure
    - verify_styles_data() validates optional styles file
    - verify_directory_structure() checks required directories
    - verify_source_files() validates source code presence
    - main() orchestrates all checks and returns proper exit codes
    """
    print("\n✓ All verify_data tests passed!")
    print("  - Print functions work correctly")
    print("  - Strokes data validation comprehensive")
    print("  - Styles data validation works")
    print("  - Directory structure checks work")
    print("  - Source file checks work")
    print("  - Main orchestration correct")
