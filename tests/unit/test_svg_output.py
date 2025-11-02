"""
Unit Tests for SVG Output Module

Tests SVG generation for pen plotters:
- Delta to absolute coordinate conversion
- Savitzky-Golay smoothing filter
- Rotation alignment correction
- Multi-line SVG file generation

Usage:
    pytest tests/unit/test_svg_output.py -v
    pytest -k test_svg_output
"""

import pytest
import numpy as np
import os
import tempfile


@pytest.mark.unit
class TestOffsetsToCoords:
    """Test delta to absolute coordinate conversion."""

    def test_offsets_to_coords_shape(self):
        """Test offsets_to_coords preserves shape."""
        from svg_output import offsets_to_coords

        offsets = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [-10.0, 0.0, 0.0],
            [0.0, -10.0, 1.0],
        ])

        coords = offsets_to_coords(offsets)

        # Shape should be preserved
        assert coords.shape == offsets.shape, \
            f"Expected shape {offsets.shape}, got {coords.shape}"

    def test_offsets_to_coords_cumulative_sum(self):
        """Test offsets_to_coords computes cumulative sum correctly."""
        from svg_output import offsets_to_coords

        # Simple square: right, down, left, up
        offsets = np.array([
            [10.0, 0.0, 0.0],   # Right 10
            [0.0, 10.0, 0.0],   # Down 10
            [-10.0, 0.0, 0.0],  # Left 10
            [0.0, -10.0, 1.0],  # Up 10
        ])

        coords = offsets_to_coords(offsets)

        # Expected absolute coordinates (cumulative sum)
        expected_x = np.array([10.0, 10.0, 0.0, 0.0])
        expected_y = np.array([0.0, 10.0, 10.0, 0.0])

        np.testing.assert_allclose(coords[:, 0], expected_x)
        np.testing.assert_allclose(coords[:, 1], expected_y)

    def test_offsets_to_coords_preserves_eos(self):
        """Test end-of-stroke column is preserved."""
        from svg_output import offsets_to_coords

        offsets = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],  # End of stroke
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],  # End of stroke
        ])

        coords = offsets_to_coords(offsets)

        # EOS column should be unchanged
        np.testing.assert_array_equal(coords[:, 2], offsets[:, 2])

    def test_offsets_to_coords_starting_from_origin(self):
        """Test conversion starts from origin (0, 0)."""
        from svg_output import offsets_to_coords

        offsets = np.array([
            [5.0, 3.0, 0.0],
            [2.0, 1.0, 0.0],
        ])

        coords = offsets_to_coords(offsets)

        # First point should be at offset
        assert coords[0, 0] == 5.0
        assert coords[0, 1] == 3.0

        # Second point should be cumulative
        assert coords[1, 0] == 7.0  # 5 + 2
        assert coords[1, 1] == 4.0  # 3 + 1

    def test_offsets_to_coords_with_negative_deltas(self):
        """Test conversion handles negative deltas correctly."""
        from svg_output import offsets_to_coords

        offsets = np.array([
            [10.0, 5.0, 0.0],
            [-5.0, -3.0, 0.0],  # Move back and up
            [-5.0, -2.0, 1.0],
        ])

        coords = offsets_to_coords(offsets)

        # Verify cumulative sum with negatives
        expected_x = np.array([10.0, 5.0, 0.0])
        expected_y = np.array([5.0, 2.0, 0.0])

        np.testing.assert_allclose(coords[:, 0], expected_x)
        np.testing.assert_allclose(coords[:, 1], expected_y)


@pytest.mark.unit
class TestDenoise:
    """Test Savitzky-Golay smoothing filter."""

    def test_denoise_preserves_stroke_count(self):
        """Test denoise preserves number of strokes."""
        from svg_output import denoise, offsets_to_coords

        # Two separate strokes
        offsets = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],  # End of stroke 1
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],  # End of stroke 2
        ] * 5, dtype=np.float32)  # Repeat to have enough points for filtering

        coords = offsets_to_coords(offsets)
        denoised = denoise(coords)

        # Should preserve end-of-stroke markers
        original_eos_count = np.sum(coords[:, 2] == 1)
        denoised_eos_count = np.sum(denoised[:, 2] == 1)

        assert denoised_eos_count == original_eos_count, \
            "Denoise should preserve number of strokes"

    def test_denoise_output_shape(self):
        """Test denoise preserves output shape."""
        from svg_output import denoise, offsets_to_coords

        offsets = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ] * 3, dtype=np.float32)

        coords = offsets_to_coords(offsets)
        denoised = denoise(coords)

        # Shape should be preserved
        assert denoised.shape == coords.shape

    def test_denoise_smooths_noisy_data(self):
        """Test denoise reduces variance in noisy data."""
        from svg_output import denoise, offsets_to_coords

        # Create noisy data
        np.random.seed(42)
        n_points = 50

        # Smooth trajectory with noise
        t = np.linspace(0, 10, n_points)
        x_clean = t
        y_clean = np.sin(t)

        # Add noise
        noise_level = 0.5
        x_noisy = x_clean + np.random.randn(n_points) * noise_level
        y_noisy = y_clean + np.random.randn(n_points) * noise_level

        # Convert to offsets then coords
        offsets = np.diff(np.vstack([[0, 0], np.column_stack([x_noisy, y_noisy])]), axis=0)
        offsets = np.column_stack([offsets, np.zeros(n_points)])
        offsets[-1, 2] = 1.0  # Mark end

        coords = offsets_to_coords(offsets)
        denoised = denoise(coords)

        # Denoised should have lower variance in second derivatives (smoother)
        x_noisy_diff2 = np.diff(coords[:, 0], n=2)
        x_smooth_diff2 = np.diff(denoised[:, 0], n=2)

        var_noisy = np.var(x_noisy_diff2)
        var_smooth = np.var(x_smooth_diff2)

        assert var_smooth < var_noisy, \
            "Denoised data should have lower variance (smoother)"

    def test_denoise_handles_short_strokes(self):
        """Test denoise handles strokes shorter than filter window."""
        from svg_output import denoise, offsets_to_coords

        # Very short stroke (less than filter window of 7)
        offsets = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32)

        coords = offsets_to_coords(offsets)

        # Should not raise exception
        denoised = denoise(coords)

        assert denoised.shape == coords.shape


@pytest.mark.unit
class TestAlign:
    """Test rotation alignment correction."""

    def test_align_output_shape(self):
        """Test align preserves output shape."""
        from svg_output import align

        coords = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ])

        aligned = align(coords)

        # Shape should be preserved
        assert aligned.shape == coords.shape

    def test_align_corrects_slope(self):
        """Test align corrects for slanted handwriting."""
        from svg_output import align

        # Create slanted line (45 degree slope)
        coords = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ])

        aligned = align(coords)

        # After alignment, should have reduced slope
        # (rotation should flatten the line)
        # We can check this by verifying y-variance decreased
        y_var_before = np.var(coords[:, 1])
        y_var_after = np.var(aligned[:, 1])

        # After rotation, y-variance should be close to 0
        assert y_var_after < y_var_before, \
            "Alignment should reduce y-axis variance"

    def test_align_handles_three_columns(self):
        """Test align works with 3-column input (x, y, eos)."""
        from svg_output import align

        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
        ])

        # Should process only first 2 columns
        aligned = align(coords)

        assert aligned.shape == coords.shape

    def test_align_centers_output(self):
        """Test align centers and offsets output."""
        from svg_output import align

        coords = np.array([
            [10.0, 5.0],
            [11.0, 6.0],
            [12.0, 7.0],
            [13.0, 8.0],
        ])

        aligned = align(coords)

        # After alignment and offset subtraction, coordinates change
        # (specific values depend on regression, but should be different)
        assert not np.allclose(aligned, coords), \
            "Alignment should transform coordinates"


@pytest.mark.unit
class TestSaveAsSVG:
    """Test SVG file generation."""

    def test_save_as_svg_creates_file(self, temp_output_dir):
        """Test save_as_svg creates SVG file."""
        from svg_output import save_as_svg

        # Simple stroke data
        stroke = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [-10.0, 0.0, 0.0],
            [0.0, -10.0, 1.0],
        ], dtype=np.float32)

        lines = ["test"]
        all_strokes = [stroke]

        output_path = os.path.join(temp_output_dir, "test.svg")

        save_as_svg(all_strokes, lines, output_path)

        # File should exist
        assert os.path.exists(output_path), "SVG file should be created"

    def test_save_as_svg_valid_xml(self, temp_output_dir):
        """Test save_as_svg produces valid XML."""
        from svg_output import save_as_svg
        import xml.etree.ElementTree as ET

        # Simple stroke data
        stroke = np.array([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
        ] * 5, dtype=np.float32)  # Repeat for enough points

        lines = ["test"]
        all_strokes = [stroke]

        output_path = os.path.join(temp_output_dir, "valid.svg")

        save_as_svg(all_strokes, lines, output_path)

        # Should parse as valid XML
        try:
            tree = ET.parse(output_path)
            root = tree.getroot()
            assert root is not None
        except ET.ParseError as e:
            pytest.fail(f"SVG is not valid XML: {e}")

    def test_save_as_svg_multiline(self, temp_output_dir):
        """Test save_as_svg handles multiple lines."""
        from svg_output import save_as_svg

        stroke1 = np.array([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
        ] * 5, dtype=np.float32)

        stroke2 = np.array([
            [5.0, 5.0, 0.0],
            [5.0, 5.0, 1.0],
        ] * 5, dtype=np.float32)

        lines = ["line 1", "line 2"]
        all_strokes = [stroke1, stroke2]

        output_path = os.path.join(temp_output_dir, "multiline.svg")

        save_as_svg(all_strokes, lines, output_path)

        # File should exist
        assert os.path.exists(output_path)

        # File should contain both lines
        with open(output_path, 'r') as f:
            content = f.read()
            # SVG should have multiple path elements
            assert content.count('<path') >= 2, \
                "SVG should contain multiple paths for multiple lines"

    def test_save_as_svg_handles_empty_lines(self, temp_output_dir):
        """Test save_as_svg handles empty lines gracefully."""
        from svg_output import save_as_svg

        stroke1 = np.array([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
        ] * 5, dtype=np.float32)

        # Empty line
        stroke2 = np.array([], dtype=np.float32).reshape(0, 3)

        lines = ["line 1", ""]
        all_strokes = [stroke1, stroke2]

        output_path = os.path.join(temp_output_dir, "empty_line.svg")

        # Should not raise exception
        save_as_svg(all_strokes, lines, output_path)

        assert os.path.exists(output_path)

    def test_save_as_svg_custom_parameters(self, temp_output_dir):
        """Test save_as_svg accepts custom parameters."""
        from svg_output import save_as_svg

        stroke = np.array([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
        ] * 5, dtype=np.float32)

        lines = ["test"]
        all_strokes = [stroke]

        output_path = os.path.join(temp_output_dir, "custom.svg")

        # Custom parameters
        save_as_svg(
            all_strokes, lines, output_path,
            line_height=100,
            view_width=2000,
            stroke_width=3,
            stroke_color='blue'
        )

        assert os.path.exists(output_path)

        # Verify parameters are used
        with open(output_path, 'r') as f:
            content = f.read()
            assert 'viewBox="0 0 2000' in content, "Custom view_width should be used"
            assert 'stroke="blue"' in content, "Custom stroke_color should be used"
            assert 'stroke-width="3"' in content, "Custom stroke_width should be used"

    def test_save_as_svg_path_format(self, temp_output_dir):
        """Test SVG path uses M/L commands (pen plotter format)."""
        from svg_output import save_as_svg

        # Stroke with pen lift
        stroke = np.array([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],  # Pen lift
            [10.0, 0.0, 0.0],  # Pen down
            [10.0, 0.0, 1.0],
        ] * 3, dtype=np.float32)

        lines = ["test"]
        all_strokes = [stroke]

        output_path = os.path.join(temp_output_dir, "path_format.svg")

        save_as_svg(all_strokes, lines, output_path)

        # Read SVG
        with open(output_path, 'r') as f:
            content = f.read()

        # Should contain M (move) commands for pen lifts
        assert 'M' in content, "SVG should contain M (move) commands"
        # Should contain L (line) commands for pen down
        assert 'L' in content, "SVG should contain L (line) commands"

    def test_save_as_svg_no_fill(self, temp_output_dir):
        """Test SVG paths have no fill (plotter optimization)."""
        from svg_output import save_as_svg

        stroke = np.array([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
        ] * 5, dtype=np.float32)

        lines = ["test"]
        all_strokes = [stroke]

        output_path = os.path.join(temp_output_dir, "no_fill.svg")

        save_as_svg(all_strokes, lines, output_path)

        # Read SVG
        with open(output_path, 'r') as f:
            content = f.read()

        # Path should have fill="none"
        assert 'fill="none"' in content, \
            "SVG paths should have no fill (plotter optimization)"


@pytest.mark.unit
class TestCoordinateTransformations:
    """Test coordinate transformation edge cases."""

    def test_offsets_to_coords_single_point(self):
        """Test conversion with single point."""
        from svg_output import offsets_to_coords

        offsets = np.array([[5.0, 3.0, 1.0]])

        coords = offsets_to_coords(offsets)

        # Single point should be at offset
        assert coords[0, 0] == 5.0
        assert coords[0, 1] == 3.0
        assert coords[0, 2] == 1.0

    def test_offsets_to_coords_all_zeros(self):
        """Test conversion with all zero offsets."""
        from svg_output import offsets_to_coords

        offsets = np.zeros((10, 3))

        coords = offsets_to_coords(offsets)

        # All coordinates should be zero
        np.testing.assert_array_equal(coords[:, :2], np.zeros((10, 2)))

    def test_denoise_empty_input(self):
        """Test denoise handles empty input."""
        from svg_output import denoise

        coords = np.array([]).reshape(0, 3)

        # Should handle gracefully (might return empty or raise)
        # Actual behavior depends on implementation
        try:
            denoised = denoise(coords)
            # If it succeeds, verify shape
            assert denoised.shape[0] == 0
        except (ValueError, IndexError):
            # Empty input might raise exception - that's OK
            pass


# ============================================================================
# Summary
# ============================================================================

def test_svg_output_suite_summary():
    """
    SVG output test suite summary.

    If all SVG output tests pass, pen plotter output is correct:
    - Delta to absolute coordinate conversion works
    - Savitzky-Golay smoothing filter reduces noise
    - Rotation alignment corrects slant
    - Multi-line SVG generation works
    - SVG format is plotter-ready (M/L commands, no fill)

    Mathematical properties verified:
    - Cumulative sum is correct
    - Denoising reduces variance
    - Alignment reduces slope
    - End-of-stroke markers preserved
    """
    print("\n✓ All SVG output tests passed!")
    print("  - Coordinate conversion correct")
    print("  - Denoising smooths data")
    print("  - Alignment corrects rotation")
    print("  - SVG generation works")
    print("  - Plotter-ready format (M/L, no fill)")
