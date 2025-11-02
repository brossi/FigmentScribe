"""
SVG output module for handwriting synthesis.

Ported from sjvasquez/handwriting-synthesis, optimized for pen plotter output.
"""

import numpy as np
from scipy.signal import savgol_filter
import svgwrite


def offsets_to_coords(offsets):
    """
    Convert from offsets (deltas) to absolute coordinates.

    Args:
        offsets: Array of shape (n_points, 3) with columns [Δx, Δy, eos]

    Returns:
        coords: Array of shape (n_points, 3) with columns [x, y, eos]
    """
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)


def denoise(coords):
    """
    Smoothing filter to mitigate artifacts using Savitzky-Golay filter.

    Args:
        coords: Array of shape (n_points, 3) with columns [x, y, eos]

    Returns:
        smoothed_coords: Denoised coordinates
    """
    # Split into individual strokes at end-of-stroke markers
    coords = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
    new_coords = []

    for stroke in coords:
        if len(stroke) != 0:
            # Apply Savitzky-Golay filter (polynomial order 3, window size 7)
            x_new = savgol_filter(stroke[:, 0], 7, 3, mode='nearest')
            y_new = savgol_filter(stroke[:, 1], 7, 3, mode='nearest')
            xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])
            stroke = np.concatenate([xy_coords, stroke[:, 2].reshape(-1, 1)], axis=1)
            new_coords.append(stroke)

    coords = np.vstack(new_coords)
    return coords


def align(coords):
    """
    Corrects for global slant/offset in handwriting strokes using linear regression.

    Args:
        coords: Array of shape (n_points, 2) or (n_points, 3) with coordinates

    Returns:
        aligned_coords: Rotation-corrected coordinates
    """
    coords = np.copy(coords)
    X, Y = coords[:, 0].reshape(-1, 1), coords[:, 1].reshape(-1, 1)
    X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)

    # Linear regression to find slope
    offset, slope = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y).squeeze()
    theta = np.arctan(slope)

    # Rotation matrix
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )

    coords[:, :2] = np.dot(coords[:, :2], rotation_matrix) - offset
    return coords


def save_as_svg(all_strokes, lines, filename, line_height=60, view_width=1000,
                stroke_width=2, stroke_color='black'):
    """
    Generate SVG file from multi-line handwriting strokes.

    Optimized for pen plotter output: clean paths, no fills, single color.

    Args:
        all_strokes: List of stroke arrays, one per line
        lines: List of text strings (for reference/comments)
        filename: Output .svg filename
        line_height: Vertical spacing between lines (pixels)
        view_width: SVG viewbox width (pixels)
        stroke_width: Pen width (pixels)
        stroke_color: Stroke color (default: black)
    """
    view_height = line_height * (len(lines) + 1)

    # Create SVG drawing
    dwg = svgwrite.Drawing(filename=filename)
    dwg.viewbox(width=view_width, height=view_height)
    dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

    # Starting Y position
    initial_coord = np.array([0, -(3 * line_height / 4)])

    for offsets, line in zip(all_strokes, lines):
        # Handle empty lines
        if not line or len(offsets) == 0:
            initial_coord[1] -= line_height
            continue

        # Scale strokes slightly for readability
        offsets = np.copy(offsets)
        offsets[:, :2] *= 1.5

        # Convert offsets to coordinates
        strokes = offsets_to_coords(offsets)

        # Apply denoising and alignment
        strokes = denoise(strokes)
        strokes[:, :2] = align(strokes[:, :2])

        # Flip Y axis (SVG coordinates)
        strokes[:, 1] *= -1

        # Position the line
        strokes[:, :2] -= strokes[:, :2].min() + initial_coord
        # Center horizontally
        strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2

        # Create SVG path
        prev_eos = 1.0
        p = "M{},{} ".format(0, 0)
        for x, y, eos in zip(*strokes.T):
            # 'M' for pen lift (move), 'L' for pen down (line)
            p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
            prev_eos = eos

        # Add path to SVG (optimized for plotters: stroke only, no fill)
        path = svgwrite.path.Path(p)
        path = path.stroke(color=stroke_color, width=stroke_width, linecap='round').fill("none")
        dwg.add(path)

        # Move to next line position
        initial_coord[1] -= line_height

    # Save SVG file
    dwg.save()
    print(f"SVG saved to {filename}")
    print(f"  Lines: {len(lines)}")
    print(f"  Dimensions: {view_width} × {view_height} pixels")
    print(f"  Ready for gcode conversion (e.g., vpype, svg2gcode)")
