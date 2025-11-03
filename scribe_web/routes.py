"""
Scribe Web UI - Routes

URL endpoints for the web interface.

Routes:
    / (GET) - Main generator page
    /generate (POST) - Generate handwriting
    /download/<filename> (GET) - Download generated file
"""

from flask import render_template, request, jsonify, send_file
from app import app, model, model_args, OUTPUTS_DIR
import uuid
from pathlib import Path
import time


# Style metadata (name and description for each style)
STYLE_INFO = [
    {'id': 0, 'name': 'Style 0', 'description': 'Elegant cursive'},
    {'id': 1, 'name': 'Style 1', 'description': 'Neat print'},
    {'id': 2, 'name': 'Style 2', 'description': 'Casual script'},
    {'id': 3, 'name': 'Style 3', 'description': 'Formal handwriting'},
    {'id': 4, 'name': 'Style 4', 'description': 'Artistic flourish'},
    {'id': 5, 'name': 'Style 5', 'description': 'Compact writing'},
    {'id': 6, 'name': 'Style 6', 'description': 'Flowing script'},
    {'id': 7, 'name': 'Style 7', 'description': 'Bold strokes'},
    {'id': 8, 'name': 'Style 8', 'description': 'Light touch'},
    {'id': 9, 'name': 'Style 9', 'description': 'Classic style'},
    {'id': 10, 'name': 'Style 10', 'description': 'Modern print'},
    {'id': 11, 'name': 'Style 11', 'description': 'Expressive hand'},
    {'id': 12, 'name': 'Style 12', 'description': 'Traditional script'},
]


@app.route('/')
@app.route('/index')
def index():
    """Main generator page."""
    return render_template('index.html', styles=STYLE_INFO)


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate handwriting from text input.

    Expects JSON:
        {
            "text": "Multi-line\ntext\ninput",
            "style": 0-12 (style ID),
            "bias": 0.5-2.0 (neatness control),
            "format": "svg" or "png"
        }

    Returns JSON:
        {
            "success": true,
            "png_filename": "uuid.png",
            "svg_filename": "uuid.svg",
            "generation_time": 5.2
        }
    """
    start_time = time.time()

    try:
        # Parse request data
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        text = data.get('text', '').strip()
        style_id = int(data.get('style', 0))
        bias = float(data.get('bias', 1.0))
        output_format = data.get('format', 'png')

        # Validate inputs
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400

        if style_id < 0 or style_id > 12:
            return jsonify({'success': False, 'error': 'Invalid style ID (must be 0-12)'}), 400

        if bias < 0.1 or bias > 5.0:
            return jsonify({'success': False, 'error': 'Invalid bias (must be 0.1-5.0)'}), 400

        # Check if model is loaded
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500

        # Split text into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if not lines:
            return jsonify({'success': False, 'error': 'No valid text lines'}), 400

        # Generate handwriting using Scribe
        # Import here to avoid circular import
        import sample as scribe_sample

        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        png_filename = f"{file_id}.png"
        svg_filename = f"{file_id}.svg"

        png_path = OUTPUTS_DIR / png_filename
        svg_path = OUTPUTS_DIR / svg_filename

        # Call Scribe's sample_multiline function
        # Use same style and bias for all lines (user's preference from our discussion)
        styles = [style_id] * len(lines)
        biases = [bias] * len(lines)

        # Generate
        strokes_list = []
        for i, line in enumerate(lines):
            print(f"Generating line {i+1}/{len(lines)}: '{line[:30]}...'")

            # Load style states if needed
            initial_states = None
            if style_id is not None:
                initial_states = scribe_sample.load_style_state(
                    model, style_id, model_args['alphabet']
                )

            # Generate single line
            stroke = scribe_sample.sample(
                model,
                line,
                initial_states=initial_states,
                bias=biases[i],
                tsteps=700,
                alphabet=model_args['alphabet']
            )

            strokes_list.append(stroke)

        # Render to SVG and PNG
        import svg_output

        # Save SVG
        svg_output.save_as_svg(
            strokes_list,
            lines,
            str(svg_path),
            stroke_width=2.0,
            stroke_color='black',
            margin=50,
            line_spacing=80
        )

        # Convert SVG to PNG for web display
        # (For now, we'll use matplotlib to render PNG directly)
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(12, 2 * len(lines)))
        ax.set_xlim(-50, 800)
        ax.set_ylim(-50, 120 * len(lines))
        ax.axis('off')

        current_y = 0
        for stroke_data in strokes_list:
            # Convert delta to absolute coordinates
            coords = np.cumsum(stroke_data[:, :2], axis=0)

            # Find stroke breaks (where eos == 1)
            eos_indices = np.where(stroke_data[:, 2] == 1)[0]

            # Plot each continuous stroke
            start_idx = 0
            for end_idx in eos_indices:
                if end_idx > start_idx:
                    segment = coords[start_idx:end_idx+1]
                    ax.plot(segment[:, 0], -segment[:, 1] + current_y, 'k-', linewidth=2)
                start_idx = end_idx + 1

            current_y += 120  # Move down for next line

        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        generation_time = time.time() - start_time

        print(f"✓ Generated in {generation_time:.2f}s")

        return jsonify({
            'success': True,
            'png_filename': png_filename,
            'svg_filename': svg_filename,
            'generation_time': round(generation_time, 2)
        })

    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg}), 500


@app.route('/download/<filename>')
def download(filename):
    """
    Download generated file.

    Args:
        filename: Name of file to download (must be in outputs/)

    Returns:
        File download response
    """
    try:
        file_path = OUTPUTS_DIR / filename

        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404

        # Security: Ensure file is in outputs directory
        if not file_path.resolve().is_relative_to(OUTPUTS_DIR.resolve()):
            return jsonify({'error': 'Invalid file path'}), 403

        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })
