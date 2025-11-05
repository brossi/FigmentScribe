# Scribe Web UI

A simple, self-hosted web interface for generating handwritten text using Scribe's neural network models.

![Scribe Web UI Screenshot](../static/scribe_web_screenshot.png)

## Features

- 📝 **Multi-Line Text Input** - Generate documents with multiple lines
- 🎨 **13 Pre-Trained Styles** - Choose from diverse handwriting styles
- 🎚️ **Neatness Control** - Adjust bias from messy (0.5) to neat (2.0)
- 📥 **Multiple Formats** - Download as PNG (images) or SVG (pen plotters)
- 🚀 **Fast Generation** - Model loaded once at startup (~5-15 sec per generation)
- 💻 **Self-Hosted** - Runs locally, no cloud required

## Quick Start

### Prerequisites

1. **Scribe installed and working** - Make sure you can run `python3 sample.py` successfully
2. **Trained model checkpoint** - Located at `../saved/model/`
3. **Python 3.11** - Same environment as Scribe

### Installation

```bash
# Navigate to web UI directory
cd scribe_web

# Install Flask dependencies
pip install -r requirements.txt

# Run the app
python3 app.py
```

### Usage

1. Open your browser to **http://localhost:5000**
2. Enter your text (multi-line supported)
3. Select a handwriting style (0-12)
4. Adjust neatness with the bias slider
5. Click "Generate Handwriting"
6. Download PNG or SVG

## Configuration

### Model Settings

Edit `app.py` to match your trained model:

```python
model_args = {
    'rnn_size': 400,      # Must match your model
    'nmixtures': 20,      # Must match your model
    'kmixtures': 1,       # Must match your model
    'alphabet': '...',    # Must match your model
}
```

### Server Settings

```python
app.run(
    host='0.0.0.0',       # '127.0.0.1' for local only
    port=5000,            # Change port if needed
    debug=True,           # False for production
    use_reloader=False    # Keep False to avoid double model loading
)
```

## File Structure

```
scribe_web/
├── app.py                  # Flask app initialization
├── routes.py               # URL endpoints
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── templates/
│   ├── base.html           # Base template with navbar
│   └── index.html          # Main generator page
├── static/
│   ├── css/
│   │   └── style.css       # Custom styles
│   ├── js/
│   │   └── app.js          # Frontend JavaScript
│   └── style_samples/      # Style preview thumbnails (optional)
└── outputs/                # Generated files (gitignored)
```

## API Endpoints

### GET /

Main generator page (HTML interface)

### POST /generate

Generate handwriting from text

**Request:**
```json
{
    "text": "Multi-line\ntext\ninput",
    "style": 0,           // Style ID (0-12)
    "bias": 1.0,          // Neatness (0.5-2.0)
    "format": "png"       // "png" or "svg"
}
```

**Response:**
```json
{
    "success": true,
    "png_filename": "abc123.png",
    "svg_filename": "abc123.svg",
    "generation_time": 5.2
}
```

### GET /download/<filename>

Download generated file

### GET /health

Health check (returns model loaded status)

## Tips

### Improving Performance

1. **Use production WSGI server** (not Flask dev server):
   ```bash
   pip install gunicorn
   gunicorn -w 2 -b 0.0.0.0:5000 app:app
   ```

2. **Limit concurrent users** - Model is single-threaded
   - Use worker queue (Celery/RQ) for multiple users
   - Or limit to 1 user (personal tool)

3. **Clean up outputs** - Add cron job to delete old files:
   ```bash
   find scribe_web/outputs/ -type f -mtime +1 -delete
   ```

### Customization

**Change styles:**
- Edit `STYLE_INFO` in `routes.py` to rename styles
- Add thumbnails to `static/style_samples/` (optional)

**Change default text:**
- Edit placeholder in `templates/index.html`

**Change colors/styling:**
- Modify `static/css/style.css`
- Bootstrap 5 variables can be customized

## Troubleshooting

### "Model not loaded" error

**Cause:** Checkpoint not found or invalid

**Solution:**
```bash
# Check if checkpoint exists
ls -lh ../saved/model/

# Verify it's not empty
# Should see: checkpoint, model.data-*, model.index

# Check model parameters match (rnn_size, nmixtures, etc.)
```

### Generation takes > 30 seconds

**Cause:** CPU inference is slow

**Solution:**
- Use GPU-enabled TensorFlow
- Reduce `tsteps` in `routes.py` (trade quality for speed)
- Use smaller model (rnn_size=100 instead of 400)

### "Cannot connect to server"

**Cause:** Port already in use or firewall

**Solution:**
```bash
# Check if port 5000 is available
lsof -i :5000

# Try different port
python3 app.py --port 8080  # (if you add argparse)
```

### Generated handwriting looks bad

**Cause:** Model not trained properly or wrong parameters

**Solution:**
- Verify model trained successfully
- Check `model_args` match training config
- Try different styles (some may work better)
- Adjust bias slider

## Known Limitations

- **Single-threaded** - One generation at a time
- **No user accounts** - Personal use only
- **No custom style upload** - Uses 13 pre-trained styles only
- **No style gallery** - Must try each style manually
- **No generation history** - Files not persisted long-term
- **CPU-bound** - Slow without GPU

These can be addressed in future versions!

## Future Enhancements

- [ ] Per-line style/bias control
- [ ] Style gallery with thumbnails
- [ ] SVG drawing canvas (custom handwriting upload)
- [ ] Real-time preview (low-res)
- [ ] Generation history
- [ ] Batch processing (upload text file)
- [ ] PDF export
- [ ] Character profile presets
- [ ] Pressure variation toggle
- [ ] User accounts and saved styles
- [ ] Async generation with progress bar

## License

Same as parent Scribe project.

## Credits

Built on top of Scribe's TensorFlow 2.15 implementation of Alex Graves' "Generating Sequences With Recurrent Neural Networks" (2013).

**Web UI:**
- Flask 3.0
- Bootstrap 5.3
- jQuery 3.7

**ML Model:**
- TensorFlow 2.15
- 3-layer LSTM with attention
- Mixture Density Network output

## Support

For issues with:
- **Web UI**: Check this README and `app.py` logs
- **Model/generation**: See main Scribe documentation in `../CLAUDE.md`
- **Installation**: Ensure main Scribe works before using web UI

---

**Enjoy generating handwritten text!** 🖊️
