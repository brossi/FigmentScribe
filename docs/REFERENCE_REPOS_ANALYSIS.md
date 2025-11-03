# Reference Repository Analysis

This document compares three high-value handwriting synthesis repositories kept for reference against Scribe's current capabilities. These repositories were selected from 5 candidates (2 redundant repos deleted) based on unique features, code quality, and potential contributions to Scribe's development.

**Last Updated:** November 2025
**Analysis Date:** November 3, 2025

---

## Executive Summary

**Scribe Status:** Top-tier handwriting synthesis implementation with modern TensorFlow 2.15, comprehensive features (multi-line, SVG output, style priming, pen plotter support), and excellent documentation.

**Key Finding:** While Scribe is already at the forefront, three repositories offer unique features worth studying for potential future enhancements:

1. **handwriting-model** - Stroke pressure/thickness prediction (completely novel)
2. **pytorch-handwriting-synthesis-toolkit** - ONNX export & attention visualization
3. **Handwriting-synthesis** - Full web application with custom style upload

---

## Repository Comparison Matrix

| Feature | **Scribe** | **handwriting-model** | **pytorch-toolkit** | **Handwriting-synthesis** |
|---------|------------|----------------------|---------------------|---------------------------|
| **Framework** | TensorFlow 2.15 | PyTorch | PyTorch | PyTorch |
| **Last Updated** | 2025 | Aug 2024 | Apr 2023 | Dec 2020 |
| **Active Development** | ✅ | ✅ | 🟡 | ❌ |
| **LSTM Layers** | 3 (standard) | 3 (peephole) | 3 (peephole) | 3 (standard) |
| **Architecture Type** | Standard LSTM | Peephole LSTM | Peephole LSTM | Standard LSTM |
| **Style Priming** | ✅ (13 styles) | ❌ | ❌ | ✅ (custom upload) |
| **Multi-line Generation** | ✅ (per-line control) | ❌ | ✅ (txt2page) | ❌ |
| **SVG Output** | ✅ (pen plotter) | ❌ | ✅ | ✅ (web canvas) |
| **Per-line Bias Control** | ✅ | ❌ | ❌ | ❌ |
| **Character Set** | 83 chars (full) | ~80 chars | ~80 chars | ~60 chars |
| **Data Samples** | 11,916 (IAM) | ~11,000 (IAM) | ~10,000 (IAM-OnDB) | 6,000 (IAM) |
| **Test Suite** | ✅ (pytest) | ✅ (comprehensive) | ✅ (comprehensive) | ❌ |
| **Documentation** | Excellent | Good | Excellent | Minimal |
| **Colab Support** | ✅ | ❌ | ❌ | ❌ |
| | | | | |
| **UNIQUE FEATURES** | | | | |
| **Pressure/Thickness Model** | ❌ | ✅ ⭐ | ❌ | ❌ |
| **ONNX Export** | ❌ | ❌ | ✅ ⭐ | ❌ |
| **Attention Visualization** | ❌ | ❌ | ✅ ⭐ | ❌ |
| **Web UI (Flask)** | ❌ | ❌ | ❌ | ✅ ⭐ |
| **Custom Data Collection** | ❌ | ✅ (Wacom) | ❌ | ❌ |
| **JIT Compilation** | ❌ | ✅ | ✅ | ❌ |

---

## Detailed Repository Profiles

### 1. handwriting-model ⭐ TOP PRIORITY

**Origin:** Custom PyTorch implementation with major extensions
**Repository:** `reference-repos/handwriting-model/`
**Last Updated:** August 2024 (actively maintained)
**Framework:** PyTorch with JIT compilation support

#### Unique Features

**🎯 Stroke Thickness/Pressure Model (COMPLETELY NOVEL)**
- Second ML layer that predicts pen pressure from generated coordinates
- Trained on Wacom tablet pressure data
- Adds realistic stroke width variation
- No other handwriting synthesis repo has this feature
- Implementation: `pressure_model/LTSM_model.py`

**Key Technical Details:**
- Separate LSTM model for pressure prediction
- Takes generated coordinates as input
- Outputs continuous pressure values (0-1 range)
- Can be trained on custom Wacom tablet data

**Peephole LSTM Architecture**
- More sophisticated than standard LSTM
- Cell state fed directly to gates for better gradient flow
- Custom implementation in `handwriting_model/modules.py`
- Used in all 3 LSTM layers

**Custom Data Collection System**
- Web application for collecting handwriting via Wacom tablets
- Captures both coordinates and pressure data
- ETL pipeline for new datasets: `infrastrcture/data_collection/`
- Enables building custom handwriting datasets

**Comprehensive Testing**
- Full pytest suite covering models, data, loss, metrics, callbacks, utils
- Better test coverage than most academic repos
- Professional testing infrastructure

**Multiple Analysis Notebooks**
- 9 Jupyter notebooks for data exploration, thickness analysis, model comparison
- Tools for visualizing stroke thickness
- Dataset quality assessment utilities

#### Code Quality: **Excellent**
- Well-structured, modular design
- Clear separation of concerns
- Active maintenance (2024)
- Good documentation
- Infrastructure scripts for cloud training

#### Potential Ports to Scribe

**High Priority:**
1. **Pressure/Thickness Prediction Model**
   - Add as optional post-processing layer
   - Train on Wacom data or synthetic pressure
   - Major differentiator for Scribe
   - Estimated effort: Medium-High (new model to train)

2. **Peephole LSTM Architecture**
   - Replace standard LSTM with peephole variant
   - May improve generation quality
   - Requires retraining models
   - Estimated effort: Medium (architecture swap)

**Medium Priority:**
3. **Testing Infrastructure**
   - Adopt comprehensive test patterns
   - Add property-based tests for attention/MDN
   - Estimated effort: Low-Medium

4. **Jupyter Analysis Notebooks**
   - Port thickness analysis to Scribe
   - Add model comparison tools
   - Estimated effort: Low

#### Files of Interest
- `pressure_model/LTSM_model.py` - Pressure prediction model
- `handwriting_model/modules.py` - Peephole LSTM implementation
- `handwriting_model/models.py` - Main synthesis architecture
- `tests/` - Comprehensive test suite
- `notebooks/` - Analysis and visualization tools

---

### 2. pytorch-handwriting-synthesis-toolkit ⭐ HIGH PRIORITY

**Origin:** X-rayLaser's professional implementation
**Repository:** `reference-repos/pytorch-handwriting-synthesis-toolkit/`
**Last Updated:** April 2023 (stable, community PRs accepted)
**Framework:** PyTorch with ONNX export

#### Unique Features

**🎯 ONNX Export Support**
- Export trained models to ONNX format
- Enables deployment to non-PyTorch environments:
  - Web browsers (JavaScript via ONNX.js)
  - Mobile devices (iOS, Android)
  - Embedded systems
  - Edge devices
- Implementation: `export_to_onnx.py`
- Complete conversion pipeline with validation

**Key Technical Details:**
- Converts PyTorch model to platform-agnostic format
- Maintains model accuracy through conversion
- Tested export/import cycle
- Documentation for deployment scenarios

**🎯 Attention Visualization**
- `--show_weights` flag creates attention heatmaps
- Shows which characters model focuses on during generation
- `--heatmap` for mixture density visualization
- Excellent debugging tool
- Great for demos and papers
- Implementation: Visualization utilities in sampling code

**txt2page Document Generation**
- `txt2script.py` - converts full text files to multi-page documents
- Automatic line breaking and layout
- Page composition for realistic documents
- Complements Scribe's multi-line generation

**Custom Data Provider System**
- Pluggable architecture for different datasets
- Easy to add new data sources
- Well-documented provider interface: `data_providers/`
- Supports IAM-OnDB and custom formats

**Best Documentation**
- 506-line README (most comprehensive of all repos)
- Complete API documentation
- Usage examples for all tools
- Installation and troubleshooting guides

#### Code Quality: **Excellent**
- Production-ready codebase
- Comprehensive test suite (6 modules: data, loss, metrics, models, callbacks, utils)
- Professional software engineering practices
- Active community (PRs accepted recently)
- Clean modular architecture

#### Potential Ports to Scribe

**High Priority:**
1. **ONNX Export**
   - Research TensorFlow → ONNX conversion (TF2ONNX)
   - Enable web deployment for Scribe
   - Major feature for broader adoption
   - Estimated effort: Medium (new conversion pipeline)

2. **Attention Visualization**
   - Add heatmap generation to sample.py
   - Show attention window movement over text
   - Valuable debugging and demo tool
   - Estimated effort: Low-Medium (visualization layer)

**Medium Priority:**
3. **txt2page Document Generation**
   - Enhance Scribe's multi-line with full page layout
   - Automatic margin and spacing calculation
   - Estimated effort: Low-Medium

4. **Custom Data Provider Architecture**
   - Refactor utils.py to support pluggable datasets
   - Enable easy addition of new data sources
   - Estimated effort: Medium

#### Files of Interest
- `export_to_onnx.py` - ONNX export implementation
- `generate_samples.py` - Attention visualization code
- `txt2script.py` - Multi-page document generation
- `data_providers/` - Pluggable data loading system
- `tests/` - Comprehensive test patterns
- `README.md` - Excellent documentation reference

---

### 3. Handwriting-synthesis 🟡 MEDIUM PRIORITY

**Origin:** Fork with web interface by swechhachoudhary
**Repository:** `reference-repos/Handwriting-synthesis/`
**Last Updated:** December 2020 (unmaintained but functional)
**Framework:** PyTorch with Flask web app

#### Unique Features

**🎯 Full Web Application (Flask)**
- Complete interactive drawing interface
- Upload custom handwriting as "style"
- Generate text in uploaded style (custom priming)
- Real-time generation via web UI
- Mobile device support (touch input)
- **This is a complete user-facing product, not just research code**

**Key Technical Details:**
- Flask backend serving PyTorch model
- JavaScript canvas for drawing interface
- SVG path parsing from web canvas
- Device-aware downsampling (mobile vs desktop)
- Session management for user styles

**Custom Style Priming from User Input**
- Users draw handwriting on web canvas
- System extracts style and uses for generation
- SVG path parser converts canvas strokes to model format
- Dynamic style extraction (unlike Scribe's pre-trained styles)

**XML/SVG Parser**
- Custom parser for web-drawn SVG paths
- Converts canvas strokes to delta-encoded format
- Handles various drawing speeds and devices
- Implementation: SVG path processing utilities

#### Code Quality: **Good**
- Working web application (deployable)
- Decent structure with clear frontend/backend separation
- Outdated dependencies (2020)
- No comprehensive test suite
- Minimal documentation

#### Potential Ports to Scribe

**Medium Priority:**
1. **Web UI Concept**
   - Build interactive demo application
   - Allow users to generate handwriting in browser
   - Community engagement tool
   - Estimated effort: High (full web app development)

2. **Custom Style Upload**
   - Let users prime with their own handwriting
   - Dynamic style extraction from uploaded samples
   - More flexible than pre-trained styles
   - Estimated effort: Medium-High

3. **SVG Path Parsing**
   - Parse web canvas drawings into model format
   - Enable real-time style customization
   - Estimated effort: Low-Medium

**Lower Priority:**
4. **Mobile Device Handling**
   - Touch input support
   - Device-aware processing
   - Estimated effort: Medium (if web UI built)

#### Files of Interest
- `app.py` - Flask web application
- `static/` and `templates/` - Frontend interface
- SVG parsing utilities
- Style extraction code

#### Limitations
- Unmaintained since 2020
- Dependencies may be outdated
- No active community
- Keep primarily for web UI inspiration

---

## Prioritized Feature Roadmap for Scribe

Based on this analysis, here are the recommended features to consider porting to Scribe, organized by priority:

### Tier 1: High Impact, High Value

#### 1. Stroke Thickness/Pressure Model
**Source:** handwriting-model
**Impact:** ⭐⭐⭐⭐⭐ (Completely unique feature)
**Effort:** Medium-High
**Timeline:** 3-4 weeks

**Why:**
- No other handwriting synthesis tool has this
- Major differentiator for Scribe
- Adds significant realism to output
- Valuable for pen plotter output (variable line width)

**Implementation Path:**
1. Study `pressure_model/LTSM_model.py`
2. Design TensorFlow equivalent architecture
3. Generate or acquire pressure training data
4. Train pressure prediction model
5. Add as optional post-processing layer
6. Update SVG output to include stroke width

**Challenges:**
- Need pressure training data (Wacom tablet or synthetic)
- Separate model to train and maintain
- SVG format changes for variable width paths

---

#### 2. ONNX Export for Web Deployment
**Source:** pytorch-handwriting-synthesis-toolkit
**Impact:** ⭐⭐⭐⭐⭐ (Enables web/mobile deployment)
**Effort:** Medium
**Timeline:** 2-3 weeks

**Why:**
- Dramatically expands Scribe's reach (web browsers, mobile apps)
- JavaScript inference via ONNX.js
- No server required for deployment
- Community would love this feature

**Implementation Path:**
1. Research TensorFlow → ONNX conversion (tf2onnx library)
2. Test conversion with current Scribe model
3. Validate accuracy preservation
4. Create export script (like `export_to_onnx.py`)
5. Document deployment for web/mobile
6. Consider building simple web demo

**Challenges:**
- TensorFlow → ONNX compatibility (especially custom layers)
- Attention mechanism may need special handling
- MDN sampling in JavaScript/ONNX.js

**Resources:**
- tf2onnx library: https://github.com/onnx/tensorflow-onnx
- ONNX.js for browser inference: https://github.com/microsoft/onnxjs

---

#### 3. Attention Visualization
**Source:** pytorch-handwriting-synthesis-toolkit
**Impact:** ⭐⭐⭐⭐ (Great debugging tool and demo feature)
**Effort:** Low-Medium
**Timeline:** 1-2 weeks

**Why:**
- Shows model "reading" text character by character
- Excellent debugging tool for attention mechanism
- Amazing for demos, papers, presentations
- Relatively easy to implement

**Implementation Path:**
1. Modify sample.py to capture attention weights (φ values)
2. Create heatmap visualization function
3. Overlay attention on generated handwriting
4. Add `--visualize_attention` flag
5. Output annotated images showing focus

**Challenges:**
- Attention state extraction from model
- Visualization library selection (matplotlib sufficient)
- Alignment of attention with output strokes

---

### Tier 2: Medium Impact, Good Value

#### 4. Web UI Demo Application
**Source:** Handwriting-synthesis
**Impact:** ⭐⭐⭐⭐ (Community engagement)
**Effort:** High
**Timeline:** 4-6 weeks

**Why:**
- Makes Scribe accessible to non-technical users
- Great for demonstrations and outreach
- Custom style upload is powerful feature
- Increases project visibility

**Implementation Path:**
1. Design Flask or FastAPI backend
2. Create JavaScript frontend with drawing canvas
3. Implement SVG path parsing
4. Add real-time generation endpoint
5. Deploy as web demo

**Challenges:**
- Full web application development
- Real-time inference performance
- Server hosting costs
- Ongoing maintenance

**Alternative:** Consider ONNX export + static web page (lower maintenance)

---

#### 5. Peephole LSTM Architecture
**Source:** handwriting-model, pytorch-handwriting-synthesis-toolkit
**Impact:** ⭐⭐⭐ (Potential quality improvement)
**Effort:** Medium
**Timeline:** 2-3 weeks

**Why:**
- More sophisticated LSTM variant
- Better gradient flow (cell state to gates)
- Used in multiple modern implementations
- May improve generation quality

**Implementation Path:**
1. Implement PeepholeLSTM layer in TensorFlow
2. Replace standard LSTM in model.py
3. Retrain models from scratch
4. Compare quality with A/B testing
5. Keep if improvement verified

**Challenges:**
- Requires complete model retraining
- May not show significant improvement
- Backward compatibility with existing checkpoints

**Risk:** Effort may not justify quality gain (test on small scale first)

---

#### 6. Enhanced Testing Infrastructure
**Source:** handwriting-model, pytorch-handwriting-synthesis-toolkit
**Impact:** ⭐⭐⭐ (Code quality and reliability)
**Effort:** Medium
**Timeline:** 2-3 weeks

**Why:**
- More comprehensive test coverage
- Better confidence in changes
- Professional development practices
- Easier maintenance

**Implementation Path:**
1. Study test patterns from both repos
2. Add property-based tests for attention/MDN
3. Add integration tests for full pipeline
4. Add regression tests with golden outputs
5. Improve CI/CD coverage

---

### Tier 3: Lower Priority, Nice to Have

#### 7. Custom Data Provider System
**Source:** pytorch-handwriting-synthesis-toolkit
**Impact:** ⭐⭐ (Flexibility for research)
**Effort:** Medium

#### 8. txt2page Document Generation
**Source:** pytorch-handwriting-synthesis-toolkit
**Impact:** ⭐⭐ (Enhanced multi-line)
**Effort:** Low-Medium

#### 9. Jupyter Analysis Notebooks
**Source:** handwriting-model
**Impact:** ⭐⭐ (Model evaluation)
**Effort:** Low

#### 10. Custom Data Collection System
**Source:** handwriting-model
**Impact:** ⭐ (Research only)
**Effort:** High

---

## Implementation Recommendations

### Immediate Next Steps (This Week)

1. **Study pressure model implementation**
   - Read `handwriting-model/pressure_model/LTSM_model.py` thoroughly
   - Understand training data format
   - Assess feasibility for TensorFlow port
   - Determine data acquisition strategy (Wacom vs synthetic)

2. **Test ONNX conversion**
   - Install tf2onnx library: `pip install tf2onnx`
   - Attempt basic conversion of Scribe model
   - Identify conversion issues with custom layers
   - Document blockers and solutions

3. **Prototype attention visualization**
   - Extract attention weights from sample.py
   - Create simple matplotlib heatmap
   - Verify alignment with generated strokes
   - Estimate effort for full implementation

### Short Term (This Month)

- Implement attention visualization (easiest high-impact feature)
- Complete ONNX conversion feasibility study
- Begin pressure model design and data acquisition

### Medium Term (This Quarter)

- Implement pressure/thickness model
- Complete ONNX export with web demo
- Enhance test coverage

### Long Term (Next Quarter)

- Consider web UI application
- Evaluate peephole LSTM architecture
- Advanced features as needed

---

## Technical Notes

### TensorFlow → ONNX Conversion

**Key Library:** tf2onnx (https://github.com/onnx/tensorflow-onnx)

**Conversion Command:**
```bash
python -m tf2onnx.convert \
    --saved-model saved/model \
    --output scribe_model.onnx \
    --opset 13
```

**Potential Issues:**
- Custom attention layer may need manual ONNX ops
- MDN sampling should be deterministic for export
- RNN layers generally convert well

**Browser Inference:**
```javascript
// ONNX.js example
const session = new onnx.InferenceSession();
await session.loadModel('scribe_model.onnx');
const output = await session.run([inputTensor]);
```

### Pressure Model Architecture

**From handwriting-model analysis:**

```python
# Simplified architecture
class PressureModel(nn.Module):
    def __init__(self):
        self.lstm1 = LSTM(input_size=3, hidden_size=128)  # (x, y, eos)
        self.lstm2 = LSTM(input_size=128, hidden_size=64)
        self.fc = Linear(64, 1)  # Output: pressure [0, 1]

    def forward(self, strokes):
        # strokes: [batch, seq_len, 3]
        h1, _ = self.lstm1(strokes)
        h2, _ = self.lstm2(h1)
        pressure = torch.sigmoid(self.fc(h2))  # [batch, seq_len, 1]
        return pressure
```

**Training Data Format:**
```python
# Each sample: (strokes, pressure)
strokes: [n_points, 3]  # Δx, Δy, eos
pressure: [n_points, 1]  # 0.0 to 1.0 (pen pressure)
```

**Data Acquisition Options:**
1. Wacom tablet data collection (handwriting-model's approach)
2. Synthetic pressure from stroke velocity (faster = lighter)
3. Heuristic pressure from curvature (curves = heavier)

### Peephole LSTM Implementation

**Key Difference from Standard LSTM:**

```python
# Standard LSTM
f_t = σ(W_f [h_{t-1}, x_t] + b_f)

# Peephole LSTM (cell state influences gates directly)
f_t = σ(W_f [h_{t-1}, x_t] + W_{cf} c_{t-1} + b_f)
i_t = σ(W_i [h_{t-1}, x_t] + W_{ci} c_{t-1} + b_i)
o_t = σ(W_o [h_{t-1}, x_t] + W_{co} c_t + b_o)
```

**TensorFlow Implementation:**
```python
# TensorFlow doesn't have built-in peephole LSTM
# Need custom implementation or use tf.keras.layers.LSTMCell with modifications
```

**Benefits:**
- Better gradient flow (cell state → gates)
- May learn longer dependencies
- Widely used in modern RNN research

---

## Conclusion

Scribe is already a top-tier handwriting synthesis implementation. The three reference repositories offer unique features that could enhance Scribe's capabilities:

**Top Priorities:**
1. **Pressure model** (handwriting-model) - Unique differentiator
2. **ONNX export** (pytorch-toolkit) - Enables web/mobile deployment
3. **Attention visualization** (pytorch-toolkit) - Great debugging/demo tool

**Secondary:**
4. **Web UI** (Handwriting-synthesis) - Community engagement
5. **Peephole LSTM** (multiple repos) - Potential quality improvement

These repositories should be retained as valuable references for ongoing Scribe development. The features identified here represent clear enhancement opportunities without compromising Scribe's current strengths.

---

**Document Version:** 1.0
**Last Updated:** November 3, 2025
**Analysis Performed By:** Claude Code (Sonnet 4.5)
