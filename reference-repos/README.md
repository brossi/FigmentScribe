# Reference Repositories

This directory contains three carefully selected handwriting synthesis repositories kept as references for potential features and improvements to Scribe.

**Selection Criteria:** 5 repositories were evaluated, and 2 were removed as redundant/downgrades. These 3 represent the best implementations with unique features not present in Scribe.

**Detailed Analysis:** See `docs/REFERENCE_REPOS_ANALYSIS.md` for comprehensive comparison, technical details, and implementation roadmap.

---

## Repository Guide

### 1. handwriting-model/ ⭐ TOP PRIORITY

**Framework:** PyTorch
**Last Updated:** August 2024 (actively maintained)
**Author:** Custom implementation with major extensions

**Why We Keep It:**
- **Stroke thickness/pressure model** - COMPLETELY UNIQUE feature that predicts pen pressure
- Trained on Wacom tablet pressure data for realistic stroke width variation
- Peephole LSTM architecture (more sophisticated than standard LSTM)
- Comprehensive test suite and professional code quality
- Custom data collection system for Wacom tablets

**Key Unique Feature:**
Second ML layer that adds realistic stroke thickness by predicting pen pressure from generated coordinates. No other handwriting synthesis repository has this.

**Files to Study:**
- `pressure_model/LTSM_model.py` - Pressure prediction architecture
- `handwriting_model/modules.py` - Peephole LSTM implementation
- `tests/` - Comprehensive testing patterns

**Potential Ports:**
1. Pressure/thickness model (HIGH - unique differentiator)
2. Peephole LSTM architecture (MEDIUM - quality improvement)
3. Testing infrastructure patterns (MEDIUM - code quality)

---

### 2. pytorch-handwriting-synthesis-toolkit/ ⭐ HIGH PRIORITY

**Framework:** PyTorch with ONNX export
**Last Updated:** April 2023 (stable, community maintained)
**Author:** X-rayLaser (professional implementation)

**Why We Keep It:**
- **ONNX export support** - Deploy to web browsers, mobile devices, embedded systems
- **Attention visualization** - Heatmaps showing model focus on characters
- txt2page multi-line document generation with automatic layout
- Best documentation of all repos (506-line README)
- Professional software engineering and comprehensive tests

**Key Unique Features:**
1. ONNX export enables JavaScript inference (ONNX.js) for web deployment
2. Attention heatmaps for debugging and demonstrations

**Files to Study:**
- `export_to_onnx.py` - ONNX conversion implementation
- `generate_samples.py` - Attention visualization code
- `txt2script.py` - Multi-page document generation
- `README.md` - Excellent documentation reference

**Potential Ports:**
1. ONNX export for web deployment (HIGH - dramatically expands reach)
2. Attention visualization (HIGH - debugging + demos)
3. Document layout system (MEDIUM - enhanced multi-line)
4. Custom data provider architecture (MEDIUM - flexibility)

---

### 3. Handwriting-synthesis/ 🟡 MEDIUM PRIORITY

**Framework:** PyTorch with Flask web application
**Last Updated:** December 2020 (unmaintained but functional)
**Author:** swechhachoudhary (fork with web UI)

**Why We Keep It:**
- **Full web application** with interactive drawing interface
- **Custom style upload** - Users draw handwriting, system generates in that style
- Real-time generation via web UI
- Mobile device support (touch input)
- Complete user-facing product (not just research code)

**Key Unique Feature:**
Flask web app where users can draw their own handwriting on a canvas and have the model generate text in that custom style. Dynamic style extraction from user input.

**Files to Study:**
- `app.py` - Flask web application structure
- `static/` and `templates/` - Frontend drawing interface
- SVG path parsing utilities for canvas input
- Style extraction from user drawings

**Potential Ports:**
1. Web UI concept (MEDIUM - community engagement)
2. Custom style upload from user drawings (MEDIUM - flexible priming)
3. SVG path parsing from web canvas (LOW-MEDIUM)
4. Mobile/touch input handling (LOW)

**Note:** Unmaintained since 2020, keep primarily for web UI inspiration.

---

## Quick Feature Comparison

| Feature | Scribe | handwriting-model | pytorch-toolkit | Handwriting-synthesis |
|---------|--------|-------------------|-----------------|----------------------|
| **Pressure Model** | ❌ | ✅ ⭐ | ❌ | ❌ |
| **ONNX Export** | ❌ | ❌ | ✅ ⭐ | ❌ |
| **Attention Viz** | ❌ | ❌ | ✅ ⭐ | ❌ |
| **Web UI** | ❌ | ❌ | ❌ | ✅ ⭐ |
| **Peephole LSTM** | ❌ | ✅ | ✅ | ❌ |
| **Style Priming** | ✅ (13 styles) | ❌ | ❌ | ✅ (custom) |
| **Multi-line** | ✅ | ❌ | ✅ | ❌ |
| **SVG Output** | ✅ | ❌ | ✅ | ✅ |
| **Per-line Control** | ✅ | ❌ | ❌ | ❌ |
| **Test Suite** | ✅ | ✅ | ✅ | ❌ |
| **Documentation** | Excellent | Good | Excellent | Minimal |

**Legend:** ⭐ = Completely unique feature not available elsewhere

---

## Recommended Implementation Priority

Based on impact and feasibility analysis:

### Tier 1: High Impact
1. **Stroke thickness/pressure model** (handwriting-model)
2. **ONNX export** (pytorch-toolkit)
3. **Attention visualization** (pytorch-toolkit)

### Tier 2: Medium Impact
4. **Web UI demo** (Handwriting-synthesis)
5. **Peephole LSTM** (multiple repos)
6. **Enhanced testing** (multiple repos)

### Tier 3: Lower Priority
7. Custom data provider system
8. txt2page document generation
9. Analysis notebooks

---

## Usage Notes

**These are reference repositories only - DO NOT modify them directly.**

- Keep them as-is for reference and learning
- Study implementations before porting features to Scribe
- Respect original licenses (check each repo's LICENSE file)
- Credit original authors in any ports to Scribe

**For detailed analysis:** Read `docs/REFERENCE_REPOS_ANALYSIS.md`

---

## Maintenance

**Last Reviewed:** November 3, 2025
**Next Review:** When considering new features for Scribe

**If adding new reference repos:**
1. Evaluate against existing capabilities
2. Identify unique features
3. Document in REFERENCE_REPOS_ANALYSIS.md
4. Update this README

**If removing repos:**
1. Verify no unique value remains
2. Document reason in git commit
3. Update this README

---

**Status:** 3 of 5 evaluated repos retained (2 redundant repos deleted)
**Evaluation Complete:** ✅ November 3, 2025
