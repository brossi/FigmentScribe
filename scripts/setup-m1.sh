#!/bin/bash
#
# M1 Mac (Apple Silicon) Setup Script for Scribe
# Automates installation of Python dependencies and verification
#
# Usage:
#   chmod +x scripts/setup-m1.sh
#   ./scripts/setup-m1.sh
#
# Requirements:
#   - Python 3.11+ installed
#   - pip installed
#   - Xcode Command Line Tools installed
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory (parent of scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Scribe M1 Mac Setup Script${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 11 ]; then
    echo -e "${RED}ERROR: Python 3.11+ required, found $PYTHON_VERSION${NC}"
    echo "Install Python 3.11+ from https://www.python.org/downloads/"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
echo ""

# Check if virtual environment already exists
VENV_PATH="$PROJECT_ROOT/venv-m1"
if [ -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Virtual environment already exists at: $VENV_PATH${NC}"
    read -p "Delete and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Deleting existing virtual environment...${NC}"
        rm -rf "$VENV_PATH"
    else
        echo -e "${YELLOW}Using existing virtual environment${NC}"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Creating virtual environment at: $VENV_PATH${NC}"
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded to $(pip --version | awk '{print $2}')${NC}"
echo ""

# Check if requirements-m1.txt exists
if [ ! -f "$PROJECT_ROOT/requirements-m1.txt" ]; then
    echo -e "${RED}ERROR: requirements-m1.txt not found${NC}"
    echo "Expected location: $PROJECT_ROOT/requirements-m1.txt"
    exit 1
fi

# Install M1-specific dependencies
echo -e "${YELLOW}Installing TensorFlow for M1...${NC}"
pip install tensorflow-macos==2.15.0 --quiet
echo -e "${GREEN}✓ tensorflow-macos installed${NC}"

echo -e "${YELLOW}Installing Metal GPU acceleration...${NC}"
pip install tensorflow-metal==1.1.0 --quiet
echo -e "${GREEN}✓ tensorflow-metal installed${NC}"
echo ""

echo -e "${YELLOW}Installing NumPy (compatible version)...${NC}"
pip install "numpy>=1.26.4,<2.0" --quiet
echo -e "${GREEN}✓ NumPy installed${NC}"
echo ""

echo -e "${YELLOW}Installing remaining dependencies...${NC}"
pip install matplotlib==3.8.3 scipy==1.12.0 svgwrite==1.4.3 jupyter==1.0.0 --quiet
echo -e "${GREEN}✓ All production dependencies installed${NC}"
echo ""

# Ask about test dependencies
read -p "Install test dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "$PROJECT_ROOT/requirements-test.txt" ]; then
        echo -e "${YELLOW}Installing test dependencies...${NC}"
        pip install -r requirements-test.txt --quiet
        echo -e "${GREEN}✓ Test dependencies installed${NC}"
    else
        echo -e "${YELLOW}requirements-test.txt not found, skipping${NC}"
    fi
fi
echo ""

# Verification
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Verification${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Check TensorFlow import
echo -e "${YELLOW}Checking TensorFlow installation...${NC}"
if python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" 2>/dev/null; then
    echo -e "${GREEN}✓ TensorFlow imports successfully${NC}"
else
    echo -e "${RED}✗ TensorFlow import failed${NC}"
    echo "Run: python3 -c 'import tensorflow as tf; print(tf.__version__)' for details"
fi
echo ""

# Check NumPy version
echo -e "${YELLOW}Checking NumPy version...${NC}"
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "ERROR")
if [[ $NUMPY_VERSION == 1.* ]]; then
    echo -e "${GREEN}✓ NumPy version: $NUMPY_VERSION (compatible)${NC}"
elif [[ $NUMPY_VERSION == 2.* ]]; then
    echo -e "${RED}✗ NumPy version: $NUMPY_VERSION (INCOMPATIBLE with TensorFlow 2.15)${NC}"
    echo "Run: pip install \"numpy>=1.26.4,<2.0\""
else
    echo -e "${RED}✗ NumPy import failed${NC}"
fi
echo ""

# Check Metal GPU
echo -e "${YELLOW}Checking Metal GPU availability...${NC}"
GPU_CHECK=$(python3 -c "import tensorflow as tf; devices = tf.config.list_physical_devices('GPU'); print(len(devices))" 2>/dev/null || echo "ERROR")
if [[ $GPU_CHECK == "ERROR" ]]; then
    echo -e "${RED}✗ GPU check failed${NC}"
elif [ "$GPU_CHECK" -gt 0 ]; then
    echo -e "${GREEN}✓ Metal GPU detected ($GPU_CHECK device(s))${NC}"
else
    echo -e "${YELLOW}⚠ No GPU detected (CPU-only mode)${NC}"
    echo "This is normal for testing, but training will be slower"
fi
echo ""

# Verify data
if [ -f "$PROJECT_ROOT/verify_data.py" ]; then
    echo -e "${YELLOW}Verifying training data...${NC}"
    if python3 verify_data.py 2>&1 | grep -q "SUCCESS"; then
        echo -e "${GREEN}✓ Training data verified (11,916 samples)${NC}"
    else
        echo -e "${RED}✗ Data verification failed${NC}"
        echo "Run: python3 verify_data.py for details"
    fi
else
    echo -e "${YELLOW}verify_data.py not found, skipping data verification${NC}"
fi
echo ""

# Run smoke tests (if pytest installed)
if pip show pytest >/dev/null 2>&1; then
    echo -e "${YELLOW}Running smoke tests...${NC}"
    if pytest -m smoke --quiet --tb=no 2>/dev/null; then
        echo -e "${GREEN}✓ All smoke tests passed${NC}"
    else
        echo -e "${RED}✗ Some smoke tests failed${NC}"
        echo "Run: pytest -m smoke -v for details"
    fi
else
    echo -e "${YELLOW}pytest not installed, skipping smoke tests${NC}"
fi
echo ""

# Summary
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Setup Complete!${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "${GREEN}Virtual environment:${NC} $VENV_PATH"
echo -e "${GREEN}Activate with:${NC} source venv-m1/bin/activate"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Activate virtual environment: source venv-m1/bin/activate"
echo "  2. Verify data: python3 verify_data.py"
echo "  3. Run smoke tests: pytest -m smoke"
echo "  4. Generate sample: python3 sample.py --text \"Hello M1!\""
echo ""
echo -e "${YELLOW}For detailed documentation:${NC}"
echo "  - M1 Setup Guide: docs/M1_SETUP.md"
echo "  - Project Overview: CLAUDE.md"
echo "  - Training on Colab: COLAB_TRAINING.ipynb"
echo ""
echo -e "${GREEN}Happy handwriting synthesis! ✍️${NC}"
