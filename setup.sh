#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.14"
VENV_DIR=".venv"

echo "==> Creating virtual environment (Python $PYTHON_VERSION)..."
if command -v uv &>/dev/null; then
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
else
    python${PYTHON_VERSION} -m venv "$VENV_DIR"
fi

echo "==> Activating venv..."
source "$VENV_DIR/bin/activate"

echo "==> Installing dependencies..."
if command -v uv &>/dev/null; then
    uv pip install -r requirements.txt
else
    pip install -r requirements.txt
fi

echo "==> Checking ffmpeg..."
if ! command -v ffmpeg &>/dev/null; then
    echo "WARNING: ffmpeg not found. Install it for video/audio processing:"
    echo "  macOS:  brew install ffmpeg"
    echo "  Ubuntu: sudo apt install ffmpeg"
fi

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "For Modal infrastructure, run:"
echo "  bash modal_setup.sh"
