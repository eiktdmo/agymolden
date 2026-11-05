#!/usr/bin/env bash
# ====================================================================
# AgyMolden One-Click Launcher for macOS and Linux
# ====================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "========================================================"
echo "          AgyMolden Launcher (macOS / Linux)"
echo "========================================================"

# 1. Detect Python 3
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "[ERROR] Python 3 was not found on your system."
    echo "Please install Python from https://www.python.org/ or via brew / your package manager."
    read -p "Press Enter to exit..."
    exit 1
fi

# 2. Find newest .whl file
WHL_FILE=$(ls -t *.whl 2>/dev/null | head -n 1)
if [ -z "$WHL_FILE" ]; then
    echo "[ERROR] No .whl file found in this folder!"
    read -p "Press Enter to exit..."
    exit 1
fi

# 3. Create isolated virtualenv
if [ ! -d "agymolden_env" ]; then
    echo "[1/2] Setting up local isolated environment (one-time setup, ~30s)..."
    "$PY_CMD" -m venv agymolden_env
    source agymolden_env/bin/activate
    pip install --upgrade pip >/dev/null 2>&1
    echo "[2/2] Installing $WHL_FILE..."
    pip install "$WHL_FILE"
else
    source agymolden_env/bin/activate
    pip install --upgrade --force-reinstall --no-deps "$WHL_FILE" >/dev/null 2>&1
fi

echo "Starting AgyMolden..."
agymolden "$@"
exit 0
