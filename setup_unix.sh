#!/usr/bin/env bash
# ====================================================================
#  AgyMolden Linux / macOS Setup & Standalone Build Script
# ====================================================================

set -euo pipefail

echo "========================================================"
echo "    AgyMolden - Unix / Mac Setup & Executable Builder"
echo "========================================================"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH."
    exit 1
fi

echo "[1/4] Creating virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

echo "[2/4] Activating environment and installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]" pyinstaller

echo "[3/4] Compiling standalone executable using PyInstaller..."
pyinstaller agymolden.spec --noconfirm --clean

echo "========================================================"
if [ -f "dist/agymolden" ]; then
    echo "[SUCCESS] Standalone binary created at dist/agymolden"
    chmod +x dist/agymolden
elif [ -d "dist/agymolden.app" ]; then
    echo "[SUCCESS] macOS App bundle created at dist/agymolden.app"
else
    echo "[INFO] Check dist/ directory for the built application."
fi
echo "========================================================"
