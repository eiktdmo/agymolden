#!/usr/bin/env bash
# ==============================================================================
# AgyMolden Automated 1-Command Installer (Linux / macOS — no conda required)
# ==============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(pwd)"

# Keep installation completely isolated from any active conda / venv session
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE VIRTUAL_ENV PYTHONHOME PYTHONPATH || true

echo "=========================================================="
echo "         AgyMolden Automatic Source Installer             "
echo "=========================================================="
echo

# 1. Discover a clean Python 3.9+ interpreter (preferring system/Homebrew over conda)
PYTHON_BIN=""
CANDIDATES=()
if [ "$(uname -s)" = "Darwin" ]; then
    CANDIDATES+=(
        "/opt/homebrew/bin/python3"
        "/usr/local/bin/python3"
        "/Library/Frameworks/Python.framework/Versions/Current/bin/python3"
    )
else
    CANDIDATES+=("/usr/bin/python3" "/usr/local/bin/python3")
fi
CANDIDATES+=("python3" "python")

for cmd in "${CANDIDATES[@]}"; do
    resolved="$(command -v "$cmd" 2>/dev/null || true)"
    [ -n "$resolved" ] || continue
    case "$resolved" in
        *conda*|*anaconda*|*miniconda*|*mamba*) continue ;;
    esac
    if "$resolved" -c "import sys, venv; assert sys.version_info >= (3, 9)" >/dev/null 2>&1; then
        ver="$("$resolved" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
        PYTHON_BIN="$resolved"
        echo "✔ Found Python $ver: $PYTHON_BIN"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Error: Python 3.9+ (with venv support) is required."
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "Install via Homebrew:  brew install python"
        echo "Or download from:      https://www.python.org/downloads/macos/"
    else
        echo "Install on Ubuntu/Debian:  sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    fi
    exit 1
fi

# 2. Check recommended Linux OpenGL / Qt system libraries (Debian/Ubuntu)
if [ "$(uname -s)" = "Linux" ] && command -v apt-get >/dev/null 2>&1; then
    MISSING_PKGS=""
    for lib in libgl1 libglx-mesa0 libegl1 libxkbcommon-x11-0 libxcb-cursor0; do
        if ! dpkg -s "$lib" >/dev/null 2>&1; then
            MISSING_PKGS="$MISSING_PKGS $lib"
        fi
    done
    if [ -n "$MISSING_PKGS" ]; then
        echo "Notice: Missing recommended OpenGL/Qt system packages:$MISSING_PKGS"
        if [ -t 0 ] && [ "${EUID:-$(id -u)}" -ne 0 ]; then
            read -p "Install them now with sudo apt? [Y/n] " -n 1 -r
            echo
            if [[ ${REPLY:-Y} =~ ^[Yy]$ ]] || [ -z "${REPLY:-}" ]; then
                sudo apt-get update && sudo apt-get install -y $MISSING_PKGS || echo "Continuing without sudo packages..."
            fi
        fi
    fi
fi

# 3. Create or reuse local .venv (never touches global or conda Python)
echo
echo "Setting up isolated virtual environment in .venv..."
if command -v uv >/dev/null 2>&1; then
    echo "⚡ Detected 'uv' — using fast installation path!"
    [ -d ".venv" ] || uv venv .venv --python "$PYTHON_BIN"
    echo "Installing core AgyMolden dependencies..."
    uv pip install --python .venv/bin/python -e .
    echo "Installing optional chemistry extras (best-effort)..."
    for pkg in basis_set_exchange rdkit spglib ase pyscf; do
        uv pip install --python .venv/bin/python "$pkg" >/dev/null 2>&1 && echo "  ✔ $pkg" || echo "  • $pkg skipped (optional)"
    done
else
    if [ ! -d ".venv" ]; then
        "$PYTHON_BIN" -m venv .venv || {
            echo "❌ Failed to create .venv. On Ubuntu/Debian run: sudo apt install -y python3-venv"
            exit 1
        }
    fi
    chmod +x .venv/bin/pip .venv/bin/pip3 2>/dev/null || true
    .venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1 || true

    echo "Installing core AgyMolden dependencies..."
    .venv/bin/python -m pip install -e .

    echo "Installing optional chemistry extras (best-effort)..."
    for pkg in basis_set_exchange rdkit spglib ase pyscf; do
        .venv/bin/python -m pip install "$pkg" >/dev/null 2>&1 && echo "  ✔ $pkg" || echo "  • $pkg skipped (optional)"
    done
fi

# Optional standalone build flag (--build-standalone)
if [ "${1:-}" = "--build-standalone" ]; then
    echo
    echo "Building standalone executable via PyInstaller..."
    .venv/bin/python -m pip install pyinstaller
    .venv/bin/pyinstaller agymolden.spec --noconfirm --clean
fi

# 4. Verify installation
echo
.venv/bin/python -c "import agymolden, PyQt6, numpy, scipy, matplotlib; print('✔ Verified AgyMolden', agymolden.__version__)"

# 5. Create launcher symlinks and Linux desktop entry
chmod +x run.py
mkdir -p "$HOME/.local/bin"
if [ -x "$PROJECT_DIR/.venv/bin/agymolden" ]; then
    ln -sf "$PROJECT_DIR/.venv/bin/agymolden" "$HOME/.local/bin/agymolden"
    echo "✔ Linked command: $HOME/.local/bin/agymolden"
    case ":$PATH:" in
        *":$HOME/.local/bin:"*) ;;
        *) echo "  Note: Add '$HOME/.local/bin' to your PATH to run 'agymolden' from any folder." ;;
    esac
fi

if [ "$(uname -s)" = "Linux" ] && [ -f "./make_desktop_entry.sh" ]; then
    echo "Creating Linux desktop/application menu entry..."
    ./make_desktop_entry.sh || true
fi

echo
echo "=========================================================="
echo "✔ Installation completed successfully!"
echo
echo "Launch AgyMolden anytime with:"
echo "   ./run.py          (from this folder)"
echo "   agymolden         (from anywhere if ~/.local/bin is in PATH)"
echo "=========================================================="
