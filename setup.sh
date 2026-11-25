#!/usr/bin/env bash
# ==============================================================================
# AgyMolden Automated Source Installer (Linux / macOS)
# ==============================================================================
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(pwd)"

echo "=========================================================="
echo "         AgyMolden Automatic Source Installer             "
echo "=========================================================="
echo

# 1. Check for Python 3
PYTHON_BIN=""
for cmd in /usr/bin/python3 /usr/local/bin/python3 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null || true)
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || true)
        if [ "$major" = "3" ] && [ "$minor" -ge 9 ]; then
            PYTHON_BIN="$cmd"
            echo "✔ Found compatible Python: $cmd (version $ver)"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Error: Python 3.9 or newer is required."
    echo "Please install Python 3.9+ using your package manager (e.g., sudo apt install python3 python3-venv python3-pip)."
    exit 1
fi

# 2. Check & offer to install Linux system libraries if on Debian/Ubuntu
if [ "$(uname -s)" = "Linux" ] && command -v apt-get >/dev/null 2>&1; then
    MISSING_PKGS=""
    for lib in libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0; do
        if ! dpkg -s "$lib" >/dev/null 2>&1; then
            MISSING_PKGS="$MISSING_PKGS $lib"
        fi
    done
    if [ -n "$MISSING_PKGS" ]; then
        echo "Notice: Missing recommended system OpenGL/Qt packages:$MISSING_PKGS"
        if [ -t 0 ] && [ "$EUID" -ne 0 ]; then
            read -p "Would you like to install them with sudo apt? [Y/n] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]] || [ -z "$REPLY" ]; then
                sudo apt-get update && sudo apt-get install -y $MISSING_PKGS || echo "Continuing without sudo packages..."
            fi
        fi
    fi
fi

# 3. Create or reuse virtual environment (prefers ultra-fast 'uv' if available)
echo
echo "Setting up Python virtual environment in .venv..."
if command -v uv >/dev/null 2>&1; then
    echo "⚡ Detected 'uv' package manager — using fast installation path!"
    if [ ! -d ".venv" ]; then
        uv venv .venv --python "$PYTHON_BIN"
    fi
    uv pip install --python .venv/bin/python -e ".[all]" || uv pip install --python .venv/bin/python -e .
else
    if [ ! -d ".venv" ]; then
        "$PYTHON_BIN" -m venv .venv || {
            echo "❌ Failed to create .venv. Try: sudo apt install python3-venv"
            exit 1
        }
    fi

    # 4. Install / upgrade pip and package with full source support
    echo "Installing dependencies..."
    chmod +x .venv/bin/pip .venv/bin/pip3 2>/dev/null || true
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -e ".[all,qm]" || .venv/bin/python -m pip install -e ".[all]" || .venv/bin/python -m pip install -e .
fi

# Optional standalone build flag (--build-standalone)
if [ "${1:-}" = "--build-standalone" ]; then
    echo
    echo "Building standalone executable via PyInstaller..."
    .venv/bin/python -m pip install pyinstaller
    .venv/bin/pyinstaller agymolden.spec --noconfirm --clean
fi

# 5. Create launcher symlinks and desktop entry if on Linux
chmod +x run.py

mkdir -p "$HOME/.local/bin"
if [ -x "$PROJECT_DIR/.venv/bin/agymolden" ]; then
    ln -sf "$PROJECT_DIR/.venv/bin/agymolden" "$HOME/.local/bin/agymolden"
    echo "✔ Linked global command: $HOME/.local/bin/agymolden"
    case ":$PATH:" in
        *":$HOME/.local/bin:"*) ;;
        *) echo "  Note: Add '$HOME/.local/bin' to your PATH to run 'agymolden' from anywhere." ;;
    esac
fi

if [ "$(uname -s)" = "Linux" ]; then
    if [ -f "./make_desktop_entry.sh" ]; then
        echo
        echo "Creating Linux desktop shortcut and application menu entry..."
        ./make_desktop_entry.sh || true
    fi
fi

echo
echo "=========================================================="
echo "✔ Installation completed successfully!"
echo
echo "You can launch AgyMolden at any time with:"
echo "   ./run.py               (from this directory)"
echo "or activate the environment:"
echo "   source .venv/bin/activate && agymolden"
echo "=========================================================="
