#!/usr/bin/env bash
# ====================================================================
# AgyMolden One-Click Launcher for macOS and Linux
# ====================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# 1. If double-clicked in GUI without an interactive terminal, spawn one so progress is visible
if [ ! -t 0 ] && [ -z "$AGY_SPAWNED_TERM" ]; then
    export AGY_SPAWNED_TERM=1
    if command -v gnome-terminal &>/dev/null; then
        gnome-terminal -- bash "$DIR/Run_Mac_or_Linux.sh" "$@"
        exit 0
    elif command -v x-terminal-emulator &>/dev/null; then
        x-terminal-emulator -e bash "$DIR/Run_Mac_or_Linux.sh" "$@"
        exit 0
    elif command -v konsole &>/dev/null; then
        konsole -e bash "$DIR/Run_Mac_or_Linux.sh" "$@"
        exit 0
    elif command -v xfce4-terminal &>/dev/null; then
        xfce4-terminal -e "bash '$DIR/Run_Mac_or_Linux.sh' $@"
        exit 0
    elif command -v xterm &>/dev/null; then
        xterm -e bash "$DIR/Run_Mac_or_Linux.sh" "$@"
        exit 0
    elif [ "$(uname)" = "Darwin" ]; then
        osascript -e "tell application \"Terminal\" to do script \"bash '$DIR/Run_Mac_or_Linux.sh'\""
        exit 0
    fi
fi

echo "========================================================"
echo "          AgyMolden Launcher (macOS / Linux)"
echo "========================================================"

# 2. Detect Python 3
PY_CMD=""
for cand in python3 python python3.14 python3.13 python3.12 python3.11 python3.10 python3.9; do
    if command -v "$cand" &>/dev/null; then
        # Check if version is 3.x
        VER=$("$cand" -c "import sys; print(sys.version_info[0])" 2>/dev/null || true)
        if [ "$VER" = "3" ]; then
            PY_CMD="$cand"
            break
        fi
    fi
done

if [ -z "$PY_CMD" ]; then
    echo "[ERROR] Python 3 was not found on your system."
    echo "Please install Python 3 using your package manager (e.g. sudo apt install python3)"
    echo "or download it from https://www.python.org/"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Found Python: $("$PY_CMD" --version)"

# 3. Find newest .whl file in this directory
WHL_FILE=$(ls -t *.whl 2>/dev/null | head -n 1)
if [ -z "$WHL_FILE" ]; then
    echo "[ERROR] No .whl file found in directory: $DIR"
    read -p "Press Enter to exit..."
    exit 1
fi
echo "Using package: $WHL_FILE"

# 4. Check / Create local virtual environment
NEED_INSTALL=0
if [ ! -d "agymolden_env" ] || [ ! -f "agymolden_env/bin/agymolden" ]; then
    NEED_INSTALL=1
    echo ""
    echo "[1/2] Setting up local isolated environment in ./agymolden_env..."
    rm -rf agymolden_env
    if ! "$PY_CMD" -m venv agymolden_env 2>/dev/null; then
        echo "[WARNING] '$PY_CMD -m venv' failed (python3-venv might not be installed)."
        echo "Attempting fallback using virtualenv..."
        if command -v virtualenv &>/dev/null; then
            virtualenv -p "$PY_CMD" agymolden_env
        else
            echo "[ERROR] Could not create virtual environment."
            echo "On Debian/Ubuntu, please run: sudo apt install python3-venv"
            read -p "Press Enter to exit..."
            exit 1
        fi
    fi
fi

source agymolden_env/bin/activate

if [ "$NEED_INSTALL" -eq 1 ]; then
    echo "[2/2] Installing dependencies for AgyMolden (PyQt6, NumPy, SciPy, Matplotlib)..."
    echo "This one-time installation takes ~30-60 seconds depending on your connection."
    pip install --upgrade pip >/dev/null 2>&1 || true
    if ! pip install "$WHL_FILE"; then
        echo ""
        echo "[ERROR] Failed to install $WHL_FILE."
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "Installation complete!"
else
    # Update wheel if newer wheel was placed in directory
    pip install --no-deps --upgrade "$WHL_FILE" >/dev/null 2>&1 || true
fi

echo ""
echo "🚀 Starting AgyMolden..."
agymolden "$@"
RET=$?

if [ $RET -ne 0 ]; then
    echo ""
    echo "[INFO] AgyMolden exited with code $RET."
    read -p "Press Enter to close this window..."
fi
exit $RET
