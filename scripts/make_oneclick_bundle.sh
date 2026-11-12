#!/usr/bin/env bash
# ====================================================================
# Package AgyMolden into wheel and One-Click Bundle
# ====================================================================
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Detect python interpreter
CANDIDATE_PYTHONS=(
    "$VIRTUAL_ENV/bin/python"
    ".venv/bin/python"
    "venv/bin/python"
    "$(command -v python3 2>/dev/null || true)"
    "$(command -v python 2>/dev/null || true)"
)

PYTHON_BIN=""
for py in "${CANDIDATE_PYTHONS[@]}"; do
    if [ -n "$py" ] && [ -x "$py" ]; then
        PYTHON_BIN="$py"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] No usable Python interpreter found!"
    exit 1
fi

echo "==> Building AgyMolden wheel using $PYTHON_BIN..."
mkdir -p dist
if "$PYTHON_BIN" -m build --version &>/dev/null; then
    "$PYTHON_BIN" -m build --wheel --outdir dist/
else
    "$PYTHON_BIN" -m pip wheel --no-deps . -w dist/
fi

VERSION=$("$PYTHON_BIN" -c "import agymolden; print(agymolden.__version__)")
WHL="dist/agymolden-${VERSION}-py3-none-any.whl"

if [ ! -f "$WHL" ]; then
    echo "[ERROR] Expected wheel $WHL was not produced!"
    exit 1
fi

echo "==> Built: $WHL"

# Recreate AgyMolden_OneClick_Bundle
BUNDLE_DIR="dist/AgyMolden_OneClick_Bundle"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

cp bundle_template/README.txt "$BUNDLE_DIR/"
cp bundle_template/Run_Mac_or_Linux.sh "$BUNDLE_DIR/"
cp bundle_template/Run_Windows.bat "$BUNDLE_DIR/"
chmod +x "$BUNDLE_DIR/Run_Mac_or_Linux.sh"
cp "$WHL" "$BUNDLE_DIR/"

echo "==> Packaging AgyMolden_OneClick_Bundle.zip..."
rm -f AgyMolden_OneClick_Bundle.zip dist/AgyMolden_OneClick_Bundle.zip
(cd dist && zip -r ../AgyMolden_OneClick_Bundle.zip AgyMolden_OneClick_Bundle)
cp -f AgyMolden_OneClick_Bundle.zip dist/AgyMolden_OneClick_Bundle.zip

# If parent Desktop directory exists, keep its zip and unzipped bundle synchronized
if [ -d "/home/andres/Desktop" ]; then
    cp -f AgyMolden_OneClick_Bundle.zip /home/andres/Desktop/AgyMolden_OneClick_Bundle.zip
    unzip -qo AgyMolden_OneClick_Bundle.zip -d /home/andres/Desktop/
    find /home/andres/Desktop/AgyMolden_OneClick_Bundle -name "agymolden-*.whl" ! -name "agymolden-${VERSION}-py3-none-any.whl" -delete 2>/dev/null || true
fi

echo "==> Successfully created and synced AgyMolden_OneClick_Bundle.zip with version ${VERSION}!"
