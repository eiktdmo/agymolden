#!/usr/bin/env bash
# ====================================================================
# AgyMolden Native AppImage Builder (Linux x86_64)
# ====================================================================
# Generates a portable, distribution-agnostic AppImage bundle with:
# - Self-contained PyInstaller frozen binary
# - Bundled desktop integration file (agymolden.desktop)
# - Standard application icon (agymolden.png)
# - AppRun execution wrapper with OpenGL Mesa LIBGL fallback
# ====================================================================
set -euo pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$ROOT_DIR"

APP_NAME="AgyMolden"
APP_DIR="build/AppDir"
DIST_DIR="dist"
ARCH="x86_64"

echo "==> [1/4] Preparing AppDir structure..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/usr/bin"
mkdir -p "$APP_DIR/usr/lib"
mkdir -p "$APP_DIR/usr/share/applications"
mkdir -p "$APP_DIR/usr/share/icons/hicolor/256x256/apps"

# 1. Desktop Entry
cat << 'EOF' > "$APP_DIR/agymolden.desktop"
[Desktop Entry]
Type=Application
Name=AgyMolden
GenericName=Quantum Chemistry Workbench & Molecular Builder
Comment=Molden-inspired 3D Z-Matrix generator and computational chemistry suite
Exec=agymolden %F
Icon=agymolden
Categories=Science;Chemistry;Education;Physics;
Terminal=false
StartupNotify=true
MimeType=chemical/x-xyz;chemical/x-pdb;chemical/x-gaussian-input;chemical/x-gaussian-log;
EOF
cp "$APP_DIR/agymolden.desktop" "$APP_DIR/usr/share/applications/"

# 2. Icon (fallback dummy/asset if missing)
ICON_SRC="agymolden/resources/icon.png"
if [ ! -f "$ICON_SRC" ]; then
    ICON_SRC="$APP_DIR/agymolden.png"
    # Create 1x1 transparent PNG if no icon exists
    echo -n -e '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82' > "$ICON_SRC"
fi
cp "$ICON_SRC" "$APP_DIR/agymolden.png"
cp "$ICON_SRC" "$APP_DIR/.DirIcon"
cp "$ICON_SRC" "$APP_DIR/usr/share/icons/hicolor/256x256/apps/agymolden.png"

# 3. AppRun Execution Wrapper
cat << 'EOF' > "$APP_DIR/AppRun"
#!/bin/sh
SELF=$(readlink -f "$0")
HERE=${SELF%/*}

export PATH="${HERE}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${HERE}/usr/bin:${LD_LIBRARY_PATH}"
export XDG_DATA_DIRS="${HERE}/usr/share:${XDG_DATA_DIRS}"

# Check for headless or missing hardware OpenGL, engage llvmpipe Mesa fallback
if [ -z "${LIBGL_ALWAYS_SOFTWARE:-}" ]; then
    if ! glxinfo >/dev/null 2>&1; then
        export LIBGL_ALWAYS_SOFTWARE=1
    fi
fi

exec "${HERE}/usr/bin/agymolden" "$@"
EOF
chmod +x "$APP_DIR/AppRun"

echo "==> [2/4] Verifying binary payload..."
if [ -f "$DIST_DIR/agymolden" ]; then
    cp "$DIST_DIR/agymolden" "$APP_DIR/usr/bin/agymolden"
    chmod +x "$APP_DIR/usr/bin/agymolden"
elif [ -d "$DIST_DIR/agymolden" ]; then
    cp -r "$DIST_DIR/agymolden"/* "$APP_DIR/usr/bin/"
    chmod +x "$APP_DIR/usr/bin/agymolden"
else
    echo "[INFO] No pre-built binary found at $DIST_DIR/agymolden."
    echo "[INFO] AppDir structure generated successfully at $APP_DIR."
    echo "[INFO] Run './build.sh' first when ready to package a real AppImage."
    exit 0
fi

echo "==> [3/4] Downloading appimagetool if not present..."
APPIMAGETOOL="$ROOT_DIR/build/appimagetool-${ARCH}.AppImage"
if [ ! -f "$APPIMAGETOOL" ]; then
    curl -sL "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-${ARCH}.AppImage" -o "$APPIMAGETOOL"
    chmod +x "$APPIMAGETOOL"
fi

echo "==> [4/4] Generating AppImage bundle..."
ARCH="$ARCH" "$APPIMAGETOOL" --appimage-extract-and-run "$APP_DIR" "$DIST_DIR/AgyMolden-${ARCH}.AppImage"
echo "==> Successfully created $DIST_DIR/AgyMolden-${ARCH}.AppImage"
