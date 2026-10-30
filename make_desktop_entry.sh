#!/usr/bin/env bash
# Create an application-menu launcher for this copy of AgyMolden (Linux).
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${XDG_DATA_HOME:-$HOME/.local/share}/applications/agymolden.desktop"
mkdir -p "$(dirname "$OUT")"
cat > "$OUT" <<DESK
[Desktop Entry]
Version=1.0
Type=Application
Name=AgyMolden Z-Matrix Builder
Comment=Molden-inspired 3D Molecular Z-Matrix Editor
Exec=$DIR/run.py %f
Path=$DIR
Terminal=false
Categories=Science;Chemistry;Education;
DESK
chmod +x "$DIR/run.py"
echo "Created $OUT"
