#!/usr/bin/env bash
# Build a standalone executable into dist/ (must be run on the target OS).
# Pass --onedir to build a fast-startup directory bundle instead of a single file.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -d .venv ] || ./setup.sh
.venv/bin/pip install pyinstaller
if [ "${1:-}" = "--onedir" ]; then
    AGYMOLDEN_ONEDIR=1 .venv/bin/pyinstaller --noconfirm agymolden.spec
    echo "Directory bundle: dist/agymolden/"
else
    .venv/bin/pyinstaller --noconfirm agymolden.spec
    echo "Executable: dist/agymolden"
fi
