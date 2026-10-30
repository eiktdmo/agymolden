#!/usr/bin/env python3
"""Launch AgyMolden from source:  ./run_agymolden.py [file]

Automatically uses the local .venv or ~/.venv-agymoldenshared if present.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def _find_venv_python():
    sub = os.path.join("Scripts", "python.exe") if os.name == "nt" else os.path.join("bin", "python")
    candidates = [
        os.path.join(ROOT, ".venv", sub),
        os.path.expanduser(os.path.join("~", ".venv-agymoldenshared", sub)),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


if __name__ == "__main__":
    venv_py = _find_venv_python()
    if (
        venv_py
        and os.path.abspath(sys.executable) != os.path.abspath(venv_py)
        and not sys.prefix.startswith(os.path.join(ROOT, ".venv"))
        and ".venv-agymoldenshared" not in sys.prefix
    ):
        sys.exit(subprocess.call([venv_py, os.path.abspath(__file__)] + sys.argv[1:]))
    sys.path.insert(0, ROOT)
    from agymolden import main
    main()
