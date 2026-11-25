#!/usr/bin/env python3
"""Launch AgyMolden from source:  python run.py [file]

Uses the local .venv if present, otherwise the current Python.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Fix for Linux (especially with Conda environments):
# Conda ships an older bundled libstdc++.so.6 that lacks recent GLIBCXX symbols required by
# modern Linux Mesa / Intel / AMD / NVIDIA OpenGL drivers (e.g., libLLVM, libEGL_mesa, libglx_mesa).
# If running under Linux with an outdated conda libstdc++, re-exec with system libstdc++ preloaded.
if sys.platform.startswith("linux"):
    if not os.environ.get("_AGYMOLDEN_LIBSTDCPP_CHECKED"):
        os.environ["_AGYMOLDEN_LIBSTDCPP_CHECKED"] = "1"
        for sys_lib in ["/usr/lib/x86_64-linux-gnu/libstdc++.so.6", "/usr/lib64/libstdc++.so.6", "/usr/lib/libstdc++.so.6"]:
            if os.path.isfile(sys_lib):
                current_preload = os.environ.get("LD_PRELOAD", "")
                if sys_lib not in current_preload:
                    # Remember the user's own value so child engines (ORCA etc.) don't inherit our preload
                    os.environ["_AGYMOLDEN_ORIG_LD_PRELOAD"] = current_preload
                    os.environ["LD_PRELOAD"] = f"{sys_lib}:{current_preload}".strip(":")
                    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])
                break


def _venv_python():
    # If the user is currently inside an active virtual environment (e.g. andres-venv), respect it
    if os.environ.get("VIRTUAL_ENV"):
        venv_dir = os.environ["VIRTUAL_ENV"]
        for sub in [os.path.join("bin", "python"), os.path.join("Scripts", "python.exe")]:
            active_py = os.path.join(venv_dir, sub)
            if os.path.isfile(active_py):
                return active_py

    # Otherwise fallback to local .venv in the project folder
    sub = os.path.join("Scripts", "python.exe") if os.name == "nt" else os.path.join("bin", "python")
    path = os.path.join(ROOT, ".venv", sub)
    return path if os.path.isfile(path) else None


if __name__ == "__main__":
    venv = _venv_python()
    if venv is None and not os.path.isdir(os.path.join(ROOT, ".venv")):
        setup_script = os.path.join(ROOT, "setup.bat" if os.name == "nt" else "setup.sh")
        if os.path.isfile(setup_script):
            print("AgyMolden is not installed yet. Running automatic setup first...")
            if os.name == "nt":
                subprocess.call([setup_script])
            else:
                subprocess.call(["/bin/bash", setup_script])
            venv = _venv_python()

    if venv and os.path.abspath(sys.executable) != os.path.abspath(venv) and not sys.prefix.startswith(os.path.join(ROOT, ".venv")):
        sys.exit(subprocess.call([venv, os.path.abspath(__file__)] + sys.argv[1:]))
    sys.path.insert(0, ROOT)
    from agymolden import main
    main()
