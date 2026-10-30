"""Locate bundled data files (works from source, pip install and PyInstaller)."""

import os
import sys
from typing import Optional


def find_resource(name: str) -> Optional[str]:
    """Return the absolute path of a bundled file, or None if not found."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if getattr(sys, "_MEIPASS", None):  # PyInstaller bundle
        candidates.append(os.path.join(sys._MEIPASS, name))
    candidates.append(os.path.join(os.path.dirname(here), name))  # repo root
    candidates.append(os.path.join(here, name))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None
