#!/usr/bin/env python3
"""Executable entrypoint script for PyInstaller build."""

import sys
import os

# Ensure current working directory / repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agymolden import main

if __name__ == "__main__":
    main()
