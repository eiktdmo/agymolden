"""
newBRORSEN Calculation Runner.

Handles running newBRORSEN wrapper scripts across different Conda environments,
streams live output, and parses optimization cycles (geometry, energy, gradient)
in real-time as they are printed to stdout.
"""

import os
import re
import sys
import glob
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from PyQt6 import QtCore

from .base import BaseCalcRunner, JobStatus, OptimizationStep
from agymolden.elements import ELEMENT_DATA

log = logging.getLogger(__name__)


_ENV_CACHE: Optional[List[Tuple[str, str]]] = None
_WRAPPER_CACHE: Optional[List[str]] = None


def detect_conda_environments(force_refresh: bool = False) -> List[Tuple[str, str]]:
    """
    Detect available Python environments (virtual environments, conda envs, user programs).
    Returns list of (display_label, python_binary_path). Uses cached results for instant loading.
    """
    global _ENV_CACHE
    if _ENV_CACHE is not None and not force_refresh:
        return list(_ENV_CACHE)

    envs: List[Tuple[str, str]] = []
    seen_paths = set()

    def add_env(label: str, path: str):
        if not path:
            return
        abs_p = os.path.abspath(path)
        if abs_p not in seen_paths and os.path.isfile(abs_p) and os.access(abs_p, os.X_OK):
            seen_paths.add(abs_p)
            envs.append((label, abs_p))

    # 1. Current running Python
    curr_py = sys.executable
    add_env("Current Python Interpreter", curr_py)

    # 2. Check active VIRTUAL_ENV or CONDA_PREFIX environment variables
    env_venv = os.environ.get("VIRTUAL_ENV")
    if env_venv:
        v_py = os.path.join(env_venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(env_venv, "bin", "python")
        add_env(f"Active Virtualenv ({os.path.basename(env_venv)})", v_py)

    env_conda = os.environ.get("CONDA_PREFIX")
    if env_conda:
        c_py = os.path.join(env_conda, "Scripts", "python.exe") if os.name == "nt" else os.path.join(env_conda, "bin", "python")
        add_env(f"Active Conda ({os.path.basename(env_conda)})", c_py)

    home = os.path.expanduser("~")

    # 3. Known virtual environments in common software / user directories
    custom_patterns = [
        os.path.join(home, "programs", "*", "andres-venv", "bin", "python"),
        os.path.join(home, "program", "*", "andres-venv", "bin", "python"),
        os.path.join(home, "programs", "*", "bin", "python"),
        os.path.join(home, "programs", "*", "Scripts", "python.exe"),
        os.path.join(home, "programs", "*", "*", "bin", "python"),
        os.path.join(home, "program", "*", "*", "bin", "python"),
        os.path.join(home, "*", "andres-venv", "bin", "python"),
        os.path.join(home, ".venv", "bin", "python"),
        os.path.join(home, "venv", "bin", "python"),
    ]
    for pat in custom_patterns:
        for py_path in glob.glob(pat):
            parts = py_path.split(os.sep)
            if len(parts) >= 4 and parts[-1] in ("python", "python.exe") and parts[-2] in ("bin", "Scripts"):
                env_title = f"{parts[-4]}/{parts[-3]}"
            elif len(parts) >= 3 and parts[-1] in ("python", "python.exe"):
                env_title = parts[-2]
            else:
                env_title = os.path.basename(os.path.dirname(os.path.dirname(py_path)))
            add_env(f"venv: {env_title}", py_path)

    # 4. Common Conda root environments
    candidates = [
        os.path.join(home, "miniconda3"),
        os.path.join(home, "anaconda3"),
        os.path.join(home, ".conda"),
        "/opt/conda",
        "/opt/miniconda3",
    ]

    for root in candidates:
        envs_dir = os.path.join(root, "envs")
        if os.path.isdir(envs_dir):
            try:
                for name in sorted(os.listdir(envs_dir)):
                    py_path = os.path.join(envs_dir, name, "Scripts", "python.exe") if os.name == "nt" else os.path.join(envs_dir, name, "bin", "python")
                    add_env(f"conda: {name}", py_path)
            except (OSError, PermissionError):
                logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

    _ENV_CACHE = envs
    return list(envs)


def find_brorsen_wrappers(force_refresh: bool = False) -> List[str]:
    """Find known newBRORSEN wrapper scripts in workspace and user directories."""
    global _WRAPPER_CACHE
    if _WRAPPER_CACHE is not None and not force_refresh:
        return list(_WRAPPER_CACHE)

    wrappers: List[str] = []
    seen = set()

    def add_wrap(path: str):
        abs_p = os.path.abspath(path)
        if abs_p not in seen and os.path.isfile(abs_p):
            seen.add(abs_p)
            wrappers.append(abs_p)

    home = os.path.expanduser("~")
    # 0. Always check the current working directory / repo root first for local newBRORSEN scripts
    cwd = os.path.abspath(".")
    for fn in ("newBRORSEN.v04.09.py", "newBRORSEN.v04.08.py"):
        local_target = os.path.join(cwd, fn)
        if os.path.isfile(local_target):
            add_wrap(local_target)

    # Quick, targeted lookups (workspace and immediate subfolders, desktop root and direct subdirs)
    patterns = [
        os.path.abspath("./newBRORSEN*.py"),
        os.path.abspath("./*.py"),
        os.path.join(os.path.abspath("."), "*", "*.py"),
        os.path.join(home, "Desktop", "*.py"),
        os.path.join(home, "Desktop", "*", "newBRORSEN*.py"),
        os.path.join(home, "Desktop", "newBRORSEN*.py"),
        os.path.join(home, "Downloads", "newBRORSEN*.py"),
    ]
    for pat in patterns:
        for p in glob.glob(pat):
            bn = os.path.basename(p)
            if "newBRORSEN" in bn or "input_builder" in bn:
                add_wrap(p)

    _WRAPPER_CACHE = sorted(wrappers, reverse=True)
    return list(_WRAPPER_CACHE)


class BrorsenRunner(BaseCalcRunner):
    """Execution runner for newBRORSEN calculation jobs."""

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.python_bin: str = sys.executable
        self.wrapper_path: str = ""
        self.extra_flags: List[str] = []
        
        # State tracking for live output parsing
        self._curr_cycle: int = 0
        self._curr_energy: Optional[float] = None
        self._parsing_geom: bool = False
        self._geom_buffer: List[Tuple[str, float, float, float]] = []

    def configure(self, python_bin: str, wrapper_path: str, extra_flags: Optional[List[str]] = None):
        self.python_bin = python_bin
        self.wrapper_path = wrapper_path
        self.extra_flags = extra_flags or []

    def start_job(self, input_file: str, work_dir: Optional[str] = None, **kwargs) -> bool:
        if not os.path.isfile(input_file):
            self._set_status(JobStatus.FAILED, f"Input file not found: {input_file}")
            return False

        if not os.path.isfile(self.wrapper_path):
            self._set_status(JobStatus.FAILED, f"Brorsen wrapper script not found: {self.wrapper_path}")
            return False

        if not os.path.isfile(self.python_bin):
            self._set_status(JobStatus.FAILED, f"Python interpreter not found: {self.python_bin}")
            return False

        self.input_file = os.path.abspath(input_file)
        self.work_dir = work_dir or os.path.dirname(self.input_file)
        self.steps.clear()
        self._raw_log_buffer.clear()
        self._curr_cycle = 0
        self._curr_energy = None
        self._parsing_geom = False
        self._geom_buffer.clear()

        args = [self.wrapper_path] + self.extra_flags + [self.input_file]
        self._setup_process(self.python_bin, args, self.work_dir)
        self._set_status(JobStatus.RUNNING, f"Running newBRORSEN on {os.path.basename(input_file)}...")
        return True

    def _parse_line(self, line: str):
        """Parse newBRORSEN stdout for cycle steps, energies, and geometries."""
        s = line.strip()

        # Check for optimization cycle header
        # E.g.: "--- Cycle 1 ---" or "Cycle 1:" or "Step 1" or "Geometry optimization cycle 1"
        cycle_match = re.search(r"(?:Cycle|Step|Iteration)\s+(\d+)", s, re.IGNORECASE)
        if cycle_match:
            try:
                self._curr_cycle = int(cycle_match.group(1))
            except ValueError:
                pass

        # Energy detection: "Total Energy = -76.12345678" or "E(HF) = ..." or "Energy: -76.xxx"
        e_match = re.search(r"(?:Total\s+Energy|E\(.+?\)|Energy)\s*[:=]\s*([+-]?\d+\.\d+)", s, re.IGNORECASE)
        if e_match:
            try:
                self._curr_energy = float(e_match.group(1))
            except ValueError:
                pass

        # Geometry block detection
        if "Coordinates (Angstrom)" in s or "Cartesian coordinates:" in s or "Standard orientation:" in s:
            self._parsing_geom = True
            self._geom_buffer.clear()
            return

        if self._parsing_geom:
            # Check for termination of geom block
            if not s or s.startswith("---") or "Grad" in s or "Energy" in s or "Cycle" in s:
                if len(self._geom_buffer) > 0:
                    self._emit_geom_step()
                self._parsing_geom = False
                return

            # Try parsing: Atom X Y Z or Idx Atom X Y Z
            parts = s.split()
            if len(parts) >= 4:
                # Check if first item is index or symbol
                if parts[0].isdigit() and len(parts) >= 5:
                    sym, x, y, z = parts[1], parts[2], parts[3], parts[4]
                else:
                    sym, x, y, z = parts[0], parts[1], parts[2], parts[3]

                # Verify valid element symbol
                sym_clean = sym.strip().capitalize()
                if sym_clean in ELEMENT_DATA or sym.upper() in ("EP", "POS", "MU"):
                    try:
                        fx, fy, fz = float(x), float(y), float(z)
                        self._geom_buffer.append((sym, fx, fy, fz))
                    except ValueError:
                        pass

        # Convergence detection
        if "Optimization converged" in s or "CONVERGENCE REACHED" in s or "Optimization completed successfully" in s:
            self._set_status(JobStatus.CONVERGED, "Optimization converged!")
            if self.steps:
                self.steps[-1].is_converged = True
                self.step_updated.emit(self.steps[-1])

    def _emit_geom_step(self):
        if not self._geom_buffer:
            return
        syms = [item[0] for item in self._geom_buffer]
        coords = np.array([[item[1], item[2], item[3]] for item in self._geom_buffer], dtype=float)

        step = OptimizationStep(
            cycle=self._curr_cycle,
            energy=self._curr_energy,
            symbols=syms,
            coords=coords,
        )
        self.steps.append(step)
        self.step_updated.emit(step)
