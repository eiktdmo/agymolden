"""
Universal Quantum Chemistry Backend Runner (Gaussian, PySCF, openLOWDIN, xTB, Q-Chem).

Provides live process execution, stdout/stderr capture to an output file, and
on-the-fly parsing of SCF energies and geometry optimization steps across standard
quantum chemistry codes.
"""

import os
import re
import logging
from typing import List, Optional, Tuple
import numpy as np
from PyQt6 import QtCore

from .base import BaseCalcRunner, JobStatus, OptimizationStep
from agymolden.elements import ELEMENT_DATA
from agymolden.qm_detect import detect_qm_executable

log = logging.getLogger(__name__)


class GenericQMRunner(BaseCalcRunner):
    """
    Execution runner for Gaussian (g16/g09), PySCF (.py scripts), openLOWDIN, xTB, and Q-Chem.
    Automatically resolves the executable via `qm_detect` and streams live output.
    """

    def __init__(self, code_key: str = "gaussian", parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.code_key: str = code_key.lower()
        self.custom_route: str = ""
        self._curr_cycle: int = 0
        self._curr_energy: Optional[float] = None
        self._parsing_gaussian_geom: bool = False
        self._gaussian_skip_lines: int = 0
        self._geom_buffer: List[Tuple[str, float, float, float]] = []
        self._out_fh = None

    def configure(self, code_key: str, custom_route: str = ""):
        self.code_key = code_key.lower()
        self.custom_route = custom_route.strip()

    def _resolve_command(self, input_file: str) -> Tuple[str, List[str], str]:
        """Return (executable, args_list, output_log_path) for the configured QM code."""
        info = detect_qm_executable(self.code_key, custom_route=self.custom_route or None)
        route_str = info.resolved_path or info.binary_name.split()[0]
        tokens = route_str.split()
        exe = tokens[0] if tokens else self.code_key
        extra_args = tokens[1:]

        stem = os.path.splitext(input_file)[0]
        key = info.code_key

        if key == "gaussian":
            log_file = f"{stem}.log"
            args = extra_args + [input_file]
        elif key == "pyscf":
            log_file = f"{stem}.out"
            args = extra_args + ["-u", input_file]
        elif key == "xtb":
            log_file = f"{stem}.out"
            args = extra_args + [input_file]
        elif key == "qchem":
            log_file = f"{stem}.out"
            args = extra_args + [input_file, log_file]
        elif key == "openlowdin":
            log_file = f"{stem}.out"
            args = extra_args + ["-i", input_file]
        else:
            log_file = f"{stem}.out"
            args = extra_args + [input_file]

        return exe, args, log_file

    def start_job(self, input_file: str, work_dir: Optional[str] = None, **kwargs) -> bool:
        if not os.path.isfile(input_file):
            self._set_status(JobStatus.FAILED, f"Input file not found: {input_file}")
            return False

        self.input_file = os.path.abspath(input_file)
        self.work_dir = work_dir or os.path.dirname(self.input_file)
        exe, args, self.log_file = self._resolve_command(self.input_file)

        self.steps.clear()
        self._raw_log_buffer.clear()
        self._curr_cycle = 0
        self._curr_energy = None
        self._parsing_gaussian_geom = False
        self._geom_buffer.clear()

        try:
            self._out_fh = open(self.log_file, "w", encoding="utf-8")
        except Exception as e:
            log.warning(f"Could not open {self.log_file} for writing: {e}")
            self._out_fh = None

        self._setup_process(exe, args, self.work_dir)
        self._set_status(JobStatus.RUNNING, f"Running {os.path.basename(exe)} on {os.path.basename(input_file)}...")
        return True

    def _on_process_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus):
        if self._out_fh:
            try:
                self._out_fh.flush()
                self._out_fh.close()
            except Exception:
                logging.getLogger(__name__).debug("Ignored exception", exc_info=True)
            self._out_fh = None
        super()._on_process_finished(exit_code, exit_status)

    def _parse_line(self, line: str):
        if self._out_fh:
            try:
                self._out_fh.write(line + "\n")
                self._out_fh.flush()
            except Exception:
                logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

        s = line.strip()

        # 1. Gaussian / PySCF / xTB Energy detection
        # Gaussian: "SCF Done:  E(RB3LYP) =  -76.4089532145     A.U. after    9 cycles"
        m_g_scf = re.search(r"SCF\s+Done:\s+E\([^)]+\)\s*=\s*([+-]?\d+\.\d+)", s)
        if m_g_scf:
            self._curr_energy = float(m_g_scf.group(1))
            self._curr_cycle += 1
            if self.steps:
                self.steps[-1].energy = self._curr_energy
                self.step_updated.emit(self.steps[-1])
            return

        # PySCF: "converged SCF energy = -76.026760675509" or "SCF Total Energy: -76.0267"
        m_pyscf = re.search(r"(?:converged\s+SCF\s+energy\s*=|SCF\s+Total\s+Energy:)\s*([+-]?\d+\.\d+)", s, re.IGNORECASE)
        if m_pyscf:
            self._curr_energy = float(m_pyscf.group(1))
            self._curr_cycle += 1
            step = OptimizationStep(cycle=self._curr_cycle, energy=self._curr_energy)
            self.steps.append(step)
            self.step_updated.emit(step)
            return

        # xTB: ":: total energy             -4.070671346672 Eh   ::"
        m_xtb = re.search(r"total\s+energy\s+([+-]?\d+\.\d+)\s+Eh", s, re.IGNORECASE)
        if m_xtb:
            self._curr_energy = float(m_xtb.group(1))
            self._curr_cycle += 1
            step = OptimizationStep(cycle=self._curr_cycle, energy=self._curr_energy)
            self.steps.append(step)
            self.step_updated.emit(step)
            return

        # 2. Gaussian Standard / Input Orientation coordinates
        if "Standard orientation:" in s or "Input orientation:" in s:
            self._parsing_gaussian_geom = True
            self._gaussian_skip_lines = 4
            self._geom_buffer.clear()
            return

        if self._parsing_gaussian_geom:
            if self._gaussian_skip_lines > 0:
                self._gaussian_skip_lines -= 1
                return
            if s.startswith("----"):
                if self._geom_buffer:
                    self._emit_geom_step()
                self._parsing_gaussian_geom = False
                return
            parts = s.split()
            if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
                at_num = int(parts[1])
                sym = next((k for k, v in ELEMENT_DATA.items() if v.get("atomic_number") == at_num), "C")
                try:
                    fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                    self._geom_buffer.append((sym, fx, fy, fz))
                except ValueError:
                    pass

        if "Optimization completed." in s or "CONVRG" in s or "Stationary point found" in s:
            self._set_status(JobStatus.CONVERGED, "Optimization converged!")

    def _emit_geom_step(self):
        if not self._geom_buffer:
            return
        syms = [item[0] for item in self._geom_buffer]
        coords = np.array([[item[1], item[2], item[3]] for item in self._geom_buffer], dtype=float)
        step = OptimizationStep(
            cycle=max(1, self._curr_cycle),
            energy=self._curr_energy,
            symbols=syms,
            coords=coords,
        )
        self.steps.append(step)
        self.step_updated.emit(step)
