"""
Universal Quantum Chemistry Backend Runner (Gaussian, PySCF, openLOWDIN, xTB, Q-Chem).

Provides live process execution, stdout/stderr capture to an output file, and
on-the-fly parsing of SCF energies and geometry optimization steps across standard
quantum chemistry codes.
"""

import os
import re
import shlex
import shutil
import logging
from typing import List, Optional, Tuple, Dict, Any
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
        self.program_name: str = self.code_key
        self.custom_route: str = ""
        self._curr_cycle: int = 0
        self._curr_energy: Optional[float] = None
        self._parsing_gaussian_geom: bool = False
        self._parsing_pyscf_geom: bool = False
        self._gaussian_skip_lines: int = 0
        self._geom_buffer: List[Tuple[str, float, float, float]] = []
        self._out_fh = None
        self.python_bin: Optional[str] = None
        self.stdin_file: Optional[str] = None

    def configure(self, code_key: str, custom_route: str = "", python_bin: Optional[str] = None):
        self.code_key = code_key.lower()
        self.program_name = self.code_key
        self.custom_route = custom_route.strip()
        if python_bin:
            self.python_bin = python_bin

    def _get_gaussian_env(self, exe: str, work_dir: str) -> Dict[str, str]:
        """
        Synthesize a complete, self-healing Gaussian execution environment.
        Ensures GAUSS_EXEDIR, g16root/g09root, G16BASIS, GAUSS_SCRDIR, PATH, and LD_LIBRARY_PATH
        are properly configured even when launched from non-login desktop sessions.
        """
        env: Dict[str, str] = {}
        exe_path = os.path.abspath(exe) if os.path.isabs(exe) else (shutil.which(exe) or exe)
        if os.path.isfile(exe_path):
            g_dir = os.path.dirname(exe_path)
            g_root = os.path.dirname(g_dir)
            is_g16 = "g16" in exe_path.lower()
            is_g09 = "g09" in exe_path.lower()
            is_g03 = "g03" in exe_path.lower()

            root_var = "g16root" if is_g16 else ("g09root" if is_g09 else "g03root")
            basis_var = "G16BASIS" if is_g16 else ("G09BASIS" if is_g09 else "G03BASIS")

            if not os.environ.get("GAUSS_EXEDIR"):
                env["GAUSS_EXEDIR"] = f"{g_dir}/bsd:{g_dir}"
                env[root_var] = g_root
                env["GAUSS_LEXEDIR"] = f"{g_dir}/linda-exe"
                env["GAUSS_ARCHDIR"] = f"{g_dir}/arch"
                env["GAUSS_BSDDIR"] = f"{g_dir}/bsd"
                env[basis_var] = f"{g_dir}/basis"

            scr_dir = os.environ.get("GAUSS_SCRDIR") or os.path.join(work_dir, "scratch")
            try:
                os.makedirs(scr_dir, exist_ok=True)
            except Exception:
                scr_dir = work_dir
            env["GAUSS_SCRDIR"] = scr_dir

            curr_path = os.environ.get("PATH", "")
            if g_dir not in curr_path:
                env["PATH"] = f"{g_root}/gauopen:{g_dir}/bsd:{g_dir}:{curr_path}"

            curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if g_dir not in curr_ld:
                env["LD_LIBRARY_PATH"] = f"{g_dir}/bsd:{g_dir}:{curr_ld}"

            env["PGI_TERM"] = "trace,abort"

        return env

    def _resolve_command(self, input_file: str) -> Tuple[str, List[str], str, bool]:
        """Return (executable, args_list, output_log_path, use_stdin) for the configured QM code."""
        info = detect_qm_executable(self.code_key, custom_route=self.custom_route or None)
        route_str = (info.resolved_path or info.binary_name.split()[0]).strip()
        if os.path.isfile(route_str):
            exe = route_str
            extra_args = []
        else:
            tokens = shlex.split(route_str, posix=(os.name != "nt")) if route_str else []
            exe = tokens[0] if tokens else self.code_key
            extra_args = tokens[1:]

        stem = os.path.splitext(input_file)[0]
        key = info.code_key
        use_stdin = False

        if key == "gaussian":
            log_file = f"{stem}.log"
            # On Linux/macOS with standard g16/g09, piping via stdin streams stdout live in real time!
            if not extra_args and os.name != "nt":
                args = []
                use_stdin = True
            else:
                args = extra_args + [input_file]
                use_stdin = False
        elif key == "pyscf":
            if self.python_bin and os.path.isfile(self.python_bin):
                exe = self.python_bin
                extra_args = []
            log_file = f"{stem}.out"
            args = extra_args + ["-u", input_file]
        elif key == "xtb":
            log_file = f"{stem}.out"
            args = extra_args + [input_file]
        elif key == "qchem":
            log_file = f"{stem}.out"
            args = extra_args + [input_file, log_file]
        elif key == "psi4":
            log_file = f"{stem}.out"
            args = extra_args + [input_file, log_file]
        elif key == "openlowdin":
            log_file = f"{stem}.out"
            args = extra_args + ["-i", input_file]
        else:
            log_file = f"{stem}.out"
            args = extra_args + [input_file]

        return exe, args, log_file, use_stdin

    def start_job(self, input_file: str, work_dir: Optional[str] = None, **kwargs) -> bool:
        if not os.path.isfile(input_file):
            self._set_status(JobStatus.FAILED, f"Input file not found: {input_file}")
            return False

        self.input_file = os.path.abspath(input_file)
        self.work_dir = work_dir or os.path.dirname(self.input_file)
        exe, args, self.log_file, use_stdin = self._resolve_command(self.input_file)

        self.steps.clear()
        self._raw_log_buffer.clear()
        self._curr_cycle = 0
        self._curr_energy = None
        self._parsing_gaussian_geom = False
        self._parsing_pyscf_geom = False
        self._geom_buffer.clear()

        # Gaussian self-healing environment setup
        extra_env = None
        if self.code_key == "gaussian":
            extra_env = self._get_gaussian_env(exe, self.work_dir)

        # Output file handle management
        if use_stdin:
            self.stdin_file = self.input_file
            try:
                self._out_fh = open(self.log_file, "w", encoding="utf-8")
            except Exception as e:
                log.warning(f"Could not open {self.log_file} for writing: {e}")
                self._out_fh = None
        else:
            self.stdin_file = None
            self._out_fh = None

        self._setup_process(exe, args, self.work_dir, extra_env=extra_env)
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

        # If output was written directly by Gaussian to self.log_file (when not using stdin),
        # parse the entire file to ensure all steps are captured!
        if self.code_key == "gaussian" and os.path.isfile(self.log_file) and not self.steps:
            try:
                with open(self.log_file, "r", encoding="utf-8", errors="replace") as f_tail:
                    for line in f_tail:
                        self._parse_line(line.rstrip("\r\n"))
            except Exception as e:
                log.debug(f"Failed to post-parse log file {self.log_file}: {e}")

        # Post-job hook for Gaussian: run formchk on .chk if available
        if exit_code == 0 and self.code_key == "gaussian" and self.input_file:
            try:
                stem = os.path.splitext(self.input_file)[0]
                for chk_cand in (f"{stem}.chk", os.path.join(self.work_dir, "checkpoint.chk")):
                    if os.path.isfile(chk_cand):
                        formchk_bin = shutil.which("formchk")
                        if formchk_bin:
                            fchk_out = f"{stem}.fchk"
                            import subprocess
                            subprocess.run([formchk_bin, chk_cand, fchk_out], cwd=self.work_dir, capture_output=True, timeout=30)
                        break
            except Exception as e:
                log.debug(f"formchk post-processing warning: {e}")

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
            else:
                step = OptimizationStep(cycle=self._curr_cycle, energy=self._curr_energy)
                self.steps.append(step)
                self.step_updated.emit(step)
            return

        m_g_post = re.search(r"(?:EUMP2|E\(CORR\)|CCSD\(T\)?)\s*=\s*([+-]?\d+\.\d+)", s, re.IGNORECASE)
        if m_g_post:
            self._curr_energy = float(m_g_post.group(1))
            if self.steps:
                self.steps[-1].energy = self._curr_energy
                self.step_updated.emit(self.steps[-1])
            else:
                step = OptimizationStep(cycle=max(1, self._curr_cycle), energy=self._curr_energy)
                self.steps.append(step)
                self.step_updated.emit(step)
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

        # openLOWDIN: "TOTAL ENERGY = -76.026760675" or "Total Energy: -76.0267"
        m_lowdin = re.search(r"(?:TOTAL\s+ENERGY\s*=|Total\s+Energy\s*[:=]|E\(TOTAL\)\s*=)\s*([+-]?\d+\.\d+)", s, re.IGNORECASE)
        if m_lowdin:
            self._curr_energy = float(m_lowdin.group(1))
            self._curr_cycle += 1
            step = OptimizationStep(cycle=self._curr_cycle, energy=self._curr_energy)
            self.steps.append(step)
            self.step_updated.emit(step)
            return

        # 2a. PySCF / openLOWDIN Cartesian Coordinates (Angstrom) block
        if "Cartesian Coordinates (Angstrom):" in s:
            self._parsing_pyscf_geom = True
            self._geom_buffer.clear()
            return

        if self._parsing_pyscf_geom:
            if not s or s.startswith("---"):
                if self._geom_buffer:
                    self._emit_geom_step()
                self._parsing_pyscf_geom = False
                return
            parts = s.split()
            if len(parts) >= 4 and parts[0].capitalize() in ELEMENT_DATA:
                try:
                    fx, fy, fz = float(parts[1]), float(parts[2]), float(parts[3])
                    self._geom_buffer.append((parts[0].capitalize(), fx, fy, fz))
                except ValueError:
                    pass
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
