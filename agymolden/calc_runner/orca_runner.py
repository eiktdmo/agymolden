"""
ORCA Calculation Runner.

Handles running ORCA jobs, streaming output, parsing SCF iterations,
and updating geometries during geometry optimizations.
"""

import os
import re
import sys
import shutil
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from PyQt6 import QtCore

from .base import BaseCalcRunner, JobStatus, OptimizationStep
from agymolden.elements import ELEMENT_DATA

log = logging.getLogger(__name__)


def find_orca_binary() -> Optional[str]:
    """Find a system or locally installed ORCA executable (saved route, PATH, then common install folders)."""
    from agymolden.qm_detect import detect_qm_executable
    info = detect_qm_executable("orca")
    if info.found and info.resolved_path:
        return info.resolved_path
    return shutil.which("orca")


class OrcaRunner(BaseCalcRunner):
    """Execution runner for ORCA quantum chemistry jobs."""

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.orca_bin: str = find_orca_binary() or "orca"
        self._curr_cycle: int = 0
        self._curr_energy: Optional[float] = None
        self._parsing_geom: bool = False
        self._geom_buffer: List[Tuple[str, float, float, float]] = []
        self._out_fh = None

    def configure(self, orca_bin: str):
        self.orca_bin = orca_bin

    def start_job(self, input_file: str, work_dir: Optional[str] = None, **kwargs) -> bool:
        if not os.path.isfile(input_file):
            self._set_status(JobStatus.FAILED, f"Input file not found: {input_file}")
            return False

        self.input_file = os.path.abspath(input_file)
        self.work_dir = work_dir or os.path.dirname(self.input_file)
        stem = os.path.splitext(self.input_file)[0]
        self.log_file = f"{stem}.out"
        self.steps.clear()
        self._raw_log_buffer.clear()
        self._curr_cycle = 0
        self._curr_energy = None
        self._parsing_geom = False
        self._geom_buffer.clear()

        # Remove stale guess (.ges, .gbw) or temp files from previous runs with different geometry
        # unless the input explicitly requested MORead or %moinp
        try:
            with open(self.input_file, "r", encoding="utf-8", errors="replace") as f_in:
                inp_text = f_in.read().lower()
            if "moread" not in inp_text and "moinp" not in inp_text:
                for stale_ext in (".ges", ".gbw", ".prop", ".molden", ".molden.input"):
                    stale = f"{stem}{stale_ext}"
                    if os.path.isfile(stale):
                        try:
                            os.remove(stale)
                        except OSError:
                            pass
                stem_base = os.path.basename(stem)
                for f_name in os.listdir(self.work_dir):
                    if f_name.startswith(stem_base) and f_name.endswith(".tmp"):
                        try:
                            os.remove(os.path.join(self.work_dir, f_name))
                        except OSError:
                            pass
        except Exception:
            logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

        try:
            self._out_fh = open(self.log_file, "w", encoding="utf-8")
        except OSError as e:
            log.warning(f"Could not open {self.log_file} for writing: {e}")
            self._out_fh = None

        # Refresh binary route in case user configured it after OrcaRunner was instantiated
        if not self.orca_bin or self.orca_bin == "orca":
            self.orca_bin = find_orca_binary() or "orca"

        # Parse route command (allows user-specified custom binary, folder, or wrapper such as mpirun)
        # Note: ORCA requires its full absolute path when running parallel (%pal nprocs N) jobs!
        import shlex
        raw_bin = (self.orca_bin or "orca").strip()
        expanded_raw = os.path.expanduser(raw_bin)
        if os.path.isdir(expanded_raw):
            for cand in (os.path.join(expanded_raw, "orca"), os.path.join(expanded_raw, "bin", "orca")):
                if os.path.isfile(cand):
                    expanded_raw = cand
                    break
        if os.path.isfile(expanded_raw):
            route_cmd = [expanded_raw]
        else:
            try:
                route_cmd = shlex.split(raw_bin)
            except ValueError:
                route_cmd = raw_bin.split()

        exec_bin = os.path.expanduser(route_cmd[0]) if route_cmd else "orca"
        if not os.path.isabs(exec_bin):
            resolved = shutil.which(exec_bin)
            if resolved:
                exec_bin = os.path.abspath(resolved)
        if not (os.path.isfile(exec_bin) or shutil.which(exec_bin)):
            self._set_status(
                JobStatus.FAILED,
                f"ORCA executable not found ('{exec_bin}'). Set its path in QM Engines or Quick ORCA.",
            )
            return False

        extra_args = [
            (os.path.abspath(shutil.which(tok)) if (tok.endswith("orca") and not os.path.isabs(tok) and shutil.which(tok)) else tok)
            for tok in route_cmd[1:]
        ]
        args = extra_args + [self.input_file]
        extra_env = self._resolve_orca_mpi_env()
        self._setup_process(exec_bin, args, self.work_dir, extra_env=extra_env)
        self._set_status(JobStatus.RUNNING, f"Running ORCA on {os.path.basename(input_file)}...")
        return True

    @staticmethod
    def _resolve_orca_mpi_env() -> Optional[Dict[str, str]]:
        """Locate an OpenMPI 4.1 environment (e.g. conda env 'orca_mpi') if libmpi.so.40 / mpirun is not in PATH."""
        if sys.platform != "linux":
            return None
        home = os.path.expanduser("~")
        candidates = [
            os.path.join(home, "anaconda3", "envs", "orca_mpi"),
            os.path.join(home, "miniconda3", "envs", "orca_mpi"),
            os.path.join(home, "mambaforge", "envs", "orca_mpi"),
            os.path.join(home, "miniforge3", "envs", "orca_mpi"),
        ]
        for env_root in candidates:
            bin_dir = os.path.join(env_root, "bin")
            lib_dir = os.path.join(env_root, "lib")
            if os.path.isfile(os.path.join(bin_dir, "mpirun")) and os.path.isfile(os.path.join(lib_dir, "libmpi.so.40")):
                cur_path = os.environ.get("PATH", "")
                cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
                new_path = f"{bin_dir}:{cur_path}" if cur_path else bin_dir
                new_ld = f"{lib_dir}:{cur_ld}" if cur_ld else lib_dir
                return {"PATH": new_path, "LD_LIBRARY_PATH": new_ld}
        return None

    def _on_process_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus):
        if self._out_fh:
            try:
                self._out_fh.flush()
                self._out_fh.close()
            except (OSError, IOError):
                logging.getLogger(__name__).debug("Ignored exception", exc_info=True)
            self._out_fh = None

        # Automatically generate .molden.input (and a .molden copy) from .gbw via orca_2mkl if available
        if exit_code == 0 and self.input_file:
            stem = os.path.splitext(self.input_file)[0]
            gbw_file = f"{stem}.gbw"
            exec_token = (self.orca_bin or "orca").split()[0]
            resolved_orca = shutil.which(exec_token) or (exec_token if os.path.isabs(exec_token) else "")
            orca_dir = os.path.dirname(resolved_orca) if resolved_orca else ""
            orca_2mkl = (os.path.join(orca_dir, "orca_2mkl") if orca_dir else None) or shutil.which("orca_2mkl")
            if os.path.isfile(gbw_file) and orca_2mkl and os.path.isfile(orca_2mkl):
                try:
                    import subprocess
                    stem_base = os.path.basename(stem)
                    res_mkl = subprocess.run([orca_2mkl, stem_base, "-molden"], cwd=self.work_dir, capture_output=True, timeout=30)
                    molden_inp = f"{stem}.molden.input"
                    molden_std = f"{stem}.molden"
                    if not os.path.isfile(molden_inp):
                        # Also check if it was output with full path
                        alt_inp = os.path.join(self.work_dir, f"{stem_base}.molden.input")
                        if os.path.isfile(alt_inp):
                            molden_inp = alt_inp
                    if os.path.isfile(molden_inp):
                        shutil.copy2(molden_inp, molden_std)
                except (subprocess.SubprocessError, OSError) as e:
                    log.warning(f"orca_2mkl conversion warning: {e}")

        super()._on_process_finished(exit_code, exit_status)

    def _parse_line(self, line: str):
        if self._out_fh:
            try:
                self._out_fh.write(line + "\n")
                self._out_fh.flush()
            except (OSError, IOError):
                logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

        s = line.strip()

        # Optimization cycle: "OPTIMIZATION CYCLE   1"
        cycle_match = re.search(r"OPTIMIZATION\s+CYCLE\s+(\d+)", s, re.IGNORECASE)
        if cycle_match:
            try:
                self._curr_cycle = int(cycle_match.group(1))
            except ValueError:
                pass

        # Energy: "FINAL SINGLE POINT ENERGY      -76.12345678"
        e_match = re.search(r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s+([+-]?\d+\.\d+)", s, re.IGNORECASE)
        if e_match:
            try:
                self._curr_energy = float(e_match.group(1))
                if self.steps and self.steps[-1].cycle == self._curr_cycle:
                    self.steps[-1].energy = self._curr_energy
            except ValueError:
                pass

        # Coordinates: "CARTESIAN COORDINATES (ANGSTROEM)"
        if "CARTESIAN COORDINATES (ANGSTROEM)" in s:
            self._parsing_geom = True
            self._geom_buffer.clear()
            return

        if self._parsing_geom:
            if not s or "CARTESIAN" in s or "---" in s:
                if len(self._geom_buffer) > 0:
                    self._emit_geom_step()
                    self._parsing_geom = False
                return

            parts = s.split()
            if len(parts) >= 4:
                sym, x, y, z = parts[0], parts[1], parts[2], parts[3]
                if sym.capitalize() in ELEMENT_DATA:
                    try:
                        fx, fy, fz = float(x), float(y), float(z)
                        self._geom_buffer.append((sym, fx, fy, fz))
                    except ValueError:
                        pass

        # Convergence
        if re.search(r"\*{3}\s*THE\s+OPTIMIZATION\s+HAS\s+CONVERGED\s*\*{3}", s, re.IGNORECASE):
            self._set_status(JobStatus.CONVERGED, "ORCA optimization converged!")
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
