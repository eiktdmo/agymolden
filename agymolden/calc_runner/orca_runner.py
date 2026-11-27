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

    _mpi_health_cache: Optional[Tuple[bool, str]] = None

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.orca_bin: str = find_orca_binary() or "orca"
        self._curr_cycle: int = 0
        self._curr_energy: Optional[float] = None
        self._parsing_geom: bool = False
        self._geom_buffer: List[Tuple[str, float, float, float]] = []
        self._out_fh = None
        self._serial_fallback_used: bool = False
        self._requested_nprocs: int = 1
        self._last_exec_bin: str = ""
        self._last_extra_args: List[str] = []

    def configure(self, orca_bin: str):
        self.orca_bin = orca_bin

    def start_job(self, input_file: str, work_dir: Optional[str] = None, **kwargs) -> bool:
        if not os.path.isfile(input_file):
            self._set_status(JobStatus.FAILED, f"Input file not found: {input_file}")
            return False

        is_serial_retry = bool(kwargs.get("_is_serial_retry", False))
        skip_mpi_preflight = bool(kwargs.get("_skip_mpi_preflight", False))
        if not is_serial_retry:
            self._serial_fallback_used = False

        self.input_file = os.path.abspath(input_file)
        self.work_dir = work_dir or os.path.dirname(self.input_file)
        stem = os.path.splitext(self.input_file)[0]
        self.log_file = f"{stem}.out"
        if not is_serial_retry:
            self.steps.clear()
            self._raw_log_buffer.clear()
        self._curr_cycle = 0
        self._curr_energy = None
        self._parsing_geom = False
        self._geom_buffer.clear()

        # Remove stale guess (.ges, .gbw) or temp files from previous runs with different geometry
        # unless the input explicitly requested MORead or %moinp
        raw_inp_text = ""
        try:
            with open(self.input_file, "r", encoding="utf-8", errors="replace") as f_in:
                raw_inp_text = f_in.read()
            inp_text = raw_inp_text.lower()
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

        self._requested_nprocs = self._extract_nprocs(raw_inp_text)

        try:
            self._out_fh = open(self.log_file, "a" if is_serial_retry else "w", encoding="utf-8")
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
        # ORCA truncates paths containing spaces (e.g. "Cannot open GBW file: /home/u/My"), so pass
        # the input relative to the process working directory (just the basename in the default case).
        try:
            input_arg = os.path.relpath(self.input_file, self.work_dir)
        except ValueError:  # different drive on Windows
            input_arg = self.input_file
        self._last_exec_bin = exec_bin
        self._last_extra_args = list(extra_args)

        if self._requested_nprocs > 1 and not skip_mpi_preflight:
            mpi_ok, mpi_msg = self.check_mpi_health(orca_bin=exec_bin)
            if not mpi_ok:
                backup_path = f"{stem}.parallel.inp"
                if self._fallback_input_to_serial(raw_inp_text, backup_path):
                    self._serial_fallback_used = True
                    self._requested_nprocs = 1
                    notice = (
                        f"[AgyMolden] ⚠️ Parallel MPI unavailable ({mpi_msg}). "
                        f"Automatically switched to 1-core serial mode "
                        f"(original deck saved to {os.path.basename(backup_path)})."
                    )
                    self._raw_log_buffer.append(notice)
                    self.line_received.emit(notice, False)
                    if self._out_fh:
                        try:
                            self._out_fh.write(notice + "\n")
                            self._out_fh.flush()
                        except OSError:
                            pass

        args = extra_args + [input_arg]
        extra_env = self._resolve_orca_mpi_env(orca_bin=exec_bin)
        self._setup_process(exec_bin, args, self.work_dir, extra_env=extra_env)
        self._set_status(JobStatus.RUNNING, f"Running ORCA on {os.path.basename(input_file)}...")
        return True

    @staticmethod
    def _is_conda_path(path_str: str) -> bool:
        """Return True if *path_str* belongs to a Conda/Anaconda/Mamba installation."""
        low = os.path.normpath(path_str).lower()
        parts = [seg for seg in low.replace("\\", "/").split("/") if seg]
        return any(
            key in seg
            for seg in parts
            for key in ("conda", "anaconda", "miniconda", "mambaforge", "miniforge")
        )

    @staticmethod
    def _inspect_mpirun(mpirun_path: str) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """Return (is_openmpi, (major, minor)) for *mpirun_path*."""
        import subprocess

        try:
            proc = subprocess.run(
                [mpirun_path, "--version"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            out = f"{proc.stdout or ''}\n{proc.stderr or ''}"
            low = out.lower()
            if "open mpi" not in low and "open-mpi" not in low:
                return (False, None)
            m = re.search(r"(\d+)\.(\d+)(?:\.\d+)?", out)
            if m:
                return (True, (int(m.group(1)), int(m.group(2))))
            return (True, None)
        except (OSError, subprocess.SubprocessError):
            return (False, None)

    @classmethod
    def _is_openmpi_mpirun(cls, mpirun_path: str) -> bool:
        """Return True if *mpirun_path* is an Open MPI launcher (not MPICH/Hydra)."""
        return cls._inspect_mpirun(mpirun_path)[0]

    @staticmethod
    def _has_openmpi_lib(lib_dir: str) -> bool:
        if not lib_dir or not os.path.isdir(lib_dir):
            return False
        if sys.platform == "darwin":
            return any(
                os.path.isfile(os.path.join(lib_dir, name))
                for name in ("libmpi.40.dylib", "libmpi.dylib")
            )
        return os.path.isfile(os.path.join(lib_dir, "libmpi.so.40"))

    @staticmethod
    def _inject_openmpi_mca_defaults(env_dict: Dict[str, str]) -> None:
        """Add classroom-safe OpenMPI MCA defaults unless already set by the user."""
        defaults = {
            "OMPI_MCA_rmaps_base_oversubscribe": "1",
            "OMPI_MCA_btl_vader_single_copy_mechanism": "none",
            "PMIX_MCA_gds": "hash",
        }
        for key, val in defaults.items():
            if not os.environ.get(key):
                env_dict.setdefault(key, val)

    @classmethod
    def _resolve_orca_mpi_env(cls, orca_bin: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Ensure an OpenMPI 4.1.x mpirun + libmpi are used, strip Conda paths, and set safe MCA defaults."""
        if sys.platform not in ("linux", "darwin"):
            return None

        ld_var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        raw_path_parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        clean_path_parts = [p for p in raw_path_parts if not cls._is_conda_path(p)]
        raw_ld_parts = [p for p in os.environ.get(ld_var, "").split(os.pathsep) if p]
        clean_ld_parts = [p for p in raw_ld_parts if not cls._is_conda_path(p)]
        had_conda = (len(clean_path_parts) != len(raw_path_parts)) or (len(clean_ld_parts) != len(raw_ld_parts))
        clean_path_str = os.pathsep.join(clean_path_parts)

        import glob

        candidates: List[Tuple[str, str]] = []
        seen_bins: set[str] = set()

        def _find_lib_dir(prefix: str) -> str:
            for sub in ("lib", "lib64"):
                cand_lib = os.path.join(prefix, sub)
                if cls._has_openmpi_lib(cand_lib):
                    return cand_lib
            return ""

        def _add_candidate(bin_dir: str, lib_dir: str = "") -> None:
            if not bin_dir or bin_dir in seen_bins or cls._is_conda_path(bin_dir):
                return
            seen_bins.add(bin_dir)
            if not lib_dir:
                lib_dir = _find_lib_dir(os.path.dirname(os.path.normpath(bin_dir)))
            candidates.append((bin_dir, lib_dir))

        # 1. Non-system directories already on clean PATH
        for p_dir in clean_path_parts:
            if os.path.normpath(p_dir) in ("/usr/bin", "/bin"):
                continue
            cand_bin = os.path.join(p_dir, "mpirun")
            if os.path.isfile(cand_bin) and os.access(cand_bin, os.X_OK):
                _add_candidate(p_dir)

        # 2. Sibling openmpi* directories next to orca_bin (e.g. ~/Programs/openmpi*)
        if orca_bin:
            resolved_orca = os.path.abspath(os.path.expanduser(orca_bin))
            orca_dir = os.path.dirname(resolved_orca)
            parent_dir = os.path.dirname(orca_dir)
            for base_dir in (orca_dir, parent_dir):
                if base_dir and os.path.isdir(base_dir):
                    for prefix in sorted(glob.glob(os.path.join(base_dir, "openmpi*"))):
                        _add_candidate(os.path.join(prefix, "bin"))

        # 3. Standard /opt, /usr/local, and Homebrew OpenMPI installations
        for prefix in sorted(
            glob.glob("/opt/openmpi*")
            + glob.glob("/usr/local/openmpi*")
            + ["/opt/homebrew/opt/open-mpi", "/usr/local/opt/open-mpi"]
        ):
            _add_candidate(os.path.join(prefix, "bin"))

        for prefix in ("/usr/lib64/openmpi", "/usr/lib/x86_64-linux-gnu/openmpi"):
            _add_candidate(os.path.join(prefix, "bin"))

        # 4. System /usr/bin OpenMPI fallback
        _add_candidate("/usr/bin", "/usr/lib/x86_64-linux-gnu")

        best_score = -1
        chosen: Optional[Tuple[str, str]] = None
        first_clean_dir = os.path.normpath(clean_path_parts[0]) if clean_path_parts else ""
        for bin_dir, lib_dir in candidates:
            mpirun_bin = os.path.join(bin_dir, "mpirun")
            if not (os.path.isfile(mpirun_bin) and os.access(mpirun_bin, os.X_OK)):
                continue
            is_ompi, ver = cls._inspect_mpirun(mpirun_bin)
            if not is_ompi:
                continue
            score = 0
            if cls._has_openmpi_lib(lib_dir):
                score += 100
            if ver is not None:
                if ver == (4, 1):
                    score += 50
                elif ver[0] == 4:
                    score += 30
                elif ver[0] == 5:
                    score += 10
                else:
                    score += 5
            if first_clean_dir and os.path.normpath(bin_dir) == first_clean_dir:
                score += 2
            if score > best_score:
                best_score = score
                chosen = (bin_dir, lib_dir)

        if chosen is None:
            if not had_conda:
                return None
            out_env = {"PATH": clean_path_str}
            if len(clean_ld_parts) != len(raw_ld_parts):
                out_env[ld_var] = os.pathsep.join(clean_ld_parts)
            return out_env

        bin_dir, lib_dir = chosen
        new_path_parts = [bin_dir] + [p for p in clean_path_parts if p != bin_dir]
        env_out: Dict[str, str] = {"PATH": os.pathsep.join(new_path_parts)}
        if cls._has_openmpi_lib(lib_dir):
            new_ld_parts = [lib_dir] + [p for p in clean_ld_parts if p != lib_dir]
            env_out[ld_var] = os.pathsep.join(new_ld_parts)
        elif len(clean_ld_parts) != len(raw_ld_parts):
            env_out[ld_var] = os.pathsep.join(clean_ld_parts)
        cls._inject_openmpi_mca_defaults(env_out)
        return env_out

    @classmethod
    def check_mpi_health(cls, orca_bin: Optional[str] = None, force: bool = False) -> Tuple[bool, str]:
        """Run a fast cached pre-flight check to verify whether multi-core MPI works."""
        if not force and cls._mpi_health_cache is not None:
            return cls._mpi_health_cache

        if sys.platform == "win32":
            ms_mpi = shutil.which("mpiexec.exe") or shutil.which("mpiexec")
            if not ms_mpi:
                cand = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
                if os.path.isfile(cand):
                    ms_mpi = cand
            res = (True, "MS-MPI ready") if ms_mpi else (False, "MS-MPI (mpiexec.exe) not found")
            cls._mpi_health_cache = res
            return res

        import subprocess

        env_overrides = cls._resolve_orca_mpi_env(orca_bin=orca_bin) or {}
        merged_env = os.environ.copy()
        merged_env.update(env_overrides)
        mpirun = shutil.which("mpirun", path=merged_env.get("PATH", ""))
        if not mpirun or cls._is_conda_path(mpirun):
            res = (False, "OpenMPI mpirun not found (only serial 1-core mode available)")
            cls._mpi_health_cache = res
            return res

        is_ompi, ver = cls._inspect_mpirun(mpirun)
        if not is_ompi:
            res = (False, f"Non-OpenMPI launcher at {mpirun}")
            cls._mpi_health_cache = res
            return res

        true_bin = shutil.which("true") or "/bin/true"
        ver_str = f"{ver[0]}.{ver[1]}" if ver else "unknown"
        if not os.path.isfile(true_bin):
            res = (True, f"OpenMPI {ver_str} detected")
            cls._mpi_health_cache = res
            return res

        try:
            proc = subprocess.run(
                [mpirun, "-np", "2", true_bin],
                env=merged_env,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if proc.returncode == 0:
                note = f"OpenMPI {ver_str} ready"
                if ver and ver[0] >= 5:
                    note = f"OpenMPI {ver_str} ready (ORCA 5/6 prefers 4.1.x)"
                res = (True, note)
            else:
                detail = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip().splitlines()[0][:100]
                res = (False, f"OpenMPI self-test failed ({detail})")
        except (OSError, subprocess.SubprocessError) as exc:
            res = (False, f"OpenMPI self-test failed ({exc})")

        cls._mpi_health_cache = res
        return res

    @staticmethod
    def _extract_nprocs(inp_text: str) -> int:
        """Extract requested processor count from ! PALn or %pal nprocs N blocks."""
        if not inp_text:
            return 1
        max_procs = 1
        for line in inp_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("!"):
                for m in re.finditer(r"\bPAL(\d+)\b", stripped, re.IGNORECASE):
                    try:
                        max_procs = max(max_procs, int(m.group(1)))
                    except ValueError:
                        pass
        for m in re.finditer(r"%pal\b(.*?)\bend\b", inp_text, re.IGNORECASE | re.DOTALL):
            n_m = re.search(r"\bnprocs\s+(\d+)\b", m.group(1), re.IGNORECASE)
            if n_m:
                try:
                    max_procs = max(max_procs, int(n_m.group(1)))
                except ValueError:
                    pass
        return max(1, max_procs)

    @staticmethod
    def _rewrite_input_for_serial(inp_text: str) -> str:
        """Rewrite an ORCA input deck to run serially on 1 core."""
        if not inp_text:
            return inp_text
        out_lines: List[str] = []
        for line in inp_text.splitlines(keepends=True):
            if line.lstrip().startswith("!"):
                newline = "\n" if line.endswith("\n") else ""
                body = re.sub(r"\bPAL\d+\b", "", line.rstrip("\r\n"), flags=re.IGNORECASE)
                body = re.sub(r"[ \t]{2,}", " ", body).rstrip()
                out_lines.append(body + newline)
            else:
                out_lines.append(line)
        rewritten = "".join(out_lines)
        rewritten = re.sub(
            r"%pal\b.*?\bend\b",
            "%pal nprocs 1 end",
            rewritten,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return rewritten

    def _fallback_input_to_serial(self, raw_inp_text: str, backup_path: str) -> bool:
        """Save *raw_inp_text* to *backup_path* and rewrite ``self.input_file`` for 1-core execution."""
        try:
            if not os.path.exists(backup_path):
                with open(backup_path, "w", encoding="utf-8") as f_bak:
                    f_bak.write(raw_inp_text)
            serial_text = self._rewrite_input_for_serial(raw_inp_text)
            with open(self.input_file, "w", encoding="utf-8") as f_out:
                f_out.write(serial_text)
            return True
        except OSError as exc:
            log.warning("Failed to rewrite ORCA input for serial fallback: %s", exc)
            return False

    @staticmethod
    def _is_mpi_failure(raw_log: str) -> bool:
        """Return True if ORCA log/stderr indicates an MPI startup or launcher failure."""
        if not raw_log:
            return False
        return bool(
            re.search(
                r"(BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES|"
                r"ORCA finished by error termination in Startup|"
                r"There are not enough slots available|"
                r"Calling Command:\s*mpirun|"
                r"mpirun.*not found|orterun|prterun|ORTE_ERROR_LOG|PMIx|"
                r"libmpi\.so|libopen-rte\.so|mpiexec)",
                raw_log,
                re.IGNORECASE,
            )
        )

    def _on_process_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus):
        if self._out_fh:
            try:
                self._out_fh.flush()
                self._out_fh.close()
            except (OSError, IOError):
                logging.getLogger(__name__).debug("Ignored exception", exc_info=True)
            self._out_fh = None

        if exit_code != 0 and self._requested_nprocs > 1 and not self._serial_fallback_used and self.input_file:
            combined_log = "\n".join(self._raw_log_buffer)
            if self._is_mpi_failure(combined_log):
                try:
                    with open(self.input_file, "r", encoding="utf-8", errors="replace") as f_in:
                        raw_inp_text = f_in.read()
                except OSError:
                    raw_inp_text = ""
                stem = os.path.splitext(self.input_file)[0]
                backup_path = f"{stem}.parallel.inp"
                if raw_inp_text and self._fallback_input_to_serial(raw_inp_text, backup_path):
                    self._serial_fallback_used = True
                    self._requested_nprocs = 1
                    banner = (
                        f"[AgyMolden] ⚠️ ORCA parallel MPI startup failed (exit code {exit_code}). "
                        f"Automatically retrying in 1-core serial mode..."
                    )
                    self._raw_log_buffer.append(banner)
                    self.line_received.emit(banner, False)
                    if self.start_job(
                        self.input_file,
                        self.work_dir,
                        _is_serial_retry=True,
                        _skip_mpi_preflight=True,
                    ):
                        return

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
