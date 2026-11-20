"""
Quantum Chemistry Executable and Route Detection Engine.

Discovers system/local QM packages (ORCA, Gaussian, PySCF, openLOWDIN, xTB, Q-Chem, etc.),
checks their execution status, validates custom command paths or routes provided by users,
and persists configured routes in QSettings.
"""

import logging
import os
import sys
import glob
import re
import shlex
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple, NamedTuple, Set
from PyQt6 import QtCore


class QMExecutableInfo(NamedTuple):
    code_key: str
    display_name: str
    binary_name: str
    found: bool
    resolved_path: str
    version_str: str
    message: str


def _glob_candidates(*patterns: str) -> List[str]:
    """Expand glob patterns (with ~) and return matches, newest-named first."""
    out: List[str] = []
    for pat in patterns:
        out.extend(sorted(glob.glob(os.path.expanduser(pat)), reverse=True))
    return out


def _dedupe_paths(paths: List[str]) -> List[str]:
    """Return non-empty paths in order without duplicates."""
    seen: Set[str] = set()
    out: List[str] = []
    for p in paths:
        if not p:
            continue
        norm = os.path.normpath(os.path.expanduser(p))
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _env_var_candidates(var_names: Tuple[str, ...], bin_names: Tuple[str, ...]) -> List[str]:
    """Extract candidate executable paths from environment variables."""
    out: List[str] = []
    for var in var_names:
        val = os.environ.get(var, "").strip()
        if not val:
            continue
        expanded = os.path.expanduser(val)
        if os.path.isfile(expanded):
            out.append(expanded)
        elif os.path.isdir(expanded):
            for b in bin_names:
                out.append(os.path.join(expanded, b))
                out.append(os.path.join(expanded, "bin", b))
    return out


def _normalize_code_key(code_key: str) -> str:
    """Map user-facing QM package names and aliases to canonical keys."""
    k = (code_key or "").strip().lower()
    if "orca" in k:
        return "orca"
    if "gaussian" in k or "g16" in k or "g09" in k or "g03" in k:
        return "gaussian"
    if "pyscf" in k:
        return "pyscf"
    if "openlowdin" in k or "lowdin" in k or "apmo" in k:
        return "openlowdin"
    if "xtb" in k:
        return "xtb"
    if "qchem" in k or "q-chem" in k:
        return "qchem"
    if "psi4" in k:
        return "psi4"
    return k


def _probe_paths(key: str) -> List[str]:
    """Candidate executables for a QM code: env vars & PATH first, then common install locations."""
    key = _normalize_code_key(key)
    if key == "orca":
        bin_names = ("orca.exe", "orca") if os.name == "nt" else ("orca",)
        env_cands = _env_var_candidates(
            ("ORCA_BIN", "ORCA_HOME", "ORCA_DIR", "EBROOTORCA"),
            bin_names,
        )
        globs = _glob_candidates(
            "~/orca*/orca", "~/orca*/bin/orca",
            "~/Programs/orca*/orca", "~/Programs/orca*/bin/orca",
            "~/programs/orca*/orca", "~/programs/orca*/bin/orca",
            "~/Software/orca*/orca", "~/Software/orca*/bin/orca",
            "~/software/orca*/orca", "~/software/orca*/bin/orca",
            "~/Applications/orca*/orca", "~/Applications/orca*/bin/orca",
            "~/apps/orca*/orca", "~/apps/orca*/bin/orca",
            "~/Downloads/orca*/orca", "~/Desktop/orca*/orca",
            "~/bin/orca", "~/.local/bin/orca",
            "~/miniconda3/envs/*/bin/orca", "~/anaconda3/envs/*/bin/orca",
            "~/mambaforge/envs/*/bin/orca", "~/miniforge3/envs/*/bin/orca",
            "/opt/orca*/orca", "/opt/orca*/bin/orca", "/opt/conda/envs/*/bin/orca",
            "/usr/local/orca*/orca", "/usr/local/orca*/bin/orca",
            "/usr/local/share/orca*/orca",
            "/usr/local/bin/orca", "/usr/bin/orca", "/snap/bin/orca",
            "/software/orca*/orca", "/software/orca*/bin/orca",
            "/apps/orca*/orca", "/apps/orca*/bin/orca",
            "C:/orca*/orca.exe", "C:/Program Files/orca*/orca.exe",
        )
        return _dedupe_paths(env_cands + [shutil.which("orca") or ""] + globs)
    if key == "gaussian":
        bin_names = ("g16.exe", "g09.exe", "g03.exe") if os.name == "nt" else ("g16", "g09", "g03")
        env_cands = _env_var_candidates(
            ("g16root", "g09root", "g03root", "GAUSS_EXEDIR", "EBROOTGAUSSIAN"),
            bin_names,
        )
        which_cands = [
            shutil.which("g16") or "",
            shutil.which("g16.exe") or "",
            shutil.which("g09") or "",
            shutil.which("g09.exe") or "",
            shutil.which("g03") or "",
            shutil.which("g03.exe") or "",
        ]
        globs = _glob_candidates(
            "~/g16*/g16", "~/g09*/g09",
            "~/Programs/g16*/g16", "~/Programs/g09*/g09", "~/Programs/gaussian*/g*/g*",
            "~/programs/g16*/g16", "~/programs/g09*/g09", "~/programs/gaussian*/g*/g*",
            "~/Software/g16*/g16", "~/Software/gaussian*/g*/g*",
            "~/software/g16*/g16", "~/software/gaussian*/g*/g*",
            "~/bin/g16", "~/bin/g09", "~/.local/bin/g16", "~/.local/bin/g09",
            "/opt/g16*/g16", "/opt/g09*/g09", "/opt/gaussian*/g*/g*",
            "/usr/local/g16*/g16", "/usr/local/g09*/g09", "/usr/local/gaussian*/g*/g*",
            "/usr/local/bin/g16", "/usr/local/bin/g09", "/usr/bin/g16", "/usr/bin/g09",
            "/software/gaussian*/g*/g*", "/apps/gaussian*/g*/g*",
            "C:/G16W/g16.exe", "C:/G09W/g09.exe", "C:/G03W/g03.exe",
            "C:/Program Files/G16W/g16.exe", "C:/Program Files (x86)/G16W/g16.exe",
            "C:/Program Files/G09W/g09.exe", "C:/Program Files (x86)/G09W/g09.exe",
            "C:/gaussian*/g*.exe", "D:/G16W/g16.exe", "D:/G09W/g09.exe",
        )
        return _dedupe_paths(env_cands + which_cands + globs)
    if key == "pyscf":
        local_venvs = [
            sys.executable,
            os.path.abspath(".venv/bin/python"),
            os.path.abspath("venv/bin/python"),
            shutil.which("python3") or "",
            shutil.which("python") or "",
        ]
        envs = _glob_candidates(
            "~/.venv/bin/python",
            "~/venv/bin/python",
            "~/venvs/*/bin/python",
            "~/miniconda3/bin/python",
            "~/miniconda3/envs/*/bin/python",
            "~/anaconda3/bin/python",
            "~/anaconda3/envs/*/bin/python",
            "~/mambaforge/envs/*/bin/python",
            "~/miniforge3/envs/*/bin/python",
            "~/.conda/envs/*/bin/python",
            "/opt/conda/envs/*/bin/python",
            "~/Programs/*/.venv/bin/python",
        )
        return _dedupe_paths(local_venvs + envs)
    if key == "openlowdin":
        env_cands = _env_var_candidates(
            ("OPENLOWDIN_HOME", "LOWDIN_HOME", "OPENLOWDIN_DIR"),
            ("openlowdin", "lowdin"),
        )
        globs = _glob_candidates(
            "~/openlowdin*/openlowdin", "~/openlowdin*/bin/openlowdin",
            "~/Programs/openlowdin*/openlowdin", "~/Programs/openlowdin*/bin/openlowdin",
            "~/programs/openlowdin*/openlowdin", "~/programs/openlowdin*/bin/openlowdin",
            "~/Software/openlowdin*/openlowdin", "~/Software/openlowdin*/bin/openlowdin",
            "~/software/openlowdin*/openlowdin", "~/software/openlowdin*/bin/openlowdin",
            "~/bin/openlowdin", "~/.local/bin/openlowdin",
            "/opt/openlowdin*/openlowdin", "/opt/openlowdin*/bin/openlowdin",
            "/usr/local/openlowdin*/bin/openlowdin", "/usr/local/bin/openlowdin", "/usr/bin/openlowdin",
            "/software/openlowdin*/bin/openlowdin", "/apps/openlowdin*/bin/openlowdin",
        )
        return _dedupe_paths(env_cands + [shutil.which("openlowdin") or "", shutil.which("lowdin") or ""] + globs)
    if key == "xtb":
        bin_names = ("xtb.exe", "xtb") if os.name == "nt" else ("xtb",)
        env_cands = _env_var_candidates(("XTBHOME", "XTB_HOME", "EBROOTXTB"), bin_names)
        globs = _glob_candidates(
            "~/.agymolden/bin/xtb",
            "~/.agymolden/bin/xtb*/bin/xtb",
            "~/.agymolden/bin/xtb.exe",
            "~/Programs/xtb*/bin/xtb", "~/Programs/xtb*/xtb",
            "~/programs/xtb*/bin/xtb", "~/Software/xtb*/bin/xtb", "~/software/xtb*/bin/xtb",
            "~/xtb*/bin/xtb", "~/bin/xtb", "~/.local/bin/xtb",
            "~/miniconda3/envs/*/bin/xtb", "~/anaconda3/envs/*/bin/xtb",
            "~/mambaforge/envs/*/bin/xtb", "~/miniforge3/envs/*/bin/xtb",
            "~/.conda/envs/*/bin/xtb", "/opt/conda/envs/*/bin/xtb",
            "/opt/xtb*/bin/xtb", "/usr/local/xtb*/bin/xtb",
            "/usr/local/bin/xtb", "/usr/bin/xtb", "/snap/bin/xtb",
            "/software/xtb*/bin/xtb", "/apps/xtb*/bin/xtb",
        )
        return _dedupe_paths(env_cands + [shutil.which("xtb") or ""] + globs)
    if key == "qchem":
        env_cands = _env_var_candidates(("QC", "QCHEM_HOME", "EBROOTQCHEM"), ("qchem",))
        globs = _glob_candidates(
            "~/qchem*/bin/qchem", "~/qchem*/qchem",
            "~/Programs/qchem*/bin/qchem", "~/programs/qchem*/bin/qchem",
            "~/Software/qchem*/bin/qchem", "~/software/qchem*/bin/qchem",
            "~/bin/qchem", "~/.local/bin/qchem",
            "/opt/qchem*/bin/qchem", "/usr/local/qchem*/bin/qchem",
            "/usr/local/bin/qchem", "/usr/bin/qchem",
            "/software/qchem*/bin/qchem", "/apps/qchem*/bin/qchem",
        )
        return _dedupe_paths(env_cands + [shutil.which("qchem") or ""] + globs)
    if key == "psi4":
        env_cands = _env_var_candidates(("PSI4_HOME", "PSI_HOME", "EBROOTPSI4"), ("psi4",))
        globs = _glob_candidates(
            "~/miniconda3/bin/psi4",
            "~/miniconda3/envs/*/bin/psi4",
            "~/anaconda3/bin/psi4",
            "~/anaconda3/envs/*/bin/psi4",
            "~/mambaforge/envs/*/bin/psi4",
            "~/miniforge3/envs/*/bin/psi4",
            "~/.conda/envs/*/bin/psi4",
            "/opt/conda/envs/*/bin/psi4",
            "~/Programs/psi4*/bin/psi4", "~/programs/psi4*/bin/psi4",
            "~/Software/psi4*/bin/psi4", "~/software/psi4*/bin/psi4",
            "~/bin/psi4", "~/.local/bin/psi4",
            "/usr/local/bin/psi4", "/usr/bin/psi4",
            "/opt/psi4*/bin/psi4", "/software/psi4*/bin/psi4", "/apps/psi4*/bin/psi4",
        )
        return _dedupe_paths(env_cands + [shutil.which("psi4") or ""] + globs)
    return [shutil.which(key) or ""]


def _check_binary(path: str) -> bool:
    if not (path and os.path.isfile(path)):
        return False
    if os.name == "nt":
        # On Windows, any existing .exe, .bat, or .cmd file is runnable
        ext = os.path.splitext(path)[1].lower()
        if ext in (".exe", ".bat", ".cmd", ""):
            return True
    return bool(os.access(path, os.X_OK))


def _check_python_pyscf(python_path: str) -> Tuple[bool, str]:
    if not (python_path and os.path.isfile(python_path) and os.access(python_path, os.X_OK)):
        return False, ""
    try:
        proc = subprocess.run(
            [python_path, "-c", "import pyscf; print(getattr(pyscf, '__version__', 'available'))"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if proc.returncode == 0:
            ver = proc.stdout.strip()
            return True, f"PySCF v{ver}"
    except Exception:
        logging.getLogger(__name__).debug("Ignored exception", exc_info=True)
    return False, ""


def _target_binary_names(key: str) -> Tuple[str, ...]:
    """Return executable filenames to look for during filesystem search."""
    if key == "orca":
        return ("orca.exe", "orca") if os.name == "nt" else ("orca",)
    if key == "gaussian":
        return ("g16.exe", "g09.exe", "g03.exe") if os.name == "nt" else ("g16", "g09", "g03")
    if key == "pyscf":
        return ("python.exe", "python3", "python") if os.name == "nt" else ("python3", "python")
    if key == "openlowdin":
        return ("openlowdin", "lowdin")
    if key == "xtb":
        return ("xtb.exe", "xtb") if os.name == "nt" else ("xtb",)
    if key == "qchem":
        return ("qchem",)
    if key == "psi4":
        return ("psi4",)
    return (key,)


def search_qm_executable_anywhere(
    code_key: str,
    search_roots: Optional[List[str]] = None,
    max_depth: int = 6,
    timeout_sec: float = 5.0,
) -> List[str]:
    """Search common system and user directories for a QM executable.

    Performs fast probe checks first, then bounded-depth directory traversal
    across user home and standard Linux/macOS/Windows install roots while
    pruning large non-binary trees (.git, .venv, node_modules, .cache, etc.).
    """
    key = _normalize_code_key(code_key)
    bin_names = _target_binary_names(key)
    found: List[str] = []
    seen_paths: Set[str] = set()

    def _record(path: str) -> None:
        norm = os.path.normpath(os.path.abspath(path))
        if norm in seen_paths or not _check_binary(norm):
            return
        if key == "pyscf":
            ok, _ = _check_python_pyscf(norm)
            if not ok:
                return
        seen_paths.add(norm)
        found.append(norm)

    # 1. Fast known-location probes
    for cand in _probe_paths(key):
        _record(cand)

    if key == "pyscf" and found:
        return found

    # 2. Fast system locate index if available on Linux/macOS (skip for generic python/pyscf)
    if key != "pyscf" and os.name != "nt" and shutil.which("locate"):
        for bname in bin_names:
            try:
                res = subprocess.run(
                    ["locate", "-l", "15", "-b", f"\\{bname}"],
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        _record(line.strip())
            except Exception:
                logging.getLogger(__name__).debug("locate failed for %s", bname, exc_info=True)

    # 3. Bounded-depth filesystem walk across user/system roots
    if search_roots is None:
        if os.name == "nt":
            search_roots = [os.path.expanduser("~"), "C:/", "D:/", "C:/Program Files", "C:/Program Files (x86)"]
        else:
            search_roots = [
                os.path.expanduser("~"),
                "/opt",
                "/usr/local",
                "/software",
                "/apps",
                "/usr/bin",
            ]

    prune_dirs = {
        ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules",
        ".cache", ".mozilla", ".thunderbird", ".config", ".npm", ".cargo",
        "Trash", "proc", "sys", "dev", "run", "tmp", "var",
    }
    visited_dirs: Set[Tuple[int, int]] = set()
    deadline = time.monotonic() + max(0.5, timeout_sec)
    bin_set = set(bin_names)

    def _walk(dir_path: str, depth: int) -> None:
        if depth > max_depth or time.monotonic() > deadline:
            return
        try:
            st = os.stat(dir_path)
            dir_id = (st.st_dev, st.st_ino)
            if dir_id in visited_dirs:
                return
            visited_dirs.add(dir_id)
        except OSError:
            return

        try:
            with os.scandir(dir_path) as it:
                subdirs: List[str] = []
                for entry in it:
                    if time.monotonic() > deadline:
                        return
                    try:
                        if entry.name in bin_set and entry.is_file():
                            _record(entry.path)
                        elif entry.is_dir(follow_symlinks=False):
                            if entry.name not in prune_dirs and not entry.name.startswith(".local/share/Trash"):
                                subdirs.append(entry.path)
                    except OSError:
                        continue
                # Prioritize directories whose name hints at the QM package
                subdirs.sort(key=lambda p: (0 if key in os.path.basename(p).lower() else 1, os.path.basename(p).lower()))
                for sub in subdirs:
                    if time.monotonic() > deadline:
                        return
                    _walk(sub, depth + 1)
        except OSError:
            return

    for root in search_roots:
        if time.monotonic() > deadline:
            break
        expanded = os.path.expanduser(root)
        if os.path.isdir(expanded):
            _walk(expanded, 0)

    # Sort newest/versioned paths first
    found.sort(reverse=True)
    return found


def _resolve_custom_route(route_to_try: str, key: str) -> Tuple[str, str]:
    """Resolve a user-supplied route (file path, directory, or command with args).

    Returns (executable_token_or_path, normalized_route_string).
    """
    expanded = os.path.expanduser(route_to_try.strip())
    bin_names = _target_binary_names(key)

    # If the user pasted or browsed a directory (e.g. /opt/orca_6_0_1), look for binary inside
    if os.path.isdir(expanded):
        for b in bin_names:
            for cand in (os.path.join(expanded, b), os.path.join(expanded, "bin", b)):
                if os.path.isfile(cand):
                    return cand, cand
        return os.path.join(expanded, bin_names[0]), expanded

    # If the entire string is an existing file path (handles spaces without quotes)
    if os.path.isfile(expanded):
        return expanded, expanded

    # Otherwise parse as shell command (e.g. "mpirun -np 4 /opt/orca/orca")
    try:
        tokens = shlex.split(route_to_try)
    except ValueError:
        tokens = route_to_try.split()
    first_token = os.path.expanduser(tokens[0]) if tokens else expanded
    return first_token, route_to_try


def detect_qm_executable(code_key: str, custom_route: Optional[str] = None) -> QMExecutableInfo:
    """
    Detect whether the requested QM package executable is available.
    
    If `custom_route` is specified, it tests that route/path first.
    Otherwise checks saved QSettings, then system PATH, then default install paths.
    """
    code_key = code_key.lower()
    
    # Map common aliases
    if "orca" in code_key:
        key = "orca"
        disp = "ORCA"
        bin_name = "orca"
    elif "gaussian" in code_key or "g16" in code_key or "g09" in code_key:
        key = "gaussian"
        disp = "Gaussian"
        bin_name = "g16 / g09"
    elif "pyscf" in code_key:
        key = "pyscf"
        disp = "PySCF"
        bin_name = "python (with PySCF)"
    elif "openlowdin" in code_key or "lowdin" in code_key or "apmo" in code_key:
        key = "openlowdin"
        disp = "openLOWDIN"
        bin_name = "openlowdin"
    elif "xtb" in code_key:
        key = "xtb"
        disp = "xTB"
        bin_name = "xtb"
    elif "qchem" in code_key or "q-chem" in code_key:
        key = "qchem"
        disp = "Q-Chem"
        bin_name = "qchem"
    elif "psi4" in code_key:
        key = "psi4"
        disp = "Psi4"
        bin_name = "psi4"
    else:
        key = code_key
        disp = code_key.capitalize()
        bin_name = code_key

    # Retrieve saved route if no custom route was explicitly passed
    from .session_state import get_settings
    settings = get_settings("QMExecutableRoutes")
    route_to_try = (custom_route or "").strip() or settings.value(f"route_{key}", "")

    # 1. Test custom or stored route
    if route_to_try:
        first_token, normalized_route = _resolve_custom_route(route_to_try, key)
        resolved_bin = shutil.which(first_token) or (
            first_token if (os.path.isabs(first_token) or os.path.isfile(first_token)) else None
        )

        if key == "pyscf":
            ok, ver = _check_python_pyscf(first_token)
            if ok:
                return QMExecutableInfo(
                    code_key=key,
                    display_name=disp,
                    binary_name=bin_name,
                    found=True,
                    resolved_path=normalized_route,
                    version_str=ver,
                    message=f"✔ Found PySCF ({ver}) at: {normalized_route}",
                )
            elif _check_binary(first_token):
                return QMExecutableInfo(
                    code_key=key,
                    display_name=disp,
                    binary_name=bin_name,
                    found=False,
                    resolved_path=normalized_route,
                    version_str="",
                    message=f"⚠️ Python binary exists at {first_token}, but 'import pyscf' failed.",
                )
        elif resolved_bin and _check_binary(resolved_bin):
            return QMExecutableInfo(
                code_key=key,
                display_name=disp,
                binary_name=bin_name,
                found=True,
                resolved_path=normalized_route,
                version_str="",
                message=f"✔ Configured route executable found: {normalized_route}",
            )
        elif resolved_bin and os.path.isfile(resolved_bin):
            return QMExecutableInfo(
                code_key=key,
                display_name=disp,
                binary_name=bin_name,
                found=False,
                resolved_path=normalized_route,
                version_str="",
                message=f"⚠️ File exists at '{resolved_bin}' but is not executable. Run: chmod +x \"{resolved_bin}\"",
            )
        elif shutil.which(first_token):
            return QMExecutableInfo(
                code_key=key,
                display_name=disp,
                binary_name=bin_name,
                found=True,
                resolved_path=normalized_route,
                version_str="",
                message=f"✔ Found command in PATH: {normalized_route}",
            )
        else:
            return QMExecutableInfo(
                code_key=key,
                display_name=disp,
                binary_name=bin_name,
                found=False,
                resolved_path=normalized_route,
                version_str="",
                message=f"⚠️ Specified route or binary not found: '{route_to_try}'",
            )

    # 2. Automated probe in known paths and PATH
    probes = _probe_paths(key)
    for cand in probes:
        if not cand:
            continue
        if key == "pyscf":
            ok, ver = _check_python_pyscf(cand)
            if ok:
                return QMExecutableInfo(
                    code_key=key,
                    display_name=disp,
                    binary_name=bin_name,
                    found=True,
                    resolved_path=cand,
                    version_str=ver,
                    message=f"✔ Found {ver} at: {cand}",
                )
        elif _check_binary(cand):
            # Try to grab version if orca or g16
            ver_str = ""
            if key == "orca":
                ver_str = "ORCA"
                m = re.search(r"orca[_-]?(\d+)[_.](\d+)(?:[_.](\d+))?", cand, re.IGNORECASE)
                if m:
                    ver_str = "ORCA " + ".".join(g for g in m.groups() if g)
            elif key == "gaussian":
                ver_str = "Gaussian 16" if "g16" in cand else "Gaussian 09"
            return QMExecutableInfo(
                code_key=key,
                display_name=disp,
                binary_name=bin_name,
                found=True,
                resolved_path=cand,
                version_str=ver_str,
                message=f"✔ Found {disp} executable at: {cand}",
            )

    # 3. Not found
    return QMExecutableInfo(
        code_key=key,
        display_name=disp,
        binary_name=bin_name,
        found=False,
        resolved_path="",
        version_str="",
        message=f"⚠️ {disp} executable ('{bin_name}') not found in PATH or standard directories. Please enter a custom path or execution command below.",
    )


def save_qm_route(code_key: str, route: str):
    """Persist user-specified command route for a QM code in QSettings."""
    from .session_state import get_settings
    key = _normalize_code_key(code_key)
    settings = get_settings("QMExecutableRoutes")
    settings.setValue(f"route_{key}", route.strip())


def get_qm_route(code_key: str) -> str:
    """Retrieve persisted command route for a QM code."""
    from .session_state import get_settings
    key = _normalize_code_key(code_key)
    settings = get_settings("QMExecutableRoutes")
    return str(settings.value(f"route_{key}", "") or "")


def download_or_install_xtb_binary() -> Optional[str]:
    """
    Attempt to provision the Grimme GFN2-xTB binary into ~/.agymolden/bin:
      1. Checks if xtb is already available in PATH or Conda/Mamba.
      2. If Conda/Mamba is available, attempts fast environment lookup.
      3. Otherwise downloads the official static xTB release tarball from GitHub Releases
         into ~/.agymolden/bin/ and returns the executable path.
    """
    existing = detect_qm_executable("xtb")
    if existing.found and existing.resolved_path:
        return existing.resolved_path

    dest_dir = os.path.expanduser("~/.agymolden/bin")
    os.makedirs(dest_dir, exist_ok=True)

    # Check if Linux x86_64 static release can be fetched
    if sys.platform.startswith("linux"):
        import urllib.request
        import tarfile
        url = "https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz"
        archive_path = os.path.join(dest_dir, "xtb-linux.tar.xz")
        try:
            urllib.request.urlretrieve(url, archive_path)
            with tarfile.open(archive_path, "r:xz") as tar:
                tar.extractall(path=dest_dir)
            for cand in glob.glob(os.path.join(dest_dir, "xtb*", "bin", "xtb")):
                if os.path.isfile(cand):
                    os.chmod(cand, 0o755)
                    save_qm_route("xtb", cand)
                    return cand
        except Exception:
            logging.getLogger(__name__).debug("Ignored exception", exc_info=True)
    return None


class QMEnginesStatusDialog(QtCore.QObject):
    """Helper factory for the all-in-one QM Engines Detection & Configuration Dialog."""

    @staticmethod
    def show_dialog(parent=None):
        from PyQt6 import QtGui, QtWidgets

        dlg = QtWidgets.QDialog(parent)
        dlg.setWindowTitle("Quantum Chemistry Engines & Executable Routes")
        dlg.resize(760, 420)
        vbox = QtWidgets.QVBoxLayout(dlg)
        vbox.setSpacing(10)

        lbl = QtWidgets.QLabel(
            "<b>Detected Quantum Chemistry Packages & Execution Routes</b><br>"
            "<span style='color: #64748b; font-size: 11px;'>"
            "AgyMolden automatically probes your PATH, Conda environments, and standard install folders. "
            "You can override any executable path or command route below."
            "</span>"
        )
        lbl.setWordWrap(True)
        vbox.addWidget(lbl)

        codes = ["orca", "pyscf", "gaussian", "psi4", "qchem", "xtb", "openlowdin"]
        tbl = QtWidgets.QTableWidget(len(codes), 4)
        tbl.setHorizontalHeaderLabels(["QM Package", "Status", "Executable Path / Command Route", "Browse"])
        tbl.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        tbl.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        tbl.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        tbl.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        tbl.verticalHeader().setVisible(False)
        vbox.addWidget(tbl, stretch=1)

        route_edits: Dict[str, QtWidgets.QLineEdit] = {}
        status_items: Dict[str, QtWidgets.QTableWidgetItem] = {}

        def _refresh_row(row_idx: int, c_key: str):
            ed = route_edits[c_key]
            custom_txt = ed.text().strip()
            info = detect_qm_executable(c_key, custom_route=custom_txt if custom_txt else None)
            st_item = status_items[c_key]
            if info.found:
                st_item.setText("✔ Found")
                st_item.setForeground(QtGui.QBrush(QtGui.QColor("#10b981")))
            else:
                st_item.setText("⚠ Not Found")
                st_item.setForeground(QtGui.QBrush(QtGui.QColor("#f59e0b")))
            st_item.setToolTip(info.message)
            if not custom_txt and info.resolved_path:
                ed.setPlaceholderText(info.resolved_path)

        for r, key in enumerate(codes):
            info = detect_qm_executable(key)
            name_item = QtWidgets.QTableWidgetItem(info.display_name)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            tbl.setItem(r, 0, name_item)

            st_item = QtWidgets.QTableWidgetItem("")
            st_item.setFlags(st_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            tbl.setItem(r, 1, st_item)
            status_items[key] = st_item

            ed = QtWidgets.QLineEdit(get_qm_route(key))
            ed.setPlaceholderText(info.resolved_path or f"Auto-detect ({info.binary_name})")
            ed.textChanged.connect(lambda _txt, r_i=r, k=key: _refresh_row(r_i, k))
            tbl.setCellWidget(r, 2, ed)
            route_edits[key] = ed

            btn_b = QtWidgets.QPushButton("Browse...")
            def _on_browse(_checked=False, r_i=r, k=key):
                p, _ = QtWidgets.QFileDialog.getOpenFileName(dlg, f"Select {k.upper()} Executable")
                if p:
                    route_edits[k].setText(p)
                    _refresh_row(r_i, k)
            btn_b.clicked.connect(_on_browse)
            tbl.setCellWidget(r, 3, btn_b)
            _refresh_row(r, key)

        # Quick 1-Click Auto-Provision & System Search Bar
        auto_row = QtWidgets.QHBoxLayout()
        btn_search_sys = QtWidgets.QPushButton("🔍 Search System for Missing Engines")
        btn_search_sys.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 5px 10px; border-radius: 4px;")
        btn_search_sys.setToolTip("Scan your home directory, /opt, /usr/local, and system folders for QM executables")

        def _on_search_system():
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            found_any = []
            try:
                for idx_c, c_key in enumerate(codes):
                    cur_info = detect_qm_executable(c_key, custom_route=route_edits[c_key].text().strip() or None)
                    if not cur_info.found:
                        matches = search_qm_executable_anywhere(c_key, timeout_sec=4.0)
                        if matches:
                            route_edits[c_key].setText(matches[0])
                            _refresh_row(idx_c, c_key)
                            found_any.append(f"{cur_info.display_name}: {matches[0]}")
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

            if found_any:
                QtWidgets.QMessageBox.information(
                    dlg,
                    "Engines Found",
                    "Discovered and populated:\n\n" + "\n".join(found_any) + "\n\nClick Save to remember these paths.",
                )
            else:
                QtWidgets.QMessageBox.information(
                    dlg,
                    "Search Complete",
                    "No additional missing QM executables were found in standard or home directories.\n"
                    "You can paste a custom path or click 'Browse...' to select the executable directly.",
                )

        btn_search_sys.clicked.connect(_on_search_system)
        auto_row.addWidget(btn_search_sys)

        btn_auto_xtb = QtWidgets.QPushButton("⬇️ 1-Click Install GFN2-xTB (~/.agymolden/bin)")
        btn_auto_xtb.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 5px 10px; border-radius: 4px;")
        btn_auto_xtb.setToolTip("Automatically download or provision the standalone Grimme GFN2-xTB binary into ~/.agymolden/bin")

        def _on_auto_xtb():
            target_bin = download_or_install_xtb_binary()
            if target_bin:
                route_edits["xtb"].setText(target_bin)
                _refresh_row(codes.index("xtb"), "xtb")
                QtWidgets.QMessageBox.information(dlg, "GFN2-xTB Installed", f"Successfully provisioned xTB at:\n{target_bin}")
            else:
                QtWidgets.QMessageBox.warning(
                    dlg,
                    "xTB Provisioning",
                    "Could not auto-download xTB (check internet connection or install via 'conda install -c conda-forge xtb').\n"
                    "Note: AgyMolden's built-in Python GFN0-xTB engine is always active with zero external binaries!",
                )

        btn_auto_xtb.clicked.connect(_on_auto_xtb)
        auto_row.addWidget(btn_auto_xtb)
        auto_row.addStretch()
        vbox.addLayout(auto_row)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        def _on_save():
            for k, ed in route_edits.items():
                save_qm_route(k, ed.text().strip())
            if parent is not None and hasattr(parent, "_update_env_badge"):
                parent._update_env_badge()
            dlg.accept()
        btns.accepted.connect(_on_save)
        btns.rejected.connect(dlg.reject)
        vbox.addWidget(btns)
        dlg.exec()

