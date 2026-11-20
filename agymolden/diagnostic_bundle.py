"""
Diagnostic & Telemetry Bundle Exporter for AgyMolden.

Packages environment diagnostics, detected QM binaries, sanitized QSettings,
and agymolden.log into a single .zip archive for troubleshooting and support.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import zipfile
from typing import Any, Dict

_SENSITIVE_KEY_TOKENS = ("password", "passwd", "secret", "token", "api_key", "private_key")


def _sanitize_value(key: str, val: Any) -> Any:
    """Redact sensitive values while keeping non-secret diagnostics readable."""
    low = key.lower()
    if any(tok in low for tok in _SENSITIVE_KEY_TOKENS):
        return "<REDACTED>" if val else ""
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if isinstance(val, (list, tuple)):
        return [_sanitize_value(key, v) for v in val]
    if isinstance(val, dict):
        return {str(k): _sanitize_value(str(k), v) for k, v in val.items()}
    return str(val)


def collect_environment_info() -> Dict[str, Any]:
    """Collect non-sensitive system, library, and QM executable diagnostics."""
    from . import __version__
    from .qm_detect import detect_qm_executable

    qt_ver = "unknown"
    try:
        from PyQt6.QtCore import PYQT_VERSION_STR
        qt_ver = PYQT_VERSION_STR
    except Exception:
        pass

    optional_libs = {
        pkg: (importlib.util.find_spec(pkg) is not None)
        for pkg in ("numpy", "scipy", "matplotlib", "skimage", "cclib", "rdkit", "pyscf", "spglib", "ase")
    }

    qm_engines = {}
    for code in ("orca", "gaussian", "pyscf", "xtb", "openlowdin", "qchem", "psi4"):
        info = detect_qm_executable(code)
        qm_engines[code] = {
            "found": info.found,
            "resolved_path": info.resolved_path,
            "version": info.version_str,
        }

    return {
        "agymolden_version": __version__,
        "platform": sys.platform,
        "python_version": sys.version,
        "pyqt6_version": qt_ver,
        "optional_libraries": optional_libs,
        "qm_engines": qm_engines,
        "path_has_orca": shutil.which("orca") is not None,
    }


def collect_sanitized_settings() -> Dict[str, Dict[str, Any]]:
    """Read AgyMolden QSettings groups with secret fields redacted."""
    from .session_state import get_settings

    out: Dict[str, Dict[str, Any]] = {}
    for app_name in ("AgyMolden", "Preferences", "QMExecutableRoutes", "RemoteProfiles"):
        try:
            s = get_settings(app_name)
            entries: Dict[str, Any] = {}
            for k in s.allKeys():
                entries[k] = _sanitize_value(k, s.value(k))
            out[app_name] = entries
        except Exception as exc:
            out[app_name] = {"_error": str(exc)}
    return out


def export_diagnostic_bundle(output_zip_path: str) -> str:
    """Write an atomic diagnostic .zip bundle to *output_zip_path* and return its path."""
    from . import _log_file_path

    out_abs = os.path.abspath(output_zip_path)
    parent_dir = os.path.dirname(out_abs)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    tmp_path = out_abs + ".tmp"
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("environment.json", json.dumps(collect_environment_info(), indent=2))
            zf.writestr("settings_keys.json", json.dumps(collect_sanitized_settings(), indent=2))
            log_path = _log_file_path()
            if os.path.isfile(log_path):
                zf.write(log_path, arcname="agymolden.log")
            else:
                zf.writestr("agymolden.log", "(No log file created yet)\n")
        os.replace(tmp_path, out_abs)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    return out_abs
