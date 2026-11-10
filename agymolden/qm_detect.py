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
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple, NamedTuple
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


def _probe_paths(key: str) -> List[str]:
    """Candidate executables for a QM code: PATH first, then common install locations (no user-specific paths)."""
    if key == "orca":
        return [shutil.which("orca") or ""] + _glob_candidates(
            "~/orca*/orca", "~/programs/orca*/orca", "~/software/orca*/orca",
            "/opt/orca*/orca", "/usr/local/orca*/orca", "/usr/local/bin/orca",
            "C:/orca*/orca.exe", "C:/Program Files/orca*/orca.exe",
        )
    if key == "gaussian":
        return [
            shutil.which("g16") or "",
            shutil.which("g16.exe") or "",
            shutil.which("g09") or "",
            shutil.which("g09.exe") or "",
            shutil.which("g03") or "",
            shutil.which("g03.exe") or "",
        ] + _glob_candidates(
            "~/g16/g16", "~/g09/g09", "~/programs/g16*/g16", "~/programs/gaussian*/g16*/g16",
            "/opt/g16/g16", "/opt/g09/g09", "/usr/local/g16/g16", "/usr/local/g09/g09",
            "C:/G16W/g16.exe", "C:/G09W/g09.exe", "C:/G03W/g03.exe",
            "C:/Program Files/G16W/g16.exe", "C:/Program Files (x86)/G16W/g16.exe",
            "C:/Program Files/G09W/g09.exe", "C:/Program Files (x86)/G09W/g09.exe",
            "C:/gaussian*/g*.exe", "D:/G16W/g16.exe", "D:/G09W/g09.exe",
        )
    if key == "pyscf":
        local_venvs = [
            sys.executable,
            os.path.abspath(".venv/bin/python"),
            os.path.abspath("venv/bin/python"),
            shutil.which("python3") or "",
            shutil.which("python") or "",
        ]
        envs = _glob_candidates("~/.venv/bin/python")
        return [p for p in local_venvs + envs if p]
    if key == "openlowdin":
        return [shutil.which("openlowdin") or "", shutil.which("lowdin") or ""] + _glob_candidates(
            "~/programs/openlowdin*/openlowdin", "/opt/openlowdin*/bin/openlowdin",
        )
    if key == "xtb":
        return [shutil.which("xtb") or ""] + _glob_candidates(
            "~/.agymolden/bin/xtb",
            "~/.agymolden/bin/xtb*/bin/xtb",
            "~/.agymolden/bin/xtb.exe",
            "~/programs/xtb*/bin/xtb", "~/xtb*/bin/xtb", "/opt/xtb*/bin/xtb", "/usr/local/bin/xtb",
        )
    if key == "qchem":
        return [shutil.which("qchem") or ""] + _glob_candidates(
            "~/programs/qchem*/bin/qchem", "/opt/qchem*/bin/qchem",
        )
    if key == "psi4":
        return [shutil.which("psi4") or ""] + _glob_candidates(
            "~/miniconda3/bin/psi4",
            "~/anaconda3/bin/psi4",
            "~/.conda/envs/*/bin/psi4",
            "/usr/local/bin/psi4",
            "/opt/psi4*/bin/psi4",
        )
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
    settings = QtCore.QSettings("AgyMolden", "QMExecutableRoutes")
    route_to_try = (custom_route or "").strip() or settings.value(f"route_{key}", "")

    # 1. Test custom or stored route
    if route_to_try:
        # If route contains arguments (e.g. "mpirun -np 4 /opt/orca/orca" or python path)
        first_token = route_to_try.split()[0]
        resolved_bin = shutil.which(first_token) or (first_token if os.path.isabs(first_token) else None)
        
        if key == "pyscf":
            ok, ver = _check_python_pyscf(first_token)
            if ok:
                return QMExecutableInfo(
                    code_key=key,
                    display_name=disp,
                    binary_name=bin_name,
                    found=True,
                    resolved_path=route_to_try,
                    version_str=ver,
                    message=f"✔ Found PySCF ({ver}) at: {route_to_try}",
                )
            elif _check_binary(first_token):
                return QMExecutableInfo(
                    code_key=key,
                    display_name=disp,
                    binary_name=bin_name,
                    found=False,
                    resolved_path=route_to_try,
                    version_str="",
                    message=f"⚠️ Python binary exists at {first_token}, but 'import pyscf' failed.",
                )
        elif resolved_bin and _check_binary(resolved_bin):
            return QMExecutableInfo(
                code_key=key,
                display_name=disp,
                binary_name=bin_name,
                found=True,
                resolved_path=route_to_try,
                version_str="",
                message=f"✔ Configured route executable found: {route_to_try}",
            )
        elif shutil.which(first_token):
            return QMExecutableInfo(
                code_key=key,
                display_name=disp,
                binary_name=bin_name,
                found=True,
                resolved_path=route_to_try,
                version_str="",
                message=f"✔ Found command in PATH: {route_to_try}",
            )
        else:
            return QMExecutableInfo(
                code_key=key,
                display_name=disp,
                binary_name=bin_name,
                found=False,
                resolved_path=route_to_try,
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
    key = code_key.lower()
    if "orca" in key:
        key = "orca"
    elif "gaussian" in key or "g16" in key or "g09" in key:
        key = "gaussian"
    elif "pyscf" in key:
        key = "pyscf"
    elif "openlowdin" in key or "lowdin" in key or "apmo" in key:
        key = "openlowdin"
    elif "xtb" in key:
        key = "xtb"
    elif "qchem" in key:
        key = "qchem"
    settings = QtCore.QSettings("AgyMolden", "QMExecutableRoutes")
    settings.setValue(f"route_{key}", route.strip())


def get_qm_route(code_key: str) -> str:
    """Retrieve persisted command route for a QM code."""
    key = code_key.lower()
    if "orca" in key:
        key = "orca"
    elif "gaussian" in key or "g16" in key or "g09" in key:
        key = "gaussian"
    elif "pyscf" in key:
        key = "pyscf"
    elif "openlowdin" in key or "lowdin" in key or "apmo" in key:
        key = "openlowdin"
    elif "xtb" in key:
        key = "xtb"
    elif "qchem" in key:
        key = "qchem"
    settings = QtCore.QSettings("AgyMolden", "QMExecutableRoutes")
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
            pass
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

        # Quick 1-Click Auto-Provision Bar for Open-Source Engines (xTB & PySCF)
        auto_row = QtWidgets.QHBoxLayout()
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
            dlg.accept()
        btns.accepted.connect(_on_save)
        btns.rejected.connect(dlg.reject)
        vbox.addWidget(btns)
        dlg.exec()

