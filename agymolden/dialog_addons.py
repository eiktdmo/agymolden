"""In-App Extension and Dependency Manager for AgyMolden.

Allows users to inspect installed scientific backends (PySCF, RDKit, ASE, Spglib,
TorchANI, cclib, etc.), view versions, and install or update them directly
from inside the application with 1-click background package managers.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets


BACKEND_CATALOG = [
    {
        "id": "pyscf",
        "name": "PySCF",
        "category": "Quantum Chemistry",
        "description": "In-memory ab initio & DFT calculations (B3LYP, HF, MP2) directly in RAM with zero temporary file pollution.",
        "module": "pyscf",
        "pip_pkg": "pyscf",
        "platforms": ["linux", "darwin"],
    },
    {
        "id": "rdkit",
        "name": "RDKit",
        "category": "Cheminformatics & UFF",
        "description": "Force-field minimization (UFF / MMFF94), SMILES parsing, 2D depictions, and 3D conformer generation.",
        "module": "rdkit",
        "pip_pkg": "rdkit",
        "platforms": ["linux", "darwin", "win32"],
    },
    {
        "id": "ase",
        "name": "ASE (Atomic Simulation Environment)",
        "category": "Solid-State & NEB",
        "description": "Nudged Elastic Band (NEB) reaction barrier calculations, trajectory manipulations, and calculator interfaces.",
        "module": "ase",
        "pip_pkg": "ase",
        "platforms": ["linux", "darwin", "win32"],
    },
    {
        "id": "spglib",
        "name": "Spglib",
        "category": "Crystallography & PBC",
        "description": "Symmetry operations, Space Group detection, Wyckoff positions, and CIF unit cell expansions.",
        "module": "spglib",
        "pip_pkg": "spglib",
        "platforms": ["linux", "darwin", "win32"],
    },
    {
        "id": "torchani",
        "name": "TorchANI",
        "category": "Machine-Learned Potentials",
        "description": "ANI-1x/ANI-2x neural network potentials for near-DFT accuracy molecular dynamics and millisecond optimizations.",
        "module": "torchani",
        "pip_pkg": "torchani",
        "platforms": ["linux", "darwin", "win32"],
    },
    {
        "id": "cclib",
        "name": "cclib",
        "category": "Parser & Analytics",
        "description": "Multi-engine computational chemistry log parser (Gaussian, ORCA, Q-Chem, NWChem, Molpro).",
        "module": "cclib",
        "pip_pkg": "cclib",
        "platforms": ["linux", "darwin", "win32"],
    },
    {
        "id": "skimage",
        "name": "scikit-image",
        "category": "Surface Generation",
        "description": "Marching cubes isosurface triangulation for molecular orbitals, electron density, and electrostatic potential maps.",
        "module": "skimage",
        "pip_pkg": "scikit-image",
        "platforms": ["linux", "darwin", "win32"],
    },
    {
        "id": "basis_set_exchange",
        "name": "Basis Set Exchange",
        "category": "Basis Sets & QC",
        "description": "Standardized basis set definitions for hundreds of Gaussian and ORCA basis libraries.",
        "module": "basis_set_exchange",
        "pip_pkg": "basis_set_exchange",
        "platforms": ["linux", "darwin", "win32"],
    },
]


def check_module_status(module_name: str) -> Tuple[bool, str]:
    """Check if a Python package is importable and retrieve its version."""
    try:
        mod = importlib.import_module(module_name)
        ver = getattr(mod, "__version__", "Installed")
        return True, str(ver)
    except Exception:
        return False, "Not Installed"


class ExtensionInstallThread(QtCore.QThread):
    """Background worker for installing or upgrading scientific packages via pip."""
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(bool, str)

    def __init__(self, package_name: str, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.package_name = package_name

    def run(self):
        py_exe = sys.executable
        cmd = [py_exe, "-m", "pip", "install", "--upgrade", self.package_name]
        self.log_signal.emit(f"🚀 Running: {' '.join(cmd)}\n")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                self.log_signal.emit(line)
            proc.wait()

            if proc.returncode == 0:
                self.finished_signal.emit(True, f"✔ Successfully installed {self.package_name}!")
            else:
                self.finished_signal.emit(False, f"❌ Installation failed with code {proc.returncode}.")
        except Exception as e:
            self.finished_signal.emit(False, f"❌ Execution error: {e}")


class ExtensionManagerDialog(QtWidgets.QDialog):
    """Modern interactive Extension & Scientific Backend Manager."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("🧩 Add-on & Scientific Backend Manager — AgyMolden")
        self.resize(880, 580)
        self.setMinimumSize(750, 480)
        self._thread: Optional[ExtensionInstallThread] = None

        self._build_ui()
        self.refresh_table()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        # Header Info Banner
        header = QtWidgets.QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        h_layout = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>Scientific Backend & Extension Manager</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Manage optional computational chemistry engines and accelerated plugins directly from this environment.<br>"
            "Zero command-line manual steps required."
        )
        h_desc.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        h_desc.setWordWrap(True)
        h_layout.addWidget(h_title)
        h_layout.addWidget(h_desc)
        layout.addWidget(header)

        # Table of extensions
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Package", "Category", "Status / Version", "Description", "Action"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(4, 120)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #0f172a;
                color: #f8fafc;
                gridline-color: #334155;
                border: 1px solid #334155;
                border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #1e293b;
                color: #94a3b8;
                padding: 6px;
                font-weight: bold;
                border: 1px solid #334155;
            }
        """)
        layout.addWidget(self.table, 1)

        # Log & Output Console (Collapsible/Visible on install)
        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(130)
        self.txt_log.setStyleSheet("""
            QPlainTextEdit {
                background-color: #020617;
                color: #22c55e;
                font-family: monospace;
                font-size: 11px;
                border: 1px solid #334155;
                border-radius: 4px;
            }
        """)
        self.txt_log.setPlaceholderText("Installation logs will appear here...")
        layout.addWidget(self.txt_log)

        # Bottom Bar
        b_bar = QtWidgets.QHBoxLayout()
        btn_refresh = QtWidgets.QPushButton("🔄 Refresh Status")
        btn_refresh.clicked.connect(self.refresh_table)
        b_bar.addWidget(btn_refresh)

        btn_install_all_qm = QtWidgets.QPushButton("⚡ Install Recommended Suite (PySCF + RDKit + ASE)")
        btn_install_all_qm.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold;")
        btn_install_all_qm.clicked.connect(self._on_install_recommended)
        b_bar.addWidget(btn_install_all_qm)

        b_bar.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        b_bar.addWidget(btn_close)
        layout.addLayout(b_bar)

    def refresh_table(self):
        """Inspect environment and populate table with current backend status."""
        self.table.setRowCount(len(BACKEND_CATALOG))
        for row, info in enumerate(BACKEND_CATALOG):
            installed, ver_str = check_module_status(info["module"])

            # 0. Name
            item_name = QtWidgets.QTableWidgetItem(info["name"])
            item_name.setFont(QtGui.QFont("", -1, QtGui.QFont.Weight.Bold))
            self.table.setItem(row, 0, item_name)

            # 1. Category
            item_cat = QtWidgets.QTableWidgetItem(info["category"])
            self.table.setItem(row, 1, item_cat)

            # 2. Status
            status_txt = f"✔ {ver_str}" if installed else "✖ Not Installed"
            item_status = QtWidgets.QTableWidgetItem(status_txt)
            if installed:
                item_status.setForeground(QtGui.QColor("#4ade80"))
            else:
                item_status.setForeground(QtGui.QColor("#f87171"))
            self.table.setItem(row, 2, item_status)

            # 3. Description
            item_desc = QtWidgets.QTableWidgetItem(info["description"])
            self.table.setItem(row, 3, item_desc)

            # 4. Action Button
            btn = QtWidgets.QPushButton("Upgrade" if installed else "1-Click Install")
            btn.setStyleSheet(
                "background-color: #10b981; color: white; padding: 4px; font-size: 11px;"
                if not installed else
                "background-color: #475569; color: #f1f5f9; padding: 4px; font-size: 11px;"
            )
            pkg = info["pip_pkg"]
            btn.clicked.connect(lambda checked=False, p=pkg: self.start_installation(p))
            self.table.setCellWidget(row, 4, btn)

    def start_installation(self, package_name: str):
        """Launch background installation of requested package."""
        if self._thread and self._thread.isRunning():
            QtWidgets.QMessageBox.warning(self, "Busy", "Another installation is currently running. Please wait.")
            return

        self.txt_log.clear()
        self.txt_log.appendPlainText(f"--- Preparing installation for: {package_name} ---\n")
        self._thread = ExtensionInstallThread(package_name, self)
        self._thread.log_signal.connect(self.txt_log.appendPlainText)
        self._thread.finished_signal.connect(self._on_install_finished)
        self._thread.start()

    def _on_install_recommended(self):
        """Batch install PySCF, RDKit, and ASE in one go."""
        self.start_installation("pyscf rdkit ase")

    def _on_install_finished(self, success: bool, msg: str):
        self.txt_log.appendPlainText(f"\n{msg}\n")
        if success:
            QtWidgets.QMessageBox.information(self, "Installation Complete", msg)
        else:
            QtWidgets.QMessageBox.warning(self, "Installation Warning", msg)
        self.refresh_table()
