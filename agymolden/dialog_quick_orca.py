"""
Quick ORCA Calculation Dialog for undergraduate students and standard lab exercises.
Provides 1-click execution for standard calculations (Optimization, IR/Vibrations, Orbitals)
with automatic coordinate generation, charge/multiplicity defaults, and live 3D monitoring.
"""

import os
from typing import Optional, List, Tuple
from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np

from agymolden.zmatrix import ZAtom


COURSE_PRESETS = [
    {
        "name": "⚡ Geometry Optimization (B3LYP / def2-SVP)",
        "route": "! B3LYP def2-SVP Opt",
        "desc": "Standard ground-state geometry optimization. Finds the minimum energy structure.",
        "icon": "📐"
    },
    {
        "name": "🎵 Vibrations & IR Spectrum (B3LYP / def2-SVP Freq)",
        "route": "! B3LYP def2-SVP Opt Freq",
        "desc": "Optimizes geometry and computes harmonic vibrational frequencies and IR intensities.",
        "icon": "🎵"
    },
    {
        "name": "⚡ Transition State Search (OptTS + NumFreq)",
        "route": "! B3LYP def2-SVP OptTS NumFreq",
        "desc": "Optimizes to a first-order saddle point transition state and calculates frequencies to confirm 1 imaginary mode.",
        "icon": "⚡"
    },
    {
        "name": "🌐 Molecular Orbitals / HOMO-LUMO (B3LYP / def2-SVP SP)",
        "route": "! B3LYP def2-SVP SP",
        "desc": "Single point energy to calculate molecular orbital energies, HOMO-LUMO gap, and dipole moment.",
        "icon": "🌐"
    },
    {
        "name": "🌌 Anions & Diffuse Orbitals (ma-def2-SVP)",
        "route": "! B3LYP ma-def2-SVP Opt",
        "desc": "Minimally augmented diffuse basis set for anions, Rydberg states, and bound positron species.",
        "icon": "🌌"
    },
    {
        "name": "⚡ Fast Semi-empirical Tight-Binding (GFN2-xTB)",
        "route": "! GFN2-xTB Opt",
        "desc": "Extremely fast geometry optimization for large molecules using Grimme's xTB method.",
        "icon": "⚡"
    },
    {
        "name": "🛠️ Custom Route Line",
        "route": "! B3LYP 6-31G* Opt",
        "desc": "Specify your own custom ORCA keywords (e.g. ! wB97X-D3 def2-TZVP).",
        "icon": "✏️"
    }
]


class QuickOrcaDialog(QtWidgets.QDialog):
    """Simplified modal dialog for running ORCA calculations in course labs."""

    def __init__(self, parent: QtWidgets.QWidget, zatoms: List[ZAtom], coords: Optional[np.ndarray] = None):
        super().__init__(parent)
        self.setWindowTitle("🚀 Quick Run ORCA Calculation")
        self.resize(520, 420)
        self.zatoms = zatoms
        self.coords = coords

        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(14)

        # Header description
        info_box = QtWidgets.QLabel(
            "<b>Run ORCA Quantum Chemistry Calculation</b><br>"
            "<span style='color: #64748b; font-size: 11px;'>"
            "Select a course lab preset below. AgyMolden will generate the input file, "
            "run ORCA in the background, and update the 3D molecule live as it optimizes.</span>"
        )
        info_box.setWordWrap(True)
        layout.addWidget(info_box)

        # Preset selection
        grp_preset = QtWidgets.QGroupBox("Lab Calculation Preset")
        p_layout = QtWidgets.QVBoxLayout(grp_preset)
        self.combo_preset = QtWidgets.QComboBox()
        for p in COURSE_PRESETS:
            self.combo_preset.addItem(f"{p['icon']} {p['name']}", p)
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        p_layout.addWidget(self.combo_preset)

        self.lbl_preset_desc = QtWidgets.QLabel(COURSE_PRESETS[0]["desc"])
        self.lbl_preset_desc.setStyleSheet("color: #475569; font-style: italic; font-size: 11px;")
        self.lbl_preset_desc.setWordWrap(True)
        p_layout.addWidget(self.lbl_preset_desc)

        # Route editor (editable)
        route_row = QtWidgets.QHBoxLayout()
        route_row.addWidget(QtWidgets.QLabel("ORCA Route:"))
        self.txt_route = QtWidgets.QLineEdit(COURSE_PRESETS[0]["route"])
        route_row.addWidget(self.txt_route, stretch=1)
        p_layout.addLayout(route_row)

        layout.addWidget(grp_preset)

        # Molecular Charge & Multiplicity
        grp_mol = QtWidgets.QGroupBox("Molecule State & Hardware")
        m_form = QtWidgets.QFormLayout(grp_mol)
        m_form.setSpacing(8)

        charge_mult_row = QtWidgets.QHBoxLayout()
        self.spn_charge = QtWidgets.QSpinBox()
        self.spn_charge.setRange(-10, 10)
        self.spn_charge.setValue(0)
        charge_mult_row.addWidget(QtWidgets.QLabel("Charge:"))
        charge_mult_row.addWidget(self.spn_charge)

        self.spn_mult = QtWidgets.QSpinBox()
        self.spn_mult.setRange(1, 10)
        self.spn_mult.setValue(1)
        charge_mult_row.addWidget(QtWidgets.QLabel("Multiplicity:"))
        charge_mult_row.addWidget(self.spn_mult)
        charge_mult_row.addStretch()

        m_form.addRow("Electronic State:", charge_mult_row)

        # Parity validation feedback label
        self.lbl_parity_warning = QtWidgets.QLabel("")
        self.lbl_parity_warning.setWordWrap(True)
        m_form.addRow("Spin Parity:", self.lbl_parity_warning)

        self.spn_charge.valueChanged.connect(self._validate_charge_multiplicity)
        self.spn_mult.valueChanged.connect(self._validate_charge_multiplicity)

        # Processors and memory
        hw_row = QtWidgets.QHBoxLayout()
        self.spn_cores = QtWidgets.QSpinBox()
        self.spn_cores.setRange(1, 64)
        self.spn_cores.setValue(1)  # Default 1 core for safe student execution without MPI errors
        hw_row.addWidget(QtWidgets.QLabel("Cores (%pal):"))
        hw_row.addWidget(self.spn_cores)

        self.spn_mem = QtWidgets.QSpinBox()
        self.spn_mem.setRange(500, 32000)
        self.spn_mem.setSingleStep(500)
        self.spn_mem.setValue(2000)
        self.spn_mem.setSuffix(" MB")
        hw_row.addWidget(QtWidgets.QLabel("Memory per core:"))
        hw_row.addWidget(self.spn_mem)
        hw_row.addStretch()

        m_form.addRow("Hardware Resources:", hw_row)

        # Job Name & Working Directory
        curr_dir = os.path.abspath(".")
        self.txt_filename = QtWidgets.QLineEdit("orca_calc.inp")
        m_form.addRow("Input File Name:", self.txt_filename)

        dir_row = QtWidgets.QHBoxLayout()
        self.txt_dir = QtWidgets.QLineEdit(curr_dir)
        dir_row.addWidget(self.txt_dir, stretch=1)
        btn_browse = QtWidgets.QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse_dir)
        dir_row.addWidget(btn_browse)
        m_form.addRow("Working Directory:", dir_row)

        layout.addWidget(grp_mol)

        # Dialog Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch()

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        self.btn_run = QtWidgets.QPushButton("🚀 Run ORCA Now")
        self.btn_run.setStyleSheet(
            "background-color: #2563eb; color: white; font-weight: bold; font-size: 13px; padding: 6px 16px; border-radius: 4px;"
        )
        self.btn_run.clicked.connect(self._on_run_clicked)
        btn_box.addWidget(self.btn_run)

        layout.addLayout(btn_box)

        # Validate charge and spin multiplicity at start
        self._validate_charge_multiplicity()

    def _validate_charge_multiplicity(self):
        """Validates that electron count parity matches the selected spin multiplicity."""
        if not self.zatoms:
            self.lbl_parity_warning.setText("<span style='color: #64748b;'>No molecule loaded</span>")
            return

        from .elements import ELEMENT_DATA
        total_atomic_number = 0
        for a in self.zatoms:
            sym = getattr(a, "symbol", "") or ""
            data = ELEMENT_DATA.get(sym.capitalize(), {})
            z_num = data.get("atomic_number", 0)
            total_atomic_number += z_num

        charge = self.spn_charge.value()
        mult = self.spn_mult.value()
        num_electrons = total_atomic_number - charge

        # Spin parity rule:
        # If N_electrons is even, (mult - 1) must be even -> mult must be odd (Singlet, Triplet, ...)
        # If N_electrons is odd, (mult - 1) must be odd -> mult must be even (Doublet, Quartet, ...)
        is_even_e = (num_electrons % 2 == 0)
        is_even_mult = (mult % 2 == 0)

        if (is_even_e and is_even_mult) or (not is_even_e and not is_even_mult):
            expected = "odd (Singlet=1, Triplet=3, ...)" if is_even_e else "even (Doublet=2, Quartet=4, ...)"
            self.lbl_parity_warning.setText(
                f"<span style='color: #dc2626; font-weight: bold;'>⚠️ Spin mismatch:</span> "
                f"<span style='color: #dc2626;'>{num_electrons} e⁻ requires {expected}. ORCA will fail.</span>"
            )
        else:
            self.lbl_parity_warning.setText(
                f"<span style='color: #16a34a; font-weight: bold;'>✓ Valid state</span> "
                f"<span style='color: #16a34a;'>({num_electrons} e⁻, mult={mult})</span>"
            )

    def _on_preset_changed(self, idx: int):
        data = self.combo_preset.currentData()
        if data:
            self.txt_route.setText(data["route"])
            self.lbl_preset_desc.setText(data["desc"])

    def _on_browse_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Working Directory", self.txt_dir.text())
        if d:
            self.txt_dir.setText(d)

    def _on_run_clicked(self):
        if not self.zatoms:
            QtWidgets.QMessageBox.warning(self, "No Molecule", "Please build or load a molecule first before running ORCA.")
            return

        route_line = self.txt_route.text().strip()
        if not route_line.startswith("!"):
            route_line = "! " + route_line

        fname = self.txt_filename.text().strip()
        if not fname.endswith(".inp"):
            fname += ".inp"

        target_dir = self.txt_dir.text().strip() or os.path.abspath(".")
        target_path = os.path.join(target_dir, fname)

        # Assemble coordinates
        lines = [f"{route_line}"]
        nprocs = self.spn_cores.value()
        if nprocs > 1:
            lines.append(f"%pal nprocs {nprocs} end")
        lines.append(f"%maxcore {self.spn_mem.value()}")
        lines.append("")
        lines.append(f"* xyz {self.spn_charge.value()} {self.spn_mult.value()}")

        # Coordinates from array or ZAtom definitions
        if self.coords is not None and len(self.coords) == len(self.zatoms):
            for atom, pos in zip(self.zatoms, self.coords):
                lines.append(f"  {atom.symbol:<2}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}")
        else:
            for atom in self.zatoms:
                lines.append(f"  {atom.symbol:<2}  {atom.x:12.6f}  {atom.y:12.6f}  {atom.z:12.6f}")
        lines.append("*\n")

        content = "\n".join(lines)
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Could not write ORCA input file:\n{e}")
            return

        self.saved_input_path = target_path
        self.accept()
