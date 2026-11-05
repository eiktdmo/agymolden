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
        self.txt_route.textChanged.connect(self._validate_charge_multiplicity)

        # Solvent & Dispersion Environment
        solv_disp_row = QtWidgets.QHBoxLayout()
        self.combo_solvent = QtWidgets.QComboBox()
        self.combo_solvent.addItems([
            "None (Gas Phase)", "Water (CPCM)", "Acetonitrile (CPCM)",
            "Dichloromethane (CPCM)", "Chloroform (CPCM)", "DMSO (CPCM)",
            "THF (CPCM)", "Toluene (CPCM)", "Ethanol (CPCM)", "Methanol (CPCM)"
        ])
        solv_disp_row.addWidget(QtWidgets.QLabel("Solvent:"))
        solv_disp_row.addWidget(self.combo_solvent, stretch=1)

        self.combo_dispersion = QtWidgets.QComboBox()
        self.combo_dispersion.addItems(["None / In Route", "D3BJ", "D4"])
        solv_disp_row.addWidget(QtWidgets.QLabel("Dispersion:"))
        solv_disp_row.addWidget(self.combo_dispersion)
        m_form.addRow("Environment:", solv_disp_row)

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

    def _run_preflight_sanity(self) -> dict:
        """Run chemistry_checker pre-flight sanity checks on current molecule, charge, multiplicity, and route."""
        from .chemistry_checker import check_calculation_sanity
        symbols = [getattr(a, "symbol", "C") for a in self.zatoms]
        coords_arr = self.coords
        if coords_arr is None or len(coords_arr) != len(symbols):
            from .zmatrix import zmatrix_to_cartesian
            try:
                coords_arr = zmatrix_to_cartesian(self.zatoms)
            except Exception:
                coords_arr = np.zeros((len(symbols), 3))
        route_str = self.txt_route.text().strip() if hasattr(self, "txt_route") else ""
        return check_calculation_sanity(
            symbols=symbols,
            coords=coords_arr,
            charge=self.spn_charge.value(),
            multiplicity=self.spn_mult.value(),
            basis_set=route_str,
            qm_package="ORCA",
        )

    def _validate_charge_multiplicity(self):
        """Validates electron count parity, interatomic clashes, and basis suitability."""
        if not self.zatoms:
            self.lbl_parity_warning.setText("<span style='color: #64748b;'>No molecule loaded</span>")
            return

        res = self._run_preflight_sanity()
        info = res["info"]
        num_electrons = info["n_electrons"]
        mult = info["multiplicity"]

        if res["errors"]:
            err_msg = res["errors"][0]
            prefix = "⚠️ Spin mismatch / Pre-flight Error:" if "Spin state" in err_msg else "⚠️ Pre-flight Error:"
            self.lbl_parity_warning.setText(
                f"<span style='color: #dc2626; font-weight: bold;'>{prefix}</span> "
                f"<span style='color: #dc2626;'>{err_msg}</span>"
            )
        elif res["warnings"]:
            warn_msg = res["warnings"][0]
            self.lbl_parity_warning.setText(
                f"<span style='color: #d97706; font-weight: bold;'>⚠️ Advisory ({num_electrons} e⁻, mult={mult}):</span> "
                f"<span style='color: #d97706;'>{warn_msg}</span>"
            )
        else:
            self.lbl_parity_warning.setText(
                f"<span style='color: #16a34a; font-weight: bold;'>✓ Valid state</span> "
                f"<span style='color: #16a34a;'>({info['formula']}, {num_electrons} e⁻, mult={mult})</span>"
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

        sanity = self._run_preflight_sanity()
        issues = list(sanity.get("errors", [])) + list(sanity.get("warnings", []))
        if issues:
            ans = QtWidgets.QMessageBox.warning(
                self,
                "Pre-Flight Chemistry Sanity Check",
                "The following potential issues were detected in your ORCA setup:\n\n• "
                + "\n• ".join(issues)
                + "\n\nDo you want to launch the ORCA calculation anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        route_line = self.txt_route.text().strip()
        if not route_line.startswith("!"):
            route_line = "! " + route_line

        # Append dispersion if selected and not already in route
        if hasattr(self, "combo_dispersion"):
            disp = self.combo_dispersion.currentText()
            if disp in ("D3BJ", "D4") and disp.lower() not in route_line.lower():
                route_line += f" {disp}"

        # Append solvent if selected and not already in route
        if hasattr(self, "combo_solvent"):
            solv = self.combo_solvent.currentText()
            if solv != "None (Gas Phase)" and "cpcm" not in route_line.lower() and "smd" not in route_line.lower():
                solv_name = solv.split()[0]
                if solv_name == "Dichloromethane":
                    solv_name = "CH2Cl2"
                route_line += f" CPCM({solv_name})"

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
