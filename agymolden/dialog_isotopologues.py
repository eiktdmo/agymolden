"""Isotopologue Vibrational Frequency Shifts & Kinetic Isotope Effect (KIE) Dialog."""

from typing import Dict, Optional, Callable
from PyQt6 import QtWidgets

from .vibrations import (
    VibrationalAnalysis,
    COMMON_ISOTOPE_MASSES,
    compute_isotopologue_vibrational_shifts,
)


class IsotopologueDialog(QtWidgets.QDialog):
    """Calculate vibrational frequencies and shifts upon isotopic substitution (H -> D, 12C -> 13C, etc.)."""

    def __init__(self, parent=None, vibrational_analysis: Optional[VibrationalAnalysis] = None, on_apply_shifts: Optional[Callable] = None):
        super().__init__(parent)
        self.setWindowTitle("⚖️ Isotopologue Vibrational Shifts & KIE (Telloni–Redlich)")
        self.resize(760, 520)

        self.analysis = vibrational_analysis
        self.on_apply_shifts = on_apply_shifts
        self._spin_boxes: Dict[int, QtWidgets.QDoubleSpinBox] = {}

        self._init_ui()
        self._recalc()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        desc = QtWidgets.QLabel(
            "<b>Telloni–Redlich Isotopic Frequency Shifter & Kinetic Isotope Effect (KIE) Analyzer</b><br>"
            "<span style='color: #94a3b8; font-size: 11px;'>"
            "Assign custom isotopic masses (e.g., ¹H ➔ ²D / ³T, ¹²C ➔ ¹³C, ¹⁶O ➔ ¹⁸O). "
            "Re-diagonalizes the mass-weighted Hessian in normal-mode space to calculate shifted frequencies and ZPE changes.</span>"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Upper form: Quick element substitutions
        quick_box = QtWidgets.QGroupBox("1-Click Bulk Isotopic Labeling")
        quick_layout = QtWidgets.QHBoxLayout(quick_box)

        btn_d = QtWidgets.QPushButton("Perdeuterate (All H ➔ ²D)")
        btn_d.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 4px 8px;")
        btn_d.clicked.connect(lambda: self._set_element_mass("H", 2.0141017781))
        quick_layout.addWidget(btn_d)

        btn_13c = QtWidgets.QPushButton("¹³C Labeling (All C ➔ ¹³C)")
        btn_13c.setStyleSheet("background-color: #0d9488; color: white; font-weight: bold; padding: 4px 8px;")
        btn_13c.clicked.connect(lambda: self._set_element_mass("C", 13.0033548352))
        quick_layout.addWidget(btn_13c)

        btn_18o = QtWidgets.QPushButton("¹⁸O Labeling (All O ➔ ¹⁸O)")
        btn_18o.setStyleSheet("background-color: #7c3aed; color: white; font-weight: bold; padding: 4px 8px;")
        btn_18o.clicked.connect(lambda: self._set_element_mass("O", 17.9991604))
        quick_layout.addWidget(btn_18o)

        btn_reset = QtWidgets.QPushButton("↺ Reset Natural Abundance")
        btn_reset.clicked.connect(self._reset_all)
        quick_layout.addWidget(btn_reset)

        quick_layout.addStretch()
        layout.addWidget(quick_box)

        # Middle: Table of atoms and isotopic masses
        n_atoms = len(self.analysis.symbols) if self.analysis else 0
        tbl_atoms = QtWidgets.QTableWidget(n_atoms, 3)
        tbl_atoms.setHorizontalHeaderLabels(["Atom #", "Element", "Isotopic Mass (amu)"])
        tbl_atoms.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        tbl_atoms.verticalHeader().setVisible(False)
        tbl_atoms.setMaximumHeight(160)

        for i, sym in enumerate(self.analysis.symbols if self.analysis else []):
            tbl_atoms.setItem(i, 0, QtWidgets.QTableWidgetItem(f"#{i+1}"))
            tbl_atoms.setItem(i, 1, QtWidgets.QTableWidgetItem(sym))

            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(1.0, 300.0)
            spin.setDecimals(5)
            spin.setSingleStep(1.0)
            def_m = self.analysis.masses[i] if self.analysis and self.analysis.masses is not None else 1.0
            spin.setValue(float(def_m))
            spin.valueChanged.connect(self._recalc)
            self._spin_boxes[i] = spin
            tbl_atoms.setCellWidget(i, 2, spin)

        layout.addWidget(tbl_atoms)

        # Output Text
        self.txt_out = QtWidgets.QTextEdit()
        self.txt_out.setReadOnly(True)
        self.txt_out.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.txt_out, stretch=1)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("📋 Copy Shifts to Clipboard")
        btn_copy.clicked.connect(self._copy_shifts)
        btn_row.addWidget(btn_copy)

        btn_row.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _set_element_mass(self, elem: str, target_mass: float):
        if not self.analysis:
            return
        for i, sym in enumerate(self.analysis.symbols):
            if sym.upper() == elem.upper() and i in self._spin_boxes:
                self._spin_boxes[i].blockSignals(True)
                self._spin_boxes[i].setValue(target_mass)
                self._spin_boxes[i].blockSignals(False)
        self._recalc()

    def _reset_all(self):
        if not self.analysis or self.analysis.masses is None:
            return
        for i in range(len(self.analysis.symbols)):
            if i in self._spin_boxes:
                self._spin_boxes[i].blockSignals(True)
                self._spin_boxes[i].setValue(float(self.analysis.masses[i]))
                self._spin_boxes[i].blockSignals(False)
        self._recalc()

    def _recalc(self):
        if not self.analysis:
            return
        custom = [self._spin_boxes[i].value() for i in range(len(self.analysis.symbols))]
        res = compute_isotopologue_vibrational_shifts(self.analysis, custom, temperature_k=298.15)

        lines = [
            "=" * 76,
            "ISOTOPOLOGUE VIBRATIONAL ANALYSIS & KINETIC ISOTOPE EFFECTS (T = 298.15 K)",
            "=" * 76,
            f"Original ZPE (zero-point energy)      : {res.zpe_original_kcal:.3f} kcal/mol  ({res.zpe_original_kcal/627.509:.6f} Ha)",
            f"Isotopologue ZPE                       : {res.zpe_isotope_kcal:.3f} kcal/mol  ({res.zpe_isotope_kcal/627.509:.6f} Ha)",
            f"Delta ZPE (Isotope - Original)         : {res.delta_zpe_kcal:.3f} kcal/mol  ({res.delta_zpe_kcal*4.184:.3f} kJ/mol)",
            f"Zero-Point KIE Factor exp(-dZPE/RT)   : {res.kie_factor:.4f}",
            "-" * 76,
            f"{'Mode':<6} {'Orig Freq (cm^-1)':<20} {'Shifted Freq (cm^-1)':<22} {'Shift Delta_nu':<16} {'% Shift':<10}",
            "-" * 76,
        ]
        for f_orig, f_iso, d_nu, pct in zip(
            res.frequencies_original,
            res.frequencies_isotope,
            res.shifts_cm,
            res.percent_shifts,
        ):
            lines.append(f"{f_orig:12.2f}        {f_iso:12.2f}            {d_nu:+8.2f}         {pct:6.2f} %")
        lines.append("=" * 76)

        self.txt_out.setPlainText("\n".join(lines))

    def _copy_shifts(self):
        QtWidgets.QApplication.clipboard().setText(self.txt_out.toPlainText())
