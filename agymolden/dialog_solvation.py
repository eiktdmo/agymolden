"""Interactive Continuum Solvation Models (SMD / PCM) Dialog."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .solvation import SOLVENT_DATABASE, estimate_solvation_energy, SolvationResult


class SolvationDialog(QtWidgets.QDialog):
    """Interactive Continuum Solvation & Thermodynamic Free Energy Estimator Dialog."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        symbols: Optional[List[str]] = None,
        coords: Optional[np.ndarray] = None,
        dipole_debye: Optional[float] = None,
        partial_charges: Optional[np.ndarray] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("💧 Continuum Solvation & Free Energy (SMD / PCM / CPCM)")
        self.resize(780, 560)
        self.setMinimumSize(680, 480)

        self.symbols = list(symbols) if symbols else []
        self.coords = np.array(coords, copy=True) if coords is not None else np.empty((0, 3))
        self.dipole_debye = dipole_debye
        self.partial_charges = partial_charges

        self._build_ui()
        self._recalculate()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header Frame
        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>Continuum Solvation Free Energy Estimator (SMD / CPCM / PCM)</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Computes Solvent Accessible Surface Area (SASA), molecular cavity volume, electrostatic reaction field dielectric response (ΔG_el), "
            "and cavitation/dispersion terms (ΔG_cds) to determine total free energy of solvation (ΔG_solv)."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_desc.setWordWrap(True)
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        # Solvent Selection Row
        s_box = QtWidgets.QGroupBox("Solvent Environment")
        s_lay = QtWidgets.QHBoxLayout(s_box)

        s_lay.addWidget(QtWidgets.QLabel("<b>Select Solvent:</b>"))
        self.combo_solvent = QtWidgets.QComboBox()
        for k, solv in SOLVENT_DATABASE.items():
            self.combo_solvent.addItem(f"{solv.name} (ε = {solv.dielectric_constant:.1f}, {solv.formula})", k)
        self.combo_solvent.currentIndexChanged.connect(self._recalculate)
        s_lay.addWidget(self.combo_solvent, 1)

        # Dipole input
        s_lay.addWidget(QtWidgets.QLabel("Dipole (Debye):"))
        self.spin_dipole = QtWidgets.QDoubleSpinBox()
        self.spin_dipole.setRange(0.0, 50.0)
        self.spin_dipole.setDecimals(2)
        self.spin_dipole.setValue(self.dipole_debye if self.dipole_debye is not None else 1.85)
        self.spin_dipole.valueChanged.connect(self._recalculate)
        s_lay.addWidget(self.spin_dipole)

        layout.addWidget(s_box)

        # Results Table
        self.tbl_results = QtWidgets.QTableWidget()
        self.tbl_results.setColumnCount(2)
        self.tbl_results.setHorizontalHeaderLabels(["Solvation Property", "Calculated Value"])
        self.tbl_results.horizontalHeader().setStretchLastSection(True)
        self.tbl_results.verticalHeader().setVisible(False)
        self.tbl_results.setStyleSheet("background-color: #020617; color: #f8fafc; gridline-color: #1e293b;")
        layout.addWidget(self.tbl_results, 1)

        # Bottom Bar
        b_bar = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("📋 Copy Solvation Summary for SI")
        btn_copy.clicked.connect(self._on_copy_summary)
        b_bar.addWidget(btn_copy)

        b_bar.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        b_bar.addWidget(btn_close)
        layout.addLayout(b_bar)

    def _recalculate(self):
        solv_key = self.combo_solvent.currentData()
        mu = self.spin_dipole.value()

        res = estimate_solvation_energy(
            self.symbols,
            self.coords,
            solvent_key=solv_key,
            partial_charges=self.partial_charges,
            dipole_debye=mu,
        )

        rows = [
            ("Solvent", f"{res.solvent_name} (ε = {res.epsilon:.2f})"),
            ("Solvent Accessible Surface Area (SASA)", f"{res.sasa_angstrom2:.2f} Å²"),
            ("Molecular Cavity Volume", f"{res.volume_angstrom3:.2f} Å³"),
            ("Electrostatic Dielectric Solvation (ΔG_el)", f"{res.delta_g_el_kcal:+.2f} kcal/mol"),
            ("Cavitation & Dispersion (ΔG_cds)", f"{res.delta_g_cds_kcal:+.2f} kcal/mol"),
            ("Total Free Energy of Solvation (ΔG_solv)", f"<b>{res.delta_g_solv_kcal:+.2f} kcal/mol</b>"),
            ("Total Free Energy of Solvation (SI units)", f"<b>{res.delta_g_solv_kj:+.2f} kJ/mol</b>"),
        ]

        self.tbl_results.setRowCount(len(rows))
        for row, (prop, val) in enumerate(rows):
            item_p = QtWidgets.QTableWidgetItem(prop)
            item_v = QtWidgets.QTableWidgetItem()
            # If bold html tag, show plain text or formatted
            clean_v = val.replace("<b>", "").replace("</b>", "")
            item_v.setText(clean_v)
            if "Total" in prop:
                item_v.setFont(QtGui.QFont("", -1, QtGui.QFont.Weight.Bold))
                item_v.setForeground(QtGui.QColor("#38bdf8"))
            self.tbl_results.setItem(row, 0, item_p)
            self.tbl_results.setItem(row, 1, item_v)

    def _on_copy_summary(self):
        lines = ["# Continuum Solvation Free Energy (SMD / PCM / CPCM) Summary\n"]
        for r in range(self.tbl_results.rowCount()):
            p = self.tbl_results.item(r, 0).text()
            v = self.tbl_results.item(r, 1).text()
            lines.append(f"{p}: {v}")
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        QtWidgets.QMessageBox.information(self, "Copied", "Solvation properties copied to clipboard.")
