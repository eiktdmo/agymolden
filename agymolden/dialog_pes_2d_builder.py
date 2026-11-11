"""Interactive 2D Relaxed Potential Energy Surface (PES) Scan Grid Builder Dialog."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .pes_2d_tools import (
    ScanCoordinateDef,
    generate_orca_2d_scan_input,
    generate_gaussian_2d_scan_input,
)


class PES2DScanBuilderDialog(QtWidgets.QDialog):
    """2D Relaxed PES Scan Grid Builder & Input Generator."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        symbols: Optional[List[str]] = None,
        coords: Optional[np.ndarray] = None,
        selected_atoms: Optional[List[int]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("🗺️ 2D Relaxed Potential Energy Surface (PES) Scan Builder")
        self.resize(760, 600)
        self.setMinimumSize(680, 500)

        self.symbols = list(symbols) if symbols else []
        self.coords = np.array(coords, copy=True) if coords is not None else np.zeros((len(self.symbols), 3))
        self.selected_atoms = list(selected_atoms) if selected_atoms else []

        self._build_ui()
        self._update_preview()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>2D Relaxed Potential Energy Surface (PES) Grid Scan Builder</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Constructs two-dimensional coordinate scan grids across two independent internal degrees of freedom "
            "(Bond distances, Angles, or Dihedrals) and generates production-ready ORCA and Gaussian ModRedundant decks."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_desc.setWordWrap(True)
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        # Coordinate 1 Box
        box1 = QtWidgets.QGroupBox("Scan Coordinate 1 (X-Axis)")
        l1 = QtWidgets.QGridLayout(box1)

        l1.addWidget(QtWidgets.QLabel("Type:"), 0, 0)
        self.combo_type1 = QtWidgets.QComboBox()
        self.combo_type1.addItems(["Bond Distance (Å)", "Bond Angle (°)", "Dihedral Angle (°)"])
        self.combo_type1.currentIndexChanged.connect(self._update_preview)
        l1.addWidget(self.combo_type1, 0, 1)

        l1.addWidget(QtWidgets.QLabel("Atom Indices (0-based, space-separated):"), 0, 2)
        self.txt_indices1 = QtWidgets.QLineEdit("0 1")
        self.txt_indices1.textChanged.connect(self._update_preview)
        l1.addWidget(self.txt_indices1, 0, 3)

        l1.addWidget(QtWidgets.QLabel("Start:"), 1, 0)
        self.spin_start1 = QtWidgets.QDoubleSpinBox()
        self.spin_start1.setRange(-360.0, 360.0)
        self.spin_start1.setValue(1.10)
        self.spin_start1.valueChanged.connect(self._update_preview)
        l1.addWidget(self.spin_start1, 1, 1)

        l1.addWidget(QtWidgets.QLabel("End:"), 1, 2)
        self.spin_end1 = QtWidgets.QDoubleSpinBox()
        self.spin_end1.setRange(-360.0, 360.0)
        self.spin_end1.setValue(2.20)
        self.spin_end1.valueChanged.connect(self._update_preview)
        l1.addWidget(self.spin_end1, 1, 3)

        l1.addWidget(QtWidgets.QLabel("Steps:"), 1, 4)
        self.spin_steps1 = QtWidgets.QSpinBox()
        self.spin_steps1.setRange(2, 50)
        self.spin_steps1.setValue(10)
        self.spin_steps1.valueChanged.connect(self._update_preview)
        l1.addWidget(self.spin_steps1, 1, 5)

        layout.addWidget(box1)

        # Coordinate 2 Box
        box2 = QtWidgets.QGroupBox("Scan Coordinate 2 (Y-Axis)")
        l2 = QtWidgets.QGridLayout(box2)

        l2.addWidget(QtWidgets.QLabel("Type:"), 0, 0)
        self.combo_type2 = QtWidgets.QComboBox()
        self.combo_type2.addItems(["Dihedral Angle (°)", "Bond Distance (Å)", "Bond Angle (°)"])
        self.combo_type2.currentIndexChanged.connect(self._update_preview)
        l2.addWidget(self.combo_type2, 0, 1)

        l2.addWidget(QtWidgets.QLabel("Atom Indices (0-based, space-separated):"), 0, 2)
        self.txt_indices2 = QtWidgets.QLineEdit("0 1 2 3" if len(self.symbols) >= 4 else "1 2")
        self.txt_indices2.textChanged.connect(self._update_preview)
        l2.addWidget(self.txt_indices2, 0, 3)

        l2.addWidget(QtWidgets.QLabel("Start:"), 1, 0)
        self.spin_start2 = QtWidgets.QDoubleSpinBox()
        self.spin_start2.setRange(-360.0, 360.0)
        self.spin_start2.setValue(0.0)
        self.spin_start2.valueChanged.connect(self._update_preview)
        l2.addWidget(self.spin_start2, 1, 1)

        l2.addWidget(QtWidgets.QLabel("End:"), 1, 2)
        self.spin_end2 = QtWidgets.QDoubleSpinBox()
        self.spin_end2.setRange(-360.0, 360.0)
        self.spin_end2.setValue(180.0)
        self.spin_end2.valueChanged.connect(self._update_preview)
        l2.addWidget(self.spin_end2, 1, 3)

        l2.addWidget(QtWidgets.QLabel("Steps:"), 1, 4)
        self.spin_steps2 = QtWidgets.QSpinBox()
        self.spin_steps2.setRange(2, 50)
        self.spin_steps2.setValue(10)
        self.spin_steps2.valueChanged.connect(self._update_preview)
        l2.addWidget(self.spin_steps2, 1, 5)

        layout.addWidget(box2)

        # Deck Generator Settings
        sett_box = QtWidgets.QHBoxLayout()
        sett_box.addWidget(QtWidgets.QLabel("QM Engine:"))
        self.combo_engine = QtWidgets.QComboBox()
        self.combo_engine.addItems(["ORCA (%geom Scan)", "Gaussian (Opt=ModRedundant)"])
        self.combo_engine.currentIndexChanged.connect(self._update_preview)
        sett_box.addWidget(self.combo_engine)

        sett_box.addWidget(QtWidgets.QLabel("Charge:"))
        self.spin_chg = QtWidgets.QSpinBox()
        self.spin_chg.setRange(-10, 10)
        self.spin_chg.setValue(0)
        self.spin_chg.valueChanged.connect(self._update_preview)
        sett_box.addWidget(self.spin_chg)

        sett_box.addWidget(QtWidgets.QLabel("Mult:"))
        self.spin_mult = QtWidgets.QSpinBox()
        self.spin_mult.setRange(1, 10)
        self.spin_mult.setValue(1)
        self.spin_mult.valueChanged.connect(self._update_preview)
        sett_box.addWidget(self.spin_mult)
        sett_box.addStretch()
        layout.addLayout(sett_box)

        # Text Deck Preview
        self.txt_deck = QtWidgets.QPlainTextEdit()
        self.txt_deck.setFont(QtGui.QFont("DejaVu Sans Mono", 9))
        layout.addWidget(self.txt_deck)

        # Bottom Buttons
        btn_box = QtWidgets.QHBoxLayout()
        self.lbl_grid_info = QtWidgets.QLabel()
        self.lbl_grid_info.setStyleSheet("color: #4ade80; font-weight: bold;")
        btn_box.addWidget(self.lbl_grid_info)
        btn_box.addStretch()

        btn_copy = QtWidgets.QPushButton("📋 Copy Input Deck")
        btn_copy.clicked.connect(self._copy_deck)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_copy)
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)

    def _parse_coord(self, combo: QtWidgets.QComboBox, txt_idx: QtWidgets.QLineEdit, s_val: float, e_val: float, steps: int) -> ScanCoordinateDef:
        txt = combo.currentText()
        if "Distance" in txt:
            c_type = "distance"
        elif "Angle" in txt and "Dihedral" not in txt:
            c_type = "angle"
        else:
            c_type = "dihedral"

        parts = txt_idx.text().strip().split()
        indices = tuple(int(p) for p in parts if p.isdigit())
        return ScanCoordinateDef(coord_type=c_type, atom_indices=indices, start_val=s_val, end_val=e_val, num_steps=steps)

    def _update_preview(self):
        try:
            s1 = self._parse_coord(self.combo_type1, self.txt_indices1, self.spin_start1.value(), self.spin_end1.value(), self.spin_steps1.value())
            s2 = self._parse_coord(self.combo_type2, self.txt_indices2, self.spin_start2.value(), self.spin_end2.value(), self.spin_steps2.value())

            chg = self.spin_chg.value()
            mult = self.spin_mult.value()

            tot_pts = (s1.num_steps + 1) * (s2.num_steps + 1)
            self.lbl_grid_info.setText(f"Grid Size: {s1.num_steps + 1} × {s2.num_steps + 1} = {tot_pts} stationary points")

            if "ORCA" in self.combo_engine.currentText():
                deck = generate_orca_2d_scan_input(self.symbols, self.coords, s1, s2, charge=chg, multiplicity=mult)
            else:
                deck = generate_gaussian_2d_scan_input(self.symbols, self.coords, s1, s2, charge=chg, multiplicity=mult)

            self.txt_deck.setPlainText(deck)
        except Exception as e:
            self.txt_deck.setPlainText(f"# Configuration Error: {e}")

    def _copy_deck(self):
        QtWidgets.QApplication.clipboard().setText(self.txt_deck.toPlainText())
        QtWidgets.QMessageBox.information(self, "Copied", "2D Scan input deck copied to clipboard.")
