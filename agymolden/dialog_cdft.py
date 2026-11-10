"""Interactive Conceptual Density Functional Theory (CDFT) Workbench Dialog.

Displays global reactivity indices (Chemical Potential, Hardness, Electrophilicity)
and condensed local Fukui functions / Dual Descriptor per atom with 1-click
3D color mapping onto the molecular viewer.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .cdft import compute_global_cdft, compute_condensed_fukui, GlobalCDFTDescriptors, LocalCDFTDescriptors


class CDFTWorkbenchDialog(QtWidgets.QDialog):
    """Interactive CDFT & Fukui Reactivity Analysis Dialog."""

    apply_dual_descriptor_to_viewer = QtCore.pyqtSignal(list, np.ndarray)  # symbols, dual_descriptor_vals

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        symbols: Optional[List[str]] = None,
        coords: Optional[np.ndarray] = None,
        homo_ev: float = -6.20,
        lumo_ev: float = -1.40,
    ):
        super().__init__(parent)
        self.setWindowTitle("⚡ Conceptual DFT & Fukui Reactivity Workbench")
        self.resize(850, 620)
        self.setMinimumSize(750, 520)

        self.symbols = list(symbols) if symbols else []
        self.coords = np.array(coords, dtype=np.float64) if coords is not None else np.empty((0, 3))
        self.homo_ev = homo_ev
        self.lumo_ev = lumo_ev

        self._build_ui()
        self._calculate_and_refresh()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header Info Banner
        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>Conceptual DFT (CDFT) Perturbative Reactivity Workbench</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Evaluate global reactivity indices (μ, η, ω, N_Nu) and condensed local Fukui functions (f+, f-, Δf) "
            "to predict electrophilic vs. nucleophilic reaction sites in real time."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_desc.setWordWrap(True)
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        # Controls Row: HOMO / LUMO Inputs
        ctrl_box = QtWidgets.QGroupBox("Frontier Orbital Input")
        c_lay = QtWidgets.QHBoxLayout(ctrl_box)

        c_lay.addWidget(QtWidgets.QLabel("HOMO (eV):"))
        self.spin_homo = QtWidgets.QDoubleSpinBox()
        self.spin_homo.setRange(-50.0, 50.0)
        self.spin_homo.setDecimals(3)
        self.spin_homo.setValue(self.homo_ev)
        self.spin_homo.valueChanged.connect(self._calculate_and_refresh)
        c_lay.addWidget(self.spin_homo)

        c_lay.addWidget(QtWidgets.QLabel("LUMO (eV):"))
        self.spin_lumo = QtWidgets.QDoubleSpinBox()
        self.spin_lumo.setRange(-50.0, 50.0)
        self.spin_lumo.setDecimals(3)
        self.spin_lumo.setValue(self.lumo_ev)
        self.spin_lumo.valueChanged.connect(self._calculate_and_refresh)
        c_lay.addWidget(self.spin_lumo)

        c_lay.addStretch()
        layout.addWidget(ctrl_box)

        # Splitter: Global Descriptors (Left) vs Local Fukui Table (Right)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left Panel: Global Descriptors Table
        left_panel = QtWidgets.QGroupBox("Global Conceptual DFT Descriptors")
        l_lay = QtWidgets.QVBoxLayout(left_panel)
        self.tbl_global = QtWidgets.QTableWidget()
        self.tbl_global.setColumnCount(2)
        self.tbl_global.setHorizontalHeaderLabels(["Descriptor", "Value"])
        self.tbl_global.horizontalHeader().setStretchLastSection(True)
        self.tbl_global.verticalHeader().setVisible(False)
        self.tbl_global.setStyleSheet("background-color: #020617; color: #f8fafc; gridline-color: #1e293b;")
        l_lay.addWidget(self.tbl_global)
        splitter.addWidget(left_panel)

        # Right Panel: Condensed Local Fukui Functions
        right_panel = QtWidgets.QGroupBox("Condensed Atomic Fukui Functions & Dual Descriptor")
        r_lay = QtWidgets.QVBoxLayout(right_panel)
        self.tbl_local = QtWidgets.QTableWidget()
        self.tbl_local.setColumnCount(5)
        self.tbl_local.setHorizontalHeaderLabels(["Atom", "f+ (Nu attack)", "f- (E+ attack)", "f0 (Radical)", "Δf (Dual)"])
        self.tbl_local.horizontalHeader().setStretchLastSection(True)
        self.tbl_local.verticalHeader().setVisible(False)
        self.tbl_local.setStyleSheet("background-color: #020617; color: #f8fafc; gridline-color: #1e293b;")
        r_lay.addWidget(self.tbl_local)

        # Button to paint viewport by dual descriptor
        btn_paint = QtWidgets.QPushButton("🎨 Color 3D Viewport by Dual Descriptor (Δf)")
        btn_paint.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px;")
        btn_paint.setToolTip("Color atoms: Blue = Electrophilic site (favors Nu attack), Red = Nucleophilic site (favors E+ attack)")
        btn_paint.clicked.connect(self._on_apply_to_viewer)
        r_lay.addWidget(btn_paint)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        # Bottom Bar
        b_bar = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("📋 Copy CDFT Summary for SI")
        btn_copy.clicked.connect(self._on_copy_summary)
        b_bar.addWidget(btn_copy)

        b_bar.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        b_bar.addWidget(btn_close)
        layout.addLayout(b_bar)

    def _calculate_and_refresh(self):
        h = self.spin_homo.value()
        l = self.spin_lumo.value()

        # Global
        g = compute_global_cdft(h, l)

        global_rows = [
            ("HOMO Energy", f"{g.homo_ev:+.3f} eV"),
            ("LUMO Energy", f"{g.lumo_ev:+.3f} eV"),
            ("Fundamental Gap (ΔE)", f"{g.gap_ev:.3f} eV"),
            ("Ionization Potential (I ≈ -HOMO)", f"{g.ip_ev:.3f} eV"),
            ("Electron Affinity (A ≈ -LUMO)", f"{g.ea_ev:.3f} eV"),
            ("Chemical Potential (μ)", f"{g.chemical_potential_mu:+.3f} eV"),
            ("Mulliken Electronegativity (χ)", f"{g.electronegativity_chi:.3f} eV"),
            ("Chemical Hardness (η)", f"{g.hardness_eta:.3f} eV"),
            ("Chemical Softness (S)", f"{g.softness_s:.3f} eV⁻¹"),
            ("Electrophilicity Index (ω)", f"{g.electrophilicity_omega:.3f} eV"),
            ("Nucleophilicity Index (N_Nu)", f"{g.nucleophilicity_n:+.3f} eV"),
            ("Net Electro-Accepting (ω+)", f"{g.omega_plus:.3f} eV"),
            ("Net Electro-Donating (ω-)", f"{g.omega_minus:.3f} eV"),
        ]

        self.tbl_global.setRowCount(len(global_rows))
        for row, (name, val) in enumerate(global_rows):
            item_name = QtWidgets.QTableWidgetItem(name)
            item_val = QtWidgets.QTableWidgetItem(val)
            item_val.setFont(QtGui.QFont("", -1, QtGui.QFont.Weight.Bold))
            self.tbl_global.setItem(row, 0, item_name)
            self.tbl_global.setItem(row, 1, item_val)

        # Local Fukui
        loc = compute_condensed_fukui(self.symbols, self.coords)
        self.local_res = loc
        n = len(self.symbols)

        self.tbl_local.setRowCount(n)
        for i in range(n):
            sym = f"{self.symbols[i]}{i+1}"
            f_p = loc.f_plus[i]
            f_m = loc.f_minus[i]
            f_0 = loc.f_zero[i]
            df = loc.dual_descriptor[i]

            item_sym = QtWidgets.QTableWidgetItem(sym)
            item_p = QtWidgets.QTableWidgetItem(f"{f_p:.4f}")
            item_m = QtWidgets.QTableWidgetItem(f"{f_m:.4f}")
            item_0 = QtWidgets.QTableWidgetItem(f"{f_0:.4f}")

            # Color code dual descriptor: Blue (>0 electrophilic), Red (<0 nucleophilic)
            df_str = f"{df:+.4f}"
            item_df = QtWidgets.QTableWidgetItem(df_str)
            item_df.setFont(QtGui.QFont("", -1, QtGui.QFont.Weight.Bold))
            if df > 0.01:
                item_df.setForeground(QtGui.QColor("#38bdf8"))  # Cyan / Blue
            elif df < -0.01:
                item_df.setForeground(QtGui.QColor("#f87171"))  # Red
            else:
                item_df.setForeground(QtGui.QColor("#94a3b8"))  # Slate

            self.tbl_local.setItem(i, 0, item_sym)
            self.tbl_local.setItem(i, 1, item_p)
            self.tbl_local.setItem(i, 2, item_m)
            self.tbl_local.setItem(i, 3, item_0)
            self.tbl_local.setItem(i, 4, item_df)

    def _on_apply_to_viewer(self):
        if hasattr(self, "local_res") and len(self.local_res.dual_descriptor) > 0:
            self.apply_dual_descriptor_to_viewer.emit(self.symbols, self.local_res.dual_descriptor)
            if not getattr(self, "_suppress_msg", False):
                QtWidgets.QMessageBox.information(
                    self,
                    "Dual Descriptor Applied",
                    "Successfully mapped Dual Descriptor (Δf) onto the 3D viewport:\n"
                    "• Blue (Δf > 0): Electrophilic site (susceptible to nucleophilic attack)\n"
                    "• Red (Δf < 0): Nucleophilic site (susceptible to electrophilic attack)"
                )

    def _on_copy_summary(self):
        lines = ["# Conceptual DFT (CDFT) Reactivity Summary\n"]
        for r in range(self.tbl_global.rowCount()):
            desc = self.tbl_global.item(r, 0).text()
            val = self.tbl_global.item(r, 1).text()
            lines.append(f"{desc}: {val}")

        lines.append("\n## Condensed Atomic Fukui Functions & Dual Descriptor:")
        lines.append("Atom\tf+\tf-\tf0\tΔf")
        for r in range(self.tbl_local.rowCount()):
            a = self.tbl_local.item(r, 0).text()
            fp = self.tbl_local.item(r, 1).text()
            fm = self.tbl_local.item(r, 2).text()
            f0 = self.tbl_local.item(r, 3).text()
            df = self.tbl_local.item(r, 4).text()
            lines.append(f"{a}\t{fp}\t{fm}\t{f0}\t{df}")

        full_txt = "\n".join(lines)
        QtWidgets.QApplication.clipboard().setText(full_txt)
        QtWidgets.QMessageBox.information(self, "Copied", "CDFT summary and Fukui table copied to clipboard.")
