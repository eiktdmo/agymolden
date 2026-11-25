"""Interactive Frontier Molecular Orbital (FMO) Alignment & Energy Level Comparison.

Renders side-by-side or stacked electronic energy levels (HOMO-1, HOMO, LUMO, LUMO+1)
comparing two systems (e.g. Donor vs Acceptor, Reactant vs Product, Ligand vs Complex)
with interactive chemical potential (μ), hardness (η), electrophilicity index (ω),
and frontier gap (ΔE) calculations.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class FMOAlignmentDialog(QtWidgets.QDialog):
    """Frontier Molecular Orbital Alignment and Donor-Acceptor Energy Level Matcher."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        sys1_name: str = "System A (Donor)",
        sys1_homo: float = -5.80,
        sys1_lumo: float = -1.20,
        sys2_name: str = "System B (Acceptor)",
        sys2_homo: float = -7.10,
        sys2_lumo: float = -3.40,
    ):
        super().__init__(parent)
        self.setWindowTitle("⚡ Frontier Molecular Orbital (FMO) Alignment & Reactivity Matcher")
        self.resize(840, 620)
        self.setMinimumSize(700, 500)

        self.sys1_name = sys1_name
        self.sys1_homo = sys1_homo
        self.sys1_lumo = sys1_lumo

        self.sys2_name = sys2_name
        self.sys2_homo = sys2_homo
        self.sys2_lumo = sys2_lumo

        self._build_ui()
        self.update_plot()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header Info Banner
        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>Frontier Molecular Orbital (FMO) Energy Alignment</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Compare frontier orbital levels (HOMO / LUMO) side-by-side to assess donor-acceptor charge transfer, "
            "chemical hardness (η = (LUMO - HOMO)/2), electronic chemical potential (μ = (HOMO + LUMO)/2), and electrophilicity index (ω = μ² / 2η)."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_desc.setWordWrap(True)
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        # Parameter Input Controls
        ctrl_box = QtWidgets.QGroupBox("Orbital Energies (eV)")
        ctrl_lay = QtWidgets.QGridLayout(ctrl_box)

        # System 1
        ctrl_lay.addWidget(QtWidgets.QLabel("<b>System A Name:</b>"), 0, 0)
        self.txt_sys1_name = QtWidgets.QLineEdit(self.sys1_name)
        self.txt_sys1_name.textChanged.connect(self.update_plot)
        ctrl_lay.addWidget(self.txt_sys1_name, 0, 1)

        ctrl_lay.addWidget(QtWidgets.QLabel("HOMO (eV):"), 0, 2)
        self.spin_sys1_homo = QtWidgets.QDoubleSpinBox()
        self.spin_sys1_homo.setRange(-50.0, 50.0)
        self.spin_sys1_homo.setDecimals(3)
        self.spin_sys1_homo.setValue(self.sys1_homo)
        self.spin_sys1_homo.valueChanged.connect(self.update_plot)
        ctrl_lay.addWidget(self.spin_sys1_homo, 0, 3)

        ctrl_lay.addWidget(QtWidgets.QLabel("LUMO (eV):"), 0, 4)
        self.spin_sys1_lumo = QtWidgets.QDoubleSpinBox()
        self.spin_sys1_lumo.setRange(-50.0, 50.0)
        self.spin_sys1_lumo.setDecimals(3)
        self.spin_sys1_lumo.setValue(self.sys1_lumo)
        self.spin_sys1_lumo.valueChanged.connect(self.update_plot)
        ctrl_lay.addWidget(self.spin_sys1_lumo, 0, 5)

        # System 2
        ctrl_lay.addWidget(QtWidgets.QLabel("<b>System B Name:</b>"), 1, 0)
        self.txt_sys2_name = QtWidgets.QLineEdit(self.sys2_name)
        self.txt_sys2_name.textChanged.connect(self.update_plot)
        ctrl_lay.addWidget(self.txt_sys2_name, 1, 1)

        ctrl_lay.addWidget(QtWidgets.QLabel("HOMO (eV):"), 1, 2)
        self.spin_sys2_homo = QtWidgets.QDoubleSpinBox()
        self.spin_sys2_homo.setRange(-50.0, 50.0)
        self.spin_sys2_homo.setDecimals(3)
        self.spin_sys2_homo.setValue(self.sys2_homo)
        self.spin_sys2_homo.valueChanged.connect(self.update_plot)
        ctrl_lay.addWidget(self.spin_sys2_homo, 1, 3)

        ctrl_lay.addWidget(QtWidgets.QLabel("LUMO (eV):"), 1, 4)
        self.spin_sys2_lumo = QtWidgets.QDoubleSpinBox()
        self.spin_sys2_lumo.setRange(-50.0, 50.0)
        self.spin_sys2_lumo.setDecimals(3)
        self.spin_sys2_lumo.setValue(self.sys2_lumo)
        self.spin_sys2_lumo.valueChanged.connect(self.update_plot)
        ctrl_lay.addWidget(self.spin_sys2_lumo, 1, 5)

        layout.addWidget(ctrl_box)

        # Matplotlib Canvas
        self.fig = Figure(figsize=(7.0, 4.5), dpi=100, facecolor="#1e293b")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 1)

        # Bottom Bar: Summary Metrics & Actions
        b_bar = QtWidgets.QHBoxLayout()
        self.lbl_metrics = QtWidgets.QLabel("")
        self.lbl_metrics.setStyleSheet("font-size: 11px;")
        b_bar.addWidget(self.lbl_metrics, 1)

        btn_save = QtWidgets.QPushButton("💾 Export Diagram (PNG/SVG)...")
        btn_save.clicked.connect(self._on_save_diagram)
        b_bar.addWidget(btn_save)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        b_bar.addWidget(btn_close)
        layout.addLayout(b_bar)

    def update_plot(self):
        """Re-render frontier orbital ladder alignment diagram."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#0f172a")

        s1_name = self.txt_sys1_name.text().strip() or "System A"
        s1_h = self.spin_sys1_homo.value()
        s1_l = self.spin_sys1_lumo.value()

        s2_name = self.txt_sys2_name.text().strip() or "System B"
        s2_h = self.spin_sys2_homo.value()
        s2_l = self.spin_sys2_lumo.value()

        # Conceptual DFT Reactivity Indices
        gap1 = max(0.001, s1_l - s1_h)
        eta1 = gap1 / 2.0
        mu1 = (s1_h + s1_l) / 2.0
        omega1 = (mu1 ** 2) / (2.0 * eta1)

        gap2 = max(0.001, s2_l - s2_h)
        eta2 = gap2 / 2.0
        mu2 = (s2_h + s2_l) / 2.0
        omega2 = (mu2 ** 2) / (2.0 * eta2)

        # Charge transfer gap: HOMO(A) -> LUMO(B)
        gap_ab = abs(s2_l - s1_h)

        metrics_txt = (
            f"<b>{s1_name}:</b> Gap = {gap1:.2f} eV, η = {eta1:.2f} eV, μ = {mu1:+.2f} eV, ω = {omega1:.2f} eV<br>"
            f"<b>{s2_name}:</b> Gap = {gap2:.2f} eV, η = {eta2:.2f} eV, μ = {mu2:+.2f} eV, ω = {omega2:.2f} eV<br>"
            f"<b>Donor-Acceptor Gap:</b> HOMO({s1_name}) → LUMO({s2_name}) = <b>{gap_ab:.2f} eV</b>"
        )
        self.lbl_metrics.setText(metrics_txt)

        # Positions on plot
        x1, x2 = 1.0, 2.5
        w = 0.35

        # System A Levels
        ax.hlines(s1_h, x1 - w, x1 + w, colors="#38bdf8", linewidth=3.5, label=f"{s1_name} HOMO")
        ax.hlines(s1_l, x1 - w, x1 + w, colors="#f43f5e", linewidth=3.5, label=f"{s1_name} LUMO")
        ax.text(x1, s1_h - 0.4, f"HOMO\n{s1_h:+.2f} eV", ha="center", va="top", color="#38bdf8", fontweight="bold", fontsize=10)
        ax.text(x1, s1_l + 0.4, f"LUMO\n{s1_l:+.2f} eV", ha="center", va="bottom", color="#f43f5e", fontweight="bold", fontsize=10)

        # System B Levels
        ax.hlines(s2_h, x2 - w, x2 + w, colors="#38bdf8", linewidth=3.5, label=f"{s2_name} HOMO")
        ax.hlines(s2_l, x2 - w, x2 + w, colors="#f43f5e", linewidth=3.5, label=f"{s2_name} LUMO")
        ax.text(x2, s2_h - 0.4, f"HOMO\n{s2_h:+.2f} eV", ha="center", va="top", color="#38bdf8", fontweight="bold", fontsize=10)
        ax.text(x2, s2_l + 0.4, f"LUMO\n{s2_l:+.2f} eV", ha="center", va="bottom", color="#f43f5e", fontweight="bold", fontsize=10)

        # Gap indicator arrows
        ax.annotate("", xy=(x1, s1_l), xytext=(x1, s1_h), arrowprops=dict(arrowstyle="<->", color="#94a3b8", lw=1.5))
        ax.text(x1 + 0.05, (s1_h + s1_l) / 2.0, f"ΔE = {gap1:.2f} eV", color="#e2e8f0", fontsize=9, va="center")

        ax.annotate("", xy=(x2, s2_l), xytext=(x2, s2_h), arrowprops=dict(arrowstyle="<->", color="#94a3b8", lw=1.5))
        ax.text(x2 + 0.05, (s2_h + s2_l) / 2.0, f"ΔE = {gap2:.2f} eV", color="#e2e8f0", fontsize=9, va="center")

        # Cross charge-transfer dashed alignment line
        ax.plot([x1 + w, x2 - w], [s1_h, s2_l], ":", color="#22c55e", linewidth=2.0)
        ax.text((x1 + x2) / 2.0, (s1_h + s2_l) / 2.0 + 0.3, f"CT ΔE = {gap_ab:.2f} eV", color="#4ade80", fontweight="bold", fontsize=9, ha="center")

        # Styling
        ax.set_xticks([x1, x2])
        ax.set_xticklabels([s1_name, s2_name], color="#f8fafc", fontsize=11, fontweight="bold")
        ax.set_ylabel("Orbital Energy (eV)", color="#e2e8f0", fontsize=11, fontweight="bold")
        ax.set_title("Frontier Molecular Orbital Alignment & Reactivity Match", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
        ax.tick_params(colors="#94a3b8", which="both")
        ax.grid(axis="y", color="#334155", linestyle="--", alpha=0.5)

        y_min = min(s1_h, s2_h) - 1.5
        y_max = max(s1_l, s2_l) + 1.5
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0.3, 3.2)

        self.fig.tight_layout()
        self.canvas.draw()

    def _on_save_diagram(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save FMO Alignment Diagram", "fmo_alignment.svg",
            "Scalable Vector Graphics (*.svg);;Portable Network Graphics (*.png);;PDF (*.pdf)"
        )
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            QtWidgets.QMessageBox.information(self, "Diagram Saved", f"Diagram successfully exported to:\n{path}")
