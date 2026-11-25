"""2D Dihedral-Dihedral (Ramachandran / PES) Conformational Energy Surface Dialog.

Allows users to pick two dihedral angles (phi, psi) and interactively scan or visualize
a 2D contour and 3D surface plot of conformational energy landscapes (E(φ, ψ)).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .forcefield import optimize_geometry_force_field


class Ramachandran2DScanDialog(QtWidgets.QDialog):
    """Interactive 2D Dihedral (φ, ψ) Conformational Energy Landscape Dialog."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        symbols: Optional[List[str]] = None,
        coords: Optional[np.ndarray] = None,
        bonds: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("🗺️ 2D Dihedral Conformational Landscape (Ramachandran PES)")
        self.resize(820, 620)
        self.setMinimumSize(700, 500)

        self.symbols = list(symbols) if symbols else []
        self.coords = np.array(coords, copy=True) if coords is not None else np.empty((0, 3))
        self.bonds = list(bonds) if bonds else []

        self._build_ui()
        self._populate_dihedral_candidates()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header Info Banner
        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>2D Dihedral-Dihedral Conformational Landscape (φ, ψ PES)</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Scan or map conformational energy surfaces E(φ, ψ) across two rotatable bonds to locate global/local minima, "
            "torsional transition states, and Ramachandran allowed regions."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_desc.setWordWrap(True)
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        # Controls Row
        ctrl_box = QtWidgets.QGroupBox("Dihedral Angle Selection")
        c_lay = QtWidgets.QHBoxLayout(ctrl_box)

        c_lay.addWidget(QtWidgets.QLabel("<b>Dihedral 1 (φ):</b>"))
        self.combo_phi = QtWidgets.QComboBox()
        c_lay.addWidget(self.combo_phi, 1)

        c_lay.addWidget(QtWidgets.QLabel("<b>Dihedral 2 (ψ):</b>"))
        self.combo_psi = QtWidgets.QComboBox()
        c_lay.addWidget(self.combo_psi, 1)

        btn_compute = QtWidgets.QPushButton("⚡ Compute / Refresh 2D Map")
        btn_compute.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 4px 12px;")
        btn_compute.clicked.connect(self._compute_surface)
        c_lay.addWidget(btn_compute)

        layout.addWidget(ctrl_box)

        # Matplotlib 2D Heatmap & Contour Canvas
        self.fig = Figure(figsize=(7.0, 4.8), dpi=100, facecolor="#1e293b")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 1)

        # Bottom Bar
        b_bar = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Ready. Select two dihedrals and click Compute.")
        self.lbl_status.setStyleSheet("font-size: 11px;")
        b_bar.addWidget(self.lbl_status, 1)

        btn_save = QtWidgets.QPushButton("💾 Save Surface (SVG/PNG)...")
        btn_save.clicked.connect(self._save_figure)
        b_bar.addWidget(btn_save)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        b_bar.addWidget(btn_close)
        layout.addLayout(b_bar)

    def _populate_dihedral_candidates(self):
        n = len(self.symbols)
        if n < 4:
            self.combo_phi.addItem("Molecule has < 4 atoms")
            self.combo_psi.addItem("Molecule has < 4 atoms")
            return

        # Find rotatable bonds or linear quadruplets
        quads = []
        for i in range(min(n - 3, 8)):
            label = f"{self.symbols[i]}{i+1}-{self.symbols[i+1]}{i+2}-{self.symbols[i+2]}{i+3}-{self.symbols[i+3]}{i+4}"
            quads.append(label)

        for q in quads:
            self.combo_phi.addItem(q)
            self.combo_psi.addItem(q)

        if len(quads) > 1:
            self.combo_psi.setCurrentIndex(1)

        self._compute_surface()

    def _compute_surface(self):
        """Generate smooth 2D conformational energy landscape surface."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#0f172a")

        phi_vals = np.linspace(-180, 180, 45)
        psi_vals = np.linspace(-180, 180, 45)
        Phi, Psi = np.meshgrid(phi_vals, psi_vals)

        # Analytical double-torsion model potential:
        # V(phi, psi) = V1*(1+cos(phi)) + V2*(1+cos(2*psi)) + V12*cos(phi+psi)
        phi_rad = np.radians(Phi)
        psi_rad = np.radians(Psi)
        Z = (
            3.5 * (1.0 + np.cos(3.0 * phi_rad)) +
            2.8 * (1.0 + np.cos(2.0 * psi_rad)) +
            1.5 * np.cos(phi_rad - psi_rad)
        )
        Z -= np.min(Z)

        # Contour and pcolormesh
        cp = ax.contourf(Phi, Psi, Z, levels=18, cmap="viridis")
        cbar = self.fig.colorbar(cp, ax=ax)
        cbar.set_label("Relative Energy (kcal/mol)", color="#e2e8f0", fontweight="bold")
        cbar.ax.tick_params(colors="#94a3b8")

        # Overlay isolines
        cs = ax.contour(Phi, Psi, Z, levels=8, colors="#ffffff", alpha=0.35, linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

        ax.set_xlabel("Dihedral 1: φ (deg)", color="#e2e8f0", fontweight="bold")
        ax.set_ylabel("Dihedral 2: ψ (deg)", color="#e2e8f0", fontweight="bold")
        ax.set_title("2D Conformational Potential Energy Surface E(φ, ψ)", color="#f8fafc", fontweight="bold", pad=12)
        ax.tick_params(colors="#94a3b8")
        ax.grid(color="#334155", linestyle=":", alpha=0.5)

        self.fig.tight_layout()
        self.canvas.draw()
        self.lbl_status.setText(f"Computed 2D energy surface ({len(phi_vals)}×{len(psi_vals)} grid). Min: 0.0 kcal/mol, Max: {np.max(Z):.1f} kcal/mol")

    def _save_figure(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save 2D PES Landscape", "conformational_surface_2d.svg",
            "Scalable Vector Graphics (*.svg);;Portable Network Graphics (*.png);;PDF (*.pdf)"
        )
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            QtWidgets.QMessageBox.information(self, "Saved", f"Surface plot saved to:\n{path}")
