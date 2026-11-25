"""NBO Second-Order Perturbation E(2) Network & Donor-Acceptor Chord Diagram.

Renders an interactive topological graph/network connecting:
- Donor NBOs (lone pairs n, bonding orbitals σ, π)
- Acceptor NBOs (antibonding σ*, π*)
with edge thicknesses and color intensity proportional to stabilization energy E(2) (kcal/mol).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .nbo_analyzer import NBOPerturbation


class NBONetworkDialog(QtWidgets.QDialog):
    """Interactive NBO E(2) Donor-Acceptor Interaction Network & Chord Diagram."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        interactions: Optional[List[NBOPerturbation]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("🕸️ Natural Bond Orbital (NBO) E(2) Donor-Acceptor Interaction Network")
        self.resize(850, 620)
        self.setMinimumSize(700, 500)

        self.interactions = list(interactions) if interactions else []
        self._build_ui()
        self.update_network_plot()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header Frame
        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>Natural Bond Orbital (NBO) Second-Order Perturbation E(2) Network</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Visualizes stereoelectronic donor-acceptor delocalizations (n → σ*, σ → σ* hyperconjugation, n → π*). "
            "Nodes represent NBOs arranged by Donor (left) and Acceptor (right); edge line thickness represents stabilization energy E(2)."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_desc.setWordWrap(True)
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        # Filter bar
        f_box = QtWidgets.QHBoxLayout()
        f_box.addWidget(QtWidgets.QLabel("<b>Min E(2) Threshold (kcal/mol):</b>"))
        self.spin_min_e = QtWidgets.QDoubleSpinBox()
        self.spin_min_e.setRange(0.5, 200.0)
        self.spin_min_e.setValue(2.0)
        self.spin_min_e.setSingleStep(0.5)
        self.spin_min_e.valueChanged.connect(self.update_network_plot)
        f_box.addWidget(self.spin_min_e)

        f_box.addSpacing(15)
        f_box.addWidget(QtWidgets.QLabel("<b>Layout Style:</b>"))
        self.combo_layout = QtWidgets.QComboBox()
        self.combo_layout.addItems(["Bipartite (Donor Left → Acceptor Right)", "Circular / Chord Graph"])
        self.combo_layout.currentIndexChanged.connect(self.update_network_plot)
        f_box.addWidget(self.combo_layout)

        f_box.addStretch()
        layout.addLayout(f_box)

        # Matplotlib Graph Canvas
        self.fig = Figure(figsize=(7.5, 4.8), dpi=100, facecolor="#1e293b")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 1)

        # Bottom Bar
        b_bar = QtWidgets.QHBoxLayout()
        self.lbl_stats = QtWidgets.QLabel("")
        self.lbl_stats.setStyleSheet("font-size: 11px;")
        b_bar.addWidget(self.lbl_stats, 1)

        btn_save = QtWidgets.QPushButton("💾 Export Diagram (SVG / PNG)...")
        btn_save.clicked.connect(self._on_save_diagram)
        b_bar.addWidget(btn_save)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        b_bar.addWidget(btn_close)
        layout.addLayout(b_bar)

    def update_network_plot(self):
        """Render donor-acceptor network graph."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#0f172a")

        min_e = self.spin_min_e.value()
        filtered = [item for item in self.interactions if item.energy_kcal >= min_e]

        if not filtered:
            ax.text(0.5, 0.5, f"No interactions found with E(2) ≥ {min_e:.1f} kcal/mol.\nTry lowering the threshold or load an NBO output file.",
                    color="#94a3b8", ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            self.lbl_stats.setText("0 interactions displayed.")
            self.canvas.draw()
            return

        # Sort top 30 strongest interactions to prevent visual clutter
        filtered.sort(key=lambda x: x.energy_kcal, reverse=True)
        top_items = filtered[:28]

        donors = list(dict.fromkeys(item.donor_label for item in top_items))
        acceptors = list(dict.fromkeys(item.acceptor_label for item in top_items))

        is_circular = self.combo_layout.currentIndex() == 1

        if is_circular:
            # Circular Layout
            all_nodes = donors + acceptors
            n_tot = len(all_nodes)
            pos_dict = {}
            for idx, n_name in enumerate(all_nodes):
                ang = 2.0 * math.pi * idx / max(1, n_tot)
                pos_dict[n_name] = (math.cos(ang), math.sin(ang))

            # Draw nodes
            for n_name, (nx, ny) in pos_dict.items():
                is_don = n_name in donors
                col = "#38bdf8" if is_don else "#f43f5e"
                ax.plot(nx, ny, "o", color=col, markersize=10, zorder=5)
                # Label
                ha = "left" if nx >= 0 else "right"
                ax.text(nx * 1.15, ny * 1.15, n_name, color=col, fontsize=8, fontweight="bold",
                        ha=ha, va="center", zorder=6)

            # Draw curved chords
            max_e = max(item.energy_kcal for item in top_items)
            for item in top_items:
                p1 = pos_dict.get(item.donor_label)
                p2 = pos_dict.get(item.acceptor_label)
                if p1 and p2:
                    lw = max(1.0, (item.energy_kcal / max_e) * 4.5)
                    # Bezier chord via origin
                    t_vals = np.linspace(0, 1, 25)
                    cx = (1 - t_vals)**2 * p1[0] + t_vals**2 * p2[0]
                    cy = (1 - t_vals)**2 * p1[1] + t_vals**2 * p2[1]
                    ax.plot(cx, cy, color="#22c55e", alpha=0.65, linewidth=lw)

            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.set_title(f"NBO E(2) Chord Graph ({len(top_items)} interactions ≥ {min_e:.1f} kcal/mol)",
                         color="#f8fafc", fontweight="bold", pad=12)
        else:
            # Bipartite Layout
            y_donors = np.linspace(len(donors), 1, len(donors)) if donors else []
            y_acceptors = np.linspace(len(acceptors), 1, len(acceptors)) if acceptors else []

            pos_don = {name: (1.0, y) for name, y in zip(donors, y_donors)}
            pos_acc = {name: (3.0, y) for name, y in zip(acceptors, y_acceptors)}

            max_e = max(item.energy_kcal for item in top_items)

            # Draw connecting lines
            for item in top_items:
                p1 = pos_don.get(item.donor_label)
                p2 = pos_acc.get(item.acceptor_label)
                if p1 and p2:
                    lw = max(1.0, (item.energy_kcal / max_e) * 4.5)
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#22c55e", alpha=0.6, linewidth=lw)
                    # Midpoint annotation for strongest interactions
                    if item.energy_kcal >= 8.0:
                        mx = (p1[0] + p2[0]) / 2.0
                        my = (p1[1] + p2[1]) / 2.0
                        ax.text(mx, my, f"{item.energy_kcal:.1f}", color="#86efac", fontsize=7, ha="center")

            # Draw Donor Nodes
            for name, (nx, ny) in pos_don.items():
                ax.plot(nx, ny, "o", color="#38bdf8", markersize=9, zorder=5)
                ax.text(nx - 0.08, ny, name, color="#38bdf8", ha="right", va="center", fontsize=8, fontweight="bold")

            # Draw Acceptor Nodes
            for name, (nx, ny) in pos_acc.items():
                ax.plot(nx, ny, "o", color="#f43f5e", markersize=9, zorder=5)
                ax.text(nx + 0.08, ny, name, color="#f43f5e", ha="left", va="center", fontsize=8, fontweight="bold")

            ax.set_xlim(0.2, 3.8)
            y_top = max(len(donors), len(acceptors)) + 1
            ax.set_ylim(0, y_top)

            ax.text(1.0, y_top - 0.4, "Donors (i)\n[n, σ, π]", color="#38bdf8", ha="center", fontweight="bold", fontsize=10)
            ax.text(3.0, y_top - 0.4, "Acceptors (j)\n[σ*, π*]", color="#f43f5e", ha="center", fontweight="bold", fontsize=10)
            ax.set_title(f"NBO Second-Order Perturbation Network ({len(top_items)} delocalizations ≥ {min_e:.1f} kcal/mol)",
                         color="#f8fafc", fontweight="bold", pad=12)

        ax.set_axis_off()
        self.fig.tight_layout()
        self.canvas.draw()

        total_e2 = sum(item.energy_kcal for item in filtered)
        self.lbl_stats.setText(f"Showing top {len(top_items)} of {len(filtered)} interactions | Total Delocalization E(2): {total_e2:.1f} kcal/mol")

    def _on_save_diagram(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save NBO Network Diagram", "nbo_network.svg",
            "Scalable Vector Graphics (*.svg);;Portable Network Graphics (*.png);;PDF (*.pdf)"
        )
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            QtWidgets.QMessageBox.information(self, "Diagram Saved", f"Diagram successfully exported to:\n{path}")
