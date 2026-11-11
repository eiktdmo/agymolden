"""Interactive Steric & Catalysis Descriptors Dialog (%V_bur, Topographic Map, Sterimol)."""

from __future__ import annotations

import os
from typing import List, Optional
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .sterics import compute_buried_volume, compute_sterimol, StericBuriedVolumeResult, SterimolResult


class StericsDialog(QtWidgets.QDialog):
    """Interactive Catalysis Steric Map and Verloop Sterimol Analyzer Dialog."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        symbols: Optional[List[str]] = None,
        coords: Optional[np.ndarray] = None,
        selected_atoms: Optional[List[int]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("🛡️ Catalysis Sterics: Buried Volume (%V_bur) & Sterimol (L, B1, B5)")
        self.resize(920, 680)
        self.setMinimumSize(780, 560)

        self.symbols = list(symbols) if symbols else []
        self.coords = np.array(coords, copy=True) if coords is not None else np.empty((0, 3))
        self.selected_atoms = selected_atoms or []

        self._build_ui()
        self._populate_atoms()
        self._compute_all()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header Info Banner
        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>Quantitative Steric Descriptors & Topographic Catalysis Map</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Evaluate Percent Buried Volume (%V_bur), quadrant steric distribution (SambVca style), "
            "2D topographic altitude contour maps, and Verloop Sterimol parameters (L, B1, B5) along any catalytic coordination axis."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_desc.setWordWrap(True)
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        # Control Row
        ctrl_box = QtWidgets.QGroupBox("Catalytic Center & Parameters")
        c_lay = QtWidgets.QHBoxLayout(ctrl_box)

        c_lay.addWidget(QtWidgets.QLabel("<b>Coordination Center:</b>"))
        self.combo_center = QtWidgets.QComboBox()
        self.combo_center.currentIndexChanged.connect(self._compute_all)
        c_lay.addWidget(self.combo_center, 1)

        c_lay.addWidget(QtWidgets.QLabel("Sphere Radius R (Å):"))
        self.spin_radius = QtWidgets.QDoubleSpinBox()
        self.spin_radius.setRange(2.0, 8.0)
        self.spin_radius.setValue(3.5)
        self.spin_radius.setSingleStep(0.5)
        self.spin_radius.valueChanged.connect(self._compute_all)
        c_lay.addWidget(self.spin_radius)

        c_lay.addWidget(QtWidgets.QLabel("Sterimol Bond (Base → Sub):"))
        self.combo_sub = QtWidgets.QComboBox()
        self.combo_sub.currentIndexChanged.connect(self._compute_all)
        c_lay.addWidget(self.combo_sub, 1)

        layout.addWidget(ctrl_box)

        # Splitter: 2D Contour Topographic Map (Left) vs Quadrants & Sterimol Table (Right)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left: Matplotlib Topographic Map
        map_panel = QtWidgets.QFrame()
        m_lay = QtWidgets.QVBoxLayout(map_panel)
        m_lay.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(5.2, 4.6), dpi=100, facecolor="#1e293b")
        self.canvas = FigureCanvas(self.fig)
        m_lay.addWidget(self.canvas)
        splitter.addWidget(map_panel)

        # Right: Tabular Summary
        right_panel = QtWidgets.QFrame()
        r_lay = QtWidgets.QVBoxLayout(right_panel)
        r_lay.setContentsMargins(6, 0, 0, 0)

        self.tbl_metrics = QtWidgets.QTableWidget()
        self.tbl_metrics.setColumnCount(2)
        self.tbl_metrics.setHorizontalHeaderLabels(["Descriptor", "Value"])
        self.tbl_metrics.horizontalHeader().setStretchLastSection(True)
        self.tbl_metrics.verticalHeader().setVisible(False)
        self.tbl_metrics.setStyleSheet("background-color: #020617; color: #f8fafc; gridline-color: #1e293b;")
        r_lay.addWidget(self.tbl_metrics)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter, 1)

        # Bottom Bar
        b_bar = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("📋 Copy Steric Descriptors for SI")
        btn_copy.clicked.connect(self._on_copy_summary)
        b_bar.addWidget(btn_copy)

        btn_save = QtWidgets.QPushButton("💾 Save Topographic Map (SVG/PNG)...")
        btn_save.clicked.connect(self._on_save_map)
        b_bar.addWidget(btn_save)

        b_bar.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        b_bar.addWidget(btn_close)
        layout.addLayout(b_bar)

    def _populate_atoms(self):
        self.combo_center.blockSignals(True)
        self.combo_sub.blockSignals(True)
        self.combo_center.clear()
        self.combo_sub.clear()

        for i, s in enumerate(self.symbols):
            lbl = f"Atom {i+1}: {s}"
            self.combo_center.addItem(lbl, i)
            self.combo_sub.addItem(lbl, i)

        # Default center to selected atom or transition metal if present
        def_idx = 0
        if self.selected_atoms:
            def_idx = self.selected_atoms[0]
        else:
            metals = {"FE", "PD", "PT", "RH", "RU", "NI", "CU", "CO", "IR", "AU", "AG", "TI", "ZR"}
            for i, s in enumerate(self.symbols):
                if s.upper() in metals:
                    def_idx = i
                    break

        self.combo_center.setCurrentIndex(def_idx)
        if len(self.symbols) > 1:
            sub_idx = 1 if def_idx != 1 else 0
            self.combo_sub.setCurrentIndex(sub_idx)

        self.combo_center.blockSignals(False)
        self.combo_sub.blockSignals(False)

    def _compute_all(self):
        c_idx = self.combo_center.currentData()
        sub_idx = self.combo_sub.currentData()
        if c_idx is None or len(self.symbols) == 0:
            return

        r = self.spin_radius.value()
        z_axis = (c_idx, sub_idx) if (sub_idx is not None and sub_idx != c_idx) else None

        vbur_res = compute_buried_volume(self.symbols, self.coords, center_idx=c_idx, sphere_radius=r, z_axis_atoms=z_axis)
        self._last_vbur = vbur_res

        # Sterimol
        ster_res = None
        if sub_idx is not None and sub_idx != c_idx:
            try:
                ster_res = compute_sterimol(self.symbols, self.coords, atom_base_idx=c_idx, atom_substituent_idx=sub_idx)
            except Exception:
                pass
        self._last_sterimol = ster_res

        self._render_topographic_map(vbur_res)
        self._populate_table(vbur_res, ster_res)

    def _render_topographic_map(self, res: StericBuriedVolumeResult):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#0f172a")

        ext = res.grid_extent
        grid = res.steric_map_grid
        extent = [-ext, ext, -ext, ext]

        # Draw filled contours (SambVca altitude aesthetic)
        levels = np.linspace(-ext, ext, 14)
        cf = ax.contourf(grid.T, levels=levels, extent=extent, cmap="turbo", origin="lower")
        cbar = self.fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Steric Height z (Å)", color="#e2e8f0", fontweight="bold", fontsize=9)
        cbar.ax.tick_params(colors="#94a3b8")

        # Quadrant dividing crosshairs
        ax.axhline(0, color="#ffffff", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axvline(0, color="#ffffff", linestyle="--", linewidth=1.0, alpha=0.6)

        # Annotate quadrant percentages
        q = res.quadrants
        ax.text(ext * 0.5, ext * 0.8, f"NE: {q['Q1_NE']:.1f}%", color="#ffffff", fontweight="bold", fontsize=9, ha="center")
        ax.text(-ext * 0.5, ext * 0.8, f"NW: {q['Q2_NW']:.1f}%", color="#ffffff", fontweight="bold", fontsize=9, ha="center")
        ax.text(-ext * 0.5, -ext * 0.85, f"SW: {q['Q3_SW']:.1f}%", color="#ffffff", fontweight="bold", fontsize=9, ha="center")
        ax.text(ext * 0.5, -ext * 0.85, f"SE: {q['Q4_SE']:.1f}%", color="#ffffff", fontweight="bold", fontsize=9, ha="center")

        # Coordination sphere boundary
        circle = matplotlib.patches.Circle((0, 0), ext, fill=False, color="#ffffff", linewidth=1.8, linestyle=":")
        ax.add_patch(circle)

        sym = self.symbols[res.center_idx]
        ax.set_title(f"%V_bur Topographic Steric Map ({sym}{res.center_idx+1}, R={res.sphere_radius:.1f} Å)",
                     color="#f8fafc", fontweight="bold", fontsize=11, pad=10)
        ax.set_xlabel("x (Å)", color="#e2e8f0", fontweight="bold")
        ax.set_ylabel("y (Å)", color="#e2e8f0", fontweight="bold")
        ax.tick_params(colors="#94a3b8")
        ax.set_xlim(-ext * 1.05, ext * 1.05)
        ax.set_ylim(-ext * 1.05, ext * 1.05)

        self.fig.tight_layout()
        self.canvas.draw()

    def _populate_table(self, vbur: StericBuriedVolumeResult, ster: Optional[SterimolResult]):
        rows = [
            ("Coordination Center", f"{self.symbols[vbur.center_idx]}{vbur.center_idx+1}"),
            ("Sphere Radius (R)", f"{vbur.sphere_radius:.2f} Å"),
            ("Percent Buried Volume (%V_bur)", f"<b>{vbur.percent_v_bur:.2f} %</b>"),
            ("Buried Volume (V_bur)", f"{vbur.buried_volume_angstrom3:.2f} Å³"),
            ("Total Sphere Volume (V_tot)", f"{vbur.total_sphere_volume:.2f} Å³"),
            ("Quadrant 1 (NE) %V_bur", f"{vbur.quadrants['Q1_NE']:.2f} %"),
            ("Quadrant 2 (NW) %V_bur", f"{vbur.quadrants['Q2_NW']:.2f} %"),
            ("Quadrant 3 (SW) %V_bur", f"{vbur.quadrants['Q3_SW']:.2f} %"),
            ("Quadrant 4 (SE) %V_bur", f"{vbur.quadrants['Q4_SE']:.2f} %"),
        ]

        if ster:
            rows.extend([
                ("Sterimol Length (L)", f"<b>{ster.l_value:.2f} Å</b>"),
                ("Sterimol Min Width (B1)", f"<b>{ster.b1_value:.2f} Å</b>"),
                ("Sterimol Max Width (B5)", f"<b>{ster.b5_value:.2f} Å</b>"),
            ])

        self.tbl_metrics.setRowCount(len(rows))
        for r, (prop, val) in enumerate(rows):
            item_p = QtWidgets.QTableWidgetItem(prop)
            item_v = QtWidgets.QTableWidgetItem(val.replace("<b>", "").replace("</b>", ""))
            if "%V_bur" in prop or "Sterimol" in prop:
                item_v.setFont(QtGui.QFont("", -1, QtGui.QFont.Weight.Bold))
                item_v.setForeground(QtGui.QColor("#38bdf8"))
            self.tbl_metrics.setItem(r, 0, item_p)
            self.tbl_metrics.setItem(r, 1, item_v)

    def _on_copy_summary(self):
        lines = ["# Quantitative Catalysis Steric Descriptors\n"]
        for r in range(self.tbl_metrics.rowCount()):
            p = self.tbl_metrics.item(r, 0).text()
            v = self.tbl_metrics.item(r, 1).text()
            lines.append(f"{p}: {v}")
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        QtWidgets.QMessageBox.information(self, "Copied", "Steric parameters copied to clipboard.")

    def _on_save_map(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Topographic Steric Map", "steric_topographic_map.svg",
            "Scalable Vector Graphics (*.svg);;Portable Network Graphics (*.png);;PDF (*.pdf)"
        )
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            QtWidgets.QMessageBox.information(self, "Saved", f"Steric map saved to:\n{path}")
