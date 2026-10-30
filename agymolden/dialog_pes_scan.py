"""
Interactive Potential Energy Surface (PES) Scan & Morse Fit Curve Dialog (GaussView & newBRORSEN style).

Allows users to:
- Load or inspect 1D/2D PES scans and Morse/harmonic fit curves from newBRORSEN (.scan, _scan_points.txt, .out)
- Graph Energy (Hartree or kcal/mol) vs. Reaction Coordinate (Bond distance R in Å, Angle A in deg)
- Click or scrub points on the potential curve to jump the 3D molecular viewer directly to that step
"""

import os
from typing import List, Optional, Tuple
import numpy as np
from PyQt6 import QtCore, QtWidgets

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class PESScanDialog(QtWidgets.QDialog):
    point_selected = QtCore.pyqtSignal(int)  # step index

    def __init__(self, parent=None, viewer=None, trajectory_coords=None):
        super().__init__(parent)
        self.setWindowTitle("Potential Energy Surface (PES) Scan & Morse Fit (newBRORSEN style)")
        self.resize(750, 520)
        self.viewer = viewer
        self.trajectory_coords = trajectory_coords or []

        self.coordinates: List[float] = []
        self.energies_ha: List[float] = []
        self.coord_label: str = "Reaction Coordinate (Å)"
        self.active_step: int = 0

        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Top Ribbon Controls
        ribbon = QtWidgets.QHBoxLayout()
        btn_open = QtWidgets.QPushButton("📂 Open .scan / scan_points.txt...")
        btn_open.setStyleSheet("background-color: #0284c7; font-weight: bold; padding: 5px 10px;")
        btn_open.clicked.connect(self._on_open_scan_file)
        ribbon.addWidget(btn_open)

        self.combo_unit = QtWidgets.QComboBox()
        self.combo_unit.addItem("Relative Energy ΔE (kcal/mol)", "kcal")
        self.combo_unit.addItem("Relative Energy ΔE (eV)", "ev")
        self.combo_unit.addItem("Total Energy (Hartree)", "ha")
        self.combo_unit.currentIndexChanged.connect(self._replot)
        ribbon.addWidget(QtWidgets.QLabel("Energy Units:"))
        ribbon.addWidget(self.combo_unit)

        self.chk_fit = QtWidgets.QCheckBox("Show Curve Interpolation")
        self.chk_fit.setChecked(True)
        self.chk_fit.toggled.connect(self._replot)
        ribbon.addWidget(self.chk_fit)

        ribbon.addStretch()
        self.lbl_info = QtWidgets.QLabel("Click any point on the curve to update the 3D geometry.")
        self.lbl_info.setStyleSheet("color: #94a3b8; font-style: italic;")
        ribbon.addWidget(self.lbl_info)
        layout.addLayout(ribbon)

        # Matplotlib Canvas
        face = "#ffffff" if self.parent() and getattr(self.parent(), "viewer", None) and getattr(self.parent().viewer, "bg_color_mode", "") == "white" else "#1e1e24"
        self.fig = Figure(figsize=(6, 4), dpi=100, facecolor=face)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def _style_axes(self):
        is_white = self.parent() and getattr(self.parent(), "viewer", None) and getattr(self.parent().viewer, "bg_color_mode", "") == "white"
        ax_face = "#f8fafc" if is_white else "#141418"
        tick_col = "#475569" if is_white else "#94a3b8"
        lbl_col = "#0f172a" if is_white else "#cbd5e1"
        spine_col = "#cbd5e1" if is_white else "#2e2e38"
        self.ax.set_facecolor(ax_face)
        self.ax.tick_params(colors=tick_col, labelsize=8)
        self.ax.xaxis.label.set_color(lbl_col)
        self.ax.yaxis.label.set_color(lbl_col)
        for spine in self.ax.spines.values():
            spine.set_color(spine_col)
        self.ax.grid(True, color=spine_col, linestyle="--", alpha=0.6)

    def load_scan_data(self, coordinates: List[float], energies_ha: List[float], coord_label: str = "Coordinate (Å)"):
        self.coordinates = list(coordinates)
        self.energies_ha = list(energies_ha)
        self.coord_label = coord_label
        self.active_step = int(np.argmin(energies_ha)) if energies_ha else 0
        self._replot()

    def set_active_step(self, step: int):
        if 0 <= step < len(self.coordinates):
            self.active_step = step
            self._replot()

    def _on_open_scan_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open newBRORSEN PES Scan File", "",
            "Scan Files (*.scan *_scan_points.txt *.txt);;All Files (*)"
        )
        if not path:
            return
        self.load_scan_file(path)

    def load_scan_file(self, path: str):
        coords = []
        energies = []
        c_label = "Coordinate (Å)"

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        if not lines:
            return

        # Check header
        first = lines[0].split()
        if "Coordinate" in lines[0] or "Total_Ha" in lines[0]:
            lines = lines[1:]  # skip header

        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    r = float(parts[0])
                    # If TSV from write_scan_tsv: Coordinate HF CCSD CCSD(T) Total_Ha
                    e = float(parts[4]) if len(parts) >= 5 and parts[4] != "nan" else float(parts[1])
                    coords.append(r)
                    energies.append(e)
                except ValueError:
                    continue

        if coords and energies:
            self.load_scan_data(coords, energies, c_label)
            self.lbl_info.setText(f"Loaded {len(coords)} points from {os.path.basename(path)}.")

    def _replot(self):
        self.ax.clear()
        self._style_axes()

        if not self.coordinates or not self.energies_ha:
            self.ax.text(
                0.5, 0.5, "No PES scan data loaded.\nOpen a newBRORSEN .scan or _scan_points.txt file.",
                ha="center", va="center", color="#64748b", transform=self.ax.transAxes, fontsize=9
            )
            self.canvas.draw()
            return

        x = np.array(self.coordinates)
        e_ha = np.array(self.energies_ha)
        unit_mode = self.combo_unit.currentData() or "kcal"

        min_e = np.min(e_ha)
        if unit_mode == "kcal":
            y = (e_ha - min_e) * 627.509474  # Ha to kcal/mol
            y_label = "Relative Energy ΔE (kcal/mol)"
        elif unit_mode == "ev":
            y = (e_ha - min_e) * 27.211386  # Ha to eV
            y_label = "Relative Energy ΔE (eV)"
        else:
            y = e_ha
            y_label = "Total Energy (Hartree)"

        # Smooth spline interpolation if requested
        if self.chk_fit.isChecked() and len(x) >= 4:
            try:
                from scipy.interpolate import make_interp_spline
                sort_idx = np.argsort(x)
                xs = x[sort_idx]
                ys = y[sort_idx]
                x_dense = np.linspace(xs[0], xs[-1], 300)
                spl = make_interp_spline(xs, ys, k=3)
                y_dense = spl(x_dense)
                self.ax.plot(x_dense, y_dense, color="#38bdf8", lw=1.8, label="PES Profile")
            except Exception:
                self.ax.plot(x, y, color="#38bdf8", lw=1.8)
        else:
            self.ax.plot(x, y, color="#38bdf8", lw=1.8)

        # Discrete points
        self.ax.scatter(x, y, color="#0284c7", s=36, edgecolors="#f8fafc", linewidth=1.0, zorder=5)

        # Highlight currently active point
        if 0 <= self.active_step < len(x):
            act_x = x[self.active_step]
            act_y = y[self.active_step]
            self.ax.scatter([act_x], [act_y], color="#f59e0b", s=110, edgecolors="#ffffff", linewidth=2.0, zorder=6)
            self.ax.annotate(
                f"Step {self.active_step + 1}\n{act_x:.3f} Å, {act_y:+.2f}",
                (act_x, act_y), textcoords="offset points", xytext=(0, 12),
                ha="center", color="#fef08a", fontsize=8, fontweight="bold"
            )

        self.ax.set_xlabel(self.coord_label, fontsize=9)
        self.ax.set_ylabel(y_label, fontsize=9)
        self.ax.set_title("Potential Energy Surface (PES) Scan", color="#38bdf8", fontsize=10, fontweight="bold")
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_canvas_click(self, event):
        if event.xdata is None or not self.coordinates:
            return
        click_x = event.xdata
        # Find nearest point along x
        x_arr = np.array(self.coordinates)
        dists = np.abs(x_arr - click_x)
        best_idx = int(np.argmin(dists))

        self.active_step = best_idx
        self.point_selected.emit(best_idx)
        self._replot()
