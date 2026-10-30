"""
Interactive 1D Dihedral Rotor Scan Dialog with Embedded Matplotlib PES Plotter.

Provides:
- Real-time animated Potential Energy Surface (PES) curve
- Dihedral torsion angle vs. Energy (kcal/mol)
- Highlight of the active conformer on the PES plot
- Direct clicking or slider dragging to update the 3D viewport structure
"""

from typing import List, Tuple, Optional, Set, Dict
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .conformer_scan import run_dihedral_rotor_scan, DihedralScanResult


class DihedralScanDialog(QtWidgets.QDialog):
    conformer_selected = QtCore.pyqtSignal(object)  # np.ndarray coords

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        symbols: List[str],
        coords: np.ndarray,
        bonds: List[Tuple[int, int]],
        atom_indices: Tuple[int, int, int, int],
        user_bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
        frozen_indices: Optional[Set[int]] = None,
    ):
        super().__init__(parent)
        self.symbols = symbols
        self.coords = coords
        self.bonds = bonds
        self.atom_indices = atom_indices
        self.user_bond_orders = user_bond_orders or {}
        self.frozen_indices = frozen_indices or set()

        i1, i2, i3, i4 = atom_indices
        self.setWindowTitle(
            f"1D Dihedral Rotor PES Scan: {symbols[i1]}{i1+1}-{symbols[i2]}{i2+1}-{symbols[i3]}{i3+1}-{symbols[i4]}{i4+1}"
        )
        self.resize(850, 520)

        self.scan_result: Optional[DihedralScanResult] = None
        self.active_index: int = 0

        self._init_ui()
        self._run_scan()

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(8)

        # Top Controls Ribbon
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(10)

        top_row.addWidget(QtWidgets.QLabel("Steps (Resolution):"))
        self.spin_steps = QtWidgets.QSpinBox()
        self.spin_steps.setRange(8, 72)
        self.spin_steps.setValue(24)
        top_row.addWidget(self.spin_steps)

        top_row.addWidget(QtWidgets.QLabel("Relaxation Steps/Point:"))
        self.spin_relax = QtWidgets.QSpinBox()
        self.spin_relax.setRange(0, 100)
        self.spin_relax.setValue(15)
        self.spin_relax.setToolTip(
            "0 = Rigid Rotor Scan (faster);\n"
            ">0 = Relaxed PES Scan (optimizes remaining degrees of freedom at each torsion angle)"
        )
        top_row.addWidget(self.spin_relax)

        self.btn_recompute = QtWidgets.QPushButton("🔄 Recompute Scan")
        self.btn_recompute.setStyleSheet("background-color: #0d9488; font-weight: bold; padding: 4px 12px;")
        self.btn_recompute.clicked.connect(self._run_scan)
        top_row.addWidget(self.btn_recompute)

        top_row.addStretch()
        main_layout.addLayout(top_row)

        # Middle splitter: Left is Matplotlib Plot, Right is Table View
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Matplotlib PES Canvas
        plot_container = QtWidgets.QWidget()
        p_layout = QtWidgets.QVBoxLayout(plot_container)
        p_layout.setContentsMargins(0, 0, 0, 0)

        face = "#ffffff" if self.parent() and getattr(self.parent(), "viewer", None) and getattr(self.parent().viewer, "bg_color_mode", "") == "white" else "#1e1e24"
        self.fig = Figure(figsize=(5, 3.5), dpi=95, facecolor=face)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        p_layout.addWidget(self.canvas)
        splitter.addWidget(plot_container)

        # Right side: Table and Conformer Slider
        right_container = QtWidgets.QWidget()
        r_layout = QtWidgets.QVBoxLayout(right_container)
        r_layout.setContentsMargins(0, 0, 0, 0)
        r_layout.setSpacing(6)

        self.lbl_summary = QtWidgets.QLabel("Computing Potential Energy Surface...")
        is_white = self.parent() and getattr(self.parent(), "viewer", None) and getattr(self.parent().viewer, "bg_color_mode", "") == "white"
        sum_col = "#0369a1" if is_white else "#38bdf8"
        step_col = "#6d28d9" if is_white else "#a78bfa"
        self.lbl_summary.setStyleSheet(f"color: {sum_col}; font-weight: bold;")
        self.lbl_summary.setWordWrap(True)
        r_layout.addWidget(self.lbl_summary)

        # Slider for rapid conformer scrubbing
        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(QtWidgets.QLabel("Conformer:"))
        self.slider_conformer = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_conformer.setRange(0, 0)
        self.slider_conformer.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self.slider_conformer, stretch=1)
        self.lbl_step_num = QtWidgets.QLabel("1/1")
        self.lbl_step_num.setStyleSheet(f"font-weight: bold; color: {step_col};")
        slider_row.addWidget(self.lbl_step_num)
        r_layout.addLayout(slider_row)

        self.table_scan = QtWidgets.QTableWidget(0, 3)
        self.table_scan.setHorizontalHeaderLabels(["Step", "Dihedral (°)", "ΔE (kcal/mol)"])
        self.table_scan.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_scan.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_scan.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_scan.itemSelectionChanged.connect(self._on_table_row_selected)
        r_layout.addWidget(self.table_scan, stretch=1)

        splitter.addWidget(right_container)
        splitter.setSizes([480, 370])
        main_layout.addWidget(splitter, stretch=1)

        # Bottom buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_apply_min = QtWidgets.QPushButton("⚡ Set to Global Minimum Conformer")
        btn_apply_min.setStyleSheet("background-color: #059669; font-weight: bold;")
        btn_apply_min.clicked.connect(self._on_apply_minimum)
        btn_box.addWidget(btn_apply_min)

        btn_box.addStretch()

        btn_close = QtWidgets.QPushButton("Done")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        main_layout.addLayout(btn_box)

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
        self.ax.set_xlabel("Dihedral Angle (°)", fontsize=9)
        self.ax.set_ylabel("Relative Energy (kcal/mol)", fontsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(spine_col)
        self.ax.grid(True, color=spine_col, linestyle="--", alpha=0.6)

    def _run_scan(self):
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        try:
            self.scan_result = run_dihedral_rotor_scan(
                self.symbols,
                self.coords,
                self.bonds,
                self.atom_indices,
                n_steps=self.spin_steps.value(),
                relax_steps=self.spin_relax.value(),
                user_bond_orders=self.user_bond_orders,
                frozen_indices=self.frozen_indices,
            )
            res = self.scan_result
            self.slider_conformer.blockSignals(True)
            self.slider_conformer.setRange(0, len(res.points) - 1)
            self.slider_conformer.setValue(res.min_index)
            self.slider_conformer.blockSignals(False)

            # Populate table
            self.table_scan.blockSignals(True)
            self.table_scan.setRowCount(len(res.points))
            for r, pt in enumerate(res.points):
                it0 = QtWidgets.QTableWidgetItem(str(pt.step_index + 1))
                it0.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                it1 = QtWidgets.QTableWidgetItem(f"{pt.dihedral_deg:+.1f}°")
                it1.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                tag = " (Min)" if r == res.min_index else (" (Max)" if r == res.max_index else "")
                it2 = QtWidgets.QTableWidgetItem(f"{pt.relative_energy_kcal:.2f}{tag}")
                if r == res.min_index:
                    it2.setForeground(QtGui.QBrush(QtGui.QColor("#4ade80")))
                elif r == res.max_index:
                    it2.setForeground(QtGui.QBrush(QtGui.QColor("#f87171")))
                self.table_scan.setItem(r, 0, it0)
                self.table_scan.setItem(r, 1, it1)
                self.table_scan.setItem(r, 2, it2)
            self.table_scan.blockSignals(False)

            min_pt = res.points[res.min_index]
            self.lbl_summary.setText(
                f"Global Minimum: Step #{res.min_index+1} ({min_pt.dihedral_deg:+.1f}°) | "
                f"Barrier: {res.barrier_kcal:.2f} kcal/mol"
            )
            self._select_conformer(res.min_index)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _select_conformer(self, idx: int):
        if not self.scan_result or not (0 <= idx < len(self.scan_result.points)):
            return
        self.active_index = idx
        pt = self.scan_result.points[idx]
        self.lbl_step_num.setText(f"{idx + 1}/{len(self.scan_result.points)}")

        # Sync table selection
        self.table_scan.blockSignals(True)
        self.table_scan.selectRow(idx)
        self.table_scan.blockSignals(False)

        # Sync slider
        self.slider_conformer.blockSignals(True)
        self.slider_conformer.setValue(idx)
        self.slider_conformer.blockSignals(False)

        # Replot PES
        self._replot()

        # Emit geometry to 3D viewport
        self.conformer_selected.emit(pt.coords)

    def _on_slider_changed(self, val: int):
        self._select_conformer(val)

    def _on_table_row_selected(self):
        r = self.table_scan.currentRow()
        if r >= 0:
            self._select_conformer(r)

    def _on_apply_minimum(self):
        if self.scan_result:
            self._select_conformer(self.scan_result.min_index)

    def _replot(self):
        self.ax.clear()
        self._style_axes()

        if not self.scan_result or not self.scan_result.points:
            self.canvas.draw()
            return

        points = self.scan_result.points
        angles = np.array([p.dihedral_deg for p in points])
        energies = np.array([p.relative_energy_kcal for p in points])

        # Sort angles for smooth line plotting
        sort_order = np.argsort(angles)
        angles_s = angles[sort_order]
        energies_s = energies[sort_order]

        # Draw PES curve
        self.ax.plot(angles_s, energies_s, color="#0d9488", lw=2.2, label="PES Profile")
        self.ax.fill_between(angles_s, energies_s, color="#0d9488", alpha=0.18)
        self.ax.scatter(angles, energies, color="#38bdf8", s=30, zorder=3)

        # Highlight Global Min and Barrier Max
        min_pt = points[self.scan_result.min_index]
        max_pt = points[self.scan_result.max_index]
        self.ax.scatter([min_pt.dihedral_deg], [min_pt.relative_energy_kcal], color="#4ade80", s=80, marker="o", zorder=4, label="Minimum")
        self.ax.scatter([max_pt.dihedral_deg], [max_pt.relative_energy_kcal], color="#f87171", s=80, marker="^", zorder=4, label="Barrier Max")

        # Active conformer vertical marker
        active_pt = points[self.active_index]
        self.ax.scatter([active_pt.dihedral_deg], [active_pt.relative_energy_kcal], color="#f59e0b", s=130, edgecolor="#ffffff", lw=2.0, zorder=5)
        self.ax.axvline(active_pt.dihedral_deg, color="#f59e0b", linestyle=":", alpha=0.8, lw=1.5)

        self.ax.set_xlim(-185, 185)
        self.ax.set_ylim(-0.2, max(0.5, np.max(energies) * 1.15))
        is_white = self.parent() and getattr(self.parent(), "viewer", None) and getattr(self.parent().viewer, "bg_color_mode", "") == "white"
        leg_bg = "#ffffff" if is_white else "#141418"
        leg_edge = "#cbd5e1" if is_white else "#2e2e38"
        leg_text = "#0f172a" if is_white else "#cbd5e1"
        self.ax.legend(loc="upper right", facecolor=leg_bg, edgecolor=leg_edge, fontsize=8, labelcolor=leg_text)
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_plot_click(self, event):
        if event.xdata is None or not self.scan_result:
            return
        clicked_deg = event.xdata
        # Find closest point
        angles = np.array([p.dihedral_deg for p in self.scan_result.points])
        diffs = np.abs(angles - clicked_deg)
        best_idx = int(np.argmin(diffs))
        self._select_conformer(best_idx)
