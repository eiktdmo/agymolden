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
        self.max_forces: List[float] = []
        self.rms_forces: List[float] = []
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

        self.combo_metric = QtWidgets.QComboBox()
        self.combo_metric.addItem("Energy Profile (ΔE / Total E)", "energy")
        self.combo_metric.addItem("Boltzmann Population at 298 K (%)", "boltzmann")
        self.combo_metric.addItem("Step Energy Change |ΔE_step| (kcal/mol)", "de_step")
        self.combo_metric.addItem("Geometry Step RMSD (Å)", "rmsd")
        self.combo_metric.addItem("Nuclear Forces (Max & RMS Ha/Bohr)", "forces")
        self.combo_metric.currentIndexChanged.connect(self._replot)
        ribbon.addWidget(QtWidgets.QLabel("Metric:"))
        ribbon.addWidget(self.combo_metric)

        self.combo_unit = QtWidgets.QComboBox()
        self.combo_unit.addItem("Relative Energy ΔE (kcal/mol)", "kcal")
        self.combo_unit.addItem("Relative Energy ΔE (eV)", "ev")
        self.combo_unit.addItem("Total Energy (Hartree)", "ha")
        self.combo_unit.currentIndexChanged.connect(self._replot)
        ribbon.addWidget(QtWidgets.QLabel("Units:"))
        ribbon.addWidget(self.combo_unit)

        self.chk_fit = QtWidgets.QCheckBox("Show Curve Interpolation")
        self.chk_fit.setChecked(True)
        self.chk_fit.toggled.connect(self._replot)
        ribbon.addWidget(self.chk_fit)

        self.btn_export_csv = QtWidgets.QPushButton("💾 Export CSV")
        self.btn_export_csv.setToolTip("Export PES / convergence data (Energy, ΔE, RMSD, Max & RMS Force) to CSV")
        self.btn_export_csv.setStyleSheet("background-color: #0d9488; color: white; font-weight: bold; padding: 4px 8px;")
        self.btn_export_csv.clicked.connect(self._on_export_csv_clicked)
        ribbon.addWidget(self.btn_export_csv)

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

    def load_scan_data(
        self,
        coordinates: List[float],
        energies_ha: List[float],
        coord_label: str = "Coordinate (Å)",
        trajectory_coords: Optional[List[np.ndarray]] = None,
        max_forces: Optional[List[float]] = None,
        rms_forces: Optional[List[float]] = None,
    ):
        self.coordinates = list(coordinates)
        self.energies_ha = list(energies_ha)
        self.coord_label = coord_label
        if trajectory_coords is not None:
            self.trajectory_coords = list(trajectory_coords)
        if max_forces is not None:
            self.max_forces = list(max_forces)
        if rms_forces is not None:
            self.rms_forces = list(rms_forces)
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

        x = np.array(self.coordinates, dtype=np.float64)
        e_ha = np.array(self.energies_ha, dtype=np.float64)
        metric_mode = self.combo_metric.currentData() if hasattr(self, "combo_metric") else "energy"
        unit_mode = self.combo_unit.currentData() or "kcal"
        self.combo_unit.setEnabled(metric_mode == "energy")

        y2 = None
        if metric_mode == "boltzmann":
            rel_kcal = (e_ha - np.min(e_ha)) * 627.509474
            rt_kcal = 0.592484  # R * 298.15 K in kcal/mol
            weights = np.exp(-np.clip(rel_kcal / rt_kcal, 0.0, 200.0))
            tot_w = float(np.sum(weights))
            y = (100.0 * weights / tot_w) if tot_w > 0 else np.zeros_like(e_ha)
            y_label = "Boltzmann Population at 298.15 K (%)"
            plot_title = "Thermodynamic Boltzmann Conformer / Step Populations (298.15 K)"
        elif metric_mode == "de_step":
            if len(e_ha) >= 2:
                diffs = np.abs(np.diff(e_ha, prepend=e_ha[0])) * 627.509474
                diffs[0] = diffs[1] if len(diffs) > 1 else 0.0
                y = diffs
            else:
                y = np.zeros_like(e_ha)
            y_label = "Step Energy Change |ΔE_step| (kcal/mol)"
            plot_title = "Optimization Step Energy Convergence |ΔE_step|"
            self.ax.axhline(0.01, color="#22c55e", linestyle=":", linewidth=1.2, label="Tight ΔE Threshold (0.01 kcal/mol)")
        elif metric_mode == "rmsd":
            if self.trajectory_coords and len(self.trajectory_coords) == len(x):
                rmsds = [0.0]
                for idx in range(1, len(self.trajectory_coords)):
                    c_prev = np.asarray(self.trajectory_coords[idx - 1], dtype=np.float64)
                    c_curr = np.asarray(self.trajectory_coords[idx], dtype=np.float64)
                    if c_prev.shape == c_curr.shape and c_curr.size > 0:
                        rmsds.append(float(np.sqrt(np.mean(np.sum((c_curr - c_prev) ** 2, axis=1)))))
                    else:
                        rmsds.append(0.0)
                if len(rmsds) > 1:
                    rmsds[0] = rmsds[1]
                y = np.array(rmsds, dtype=np.float64)
            else:
                y = np.zeros_like(x)
            y_label = "Step-to-Step Geometry RMSD (Å)"
            plot_title = "Geometry Step Displacement RMSD (Å)"
            self.ax.axhline(0.001, color="#22c55e", linestyle=":", linewidth=1.2, label="Convergence RMSD (0.001 Å)")
        elif metric_mode == "forces":
            if self.max_forces and len(self.max_forces) == len(x):
                y = np.array(self.max_forces, dtype=np.float64)
                if self.rms_forces and len(self.rms_forces) == len(x):
                    y2 = np.array(self.rms_forces, dtype=np.float64)
            elif len(e_ha) >= 2:
                # Numerical gradient magnitude estimate |dE/ds| when analytical step forces are unavailable
                y = np.abs(np.gradient(e_ha))
            else:
                y = np.zeros_like(x)
            y_label = "Nuclear Force Norm (Ha/Bohr)"
            plot_title = "Optimization Force Convergence (Max & RMS Force)"
            self.ax.axhline(0.00045, color="#22c55e", linestyle=":", linewidth=1.2, label="Max Force Threshold (0.00045 Ha/Bohr)")
        else:
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
            plot_title = "Potential Energy Surface (PES) Scan"

        # Smooth spline interpolation if requested
        if self.chk_fit.isChecked() and len(x) >= 4 and metric_mode == "energy":
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
            lbl_main = "Max Force" if (metric_mode == "forces" and y2 is not None) else None
            self.ax.plot(x, y, color="#38bdf8", lw=1.8, label=lbl_main)
            if y2 is not None:
                self.ax.plot(x, y2, color="#a855f7", lw=1.5, linestyle="--", label="RMS Force")
                self.ax.scatter(x, y2, color="#9333ea", s=26, edgecolors="#f8fafc", linewidth=0.8, zorder=4)
                self.ax.legend(fontsize=8, framealpha=0.7)

        # Discrete points
        self.ax.scatter(x, y, color="#0284c7", s=36, edgecolors="#f8fafc", linewidth=1.0, zorder=5)

        # Highlight currently active point
        if 0 <= self.active_step < len(x):
            act_x = x[self.active_step]
            act_y = y[self.active_step]
            self.ax.scatter([act_x], [act_y], color="#f59e0b", s=110, edgecolors="#ffffff", linewidth=2.0, zorder=6)
            self.ax.annotate(
                f"Step {self.active_step + 1}\n{act_x:.3f}, {act_y:+.4g}",
                (act_x, act_y), textcoords="offset points", xytext=(0, 12),
                ha="center", color="#fef08a", fontsize=8, fontweight="bold"
            )

        self.ax.set_xlabel(self.coord_label, fontsize=9)
        self.ax.set_ylabel(y_label, fontsize=9)
        self.ax.set_title(plot_title, color="#38bdf8", fontsize=10, fontweight="bold")
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

    def get_csv_string(self) -> str:
        """Format loaded PES scan / optimization convergence steps into a CSV string."""
        if not self.coordinates or not self.energies_ha:
            return "Step,Coordinate,Energy_Ha,Rel_Energy_kcal_mol\n"
        e_arr = np.asarray(self.energies_ha, dtype=np.float64)
        min_e = float(np.min(e_arr))
        rel_arr = (e_arr - min_e) * 627.509474
        w_boltz = np.exp(-np.clip(rel_arr / 0.592484, 0.0, 200.0))
        boltz_pct = (100.0 * w_boltz / float(np.sum(w_boltz))) if float(np.sum(w_boltz)) > 0 else np.zeros_like(e_arr)
        lines = ["Step,Coordinate,Energy_Ha,Rel_Energy_kcal_mol,Boltzmann_Pop_298K_pct,Step_dE_kcal_mol,Step_RMSD_Ang,Max_Force_Ha_Bohr,RMS_Force_Ha_Bohr"]
        for i, (coord_val, e_ha) in enumerate(zip(self.coordinates, self.energies_ha)):
            rel_kcal = (e_ha - min_e) * 627.509474
            b_pct = float(boltz_pct[i])
            de_step = abs(e_ha - self.energies_ha[i - 1]) * 627.509474 if i > 0 else 0.0
            rmsd_val = 0.0
            if i > 0 and self.trajectory_coords and i < len(self.trajectory_coords):
                c_prev = np.asarray(self.trajectory_coords[i - 1], dtype=np.float64)
                c_curr = np.asarray(self.trajectory_coords[i], dtype=np.float64)
                if c_prev.shape == c_curr.shape and c_curr.size > 0:
                    rmsd_val = float(np.sqrt(np.mean(np.sum((c_curr - c_prev) ** 2, axis=1))))
            max_f = self.max_forces[i] if (self.max_forces and i < len(self.max_forces)) else 0.0
            rms_f = self.rms_forces[i] if (self.rms_forces and i < len(self.rms_forces)) else 0.0
            lines.append(
                f"{i + 1},{coord_val:.6f},{e_ha:.8f},{rel_kcal:.4f},{b_pct:.2f},{de_step:.4f},{rmsd_val:.5f},{max_f:.6f},{rms_f:.6f}"
            )
        return "\n".join(lines) + "\n"

    def _on_export_csv_clicked(self):
        if not self.coordinates or not self.energies_ha:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PES / Convergence Data to CSV", "pes_convergence.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.get_csv_string())
        self.lbl_info.setText(f"Exported {len(self.coordinates)} steps to {os.path.basename(path)}.")

