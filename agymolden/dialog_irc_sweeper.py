"""
Interactive Intrinsic Reaction Coordinate (IRC) Reaction Sweeper & Energy Profile Dialog.

Features:
- Visualizes Minimum Energy Pathways (MEP) from Reactant through Transition State (TS ‡) to Product.
- Interactive 2D Energy vs Reaction Coordinate (s) plot with real-time cursor tracking.
- Interactive scrubbing and movie playback (Bounce, Loop, Forward, Reverse) with live 3D viewer sync.
- Direct chemical thermodynamics display:
    ΔE‡_fwd (Forward Barrier), ΔE‡_rev (Reverse Barrier), ΔE_rxn (Reaction Energy, Exothermic/Endothermic).
- Fast landmark jumps: Reactant (R), Transition State (TS ‡), and Product (P).
- 1-Click Chemcraft-style geometry adoption into the main Z-Matrix editor (with Undo).
- Export to publication-ready CSV, multi-frame IRC XYZ movie, or high-res vector figure.
"""

import logging
import os
from typing import List, Optional
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

log = logging.getLogger(__name__)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .irc_sweeper import (
    HARTREE_TO_EV,
    HARTREE_TO_KCAL,
    HARTREE_TO_KJ,
    IRCPathway,
    IRCPoint,
    assemble_irc_pathway,
    parse_gaussian_irc_text,
    parse_orca_irc_text,
    parse_irc_xyz,
)


class IRCReactionSweeperDialog(QtWidgets.QDialog):
    """Interactive Intrinsic Reaction Coordinate (IRC) Sweeper Dialog."""

    step_selected = QtCore.pyqtSignal(int)  # Emits 0-based step index
    geometry_adopted = QtCore.pyqtSignal(list, np.ndarray)  # (symbols, coords) to adopt into editor

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        viewer: Optional[QtWidgets.QWidget] = None,
        pathway: Optional[IRCPathway] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("⚛️ Interactive IRC Reaction Sweeper & Energy Profile")
        self.resize(880, 620)
        self.viewer = viewer
        self.pathway = pathway

        self.current_step: int = pathway.ts_index if (pathway and pathway.points) else 0
        self.is_playing: bool = False
        self.play_direction: int = 1  # +1 forward, -1 backward (for bounce mode)
        self.unit_mode: str = "kcal"

        # Playback timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

        self._init_ui()

        if self.pathway and self.pathway.n_points > 0:
            self._update_pathway_display()

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)

        # 1. Header & Reaction Thermodynamics Banner
        self.grp_thermo = QtWidgets.QGroupBox("Reaction Thermodynamics & Activation Barriers")
        self.grp_thermo.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #334155; border-radius: 6px; margin-top: 6px; padding-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #38bdf8; }"
        )
        thermo_layout = QtWidgets.QHBoxLayout(self.grp_thermo)
        thermo_layout.setContentsMargins(8, 6, 8, 6)

        self.lbl_thermo_summary = QtWidgets.QLabel("No IRC Pathway loaded.")
        self.lbl_thermo_summary.setStyleSheet("font-size: 12px; font-weight: bold; color: #f8fafc;")
        thermo_layout.addWidget(self.lbl_thermo_summary, stretch=1)

        self.lbl_status_badge = QtWidgets.QLabel("")
        self.lbl_status_badge.setStyleSheet("padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")
        thermo_layout.addWidget(self.lbl_status_badge)

        btn_open = QtWidgets.QPushButton("📂 Open IRC File...")
        btn_open.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 4px 10px;")
        btn_open.setToolTip("Open Gaussian / ORCA IRC calculation output or multi-frame IRC XYZ file")
        btn_open.clicked.connect(self._on_open_file_dialog)
        thermo_layout.addWidget(btn_open)

        main_layout.addWidget(self.grp_thermo)

        # 2. Matplotlib Canvas
        face = "#ffffff" if self.viewer and getattr(self.viewer, "bg_color_mode", "") == "white" else "#18181b"
        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor=face)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas, stretch=1)

        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)

        # 3. Interactive Sweeping & Playback Ribbon
        playback_box = QtWidgets.QGroupBox("Reaction Coordinate Sweeper & Playback")
        playback_box.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #334155; border-radius: 6px; margin-top: 4px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #a78bfa; }"
        )
        p_layout = QtWidgets.QVBoxLayout(playback_box)
        p_layout.setContentsMargins(8, 6, 8, 6)
        p_layout.setSpacing(6)

        # Row 1: Playback Buttons & Scrubber Slider
        r1 = QtWidgets.QHBoxLayout()
        r1.setSpacing(4)

        self.btn_first = QtWidgets.QPushButton("⏮")
        self.btn_first.setFixedWidth(28)
        self.btn_first.setToolTip("Jump to Reactant (Home)")
        self.btn_first.clicked.connect(self._on_jump_reactant)
        r1.addWidget(self.btn_first)

        self.btn_prev = QtWidgets.QPushButton("◀")
        self.btn_prev.setFixedWidth(28)
        self.btn_prev.setToolTip("Previous reaction coordinate step (Left)")
        self.btn_prev.clicked.connect(self._on_step_prev)
        r1.addWidget(self.btn_prev)

        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_play.setFixedWidth(68)
        self.btn_play.setStyleSheet("background-color: #6366f1; color: white; font-weight: bold;")
        self.btn_play.setToolTip("Play/Pause reaction pathway animation (Space)")
        self.btn_play.clicked.connect(self._on_toggle_play)
        r1.addWidget(self.btn_play)

        self.btn_next = QtWidgets.QPushButton("▶")
        self.btn_next.setFixedWidth(28)
        self.btn_next.setToolTip("Next reaction coordinate step (Right)")
        self.btn_next.clicked.connect(self._on_step_next)
        r1.addWidget(self.btn_next)

        self.btn_last = QtWidgets.QPushButton("⏭")
        self.btn_last.setFixedWidth(28)
        self.btn_last.setToolTip("Jump to Product (End)")
        self.btn_last.clicked.connect(self._on_jump_product)
        r1.addWidget(self.btn_last)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        r1.addWidget(self.slider, stretch=1)

        self.lbl_step_info = QtWidgets.QLabel("Step 0/0")
        self.lbl_step_info.setStyleSheet("color: #38bdf8; font-weight: bold; min-width: 80px;")
        r1.addWidget(self.lbl_step_info)

        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItem("Bounce ⮂ (R ↔ TS ↔ P)", "bounce")
        self.combo_mode.addItem("Forward ➔ (R → TS → P)", "forward")
        self.combo_mode.addItem("Reverse ⬅ (P → TS → R)", "reverse")
        self.combo_mode.addItem("Loop 🔁", "loop")
        self.combo_mode.setToolTip("IRC animation sweep mode")
        self.combo_mode.setFixedWidth(150)
        r1.addWidget(self.combo_mode)

        self.combo_fps = QtWidgets.QComboBox()
        self.combo_fps.addItem("5 FPS (Slow)", 200)
        self.combo_fps.addItem("10 FPS (Normal)", 100)
        self.combo_fps.addItem("20 FPS (Fast)", 50)
        self.combo_fps.addItem("30 FPS (Smooth)", 33)
        self.combo_fps.setCurrentIndex(1)
        self.combo_fps.setFixedWidth(110)
        self.combo_fps.currentIndexChanged.connect(self._on_speed_changed)
        r1.addWidget(self.combo_fps)

        p_layout.addLayout(r1)

        # Row 2: Landmark Jumps & Direct Chemistry Actions
        r2 = QtWidgets.QHBoxLayout()
        r2.setSpacing(6)

        r2.addWidget(QtWidgets.QLabel("<b>Landmark Jumps:</b>"))

        btn_jump_r = QtWidgets.QPushButton("🌿 Reactant (R)")
        btn_jump_r.setStyleSheet("background-color: #15803d; color: white; font-weight: bold; padding: 2px 8px;")
        btn_jump_r.clicked.connect(self._on_jump_reactant)
        r2.addWidget(btn_jump_r)

        btn_jump_ts = QtWidgets.QPushButton("⚛️ Transition State (TS ‡)")
        btn_jump_ts.setStyleSheet("background-color: #d97706; color: white; font-weight: bold; padding: 2px 8px;")
        btn_jump_ts.clicked.connect(self._on_jump_ts)
        r2.addWidget(btn_jump_ts)

        btn_jump_p = QtWidgets.QPushButton("🔷 Product (P)")
        btn_jump_p.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 2px 8px;")
        btn_jump_p.clicked.connect(self._on_jump_product)
        r2.addWidget(btn_jump_p)

        r2.addStretch()

        r2.addWidget(QtWidgets.QLabel("Units:"))
        self.combo_units = QtWidgets.QComboBox()
        self.combo_units.addItem("Relative ΔE (kcal/mol)", "kcal")
        self.combo_units.addItem("Relative ΔE (kJ/mol)", "kj")
        self.combo_units.addItem("Relative ΔE (eV)", "ev")
        self.combo_units.addItem("Total Energy E (Hartree)", "ha")
        self.combo_units.currentIndexChanged.connect(self._on_unit_changed)
        r2.addWidget(self.combo_units)

        self.btn_adopt = QtWidgets.QPushButton("📌 Adopt Current Step")
        self.btn_adopt.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 3px 10px;")
        self.btn_adopt.setToolTip("Chemcraft style: Adopt the currently selected reaction coordinate geometry into the Z-Matrix editor")
        self.btn_adopt.clicked.connect(self._on_adopt_current_geometry)
        r2.addWidget(self.btn_adopt)

        self.btn_adopt_ts = QtWidgets.QPushButton("⚛️ Adopt TS (‡)")
        self.btn_adopt_ts.setStyleSheet("background-color: #b45309; color: white; font-weight: bold; padding: 3px 10px;")
        self.btn_adopt_ts.setToolTip("Adopt the saddle-point Transition State geometry into the Z-Matrix editor")
        self.btn_adopt_ts.clicked.connect(self._on_adopt_ts_geometry)
        r2.addWidget(self.btn_adopt_ts)

        self.btn_export_csv = QtWidgets.QPushButton("💾 Export CSV")
        self.btn_export_csv.setToolTip("Export publication-ready reaction coordinate profile to CSV")
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        r2.addWidget(self.btn_export_csv)

        self.btn_export_xyz = QtWidgets.QPushButton("🎬 Export Movie (XYZ)")
        self.btn_export_xyz.setToolTip("Export the continuous reaction path as a multi-frame XYZ movie")
        self.btn_export_xyz.clicked.connect(self._on_export_xyz)
        r2.addWidget(self.btn_export_xyz)

        p_layout.addLayout(r2)
        main_layout.addWidget(playback_box)

    def _style_axes(self):
        is_white = self.viewer and getattr(self.viewer, "bg_color_mode", "") == "white"
        ax_face = "#f8fafc" if is_white else "#0f0f12"
        tick_col = "#475569" if is_white else "#94a3b8"
        lbl_col = "#0f172a" if is_white else "#cbd5e1"
        spine_col = "#cbd5e1" if is_white else "#27272a"
        self.ax.set_facecolor(ax_face)
        self.ax.tick_params(colors=tick_col, labelsize=8)
        self.ax.xaxis.label.set_color(lbl_col)
        self.ax.yaxis.label.set_color(lbl_col)
        for spine in self.ax.spines.values():
            spine.set_color(spine_col)
        self.ax.grid(True, color=spine_col, linestyle="--", alpha=0.5)

    def load_pathway(self, pathway: IRCPathway):
        """Load an IRCPathway object into the dialog."""
        self.pathway = pathway
        self.current_step = pathway.ts_index if pathway.points else 0
        self._update_pathway_display()

    def load_from_text(self, text: str, filename: str = ""):
        """Parse and load an IRC pathway from Gaussian log, ORCA log, or XYZ text."""
        # Try Gaussian first
        pathway = parse_gaussian_irc_text(text)
        if pathway is None:
            pathway = parse_orca_irc_text(text)
        if pathway is None:
            pathway = parse_irc_xyz(text)

        if pathway and pathway.n_points > 0:
            self.load_pathway(pathway)
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Could Not Parse IRC Pathway",
                f"Could not extract Intrinsic Reaction Coordinate data from {filename or 'input'}.\n"
                "Supported formats include:\n"
                "- Gaussian IRC calculation logs\n"
                "- ORCA IRC calculation logs / _IRC_Full.xyz\n"
                "- Multi-frame IRC XYZ files with reaction coordinates and energies.",
            )

    def _update_pathway_display(self):
        if not self.pathway or self.pathway.n_points == 0:
            return

        p = self.pathway
        n = p.n_points

        # Update thermodynamics banner
        exo_str = "Exothermic (Release)" if p.is_exothermic else "Endothermic (Uphill)"
        exo_col = "#22c55e" if p.is_exothermic else "#f97316"
        self.lbl_thermo_summary.setText(
            f"<b>MEP Pathway:</b> {len(p.symbols)} atoms, {n} points | "
            f"<b>Forward Barrier ΔE‡<sub>fwd</sub>:</b> <span style='color: #f59e0b;'>+{p.delta_e_fwd_kcal:.2f} kcal/mol</span> ({p.delta_e_fwd_kcal * 4.184:.2f} kJ/mol) | "
            f"<b>Reverse Barrier ΔE‡<sub>rev</sub>:</b> <span style='color: #38bdf8;'>+{p.delta_e_rev_kcal:.2f} kcal/mol</span> ({p.delta_e_rev_kcal * 4.184:.2f} kJ/mol) | "
            f"<b>Reaction Energy ΔE<sub>rxn</sub>:</b> <span style='color: {exo_col};'>{p.delta_e_rxn_kcal:+.2f} kcal/mol</span>"
        )
        self.lbl_status_badge.setText(exo_str)
        self.lbl_status_badge.setStyleSheet(
            f"background-color: {exo_col}22; color: {exo_col}; border: 1px solid {exo_col}; padding: 3px 8px; border-radius: 4px; font-weight: bold;"
        )

        # Update slider
        self.slider.blockSignals(True)
        self.slider.setRange(0, n - 1)
        self.slider.setValue(self.current_step)
        self.slider.blockSignals(False)

        self._replot()
        self._sync_active_step()

    def _replot(self):
        self.ax.clear()
        self._style_axes()

        if not self.pathway or self.pathway.n_points == 0:
            self.ax.text(
                0.5, 0.5, "No IRC Pathway loaded.\nOpen a Gaussian, ORCA, or XYZ IRC calculation file.",
                ha="center", va="center", color="#64748b", transform=self.ax.transAxes, fontsize=10,
            )
            self.canvas.draw()
            return

        p = self.pathway
        s_arr = np.array(p.get_reaction_coords(), dtype=np.float64)
        y_vals = np.array(p.get_energies_relative(self.unit_mode), dtype=np.float64)

        unit_label = {
            "kcal": "Relative Energy ΔE (kcal/mol)",
            "kj": "Relative Energy ΔE (kJ/mol)",
            "ev": "Relative Energy ΔE (eV)",
            "ha": "Total Energy E (Hartree)",
        }.get(self.unit_mode, "Relative Energy ΔE (kcal/mol)")

        # Smooth spline interpolation if >= 4 points
        if len(s_arr) >= 4:
            try:
                from scipy.interpolate import make_interp_spline
                s_dense = np.linspace(s_arr[0], s_arr[-1], 350)
                spl = make_interp_spline(s_arr, y_vals, k=3)
                y_dense = spl(s_dense)
                self.ax.plot(s_dense, y_dense, color="#818cf8", lw=2.2, label="MEP Reaction Profile")
            except Exception:
                self.ax.plot(s_arr, y_vals, color="#818cf8", lw=2.2)
        else:
            self.ax.plot(s_arr, y_vals, color="#818cf8", lw=2.2)

        # Plot all intermediate points
        self.ax.scatter(s_arr, y_vals, color="#38bdf8", s=32, edgecolors="#0f172a", linewidth=0.8, zorder=4)

        # Highlight landmark points: Reactant (Green), TS (Amber), Product (Blue)
        r_idx = p.reactant_index
        ts_idx = p.ts_index
        p_idx = p.product_index

        if 0 <= r_idx < len(s_arr):
            self.ax.scatter([s_arr[r_idx]], [y_vals[r_idx]], color="#22c55e", s=90, edgecolors="#ffffff", linewidth=1.5, zorder=6, label="Reactant (R)")
            self.ax.annotate("R", (s_arr[r_idx], y_vals[r_idx]), textcoords="offset points", xytext=(0, -16), ha="center", color="#86efac", fontsize=9, fontweight="bold")

        if 0 <= ts_idx < len(s_arr):
            self.ax.scatter([s_arr[ts_idx]], [y_vals[ts_idx]], color="#f59e0b", s=130, edgecolors="#ffffff", linewidth=2.0, zorder=7, label="TS (‡)")
            self.ax.annotate("TS ‡", (s_arr[ts_idx], y_vals[ts_idx]), textcoords="offset points", xytext=(0, 14), ha="center", color="#fde047", fontsize=10, fontweight="bold")

        if 0 <= p_idx < len(s_arr):
            self.ax.scatter([s_arr[p_idx]], [y_vals[p_idx]], color="#0ea5e9", s=90, edgecolors="#ffffff", linewidth=1.5, zorder=6, label="Product (P)")
            self.ax.annotate("P", (s_arr[p_idx], y_vals[p_idx]), textcoords="offset points", xytext=(0, -16), ha="center", color="#7dd3fc", fontsize=9, fontweight="bold")

        # Current active step indicator
        if 0 <= self.current_step < len(s_arr):
            act_s = s_arr[self.current_step]
            act_y = y_vals[self.current_step]
            self.ax.axvline(act_s, color="#e0e7ff", linestyle=":", lw=1.2, alpha=0.8)
            self.ax.scatter([act_s], [act_y], color="#ec4899", s=140, edgecolors="#ffffff", linewidth=2.2, zorder=8)
            pt = p.points[self.current_step]
            self.ax.annotate(
                f"Step {self.current_step + 1}: {pt.reaction_coord:+.3f} {p.coord_unit}\nΔE = {act_y:+.2f}",
                (act_s, act_y),
                textcoords="offset points",
                xytext=(0, 16),
                ha="center",
                color="#fbcfe8",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#831843", edgecolor="#f472b6", alpha=0.85),
            )

        self.ax.set_xlabel(f"Intrinsic Reaction Coordinate s ({p.coord_unit})", fontsize=9, fontweight="bold")
        self.ax.set_ylabel(unit_label, fontsize=9, fontweight="bold")
        self.ax.set_title("Intrinsic Reaction Coordinate (IRC) Minimum Energy Path", color="#a5b4fc", fontsize=10, fontweight="bold")
        self.ax.legend(loc="best", fontsize=8, framealpha=0.6)
        self.fig.tight_layout()
        self.canvas.draw()

    def _sync_active_step(self):
        if not self.pathway or not (0 <= self.current_step < self.pathway.n_points):
            return

        pt = self.pathway.points[self.current_step]
        n = self.pathway.n_points
        self.lbl_step_info.setText(f"Step {self.current_step + 1}/{n}")

        # Update 3D viewer if present
        if self.viewer is not None and hasattr(self.viewer, "set_coords"):
            try:
                self.viewer.set_coords(pt.coords)
                if hasattr(self.viewer, "update"):
                    self.viewer.update()
            except Exception as e:
                log.debug("Viewer coordinate update failed in IRC sweeper: %s", e)

        # Update parent main window if parent has trajectory slider
        if self.parent() and hasattr(self.parent(), "slider_trajectory"):
            traj_slider = getattr(self.parent(), "slider_trajectory")
            if 0 <= self.current_step <= traj_slider.maximum():
                traj_slider.blockSignals(True)
                traj_slider.setValue(self.current_step)
                traj_slider.blockSignals(False)

        self.step_selected.emit(self.current_step)

    def set_active_step(self, step: int):
        if self.pathway and 0 <= step < self.pathway.n_points:
            self.current_step = step
            self.slider.blockSignals(True)
            self.slider.setValue(step)
            self.slider.blockSignals(False)
            self._sync_active_step()
            self._replot()

    def _on_slider_changed(self, value: int):
        self.current_step = value
        self._sync_active_step()
        self._replot()

    def _on_canvas_click(self, event):
        if event.xdata is None or not self.pathway or self.pathway.n_points == 0:
            return
        s_arr = np.array(self.pathway.get_reaction_coords())
        dists = np.abs(s_arr - event.xdata)
        best_idx = int(np.argmin(dists))
        self.set_active_step(best_idx)

    def _on_canvas_motion(self, event):
        if event.button == 1 and event.xdata is not None and self.pathway and self.pathway.n_points > 0:
            s_arr = np.array(self.pathway.get_reaction_coords())
            best_idx = int(np.argmin(np.abs(s_arr - event.xdata)))
            if best_idx != self.current_step:
                self.set_active_step(best_idx)

    def _on_unit_changed(self):
        self.unit_mode = self.combo_units.currentData() or "kcal"
        self._replot()

    def _on_speed_changed(self):
        ms = self.combo_fps.currentData() or 100
        if self.timer.isActive():
            self.timer.setInterval(ms)

    def _on_toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.btn_play.setText("▶ Play")
            self.btn_play.setStyleSheet("background-color: #6366f1; color: white; font-weight: bold;")
        else:
            if not self.pathway or self.pathway.n_points <= 1:
                return
            ms = self.combo_fps.currentData() or 100
            self.timer.start(ms)
            self.is_playing = True
            self.btn_play.setText("⏸ Pause")
            self.btn_play.setStyleSheet("background-color: #e11d48; color: white; font-weight: bold;")

    def _on_timer_tick(self):
        if not self.pathway or self.pathway.n_points <= 1:
            self._on_toggle_play()
            return

        mode = self.combo_mode.currentData() or "bounce"
        n = self.pathway.n_points

        if mode == "bounce":
            next_step = self.current_step + self.play_direction
            if next_step >= n:
                self.play_direction = -1
                next_step = n - 2 if n >= 2 else 0
            elif next_step < 0:
                self.play_direction = 1
                next_step = 1 if n >= 2 else 0
            self.set_active_step(next_step)
        elif mode == "forward":
            next_step = self.current_step + 1
            if next_step >= n:
                self.set_active_step(n - 1)
                self._on_toggle_play()
            else:
                self.set_active_step(next_step)
        elif mode == "reverse":
            next_step = self.current_step - 1
            if next_step < 0:
                self.set_active_step(0)
                self._on_toggle_play()
            else:
                self.set_active_step(next_step)
        elif mode == "loop":
            next_step = (self.current_step + 1) % n
            self.set_active_step(next_step)

    def _on_step_prev(self):
        if self.current_step > 0:
            self.set_active_step(self.current_step - 1)

    def _on_step_next(self):
        if self.pathway and self.current_step < self.pathway.n_points - 1:
            self.set_active_step(self.current_step + 1)

    def _on_jump_reactant(self):
        if self.pathway:
            self.set_active_step(self.pathway.reactant_index)

    def _on_jump_ts(self):
        if self.pathway:
            self.set_active_step(self.pathway.ts_index)

    def _on_jump_product(self):
        if self.pathway:
            self.set_active_step(self.pathway.product_index)

    def _on_adopt_current_geometry(self):
        if not self.pathway or not (0 <= self.current_step < self.pathway.n_points):
            return
        pt = self.pathway.points[self.current_step]
        self._adopt_geometry(pt.coords, label=f"IRC Step {self.current_step + 1}")

    def _on_adopt_ts_geometry(self):
        if not self.pathway or not (0 <= self.pathway.ts_index < self.pathway.n_points):
            return
        pt = self.pathway.points[self.pathway.ts_index]
        self._adopt_geometry(pt.coords, label="Transition State (TS ‡)")

    def _adopt_geometry(self, coords: np.ndarray, label: str = ""):
        symbols = self.pathway.symbols
        self.geometry_adopted.emit(symbols, coords)

        # Direct adoption if parent has _push_undo_state & zatoms
        parent = self.parent()
        if parent and hasattr(parent, "_push_undo_state") and hasattr(parent, "zatoms"):
            parent._push_undo_state()
            from .bonding import detect_connectivity
            detected_bonds = detect_connectivity(symbols, coords, strict=False)
            parent.zatoms = parent._cartesian_to_zatoms(symbols, coords, detected_bonds)
            parent._update_zmatrix_table()
            parent.viewer.set_atoms(parent.zatoms)
            parent.viewer.update()
            if hasattr(parent, "lbl_status"):
                parent.lbl_status.setText(f"Adopted {label} into Z-Matrix table.")

        QtWidgets.QMessageBox.information(
            self,
            "Geometry Adopted",
            f"Successfully adopted {label} ({len(symbols)} atoms) into the active editor with Undo support.",
        )

    def _on_open_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open IRC Calculation File",
            "",
            "QM IRC & Trajectory Files (*.log *.out *.xyz *.irc *.txt);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            self.load_from_text(content, filename=os.path.basename(path))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open Error", f"Failed to open file: {e}")

    def _on_export_csv(self):
        if not self.pathway or self.pathway.n_points == 0:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export IRC Profile to CSV", "irc_reaction_profile.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.pathway.export_csv())
            QtWidgets.QMessageBox.information(self, "Exported", f"Successfully exported CSV to {os.path.basename(path)}.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {e}")

    def _on_export_xyz(self):
        if not self.pathway or self.pathway.n_points == 0:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export IRC Reaction Movie", "irc_reaction_pathway.xyz", "XYZ Files (*.xyz);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.pathway.export_xyz())
            QtWidgets.QMessageBox.information(self, "Exported", f"Successfully exported XYZ movie to {os.path.basename(path)}.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export XYZ: {e}")

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_Space:
            self._on_toggle_play()
        elif event.key() == QtCore.Qt.Key.Key_Left:
            self._on_step_prev()
        elif event.key() == QtCore.Qt.Key.Key_Right:
            self._on_step_next()
        elif event.key() == QtCore.Qt.Key.Key_Home:
            self._on_jump_reactant()
        elif event.key() == QtCore.Qt.Key.Key_End:
            self._on_jump_product()
        elif event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self._on_adopt_current_geometry()
        else:
            super().keyPressEvent(event)
