"""
Interactive Microkinetics & Catalytic Cycle Simulator Dialog.

Provides:
- Visual definition of elementary catalytic steps, reactants, products, and activation barriers (kcal/mol)
- Eyring transition-state rate constant calculation at user-specified temperature
- Numerical integration of coupled non-linear reaction rate differential equations via scipy.integrate.solve_ivp
- Dynamic plotting of concentration profiles [C_i](t) over linear or log time
- Automated calculation of Catalytic Turnover Frequency (TOF, s^-1) and Turnover Number (TON)
- Export simulated concentration trajectories to CSV and publication-ready figures
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from .microkinetics import (
    ElementaryStep,
    MicrokineticsResult,
    eyring_rate_constant,
    solve_microkinetics,
)


class MicrokineticsDialog(QtWidgets.QDialog):
    """Interactive GUI for catalytic cycle and reaction network kinetics simulation."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Microkinetics & Catalytic Cycle Simulator (TOF, [C] vs. t)")
        self.resize(1020, 720)

        self.steps: List[ElementaryStep] = []
        self.initial_concs: Dict[str, float] = {}
        self.result: Optional[MicrokineticsResult] = None

        self._init_ui()
        self._load_default_catalytic_cycle()
        self._simulate()

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Left Panel: Network Setup & Step Configuration
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(420)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Header Info Card
        hdr_frame = QtWidgets.QFrame()
        hdr_frame.setStyleSheet("background-color: #0f172a; border-radius: 6px; padding: 10px;")
        hdr_box = QtWidgets.QHBoxLayout(hdr_frame)
        hdr_icon = QtWidgets.QLabel("<span style='font-size: 26px;'>📈</span>")
        hdr_box.addWidget(hdr_icon)
        hdr_text = QtWidgets.QVBoxLayout()
        hdr_text.addWidget(QtWidgets.QLabel("<b style='color: #38bdf8; font-size: 13px;'>Microkinetics Engine</b>"))
        hdr_text.addWidget(QtWidgets.QLabel("<span style='color: #94a3b8; font-size: 11px;'>Catalytic cycles, TOF & concentration dynamics</span>"))
        hdr_box.addLayout(hdr_text)
        left_layout.addWidget(hdr_frame)

        # Conditions bar (Temperature & Time)
        cond_grp = QtWidgets.QGroupBox("Reaction Conditions")
        c_grid = QtWidgets.QGridLayout(cond_grp)
        c_grid.setSpacing(6)

        c_grid.addWidget(QtWidgets.QLabel("Temp (K):"), 0, 0)
        self.spin_temp = QtWidgets.QDoubleSpinBox()
        self.spin_temp.setRange(1.0, 3000.0)
        self.spin_temp.setValue(298.15)
        self.spin_temp.setDecimals(2)
        c_grid.addWidget(self.spin_temp, 0, 1)

        c_grid.addWidget(QtWidgets.QLabel("Time span (s):"), 0, 2)
        self.spin_tmax = QtWidgets.QDoubleSpinBox()
        self.spin_tmax.setRange(0.01, 1e8)
        self.spin_tmax.setValue(3600.0)  # 1 hour
        self.spin_tmax.setDecimals(1)
        c_grid.addWidget(self.spin_tmax, 0, 3)

        c_grid.addWidget(QtWidgets.QLabel("Solver:"), 1, 0)
        self.cbo_solver = QtWidgets.QComboBox()
        self.cbo_solver.addItems(["LSODA", "Radau", "BDF", "RK45"])
        c_grid.addWidget(self.cbo_solver, 1, 1)

        c_grid.addWidget(QtWidgets.QLabel("X-Axis:"), 1, 2)
        self.cbo_scale = QtWidgets.QComboBox()
        self.cbo_scale.addItems(["Linear Time", "Log10 Time"])
        self.cbo_scale.currentIndexChanged.connect(self._replot_only)
        c_grid.addWidget(self.cbo_scale, 1, 3)

        left_layout.addWidget(cond_grp)

        # Elementary Steps Table
        steps_grp = QtWidgets.QGroupBox("Elementary Reaction Steps")
        s_lay = QtWidgets.QVBoxLayout(steps_grp)

        self.tbl_steps = QtWidgets.QTableWidget(0, 4)
        self.tbl_steps.setHorizontalHeaderLabels(["Step Name", "Reactants", "Products", "ΔG‡ (kcal/mol)"])
        self.tbl_steps.horizontalHeader().setStretchLastSection(True)
        s_lay.addWidget(self.tbl_steps)

        btn_step_row = QtWidgets.QHBoxLayout()
        btn_add_step = QtWidgets.QPushButton("➕ Add Step")
        btn_add_step.clicked.connect(self._on_add_step)
        btn_step_row.addWidget(btn_add_step)

        btn_rem_step = QtWidgets.QPushButton("➖ Remove Step")
        btn_rem_step.clicked.connect(self._on_remove_step)
        btn_step_row.addWidget(btn_rem_step)
        s_lay.addLayout(btn_step_row)

        left_layout.addWidget(steps_grp)

        # Initial Concentrations Table
        conc_grp = QtWidgets.QGroupBox("Initial Species Concentrations [C]₀ (mol/L)")
        c_lay = QtWidgets.QVBoxLayout(conc_grp)
        self.tbl_concs = QtWidgets.QTableWidget(0, 3)
        self.tbl_concs.setHorizontalHeaderLabels(["Species", "[C]₀ (M)", "Role"])
        self.tbl_concs.horizontalHeader().setStretchLastSection(True)
        c_lay.addWidget(self.tbl_concs)

        btn_conc_row = QtWidgets.QHBoxLayout()
        btn_add_conc = QtWidgets.QPushButton("➕ Add Species")
        btn_add_conc.clicked.connect(self._on_add_species)
        btn_conc_row.addWidget(btn_add_conc)

        btn_rem_conc = QtWidgets.QPushButton("➖ Remove Species")
        btn_rem_conc.clicked.connect(self._on_remove_species)
        btn_conc_row.addWidget(btn_rem_conc)
        c_lay.addLayout(btn_conc_row)

        left_layout.addWidget(conc_grp)

        # Action Buttons
        btn_sim = QtWidgets.QPushButton("🚀 Run Microkinetics Integration")
        btn_sim.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 8px; font-size: 13px;")
        btn_sim.clicked.connect(self._simulate)
        left_layout.addWidget(btn_sim)

        main_layout.addWidget(left_panel)

        # Right Panel: Output & Matplotlib Plots
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Catalytic Metrics Banner
        self.metrics_banner = QtWidgets.QFrame()
        self.metrics_banner.setStyleSheet("background-color: #1e293b; border-radius: 6px; padding: 10px; border: 1px solid #334155;")
        b_lay = QtWidgets.QHBoxLayout(self.metrics_banner)
        self.lbl_tof = QtWidgets.QLabel("<b>TOF:</b> -")
        self.lbl_tof.setStyleSheet("color: #38bdf8; font-size: 13px;")
        self.lbl_ton = QtWidgets.QLabel("<b>TON:</b> -")
        self.lbl_ton.setStyleSheet("color: #4ade80; font-size: 13px;")
        self.lbl_prod = QtWidgets.QLabel("<b>Target Product:</b> -")
        self.lbl_prod.setStyleSheet("font-size: 12px;")

        b_lay.addWidget(self.lbl_tof)
        b_lay.addWidget(self.lbl_ton)
        b_lay.addWidget(self.lbl_prod)
        b_lay.addStretch()
        right_layout.addWidget(self.metrics_banner)

        # Canvas & Toolbar
        self.fig = Figure(facecolor="#18181b", tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, right_panel)
        self.toolbar.setStyleSheet("background-color: #27272a; color: white;")

        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        # Export CSV Button
        btn_export = QtWidgets.QPushButton("💾 Export Concentration Trajectories (CSV)...")
        btn_export.clicked.connect(self._on_export_csv)
        right_layout.addWidget(btn_export)

        main_layout.addWidget(right_panel)

    def _style_axes(self):
        self.ax.set_facecolor("#09090b")
        self.ax.tick_params(colors="#94a3b8", labelsize=10)
        for spine in self.ax.spines.values():
            spine.set_color("#3f3f46")
        self.ax.grid(True, linestyle="--", alpha=0.25, color="#71717a")

    def _load_default_catalytic_cycle(self):
        """Loads a classic textbook catalytic cycle:
        Step 1: Catalyst + Substrate A -> Cat-A (barrier 14.5 kcal/mol)
        Step 2: Cat-A + Coreactant B -> Cat-P (barrier 19.8 kcal/mol - rate determining)
        Step 3: Cat-P -> Catalyst + Product P (barrier 11.2 kcal/mol)
        """
        default_steps = [
            ("Coordination (Cat + A -> Cat-A)", "Cat + A", "Cat-A", "14.5"),
            ("Coupling (Cat-A + B -> Cat-P)", "Cat-A + B", "Cat-P", "19.8"),
            ("Product Release (Cat-P -> Cat + P)", "Cat-P", "Cat + P", "11.2"),
        ]
        self.tbl_steps.setRowCount(len(default_steps))
        for r, (name, react, prod, dg) in enumerate(default_steps):
            self.tbl_steps.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            self.tbl_steps.setItem(r, 1, QtWidgets.QTableWidgetItem(react))
            self.tbl_steps.setItem(r, 2, QtWidgets.QTableWidgetItem(prod))
            self.tbl_steps.setItem(r, 3, QtWidgets.QTableWidgetItem(dg))

        default_concs = [
            ("Cat", "0.010", "Catalyst"),
            ("A", "1.000", "Reactant"),
            ("B", "1.200", "Reactant"),
            ("Cat-A", "0.000", "Intermediate"),
            ("Cat-P", "0.000", "Intermediate"),
            ("P", "0.000", "Product"),
        ]
        self.tbl_concs.setRowCount(len(default_concs))
        for r, (sp, c0, role) in enumerate(default_concs):
            self.tbl_concs.setItem(r, 0, QtWidgets.QTableWidgetItem(sp))
            self.tbl_concs.setItem(r, 1, QtWidgets.QTableWidgetItem(c0))
            self.tbl_concs.setItem(r, 2, QtWidgets.QTableWidgetItem(role))

    def _on_add_step(self):
        r = self.tbl_steps.rowCount()
        self.tbl_steps.insertRow(r)
        self.tbl_steps.setItem(r, 0, QtWidgets.QTableWidgetItem(f"Step {r + 1}"))
        self.tbl_steps.setItem(r, 1, QtWidgets.QTableWidgetItem("A"))
        self.tbl_steps.setItem(r, 2, QtWidgets.QTableWidgetItem("B"))
        self.tbl_steps.setItem(r, 3, QtWidgets.QTableWidgetItem("20.0"))

    def _on_remove_step(self):
        sel = self.tbl_steps.selectionModel().selectedRows()
        if sel:
            for s in sorted(sel, key=lambda x: x.row(), reverse=True):
                self.tbl_steps.removeRow(s.row())
        elif self.tbl_steps.rowCount() > 0:
            self.tbl_steps.removeRow(self.tbl_steps.rowCount() - 1)

    def _on_add_species(self):
        r = self.tbl_concs.rowCount()
        self.tbl_concs.insertRow(r)
        self.tbl_concs.setItem(r, 0, QtWidgets.QTableWidgetItem(f"Species_{r + 1}"))
        self.tbl_concs.setItem(r, 1, QtWidgets.QTableWidgetItem("1.000"))
        self.tbl_concs.setItem(r, 2, QtWidgets.QTableWidgetItem("Reactant"))

    def _on_remove_species(self):
        sel = self.tbl_concs.selectionModel().selectedRows()
        if sel:
            for s in sorted(sel, key=lambda x: x.row(), reverse=True):
                self.tbl_concs.removeRow(s.row())
        elif self.tbl_concs.rowCount() > 0:
            self.tbl_concs.removeRow(self.tbl_concs.rowCount() - 1)

    def _parse_stoichiometry(self, text: str) -> Dict[str, float]:
        """Parses reaction expressions like 'Cat + 2 A' into {'Cat': 1.0, 'A': 2.0}."""
        res: Dict[str, float] = {}
        tokens = [t.strip() for t in text.split("+") if t.strip()]
        for t in tokens:
            parts = t.split()
            if len(parts) == 1:
                res[parts[0]] = 1.0
            elif len(parts) >= 2:
                try:
                    coeff = float(parts[0])
                    name = " ".join(parts[1:])
                    res[name] = coeff
                except ValueError:
                    res[t] = 1.0
        return res

    def _simulate(self):
        # 1. Parse steps
        steps: List[ElementaryStep] = []
        for r in range(self.tbl_steps.rowCount()):
            name = self.tbl_steps.item(r, 0).text() if self.tbl_steps.item(r, 0) else f"Step {r+1}"
            react_txt = self.tbl_steps.item(r, 1).text() if self.tbl_steps.item(r, 1) else ""
            prod_txt = self.tbl_steps.item(r, 2).text() if self.tbl_steps.item(r, 2) else ""
            dg_txt = self.tbl_steps.item(r, 3).text() if self.tbl_steps.item(r, 3) else "20.0"

            try:
                dg_fwd = float(dg_txt)
            except ValueError:
                dg_fwd = 20.0

            r_dict = self._parse_stoichiometry(react_txt)
            p_dict = self._parse_stoichiometry(prod_txt)
            if r_dict and p_dict:
                steps.append(ElementaryStep(name, r_dict, p_dict, dg_fwd))

        # 2. Parse initial concentrations
        c0_dict: Dict[str, float] = {}
        catalyst_name: Optional[str] = None
        product_name: Optional[str] = None

        for r in range(self.tbl_concs.rowCount()):
            sp_name = self.tbl_concs.item(r, 0).text() if self.tbl_concs.item(r, 0) else ""
            c0_txt = self.tbl_concs.item(r, 1).text() if self.tbl_concs.item(r, 1) else "0.0"
            role_txt = self.tbl_concs.item(r, 2).text().lower() if self.tbl_concs.item(r, 2) else ""

            if not sp_name:
                continue
            try:
                c0_val = float(c0_txt)
            except ValueError:
                c0_val = 0.0

            c0_dict[sp_name] = c0_val
            if "cat" in role_txt:
                catalyst_name = sp_name
            elif "prod" in role_txt:
                product_name = sp_name

        if not steps:
            QtWidgets.QMessageBox.warning(self, "No Steps", "Please define at least one elementary reaction step.")
            return

        temp = self.spin_temp.value()
        tmax = self.spin_tmax.value()
        solver_method = self.cbo_solver.currentText()

        try:
            self.result = solve_microkinetics(
                steps=steps,
                initial_concentrations=c0_dict,
                time_span=(0.0, tmax),
                temperature_k=temp,
                catalyst_name=catalyst_name,
                product_name=product_name,
                method=solver_method,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Integration Error", f"Failed to solve rate ODEs:\n{exc}")
            return

        # Update Metrics Banner
        tof_str = f"{self.result.turnover_frequency_tof:.2e} s⁻¹" if self.result.turnover_frequency_tof > 0 else "N/A"
        ton_str = f"{self.result.turnover_number_ton:.2f}"
        cat_str = self.result.catalyst_species or "None"
        prod_str = self.result.product_species or "None"

        self.lbl_tof.setText(f"<b>TOF:</b> {tof_str} (Cat: {cat_str})")
        self.lbl_ton.setText(f"<b>TON:</b> {ton_str}")
        self.lbl_prod.setText(f"<b>Target Product:</b> {prod_str}")

        self._replot_only()

    def _replot_only(self):
        if not self.result:
            return

        self.ax.clear()
        self._style_axes()

        res = self.result
        t = res.time_grid
        is_log = "Log" in self.cbo_scale.currentText()

        # Modern palette
        colors = [
            "#38bdf8", "#4ade80", "#f43f5e", "#fbbf24", "#c084fc",
            "#2dd4bf", "#f97316", "#e879f9", "#94a3b8", "#a3e635"
        ]

        for idx, sp_name in enumerate(res.species_names):
            c_trace = res.concentrations[idx, :]
            color = colors[idx % len(colors)]
            self.ax.plot(t, c_trace, label=f"[{sp_name}]", color=color, linewidth=2.0)

        if is_log:
            self.ax.set_xscale("log")
            self.ax.set_xlabel("Time t (s, log scale)", color="#f1f5f9", fontsize=11, fontweight="bold")
        else:
            self.ax.set_xlabel("Time t (s)", color="#f1f5f9", fontsize=11, fontweight="bold")

        self.ax.set_ylabel("Concentration [C] (mol / L)", color="#f1f5f9", fontsize=11, fontweight="bold")
        self.ax.set_title(
            f"Concentration Trajectories [T = {self.spin_temp.value():.1f} K] (TOF = {res.turnover_frequency_tof:.2e} s⁻¹)",
            color="#38bdf8",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        self.ax.legend(loc="best", facecolor="#18181b", edgecolor="#3f3f46", labelcolor="#f1f5f9")
        self.canvas.draw()

    def _on_export_csv(self):
        if not self.result:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Microkinetics Data",
            "microkinetics_trajectories.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            res = self.result
            with open(path, "w", encoding="utf-8") as f:
                header = ["time_s"] + [f"[{name}]_M" for name in res.species_names]
                f.write(",".join(header) + "\n")
                for col_i in range(len(res.time_grid)):
                    row_vals = [f"{res.time_grid[col_i]:.6e}"]
                    for row_i in range(len(res.species_names)):
                        row_vals.append(f"{res.concentrations[row_i, col_i]:.6e}")
                    f.write(",".join(row_vals) + "\n")
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Successfully exported kinetics trajectories to:\n{path}")
