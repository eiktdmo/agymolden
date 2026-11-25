"""
Dialog for Reaction Thermochemistry Workbench (Eyring kinetics & equilibrium).
"""

from __future__ import annotations

import math
import os
from typing import List, Optional

import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets

from .thermochemistry import (
    SpeciesThermo,
    ReactionThermoResult,
    calculate_reaction_thermochemistry,
    extract_species_from_log_file,
)


class ReactionThermoDialog(QtWidgets.QDialog):
    """Interactive GUI for reaction free energy profiles, activation barriers, and rate constants."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Reaction Thermochemistry & Kinetics Workbench")
        self.resize(780, 560)
        self.species: List[SpeciesThermo] = []
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header Info Bar
        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QHBoxLayout(header)
        lbl_icon = QtWidgets.QLabel("<span style='font-size: 24px;'>⚗️</span>")
        h_lay.addWidget(lbl_icon)

        t_lay = QtWidgets.QVBoxLayout()
        t_lay.addWidget(QtWidgets.QLabel("<b style='color: #38bdf8; font-size: 14px;'>Reaction Thermochemistry & Eyring Kinetics</b>"))
        t_lay.addWidget(QtWidgets.QLabel("<span style='color: #94a3b8; font-size: 11px;'>Compute ΔG‡, ΔH‡, ΔGrxn, Keq, and reaction rate constants k(T) from QM output logs.</span>"))
        h_lay.addLayout(t_lay)
        h_lay.addStretch()

        # Temperature setting
        temp_box = QtWidgets.QHBoxLayout()
        temp_box.addWidget(QtWidgets.QLabel("<b>Temperature (K):</b>"))
        self.spin_temp = QtWidgets.QDoubleSpinBox()
        self.spin_temp.setRange(1.0, 5000.0)
        self.spin_temp.setValue(298.15)
        self.spin_temp.setDecimals(2)
        self.spin_temp.valueChanged.connect(self._recalculate)
        temp_box.addWidget(self.spin_temp)
        h_lay.addLayout(temp_box)

        layout.addWidget(header)

        # Species Table
        tbl_box = QtWidgets.QGroupBox("Reaction Components (Reactants, Transition States, Products)")
        tbl_lay = QtWidgets.QVBoxLayout(tbl_box)

        self.tbl_species = QtWidgets.QTableWidget(0, 5)
        self.tbl_species.setHorizontalHeaderLabels(["Role", "Coeff", "Species Name", "Gibbs Free Energy G (Ha)", "Log Source"])
        self.tbl_species.horizontalHeader().setStretchLastSection(True)
        self.tbl_species.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        tbl_lay.addWidget(self.tbl_species)

        # Button Bar for table
        btn_row = QtWidgets.QHBoxLayout()
        btn_add_file = QtWidgets.QPushButton("📂 Add QM Log File...")
        btn_add_file.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 4px 10px;")
        btn_add_file.clicked.connect(self._on_add_log_file)
        btn_row.addWidget(btn_add_file)

        btn_add_manual = QtWidgets.QPushButton("➕ Add Manual Row")
        btn_add_manual.clicked.connect(self._on_add_manual_row)
        btn_row.addWidget(btn_add_manual)

        btn_remove = QtWidgets.QPushButton("🗑️ Remove Selected")
        btn_remove.clicked.connect(self._on_remove_selected)
        btn_row.addWidget(btn_remove)

        btn_row.addStretch()
        tbl_lay.addLayout(btn_row)
        layout.addWidget(tbl_box)

        # Results Summary Card
        res_box = QtWidgets.QGroupBox("Calculated Thermochemical & Kinetic Results")
        res_lay = QtWidgets.QGridLayout(res_box)
        res_lay.setHorizontalSpacing(20)

        self.lbl_delta_g_rxn = QtWidgets.QLabel("ΔGrxn: —")
        self.lbl_delta_g_rxn.setStyleSheet("font-size: 13px; font-weight: bold; color: #38bdf8;")
        res_lay.addWidget(self.lbl_delta_g_rxn, 0, 0)

        self.lbl_delta_h_rxn = QtWidgets.QLabel("ΔHrxn: —")
        res_lay.addWidget(self.lbl_delta_h_rxn, 0, 1)

        self.lbl_keq = QtWidgets.QLabel("Keq: —")
        self.lbl_keq.setStyleSheet("font-size: 13px; font-weight: bold; color: #059669;")
        res_lay.addWidget(self.lbl_keq, 0, 2)

        self.lbl_delta_g_act = QtWidgets.QLabel("ΔG‡ (Barrier): —")
        self.lbl_delta_g_act.setStyleSheet("font-size: 13px; font-weight: bold; color: #f59e0b;")
        res_lay.addWidget(self.lbl_delta_g_act, 1, 0)

        self.lbl_k_rate = QtWidgets.QLabel("Rate Constant k(T): —")
        self.lbl_k_rate.setStyleSheet("font-size: 13px; font-weight: bold; color: #f43f5e;")
        res_lay.addWidget(self.lbl_k_rate, 1, 1)

        self.lbl_delta_s_rxn = QtWidgets.QLabel("ΔSrxn: —")
        res_lay.addWidget(self.lbl_delta_s_rxn, 1, 2)

        layout.addWidget(res_box)

        # Bottom Buttons
        bottom_bar = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("📋 Copy Summary for SI")
        btn_copy.clicked.connect(self._on_copy_summary)
        bottom_bar.addWidget(btn_copy)

        btn_plot = QtWidgets.QPushButton("📈 Plot Energy Profile (SVG / PNG)...")
        btn_plot.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 4px 12px;")
        btn_plot.setToolTip("Generate publication-ready 1-click Free Energy Diagram (ΔG‡ and ΔGrxn)")
        btn_plot.clicked.connect(self._on_plot_energy_profile)
        bottom_bar.addWidget(btn_plot)

        btn_arrhenius = QtWidgets.QPushButton("⚡ Arrhenius & Eyring Kinetics T-Scan...")
        btn_arrhenius.setStyleSheet("background-color: #7c3aed; color: white; font-weight: bold; padding: 4px 12px;")
        btn_arrhenius.setToolTip("Compute and plot temperature-dependent rate constants k(T) and Keq over 200 K – 1000 K")
        btn_arrhenius.clicked.connect(self._on_plot_arrhenius_kinetics)
        bottom_bar.addWidget(btn_arrhenius)

        bottom_bar.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bottom_bar.addWidget(btn_close)
        layout.addLayout(bottom_bar)

    def _on_add_log_file(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select QM Output Logs (ORCA, Gaussian, PySCF)", "",
            "QM Logs (*.out *.log *.out.gz *.log.gz);;All Files (*)"
        )
        if not paths:
            return

        for path in paths:
            try:
                # Default role heuristic: if 'ts' in name -> ts, else reactant
                role = "ts" if "ts" in os.path.basename(path).lower() else "reactant"
                species = extract_species_from_log_file(path, role=role)
                self.species.append(species)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Parse Warning", f"Could not extract thermo from {os.path.basename(path)}: {e}")

        self._refresh_table()
        self._recalculate()

    def _on_add_manual_row(self):
        row_num = len(self.species) + 1
        species = SpeciesThermo(name=f"Species {row_num}", role="product", coefficient=1.0, gibbs_ha=-100.0)
        self.species.append(species)
        self._refresh_table()
        self._recalculate()

    def _on_remove_selected(self):
        selected_rows = sorted(set(idx.row() for idx in self.tbl_species.selectedIndexes()), reverse=True)
        for r in selected_rows:
            if 0 <= r < len(self.species):
                self.species.pop(r)
        self._refresh_table()
        self._recalculate()

    def _refresh_table(self):
        self.tbl_species.blockSignals(True)
        self.tbl_species.setRowCount(len(self.species))
        for i, s in enumerate(self.species):
            # Role combo
            combo_role = QtWidgets.QComboBox()
            combo_role.addItem("Reactant", "reactant")
            combo_role.addItem("Intermediate", "intermediate")
            combo_role.addItem("Transition State (TS)", "ts")
            combo_role.addItem("Product", "product")
            idx = combo_role.findData(s.role)
            if idx >= 0:
                combo_role.setCurrentIndex(idx)
            combo_role.currentIndexChanged.connect(lambda _, row=i, cb=combo_role: self._on_role_changed(row, cb.currentData()))
            self.tbl_species.setCellWidget(i, 0, combo_role)

            # Coeff spin
            spin_coeff = QtWidgets.QDoubleSpinBox()
            spin_coeff.setRange(0.1, 100.0)
            spin_coeff.setValue(s.coefficient)
            spin_coeff.valueChanged.connect(lambda val, row=i: self._on_coeff_changed(row, val))
            self.tbl_species.setCellWidget(i, 1, spin_coeff)

            # Name item
            self.tbl_species.setItem(i, 2, QtWidgets.QTableWidgetItem(s.name))

            # Gibbs item
            g_str = f"{s.gibbs_ha:.6f}" if s.gibbs_ha is not None else "0.000000"
            self.tbl_species.setItem(i, 3, QtWidgets.QTableWidgetItem(g_str))

            # Source item
            src = os.path.basename(s.file_path) if s.file_path else "Manual"
            item_src = QtWidgets.QTableWidgetItem(src)
            item_src.setFlags(item_src.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.tbl_species.setItem(i, 4, item_src)

        self.tbl_species.cellChanged.connect(self._on_cell_edited)
        self.tbl_species.blockSignals(False)

    def _on_role_changed(self, row: int, new_role: str):
        if 0 <= row < len(self.species):
            self.species[row].role = new_role
            self._recalculate()

    def _on_coeff_changed(self, row: int, val: float):
        if 0 <= row < len(self.species):
            self.species[row].coefficient = val
            self._recalculate()

    def _on_cell_edited(self, row: int, col: int):
        if not (0 <= row < len(self.species)):
            return
        if col == 2:
            self.species[row].name = self.tbl_species.item(row, col).text()
        elif col == 3:
            try:
                val = float(self.tbl_species.item(row, col).text())
                self.species[row].gibbs_ha = val
                self._recalculate()
            except ValueError:
                pass

    def _recalculate(self):
        temp = self.spin_temp.value()
        try:
            res = calculate_reaction_thermochemistry(self.species, temperature_k=temp)
            self.lbl_delta_g_rxn.setText(f"ΔGrxn: <b>{res.delta_g_rxn_kcal:+.2f} kcal/mol</b>")
            self.lbl_delta_h_rxn.setText(f"ΔHrxn: <b>{res.delta_h_rxn_kcal:+.2f} kcal/mol</b>")
            self.lbl_delta_s_rxn.setText(f"ΔSrxn: <b>{res.delta_s_rxn_cal_mol_k:+.1f} cal/(mol·K)</b>")

            if res.k_eq == float("inf"):
                self.lbl_keq.setText("Keq: <b>> 10³⁰⁰</b>")
            elif res.k_eq == 0.0:
                self.lbl_keq.setText("Keq: <b>< 10⁻³⁰⁰</b>")
            else:
                self.lbl_keq.setText(f"Keq: <b>{res.k_eq:.3e}</b>")

            if res.has_ts and res.delta_g_act_kcal is not None:
                self.lbl_delta_g_act.setText(f"ΔG‡ (Barrier): <b>{res.delta_g_act_kcal:.2f} kcal/mol</b>")
                if res.rate_constant_k_fwd is not None:
                    tun_str = f" [κ = {res.tunneling_kappa_wigner:.2f}]" if res.tunneling_kappa_wigner and res.tunneling_kappa_wigner > 1.01 else ""
                    self.lbl_k_rate.setText(f"Rate k(T): <b>{res.rate_constant_k_fwd:.2e} s⁻¹</b>{tun_str}")
            else:
                self.lbl_delta_g_act.setText("ΔG‡ (Barrier): — (No TS)")
                self.lbl_k_rate.setText("Rate k(T): —")
        except Exception as e:
            self.lbl_delta_g_rxn.setText(f"ΔGrxn: <span style='color: #f87171;'>{e}</span>")

    def _on_copy_summary(self):
        temp = self.spin_temp.value()
        try:
            res = calculate_reaction_thermochemistry(self.species, temperature_k=temp)
            lines = [
                f"# Reaction Thermochemistry Summary (T = {temp:.2f} K)",
                f"• ΔGrxn = {res.delta_g_rxn_kcal:+.2f} kcal/mol",
                f"• ΔHrxn = {res.delta_h_rxn_kcal:+.2f} kcal/mol",
                f"• ΔSrxn = {res.delta_s_rxn_cal_mol_k:+.1f} cal/(mol·K)",
                f"• Equilibrium Constant Keq = {res.k_eq:.3e}" if res.k_eq not in (0.0, float('inf')) else f"• Keq = {res.k_eq}",
            ]
            if res.has_ts and res.delta_g_act_kcal is not None:
                lines.append(f"• Activation Barrier ΔG‡ = {res.delta_g_act_kcal:.2f} kcal/mol")
                if res.rate_constant_k_fwd is not None:
                    lines.append(f"• Eyring Rate Constant k({temp:.0f}K) = {res.rate_constant_k_fwd:.2e} s⁻¹")

            text = "\n".join(lines)
            QtWidgets.QApplication.clipboard().setText(text)
            QtWidgets.QMessageBox.information(self, "Copied", "Reaction thermochemistry summary copied to clipboard!")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Copy Error", f"Cannot compute summary: {e}")

    def _on_plot_energy_profile(self):
        """Generates a publication-ready Free Energy Reaction Coordinate Diagram in a modal viewer."""
        temp = self.spin_temp.value()
        try:
            res = calculate_reaction_thermochemistry(self.species, temperature_k=temp)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Plot Warning", f"Cannot generate energy profile: {e}")
            return

        # Prepare multi-state reaction profile:
        # If user has multiple intermediate or TS states, render full multistep profile
        intermediates = [s for s in self.species if s.role == "intermediate"]
        ts_list = [s for s in self.species if s.role == "ts"]

        if intermediates or len(ts_list) > 1:
            states = []
            g_ref = res.g_reactants_ha
            # Reactants
            states.append(("Reactants", 0.0, "#38bdf8"))
            # Sequential species in order of table
            for s in self.species:
                if s.role in ("ts", "intermediate"):
                    g_val = s.free_energy_val or 0.0
                    rel_kcal = (g_val - g_ref) * 627.509474
                    col = "#f43f5e" if s.role == "ts" else "#a855f7"
                    label = f"{s.name}‡" if s.role == "ts" else s.name
                    states.append((label, rel_kcal, col))
            # Products
            states.append(("Products", res.delta_g_rxn_kcal, "#10b981"))
        else:
            states = [
                ("Reactants", 0.0, "#38bdf8")
            ]
            if res.has_ts and res.delta_g_act_kcal is not None:
                states.append(("TS‡", res.delta_g_act_kcal, "#f43f5e"))
            states.append(("Products", res.delta_g_rxn_kcal, "#10b981"))

        # Launch Profile Dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Reaction Free Energy Coordinate Profile")
        dlg.resize(680, 520)
        dlg_lay = QtWidgets.QVBoxLayout(dlg)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(6.5, 4.8), dpi=100, facecolor="#1e293b")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#0f172a")

        xs = []
        ys = []
        w = 0.35  # platform half-width

        for idx, (label, val, col) in enumerate(states):
            x_center = float(idx + 1)
            xs.append(x_center)
            ys.append(val)
            # Horizontal energy plateau bar
            ax.hlines(val, x_center - w, x_center + w, colors=col, linewidth=3.5, label=label)
            # Text annotation
            prefix = "+" if val > 0 else ""
            txt = f"{label}\n{prefix}{val:.1f} kcal/mol" if idx > 0 else f"{label}\n0.0 kcal/mol"
            y_offset = 1.2 if val >= 0 else -2.2
            ax.text(x_center, val + y_offset, txt, ha="center", va="bottom" if val >= 0 else "top",
                    color=col, fontweight="bold", fontsize=10)

        # Dashed connecting reaction pathway splines
        for i in range(len(states) - 1):
            x1, y1 = xs[i] + w, ys[i]
            x2, y2 = xs[i+1] - w, ys[i+1]
            # Smooth cubic bezier curve connecting the platforms
            curve_x = np.linspace(x1, x2, 40)
            t = (curve_x - x1) / (x2 - x1)
            # Smoothstep interpolation
            smooth_t = t * t * (3 - 2 * t)
            curve_y = y1 + (y2 - y1) * smooth_t
            ax.plot(curve_x, curve_y, "--", color="#64748b", linewidth=1.8, alpha=0.85)

        ax.set_ylabel(f"Relative Free Energy ΔG ({temp:.0f} K) [kcal/mol]", color="#e2e8f0", fontsize=11, fontweight="bold")
        ax.set_xlabel("Reaction Coordinate", color="#e2e8f0", fontsize=11, fontweight="bold")
        ax.set_title(f"Reaction Free Energy Profile (ΔG‡ = {res.delta_g_act_kcal:.2f} kcal/mol, ΔGrxn = {res.delta_g_rxn_kcal:+.2f} kcal/mol)" if res.has_ts else f"Reaction Free Energy Profile (ΔGrxn = {res.delta_g_rxn_kcal:+.2f} kcal/mol)",
                     color="#f8fafc", fontsize=12, fontweight="bold", pad=12)

        ax.set_xticks(xs)
        ax.set_xticklabels([s[0] for s in states], color="#cbd5e1", fontweight="bold", fontsize=11)
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")

        y_min = min(ys)
        y_max = max(ys)
        span = max(y_max - y_min, 10.0)
        ax.set_ylim(y_min - 0.25 * span, y_max + 0.35 * span)
        ax.set_xlim(0.3, len(states) + 0.7)
        fig.tight_layout()

        canvas = FigureCanvasQTAgg(fig)
        dlg_lay.addWidget(canvas, stretch=1)

        btn_box = QtWidgets.QHBoxLayout()
        btn_save = QtWidgets.QPushButton("💾 Save Figure (SVG / PNG / PDF)...")
        btn_save.setStyleSheet("background-color: #0d9488; color: white; font-weight: bold; padding: 4px 12px;")

        def _save_plot():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, "Save Energy Profile", "reaction_energy_profile.svg",
                "Scalable Vector Graphics (*.svg);;Portable Network Graphics (*.png);;PDF (*.pdf)"
            )
            if path:
                # Save with white background if requested for journal papers
                fig.savefig(path, dpi=300, bbox_inches="tight")
                QtWidgets.QMessageBox.information(dlg, "Saved", f"Energy profile saved to {os.path.basename(path)}")

        btn_save.clicked.connect(_save_plot)
        btn_box.addWidget(btn_save)
        btn_box.addStretch()

        btn_dlg_close = QtWidgets.QPushButton("Close")
        btn_dlg_close.clicked.connect(dlg.accept)
        btn_box.addWidget(btn_dlg_close)
        dlg_lay.addLayout(btn_box)

        dlg.exec()

    def _on_plot_arrhenius_kinetics(self):
        """Generates temperature-dependent Arrhenius and Eyring rate constant curves k(T) and Keq(T)."""
        try:
            res_base = calculate_reaction_thermochemistry(self.species, temperature_k=298.15)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Kinetics Warning", f"Cannot calculate kinetics: {e}")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("⚡ Arrhenius & Eyring Temperature Sensitivity Workbench")
        dlg.resize(750, 560)
        dlg_lay = QtWidgets.QVBoxLayout(dlg)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(7.0, 5.0), dpi=100, facecolor="#1e293b")
        canvas = FigureCanvasQTAgg(fig)
        dlg_lay.addWidget(canvas, stretch=1)

        # Compute over T = 200 K to 1000 K
        temps = np.linspace(200.0, 1000.0, 80)
        inv_t = 1000.0 / temps  # 1000/T [K^-1]
        log_k_fwd = []
        log_k_rev = []
        log_k_tun = []
        keq_vals = []

        for T in temps:
            try:
                res_t = calculate_reaction_thermochemistry(self.species, temperature_k=T)
                # Forward rate
                kf = res_t.rate_constant_k_fwd
                kr = res_t.rate_constant_k_rev
                kf_tun = res_t.rate_constant_k_fwd_tunneled
                log_k_fwd.append(math.log10(max(1e-300, kf)) if kf is not None else float("nan"))
                log_k_rev.append(math.log10(max(1e-300, kr)) if kr is not None else float("nan"))
                log_k_tun.append(math.log10(max(1e-300, kf_tun)) if kf_tun is not None else float("nan"))
                keq_vals.append(math.log10(max(1e-300, res_t.k_eq)) if res_t.k_eq not in (float("inf"), 0.0) else float("nan"))
            except Exception:
                log_k_fwd.append(float("nan"))
                log_k_rev.append(float("nan"))
                log_k_tun.append(float("nan"))
                keq_vals.append(float("nan"))

        ax1 = fig.add_subplot(121)
        ax1.set_facecolor("#0f172a")
        if res_base.has_ts and not all(np.isnan(log_k_fwd)):
            ax1.plot(inv_t, log_k_fwd, color="#38bdf8", linewidth=2.5, label="log₁₀ k_fwd (Classical)")
            if any(not np.isnan(v) for v in log_k_tun):
                ax1.plot(inv_t, log_k_tun, color="#22c55e", linewidth=2.2, linestyle="-.", label="log₁₀ k_tun (Wigner κ)")
            if not all(np.isnan(log_k_rev)):
                ax1.plot(inv_t, log_k_rev, color="#f43f5e", linewidth=2.0, linestyle="--", label="log₁₀ k_rev")
            ax1.set_xlabel("1000 / T (K⁻¹)", color="#e2e8f0", fontweight="bold")
            ax1.set_ylabel("log₁₀ k (s⁻¹)", color="#e2e8f0", fontweight="bold")
            ax1.set_title("Eyring Rate Constants k(T)", color="#f8fafc", fontweight="bold", fontsize=11)
            ax1.legend(loc="best")
            ax1.grid(color="#334155", linestyle=":", alpha=0.6)
            ax1.tick_params(colors="#94a3b8")
        else:
            ax1.text(0.5, 0.5, "No Transition State (TS)\nspecified in species list",
                     color="#94a3b8", ha="center", va="center")

        ax2 = fig.add_subplot(122)
        ax2.set_facecolor("#0f172a")
        ax2.plot(temps, keq_vals, color="#10b981", linewidth=2.5, label="log₁₀ K_eq")
        ax2.set_xlabel("Temperature T (K)", color="#e2e8f0", fontweight="bold")
        ax2.set_ylabel("log₁₀ K_eq", color="#e2e8f0", fontweight="bold")
        ax2.set_title("Equilibrium Constant K_eq(T)", color="#f8fafc", fontweight="bold", fontsize=11)
        ax2.grid(color="#334155", linestyle=":", alpha=0.6)
        ax2.tick_params(colors="#94a3b8")

        fig.tight_layout()
        canvas.draw()

        b_box = QtWidgets.QHBoxLayout()
        btn_save = QtWidgets.QPushButton("💾 Save Kinetics Plot (SVG/PNG)...")
        def _save_kin():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save Kinetics Plot", "kinetics_arrhenius.svg", "SVG (*.svg);;PNG (*.png)")
            if path:
                fig.savefig(path, dpi=300, bbox_inches="tight")
                QtWidgets.QMessageBox.information(dlg, "Saved", f"Plot saved to:\n{path}")
        btn_save.clicked.connect(_save_kin)
        b_box.addWidget(btn_save)
        b_box.addStretch()

        btn_c = QtWidgets.QPushButton("Close")
        btn_c.clicked.connect(dlg.accept)
        b_box.addWidget(btn_c)
        dlg_lay.addLayout(b_box)

        dlg.exec()

