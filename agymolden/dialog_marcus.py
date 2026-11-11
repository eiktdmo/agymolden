"""Marcus Theory Electron Transfer & Photoredox Kinetics Dialog.

Computes:
1. Four-point internal reorganization energy (λ_in) from neutral and ionized states:
   λ_in = [E_ox(geom_red) - E_ox(geom_ox)] + [E_red(geom_ox) - E_red(geom_red)]
2. Continuum solvent outer-sphere reorganization energy (λ_out):
   λ_out = (Δe)² / (8πε_0) * (1/r_D + 1/r_A - 2/R_DA) * (1/n² - 1/ε)
3. Total reorganization energy: λ = λ_in + λ_out
4. Marcus classical activation free energy:
   ΔG‡_ET = (λ + ΔG⁰)² / (4λ)
5. Semiclassical Non-Adiabatic Electron Transfer rate constant:
   k_ET = (2π / ℏ) * |H_DA|² * (4πλ kB T)^(-1/2) * exp( - (λ + ΔG⁰)² / (4λ kB T) )
6. Interactive diabatic reactant/product Marcus parabolas showing normal and inverted regimes.
"""

from __future__ import annotations

import math
from typing import Optional
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .solvation import SOLVENT_DATABASE


def compute_marcus_kinetics(
    delta_g0_ev: float,
    lambda_in_ev: float,
    lambda_out_ev: float,
    v_da_cm1: float = 50.0,
    temp_k: float = 298.15,
) -> dict:
    """Computes Marcus electron transfer kinetics and regime."""
    lam_tot = max(0.001, lambda_in_ev + lambda_out_ev)

    # Marcus activation barrier: ΔG‡ = (λ + ΔG0)² / (4λ)
    act_ev = ((lam_tot + delta_g0_ev) ** 2) / (4.0 * lam_tot)
    act_kcal = act_ev * 23.0605

    # Non-adiabatic rate constant:
    # k_ET = (2π / ℏ) * |H_DA|² * (4πλ kB T)^(-1/2) * exp(-ΔG‡ / kB T)
    hda_ev = v_da_cm1 * 1.23984e-4
    kb_ev = 8.617333e-5  # eV/K
    kb_t = kb_ev * temp_k
    hbar_ev_s = 6.582119e-16  # eV*s

    prefactor = (2.0 * math.pi / hbar_ev_s) * (hda_ev**2)
    gauss_factor = 1.0 / math.sqrt(4.0 * math.pi * lam_tot * kb_t)
    exp_arg = -act_ev / kb_t

    if exp_arg < -600.0:
        k_et = 0.0
    elif exp_arg > 600.0:
        k_et = float("inf")
    else:
        k_et = prefactor * gauss_factor * math.exp(exp_arg)

    # Regime identification
    if abs(delta_g0_ev + lam_tot) < 0.05:
        regime = "Activationless / Optimal (-ΔG⁰ ≈ λ)"
    elif -delta_g0_ev < lam_tot:
        regime = "Normal Activation (|ΔG°| < λ)"
    else:
        regime = "Inverted Regime (-ΔG⁰ > λ)"

    return {
        "lambda_total_ev": lam_tot,
        "delta_g_act_ev": act_ev,
        "delta_g_act_kcal": act_kcal,
        "k_et_s1": k_et,
        "regime": regime,
    }


class MarcusElectronTransferDialog(QtWidgets.QDialog):
    """Interactive Marcus Theory & Photoredox Electron Transfer Workbench."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("⚡ Marcus Theory & Photoredox Electron Transfer (λ, ΔG‡_ET, k_ET)")
        self.resize(920, 680)
        self.setMinimumSize(800, 560)

        self._build_ui()
        self._recalculate()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header Frame
        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>Marcus Electron Transfer & Photoredox Reorganization Workbench</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Evaluate total reorganization energy (λ = λ_in + λ_out), driving force (ΔG⁰), "
            "activation barrier (ΔG‡_ET), and non-adiabatic rate constant (k_ET) with visual diabatic parabolas spanning normal and inverted regions."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_desc.setWordWrap(True)
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        # Controls Grid
        ctrl_box = QtWidgets.QGroupBox("Marcus Parameters")
        grid = QtWidgets.QGridLayout(ctrl_box)

        # Driving force ΔG0
        grid.addWidget(QtWidgets.QLabel("<b>Driving Force ΔG⁰ (eV):</b>"), 0, 0)
        self.spin_dg0 = QtWidgets.QDoubleSpinBox()
        self.spin_dg0.setRange(-5.0, 5.0)
        self.spin_dg0.setValue(-0.35)
        self.spin_dg0.setSingleStep(0.05)
        self.spin_dg0.setDecimals(3)
        self.spin_dg0.valueChanged.connect(self._recalculate)
        grid.addWidget(self.spin_dg0, 0, 1)

        # Internal Reorganization λ_in
        grid.addWidget(QtWidgets.QLabel("Internal Reorganization λ_in (eV):"), 0, 2)
        self.spin_lambda_in = QtWidgets.QDoubleSpinBox()
        self.spin_lambda_in.setRange(0.01, 3.0)
        self.spin_lambda_in.setValue(0.30)
        self.spin_lambda_in.setSingleStep(0.05)
        self.spin_lambda_in.setDecimals(3)
        self.spin_lambda_in.valueChanged.connect(self._recalculate)
        grid.addWidget(self.spin_lambda_in, 0, 3)

        # Electronic coupling |H_DA|
        grid.addWidget(QtWidgets.QLabel("Coupling |H_DA| (cm⁻¹):"), 0, 4)
        self.spin_hda = QtWidgets.QDoubleSpinBox()
        self.spin_hda.setRange(0.1, 5000.0)
        self.spin_hda.setValue(50.0)
        self.spin_hda.setSingleStep(10.0)
        self.spin_hda.valueChanged.connect(self._recalculate)
        grid.addWidget(self.spin_hda, 0, 5)

        # Solvent environment for outer-sphere λ_out
        grid.addWidget(QtWidgets.QLabel("Solvent (for λ_out):"), 1, 0)
        self.combo_solvent = QtWidgets.QComboBox()
        for k, s in SOLVENT_DATABASE.items():
            self.combo_solvent.addItem(f"{s.name} (ε={s.dielectric_constant:.1f}, n={s.refractive_index:.2f})", k)
        self.combo_solvent.currentIndexChanged.connect(self._recalculate)
        grid.addWidget(self.combo_solvent, 1, 1, 1, 2)

        # Donor-Acceptor distance R_DA
        grid.addWidget(QtWidgets.QLabel("D–A Distance (Å):"), 1, 3)
        self.spin_rda = QtWidgets.QDoubleSpinBox()
        self.spin_rda.setRange(3.0, 30.0)
        self.spin_rda.setValue(8.5)
        self.spin_rda.setSingleStep(0.5)
        self.spin_rda.valueChanged.connect(self._recalculate)
        grid.addWidget(self.spin_rda, 1, 4)

        # Temperature
        grid.addWidget(QtWidgets.QLabel("Temperature (K):"), 1, 5)
        self.spin_temp = QtWidgets.QDoubleSpinBox()
        self.spin_temp.setRange(10.0, 1000.0)
        self.spin_temp.setValue(298.15)
        self.spin_temp.valueChanged.connect(self._recalculate)
        grid.addWidget(self.spin_temp, 1, 6)

        layout.addWidget(ctrl_box)

        # Splitter: Left Diabatic Parabolas Plot, Right Results Table
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Matplotlib Canvas
        plot_panel = QtWidgets.QFrame()
        p_lay = QtWidgets.QVBoxLayout(plot_panel)
        p_lay.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(5.4, 4.6), dpi=100, facecolor="#1e293b")
        self.canvas = FigureCanvas(self.fig)
        p_lay.addWidget(self.canvas)
        splitter.addWidget(plot_panel)

        # Results Table
        right_panel = QtWidgets.QFrame()
        r_lay = QtWidgets.QVBoxLayout(right_panel)
        r_lay.setContentsMargins(6, 0, 0, 0)

        self.tbl_results = QtWidgets.QTableWidget()
        self.tbl_results.setColumnCount(2)
        self.tbl_results.setHorizontalHeaderLabels(["Marcus Kinetic Property", "Value"])
        self.tbl_results.horizontalHeader().setStretchLastSection(True)
        self.tbl_results.verticalHeader().setVisible(False)
        self.tbl_results.setStyleSheet("background-color: #020617; color: #f8fafc; gridline-color: #1e293b;")
        r_lay.addWidget(self.tbl_results)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter, 1)

        # Bottom Bar
        b_bar = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("📋 Copy Marcus Kinetics for SI")
        btn_copy.clicked.connect(self._on_copy_summary)
        b_bar.addWidget(btn_copy)

        btn_save = QtWidgets.QPushButton("💾 Save Marcus Parabolas (SVG/PNG)...")
        btn_save.clicked.connect(self._on_save_plot)
        b_bar.addWidget(btn_save)

        b_bar.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        b_bar.addWidget(btn_close)
        layout.addLayout(b_bar)

    def _recalculate(self):
        dg0_ev = self.spin_dg0.value()
        lam_in = self.spin_lambda_in.value()
        hda_cm1 = self.spin_hda.value()
        solv_key = self.combo_solvent.currentData() or "water"
        r_da = self.spin_rda.value()
        temp_k = self.spin_temp.value()

        solv = SOLVENT_DATABASE.get(solv_key, SOLVENT_DATABASE["water"])
        eps = solv.dielectric_constant
        n_ref = solv.refractive_index

        # Outer-sphere reorganization energy λ_out (Marcus two-sphere model):
        # λ_out = e² / (4πε0) * (1/(2a_D) + 1/(2a_A) - 1/R) * (1/n² - 1/ε)
        # e² / (4πε0) in eV * Å ≈ 14.3996
        # Assuming typical effective radii a_D = a_A ≈ 3.5 Å
        a_d, a_a = 3.5, 3.5
        gamma_factor = (1.0 / (n_ref**2)) - (1.0 / eps)
        dist_factor = (1.0 / (2.0 * a_d)) + (1.0 / (2.0 * a_a)) - (1.0 / r_da)
        dist_factor = max(0.01, dist_factor)
        lam_out = max(0.0, 14.3996 * dist_factor * gamma_factor)

        lam_tot = lam_in + lam_out
        lam_tot = max(0.01, lam_tot)

        # Marcus activation barrier: ΔG‡ = (λ + ΔG0)² / (4λ)
        act_ev = ((lam_tot + dg0_ev)**2) / (4.0 * lam_tot)
        act_kcal = act_ev * 23.0605

        # Non-adiabatic rate constant:
        # k_ET = (2π / ℏ) * |H_DA|² * (4πλ kB T)^(-1/2) * exp(-ΔG‡ / kB T)
        # 1 cm^-1 = 1.23984e-4 eV
        hda_ev = hda_cm1 * 1.23984e-4
        kb_ev = 8.617333e-5  # eV/K
        kb_t = kb_ev * temp_k
        hbar_ev_s = 6.582119e-16  # eV*s

        prefactor = (2.0 * math.pi / hbar_ev_s) * (hda_ev**2)
        gauss_factor = 1.0 / math.sqrt(4.0 * math.pi * lam_tot * kb_t)
        exp_arg = -act_ev / kb_t

        if exp_arg < -600.0:
            k_et = 0.0
        elif exp_arg > 600.0:
            k_et = float("inf")
        else:
            k_et = prefactor * gauss_factor * math.exp(exp_arg)

        # Regime identification
        if abs(dg0_ev + lam_tot) < 0.05:
            regime = "Barrierless / Optimal (-ΔG⁰ ≈ λ)"
        elif -dg0_ev < lam_tot:
            regime = "Normal Marcus Regime (-ΔG⁰ < λ)"
        else:
            regime = "Marcus Inverted Regime (-ΔG⁰ > λ)"

        self._render_parabolas(lam_tot, dg0_ev, act_ev, regime)
        self._populate_table(lam_in, lam_out, lam_tot, dg0_ev, act_ev, act_kcal, k_et, regime)

    def _render_parabolas(self, lam: float, dg0: float, act: float, regime: str):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#0f172a")

        # Reaction coordinate q from -0.5 to 1.5
        q = np.linspace(-0.4, 1.4, 200)
        # Reactant state R centered at q = 0: E_R(q) = λ * q²
        e_r = lam * (q**2)
        # Product state P centered at q = 1: E_P(q) = λ * (q - 1)² + ΔG0
        e_p = lam * ((q - 1.0)**2) + dg0

        ax.plot(q, e_r, color="#38bdf8", linewidth=2.8, label="Reactant State (D + A)")
        ax.plot(q, e_p, color="#f43f5e", linewidth=2.8, label="Product State (D⁺ + A⁻)")

        # Crossing point q_cross = 0.5 * (1 + ΔG0 / λ)
        q_cross = 0.5 * (1.0 + dg0 / lam)
        e_cross = act
        ax.plot(q_cross, e_cross, "o", color="#fbbf24", markersize=9, zorder=6, label=f"Crossing TS (ΔG‡ = {act:.2f} eV)")

        ax.set_xlabel("Nuclear Polarization Coordinate q", color="#e2e8f0", fontweight="bold")
        ax.set_ylabel("Free Energy (eV)", color="#e2e8f0", fontweight="bold")
        ax.set_title(f"Marcus Diabatic Parabolas — {regime}", color="#f8fafc", fontweight="bold", fontsize=11, pad=10)
        ax.tick_params(colors="#94a3b8")
        ax.grid(color="#334155", linestyle=":", alpha=0.6)
        ax.legend(loc="upper center", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)

        y_min = min(np.min(e_r), np.min(e_p)) - 0.2
        y_max = max(e_cross + 0.5, 1.2)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.35, 1.35)

        self.fig.tight_layout()
        self.canvas.draw()

    def _populate_table(self, lam_in, lam_out, lam_tot, dg0, act_ev, act_kcal, k_et, regime):
        rows = [
            ("Kinetic Regime", f"<b>{regime}</b>"),
            ("Driving Force (ΔG⁰)", f"{dg0:+.3f} eV ({dg0 * 23.0605:+.2f} kcal/mol)"),
            ("Internal Reorganization (λ_in)", f"{lam_in:.3f} eV ({lam_in * 23.0605:.2f} kcal/mol)"),
            ("Solvent Outer-Sphere (λ_out)", f"{lam_out:.3f} eV ({lam_out * 23.0605:.2f} kcal/mol)"),
            ("Total Reorganization Energy (λ)", f"<b>{lam_tot:.3f} eV ({lam_tot * 23.0605:.2f} kcal/mol)</b>"),
            ("Marcus Activation Barrier (ΔG‡_ET)", f"<b>{act_ev:.3f} eV ({act_kcal:.2f} kcal/mol)</b>"),
            ("Non-Adiabatic Rate Constant (k_ET)", f"<b>{k_et:.2e} s⁻¹</b>"),
        ]

        self.tbl_results.setRowCount(len(rows))
        for r, (prop, val) in enumerate(rows):
            item_p = QtWidgets.QTableWidgetItem(prop)
            item_v = QtWidgets.QTableWidgetItem(val.replace("<b>", "").replace("</b>", ""))
            if "Rate Constant" in prop or "Barrier" in prop:
                item_v.setFont(QtGui.QFont("", -1, QtGui.QFont.Weight.Bold))
                item_v.setForeground(QtGui.QColor("#38bdf8"))
            self.tbl_results.setItem(r, 0, item_p)
            self.tbl_results.setItem(r, 1, item_v)

    def _on_copy_summary(self):
        lines = ["# Marcus Theory Electron Transfer Kinetics Summary\n"]
        for r in range(self.tbl_results.rowCount()):
            p = self.tbl_results.item(r, 0).text()
            v = self.tbl_results.item(r, 1).text()
            lines.append(f"{p}: {v}")
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        QtWidgets.QMessageBox.information(self, "Copied", "Marcus kinetics parameters copied to clipboard.")

    def _on_save_plot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Marcus Parabolas Diagram", "marcus_parabolas.svg",
            "Scalable Vector Graphics (*.svg);;Portable Network Graphics (*.png);;PDF (*.pdf)"
        )
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            QtWidgets.QMessageBox.information(self, "Saved", f"Diagram saved to:\n{path}")
