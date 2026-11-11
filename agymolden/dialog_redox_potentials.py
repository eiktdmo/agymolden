"""Interactive Electrochemical Redox Potentials & Born-Haber Cycle Dialog."""

from __future__ import annotations

from typing import Optional
from PyQt6 import QtCore, QtGui, QtWidgets

from .electrochemistry import (
    REFERENCE_ELECTRODES,
    compute_redox_potential,
    RedoxPotentialResult,
)
from .solvation import SOLVENT_DATABASE


class RedoxPotentialsDialog(QtWidgets.QDialog):
    """Interactive Born-Haber Solvation Cycle & Redox Potential Estimator."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        initial_g_ox_ha: float = -150.500000,
        initial_g_red_ha: float = -150.600000,
    ):
        super().__init__(parent)
        self.setWindowTitle("⚡ Electrochemical Redox Potentials & Born-Haber Cycle")
        self.resize(760, 560)
        self.setMinimumSize(680, 480)

        self._build_ui(initial_g_ox_ha, initial_g_red_ha)
        self._recalculate()

    def _build_ui(self, g_ox: float, g_red: float):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>Electrochemical Redox Potentials (Born-Haber Thermodynamic Cycle)</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Computes standard reduction and oxidation potentials (E° vs. SHE, SCE, Ag/AgCl, and Fc/Fc⁺) "
            "by coupling gas-phase free energies (ΔG°_gas) with continuum solvation free energies (ΔG_solv) via the Born-Haber cycle."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_desc.setWordWrap(True)
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        # Input Grid
        grid_box = QtWidgets.QGroupBox("Redox Couple Parameters (Ox + n e⁻ ⇄ Red)")
        grid = QtWidgets.QGridLayout(grid_box)

        grid.addWidget(QtWidgets.QLabel("Oxidized State Gas G°(Ox) [Ha]:"), 0, 0)
        self.spin_g_ox = QtWidgets.QDoubleSpinBox()
        self.spin_g_ox.setRange(-100000.0, 100000.0)
        self.spin_g_ox.setDecimals(6)
        self.spin_g_ox.setValue(g_ox)
        self.spin_g_ox.valueChanged.connect(self._recalculate)
        grid.addWidget(self.spin_g_ox, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Reduced State Gas G°(Red) [Ha]:"), 0, 2)
        self.spin_g_red = QtWidgets.QDoubleSpinBox()
        self.spin_g_red.setRange(-100000.0, 100000.0)
        self.spin_g_red.setDecimals(6)
        self.spin_g_red.setValue(g_red)
        self.spin_g_red.valueChanged.connect(self._recalculate)
        grid.addWidget(self.spin_g_red, 0, 3)

        grid.addWidget(QtWidgets.QLabel("ΔG_solv(Ox) [kcal/mol]:"), 1, 0)
        self.spin_solv_ox = QtWidgets.QDoubleSpinBox()
        self.spin_solv_ox.setRange(-1000.0, 1000.0)
        self.spin_solv_ox.setDecimals(2)
        self.spin_solv_ox.setValue(-15.00)
        self.spin_solv_ox.valueChanged.connect(self._recalculate)
        grid.addWidget(self.spin_solv_ox, 1, 1)

        grid.addWidget(QtWidgets.QLabel("ΔG_solv(Red) [kcal/mol]:"), 1, 2)
        self.spin_solv_red = QtWidgets.QDoubleSpinBox()
        self.spin_solv_red.setRange(-1000.0, 1000.0)
        self.spin_solv_red.setDecimals(2)
        self.spin_solv_red.setValue(-65.00)
        self.spin_solv_red.valueChanged.connect(self._recalculate)
        grid.addWidget(self.spin_solv_red, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Electrons Transferred (n):"), 2, 0)
        self.spin_n = QtWidgets.QSpinBox()
        self.spin_n.setRange(1, 4)
        self.spin_n.setValue(1)
        self.spin_n.valueChanged.connect(self._recalculate)
        grid.addWidget(self.spin_n, 2, 1)

        grid.addWidget(QtWidgets.QLabel("Solvent Reference:"), 2, 2)
        self.combo_solv = QtWidgets.QComboBox()
        for k in sorted(SOLVENT_DATABASE.keys()):
            self.combo_solv.addItem(f"{k.capitalize()}", k)
        self.combo_solv.setCurrentText("Acetonitrile")
        grid.addWidget(self.combo_solv, 2, 3)

        layout.addWidget(grid_box)

        # Output Results Browser
        self.txt_result = QtWidgets.QTextBrowser()
        self.txt_result.setStyleSheet("background-color: #090d16; color: #e2e8f0; font-family: monospace; font-size: 12px;")
        layout.addWidget(self.txt_result)

        # Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("📋 Copy Table")
        btn_copy.clicked.connect(self._copy_results)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_copy)
        btn_box.addStretch()
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)

    def _recalculate(self):
        try:
            g_ox = self.spin_g_ox.value()
            g_red = self.spin_g_red.value()
            s_ox = self.spin_solv_ox.value()
            s_red = self.spin_solv_red.value()
            n = self.spin_n.value()

            res = compute_redox_potential(
                g_ox_gas_ha=g_ox,
                g_red_gas_ha=g_red,
                dg_solv_ox_kcal=s_ox,
                dg_solv_red_kcal=s_red,
                electrons_n=n,
            )

            v_she = res.e0_she_v
            v_sce = res.potentials_vs_ref.get("SCE (Sat. Calomel, +0.241 V vs SHE)", v_she - 0.241)
            v_fc = res.potentials_vs_ref.get("Fc/Fc+ (Ferrocene in MeCN, +0.641 V vs SHE)", v_she - 0.641)
            v_agcl = res.potentials_vs_ref.get("Ag/AgCl (Sat. KCl, +0.197 V vs SHE)", v_she - 0.197)

            html = f"""
            <h3>Born-Haber Thermodynamic Cycle Breakdown:</h3>
            <table cellpadding="4" style="border-collapse: collapse; width: 100%;">
              <tr style="border-bottom: 1px solid #334155;">
                <td><b>Gas-Phase Reduction Free Energy (ΔG°_gas):</b></td>
                <td><b>{res.dg_gas_kcal:+.2f} kcal/mol</b> ({res.g_red_gas_ha - res.g_ox_gas_ha:+.6f} Ha)</td>
              </tr>
              <tr style="border-bottom: 1px solid #334155;">
                <td><b>Differential Solvation Free Energy (ΔΔG_solv):</b></td>
                <td><b>{res.ddg_solv_kcal:+.2f} kcal/mol</b> (Red: {res.dg_solv_red_kcal:.1f}, Ox: {res.dg_solv_ox_kcal:.1f})</td>
              </tr>
              <tr style="border-bottom: 2px solid #38bdf8;">
                <td><b>Solution-Phase Net Reduction Free Energy (ΔG°_solv):</b></td>
                <td><b style="color: #38bdf8; font-size: 13px;">{res.dg_total_solv_kcal:+.2f} kcal/mol</b></td>
              </tr>
            </table>

            <h3 style="margin-top: 14px;">Calculated Standard Redox Potentials:</h3>
            <table cellpadding="4" style="border-collapse: collapse; width: 100%;">
              <tr><td><b>E° vs. SHE (Standard Hydrogen Electrode):</b></td><td><b style="color: #4ade80; font-size: 14px;">{v_she:+.3f} V</b></td></tr>
              <tr><td><b>E° vs. SCE (Saturated Calomel Electrode):</b></td><td><b style="color: #38bdf8; font-size: 13px;">{v_sce:+.3f} V</b></td></tr>
              <tr><td><b>E° vs. Ag/AgCl (Sat. KCl):</b></td><td><b style="color: #facc15; font-size: 13px;">{v_agcl:+.3f} V</b></td></tr>
              <tr><td><b>E° vs. Fc/Fc⁺ (Ferrocene couple in MeCN):</b></td><td><b style="color: #f43f5e; font-size: 13px;">{v_fc:+.3f} V</b></td></tr>
            </table>
            """
            self.txt_result.setHtml(html)
        except Exception as e:
            self.txt_result.setPlainText(f"Error: {e}")

    def _copy_results(self):
        clip = QtWidgets.QApplication.clipboard()
        clip.setText(self.txt_result.toPlainText())
        QtWidgets.QMessageBox.information(self, "Copied", "Redox potential table copied to clipboard.")
