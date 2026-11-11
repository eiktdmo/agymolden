"""Interactive Open-Shell Spin Contamination & Yamaguchi Approximate Projection Dialog."""

from __future__ import annotations

from typing import Optional
from PyQt6 import QtCore, QtGui, QtWidgets

from .spin_tools import evaluate_spin_contamination, compute_yamaguchi_projection


class SpinContaminationDialog(QtWidgets.QDialog):
    """Interactive Spin Contamination & Yamaguchi Approximate Spin Projection Dialog."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        multiplicity: int = 2,
        s2_calc: float = 0.775,
    ):
        super().__init__(parent)
        self.setWindowTitle("⚛️ Open-Shell Spin Diagnostics & Yamaguchi Spin Projection")
        self.resize(720, 540)
        self.setMinimumSize(640, 460)

        self.initial_mult = multiplicity
        self.initial_s2 = s2_calc

        self._build_ui()
        self._recalculate_diagnostics()
        self._recalculate_yamaguchi()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>Open-Shell Spin Diagnostics & Yamaguchi AP Projection</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Diagnoses spin contamination in unrestricted calculations (<S²>_calc vs S(S+1)), "
            "and computes Yamaguchi approximate spin-projected energies (E_AP) and Heisenberg exchange coupling (J) "
            "for broken-symmetry diradicals and open-shell complexes."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_desc.setWordWrap(True)
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        tabs = QtWidgets.QTabWidget()

        # Tab 1: Single-State Spin Contamination
        tab1 = QtWidgets.QWidget()
        t1_lay = QtWidgets.QVBoxLayout(tab1)

        f_box = QtWidgets.QGroupBox("Calculation Parameters")
        f_lay = QtWidgets.QFormLayout(f_box)

        self.spin_mult = QtWidgets.QSpinBox()
        self.spin_mult.setRange(1, 15)
        self.spin_mult.setValue(self.initial_mult)
        self.spin_mult.valueChanged.connect(self._recalculate_diagnostics)
        f_lay.addRow("Nominal Multiplicity (2S+1):", self.spin_mult)

        self.spin_s2 = QtWidgets.QDoubleSpinBox()
        self.spin_s2.setRange(0.0, 50.0)
        self.spin_s2.setDecimals(4)
        self.spin_s2.setSingleStep(0.01)
        self.spin_s2.setValue(self.initial_s2)
        self.spin_s2.valueChanged.connect(self._recalculate_diagnostics)
        f_lay.addRow("Calculated <S²> Value:", self.spin_s2)

        t1_lay.addWidget(f_box)

        self.txt_diag_result = QtWidgets.QTextBrowser()
        self.txt_diag_result.setStyleSheet("background-color: #090d16; color: #e2e8f0; font-family: monospace; font-size: 12px;")
        t1_lay.addWidget(self.txt_diag_result)
        tabs.addTab(tab1, "🛡️ <S²> Contamination Diagnostic")

        # Tab 2: Yamaguchi Broken-Symmetry Projection (BS vs HS)
        tab2 = QtWidgets.QWidget()
        t2_lay = QtWidgets.QVBoxLayout(tab2)

        g_box = QtWidgets.QGroupBox("Broken-Symmetry (BS) & High-Spin (HS) Energies")
        g_lay = QtWidgets.QGridLayout(g_box)

        g_lay.addWidget(QtWidgets.QLabel("Broken-Symmetry Energy E(BS) [Ha]:"), 0, 0)
        self.spin_ebs = QtWidgets.QDoubleSpinBox()
        self.spin_ebs.setRange(-100000.0, 100000.0)
        self.spin_ebs.setDecimals(6)
        self.spin_ebs.setValue(-150.234500)
        self.spin_ebs.valueChanged.connect(self._recalculate_yamaguchi)
        g_lay.addWidget(self.spin_ebs, 0, 1)

        g_lay.addWidget(QtWidgets.QLabel("BS <S²> Value:"), 0, 2)
        self.spin_s2bs = QtWidgets.QDoubleSpinBox()
        self.spin_s2bs.setRange(0.0, 10.0)
        self.spin_s2bs.setDecimals(4)
        self.spin_s2bs.setValue(1.0250)
        self.spin_s2bs.valueChanged.connect(self._recalculate_yamaguchi)
        g_lay.addWidget(self.spin_s2bs, 0, 3)

        g_lay.addWidget(QtWidgets.QLabel("High-Spin Energy E(HS) [Ha]:"), 1, 0)
        self.spin_ehs = QtWidgets.QDoubleSpinBox()
        self.spin_ehs.setRange(-100000.0, 100000.0)
        self.spin_ehs.setDecimals(6)
        self.spin_ehs.setValue(-150.231200)
        self.spin_ehs.valueChanged.connect(self._recalculate_yamaguchi)
        g_lay.addWidget(self.spin_ehs, 1, 1)

        g_lay.addWidget(QtWidgets.QLabel("HS <S²> Value:"), 1, 2)
        self.spin_s2hs = QtWidgets.QDoubleSpinBox()
        self.spin_s2hs.setRange(0.0, 10.0)
        self.spin_s2hs.setDecimals(4)
        self.spin_s2hs.setValue(2.0150)
        self.spin_s2hs.valueChanged.connect(self._recalculate_yamaguchi)
        g_lay.addWidget(self.spin_s2hs, 1, 3)

        t2_lay.addWidget(g_box)

        self.txt_yamaguchi_result = QtWidgets.QTextBrowser()
        self.txt_yamaguchi_result.setStyleSheet("background-color: #090d16; color: #e2e8f0; font-family: monospace; font-size: 12px;")
        t2_lay.addWidget(self.txt_yamaguchi_result)
        tabs.addTab(tab2, "⚛️ Yamaguchi Projection & J Coupling")

        layout.addWidget(tabs)

        btn_box = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("📋 Copy Summary")
        btn_copy.clicked.connect(self._copy_summary)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_copy)
        btn_box.addStretch()
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)

    def _recalculate_diagnostics(self):
        mult = self.spin_mult.value()
        s2 = self.spin_s2.value()
        res = evaluate_spin_contamination(mult, s2)

        color = "#ef4444" if res.is_contaminated else "#22c55e"
        html = f"""
        <h3>Spin Contamination Assessment:</h3>
        <ul>
          <li><b>Nominal Spin State:</b> Multiplicity = {res.multiplicity} (S = {res.spin_s:.1f}, {res.unpaired_electrons} unpaired e⁻)</li>
          <li><b>Exact Expectation Value &lt;S²&gt;_exact:</b> {res.s2_exact:.4f}</li>
          <li><b>Calculated Expectation Value &lt;S²&gt;_calc:</b> {res.s2_calc:.4f}</li>
          <li><b>Contamination Ratio:</b> <span style="color: {color}; font-weight: bold;">{res.contamination_percent:.2f}%</span></li>
        </ul>
        <p><b>Diagnosis:</b> <span style="color: {color};">{res.status_summary}</span></p>
        <p style="color: #94a3b8; font-size: 11px;">
        Rule of thumb: Contamination exceeding 10% in DFT or 15% in HF indicates significant mixing of unwanted spin multiplets.<br>
        For broken-symmetry open-shell singlets, proceed to the Yamaguchi Projection tab to correct the energy.
        </p>
        """
        self.txt_diag_result.setHtml(html)

    def _recalculate_yamaguchi(self):
        try:
            ebs = self.spin_ebs.value()
            ehs = self.spin_ehs.value()
            s2bs = self.spin_s2bs.value()
            s2hs = self.spin_s2hs.value()

            res = compute_yamaguchi_projection(ebs, ehs, s2bs, s2hs)
            html = f"""
            <h3>Yamaguchi Approximate Spin Projection (AP) Results:</h3>
            <table cellpadding="4" style="border-collapse: collapse;">
              <tr><td><b>Broken-Symmetry Energy E(BS):</b></td><td>{res.e_bs_ha:.6f} Ha</td></tr>
              <tr><td><b>High-Spin Energy E(HS):</b></td><td>{res.e_hs_ha:.6f} Ha</td></tr>
              <tr><td><b>Spin-Projected Singlet E_AP:</b></td><td><b style="color: #38bdf8;">{res.e_ap_ha:.6f} Ha</b></td></tr>
              <tr><td><b>Correction ΔE (E_AP - E_BS):</b></td><td><b>{res.e_ap_delta_kcal:+.2f} kcal/mol</b></td></tr>
              <tr><td colspan="2"><hr style="border: 0.5px solid #334155;"></td></tr>
              <tr><td><b>Heisenberg Exchange Coupling (J_Yamaguchi):</b></td><td><b style="color: #4ade80;">{res.j_yamaguchi_cm1:+.2f} cm⁻¹</b></td></tr>
              <tr><td><b>Noodleman Limit (J_Noodleman):</b></td><td>{res.j_noodleman_cm1:+.2f} cm⁻¹</td></tr>
              <tr><td><b>Magnetic Behavior:</b></td><td><b style="color: #facc15;">{res.coupling_type}</b></td></tr>
            </table>
            <p style="color: #94a3b8; font-size: 11px; margin-top: 8px;">
            Heisenberg Hamiltonian convention: <i>H = -2J · Ŝ_A · Ŝ_B</i>.<br>
            Negative J values indicate antiferromagnetic singlet ground state; positive J indicates ferromagnetic alignment.
            </p>
            """
            self.txt_yamaguchi_result.setHtml(html)
        except Exception as e:
            self.txt_yamaguchi_result.setPlainText(f"Calculation Error: {e}")

    def _copy_summary(self):
        clip = QtWidgets.QApplication.clipboard()
        txt = self.txt_diag_result.toPlainText() + "\n\n" + self.txt_yamaguchi_result.toPlainText()
        clip.setText(txt)
        QtWidgets.QMessageBox.information(self, "Copied", "Spin diagnostics summary copied to clipboard.")
