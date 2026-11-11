"""Interactive Energy Decomposition Analysis (EDA) and SAPT Interaction Dialog."""

from __future__ import annotations

import os
from typing import Optional
from PyQt6 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from .eda_tools import parse_eda_text, EDAResult


class EDADialog(QtWidgets.QDialog):
    """Energy Decomposition Analysis & Non-Covalent Interaction Inspector."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("🧩 Energy Decomposition Analysis (EDA / SAPT / ALMO)")
        self.resize(760, 580)
        self.setMinimumSize(680, 480)

        self._build_ui()
        # Default sample values (e.g. water dimer or pi-pi stacking)
        self._load_sample()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>Energy Decomposition Analysis (EDA) & SAPT Interaction Inspector</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Decomposes intermolecular interaction energies (ΔE_int) into physical terms: "
            "Electrostatics (ΔE_elstat), Pauli Repulsion (ΔE_Pauli), Orbital/Charge-Transfer (ΔE_orb), and Dispersion (ΔE_disp). "
            "Parses Q-Chem ALMO-EDA, Psi4 SAPT, and Morokuma/ADF outputs."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_desc.setWordWrap(True)
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        # File Load & Input Row
        top_row = QtWidgets.QHBoxLayout()
        btn_open = QtWidgets.QPushButton("📂 Load EDA / SAPT Log File...")
        btn_open.setStyleSheet("background-color: #0284c7; font-weight: bold; padding: 5px 12px;")
        btn_open.clicked.connect(self._on_open_file)
        top_row.addWidget(btn_open)

        btn_sample = QtWidgets.QPushButton("🧪 Load Sample Interaction (H-Bond)")
        btn_sample.clicked.connect(self._load_sample)
        top_row.addWidget(btn_sample)
        top_row.addStretch()
        layout.addLayout(top_row)

        # Plot Canvas
        self.fig = Figure(facecolor="#0f172a")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas)

        # Summary Breakdown Text
        self.txt_summary = QtWidgets.QTextBrowser()
        self.txt_summary.setFixedHeight(110)
        self.txt_summary.setStyleSheet("background-color: #090d16; color: #e2e8f0; font-family: monospace; font-size: 12px;")
        layout.addWidget(self.txt_summary)

        # Close button
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

    def _load_sample(self):
        sample_eda = EDAResult(
            method="SAPT0 / def2-TZVP",
            elstat_kcal=-8.45,
            pauli_kcal=11.20,
            orb_kcal=-3.15,
            disp_kcal=-4.60,
            total_int_kcal=-5.00,
            summary="Strong hydrogen bonding stabilized by electrostatics & orbital charge transfer.",
        )
        self._render_eda(sample_eda)

    def _on_open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open EDA / SAPT Output", "", "Output Logs (*.out *.log *.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            res = parse_eda_text(text)
            if res:
                self._render_eda(res)
            else:
                QtWidgets.QMessageBox.warning(
                    self, "EDA Parse Warning", "Could not locate ALMO-EDA or Psi4 SAPT energy blocks in file."
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Read Error", str(e))

    def _render_eda(self, res: EDAResult):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#1e293b")

        categories = [
            "Electrostatics\n(ΔE_elstat)",
            "Pauli Repulsion\n(ΔE_Pauli)",
            "Orbital / CT\n(ΔE_orb)",
            "Dispersion\n(ΔE_disp)",
            "Total Interaction\n(ΔE_int)",
        ]
        values = [res.elstat_kcal, res.pauli_kcal, res.orb_kcal, res.disp_kcal, res.total_int_kcal]
        colors = ["#38bdf8", "#f43f5e", "#22c55e", "#a855f7", "#facc15"]

        bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.5)
        ax.axhline(0, color="#64748b", linestyle="--", linewidth=1.0)

        # Label values on top of bars
        for bar, val in zip(bars, values):
            y_pos = bar.get_height() + (0.3 if val >= 0 else -0.8)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                f"{val:+.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                color="#ffffff",
                fontweight="bold",
                fontsize=10,
            )

        ax.set_ylabel("Energy Contribution (kcal/mol)", color="#cbd5e1", fontsize=10)
        ax.set_title(f"Energy Decomposition: {res.method}", color="#f8fafc", fontsize=12, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.grid(axis="y", linestyle=":", alpha=0.3, color="#64748b")

        self.fig.tight_layout()
        self.canvas.draw()

        html = f"""
        <b>Energy Decomposition Summary ({res.method}):</b><br>
        • Attractive Stabilization: <b>{res.elstat_kcal + res.orb_kcal + res.disp_kcal:.2f} kcal/mol</b>
          (Electrostatics: {res.elstat_kcal:.2f}, Orbital: {res.orb_kcal:.2f}, Dispersion: {res.disp_kcal:.2f})<br>
        • Repulsive Exchange Penalty: <b>+{res.pauli_kcal:.2f} kcal/mol</b> (Pauli repulsion)<br>
        • Net Intermolecular Binding: <b style="color: #facc15; font-size: 13px;">ΔE_int = {res.total_int_kcal:.2f} kcal/mol</b>
        """
        self.txt_summary.setHtml(html)
