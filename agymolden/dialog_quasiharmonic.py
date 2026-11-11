"""
Interactive Quasi-Harmonic Thermochemistry & Vibrational Scaling Dialog.

Provides:
- Comparison of Standard RRHO vs. Grimme mRRHO vs. Truhlar quasi-harmonic entropy
- Real-time slider for cutoff frequency omega_0 (default 100 cm^-1)
- Automatic selection or custom adjustment of Truhlar / NIST vibrational scaling factors
- Frequency spectrum preview highlighting affected low-frequency modes (< omega_0)
- Export detailed comparison tables to CSV
"""

from __future__ import annotations

import os
from typing import List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from .quasiharmonic import (
    FREQUENCY_SCALING_FACTORS,
    QuasiHarmonicResult,
    calculate_quasiharmonic_thermo,
)


class QuasiHarmonicDialog(QtWidgets.QDialog):
    """Interactive GUI for quasi-harmonic vibrational thermochemistry and entropy corrections."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        frequencies_cm1: Optional[List[float]] = None,
        log_path: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Quasi-Harmonic Thermochemistry (Grimme mRRHO & Truhlar)")
        self.resize(920, 640)

        self.frequencies: List[float] = frequencies_cm1 or []
        self.log_path = log_path
        self.result: Optional[QuasiHarmonicResult] = None

        if not self.frequencies and self.log_path and os.path.exists(self.log_path):
            self._load_frequencies_from_file(self.log_path)

        if not self.frequencies:
            # Fallback to realistic textbook frequencies containing floppy modes
            self.frequencies = [
                35.2, 58.4, 82.1, 114.0, 168.5, 245.0, 380.2, 510.0,
                820.5, 1050.0, 1220.0, 1450.0, 1720.0, 2950.0, 3050.0
            ]

        self._init_ui()
        self._recalculate()

    def _load_frequencies_from_file(self, path: str):
        try:
            from .qm_log_parser import parse_qm_log
            res = parse_qm_log(path)
            if res.vibrational_analysis and res.vibrational_analysis.frequencies:
                self.frequencies = [float(f) for f in res.vibrational_analysis.frequencies if float(f) > 0]
        except Exception:
            pass

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Left Panel: Controls & Summary
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(380)
        left_lay = QtWidgets.QVBoxLayout(left_panel)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(10)

        # Header Info Card
        hdr_frame = QtWidgets.QFrame()
        hdr_frame.setStyleSheet("background-color: #0f172a; border-radius: 6px; padding: 10px;")
        h_box = QtWidgets.QHBoxLayout(hdr_frame)
        h_icon = QtWidgets.QLabel("<span style='font-size: 26px;'>🌡️</span>")
        h_box.addWidget(h_icon)
        h_text = QtWidgets.QVBoxLayout()
        h_text.addWidget(QtWidgets.QLabel("<b style='color: #38bdf8; font-size: 13px;'>Quasi-Harmonic Entropy</b>"))
        h_text.addWidget(QtWidgets.QLabel("<span style='color: #94a3b8; font-size: 11px;'>Grimme mRRHO & Truhlar low-frequency fix</span>"))
        h_box.addLayout(h_text)
        left_lay.addWidget(hdr_frame)

        # File load button
        btn_load = QtWidgets.QPushButton("📂 Load QM Frequency Output...")
        btn_load.clicked.connect(self._on_load_log)
        left_lay.addWidget(btn_load)

        # Settings
        grp_settings = QtWidgets.QGroupBox("Thermodynamic Conditions & Scaling")
        s_form = QtWidgets.QFormLayout(grp_settings)
        s_form.setSpacing(8)

        self.spin_temp = QtWidgets.QDoubleSpinBox()
        self.spin_temp.setRange(1.0, 3000.0)
        self.spin_temp.setValue(298.15)
        self.spin_temp.valueChanged.connect(self._recalculate)
        s_form.addRow("Temperature (K):", self.spin_temp)

        self.spin_cutoff = QtWidgets.QDoubleSpinBox()
        self.spin_cutoff.setRange(10.0, 500.0)
        self.spin_cutoff.setValue(100.0)
        self.spin_cutoff.valueChanged.connect(self._recalculate)
        s_form.addRow("Cutoff ω₀ (cm⁻¹):", self.spin_cutoff)

        self.cbo_model = QtWidgets.QComboBox()
        self.cbo_model.addItem("Unscaled (1.000)", 1.0)
        for name, factor in FREQUENCY_SCALING_FACTORS.items():
            self.cbo_model.addItem(f"{name} ({factor:.4f})", factor)
        self.cbo_model.currentIndexChanged.connect(self._on_model_changed)
        s_form.addRow("Method Scaling:", self.cbo_model)

        self.spin_scaling = QtWidgets.QDoubleSpinBox()
        self.spin_scaling.setRange(0.5, 1.5)
        self.spin_scaling.setDecimals(4)
        self.spin_scaling.setSingleStep(0.005)
        self.spin_scaling.setValue(1.0000)
        self.spin_scaling.valueChanged.connect(self._recalculate)
        s_form.addRow("Custom Factor:", self.spin_scaling)

        left_lay.addWidget(grp_settings)

        # Thermochemical Results Table
        grp_results = QtWidgets.QGroupBox("Thermochemical Comparison (298.15 K)")
        r_lay = QtWidgets.QVBoxLayout(grp_results)

        self.tbl_results = QtWidgets.QTableWidget(4, 3)
        self.tbl_results.setHorizontalHeaderLabels(["Model", "S_vib (cal/mol·K)", "ΔG vs RRHO (kcal/mol)"])
        self.tbl_results.horizontalHeader().setStretchLastSection(True)
        self.tbl_results.verticalHeader().setVisible(False)
        r_lay.addWidget(self.tbl_results)

        left_lay.addWidget(grp_results)

        # Export CSV Button
        btn_csv = QtWidgets.QPushButton("💾 Export Quasi-Harmonic Summary (CSV)...")
        btn_csv.clicked.connect(self._on_export_csv)
        left_lay.addWidget(btn_csv)

        main_layout.addWidget(left_panel)

        # Right Panel: Matplotlib Figure
        right_panel = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right_panel)
        right_lay.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(facecolor="#18181b", tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, right_panel)
        self.toolbar.setStyleSheet("background-color: #27272a; color: white;")

        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        right_lay.addWidget(self.toolbar)
        right_lay.addWidget(self.canvas)
        main_layout.addWidget(right_panel)

    def _style_axes(self):
        self.ax.set_facecolor("#09090b")
        self.ax.tick_params(colors="#94a3b8", labelsize=10)
        for spine in self.ax.spines.values():
            spine.set_color("#3f3f46")
        self.ax.grid(True, linestyle="--", alpha=0.25, color="#71717a")

    def _on_model_changed(self):
        factor = float(self.cbo_model.currentData())
        self.spin_scaling.blockSignals(True)
        self.spin_scaling.setValue(factor)
        self.spin_scaling.blockSignals(False)
        self._recalculate()

    def _on_load_log(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open QM Frequency Output",
            "",
            "QM Output (*.log *.out *.txt);;All Files (*)",
        )
        if path:
            from .qm_log_parser import parse_qm_log
            res = parse_qm_log(path)
            if res.vibrational_analysis and res.vibrational_analysis.frequencies:
                self.frequencies = [float(f) for f in res.vibrational_analysis.frequencies if float(f) > 0]
                self.log_path = path
                self._recalculate()
            else:
                QtWidgets.QMessageBox.warning(self, "No Frequencies", f"No positive frequencies found in:\n{path}")

    def _recalculate(self):
        temp = self.spin_temp.value()
        cutoff = self.spin_cutoff.value()
        scaling = self.spin_scaling.value()

        self.result = calculate_quasiharmonic_thermo(
            frequencies_cm1=self.frequencies,
            temperature_k=temp,
            cutoff_cm1=cutoff,
            scaling_factor=scaling,
        )

        res = self.result

        # Populate Results Table
        rows_data = [
            ("Standard RRHO", f"{res.s_vib_standard_cal_mol_k:.2f}", "0.00 (Reference)"),
            ("Grimme mRRHO", f"{res.s_vib_grimme_cal_mol_k:.2f}", f"{res.delta_g_grimme_kcal_mol:+.3f}"),
            ("Truhlar Quasi-Harmonic", f"{res.s_vib_truhlar_cal_mol_k:.2f}", f"{res.delta_g_truhlar_kcal_mol:+.3f}"),
            ("Zero-Point Energy (ZPE)", f"{res.zpe_kcal_mol:.2f} kcal/mol", "-"),
        ]

        self.tbl_results.blockSignals(True)
        for r, (model, s_val, dg_val) in enumerate(rows_data):
            self.tbl_results.setItem(r, 0, QtWidgets.QTableWidgetItem(model))
            self.tbl_results.setItem(r, 1, QtWidgets.QTableWidgetItem(s_val))
            self.tbl_results.setItem(r, 2, QtWidgets.QTableWidgetItem(dg_val))
        self.tbl_results.blockSignals(False)

        self._replot()

    def _replot(self):
        if not self.result:
            return

        self.ax.clear()
        self._style_axes()

        res = self.result
        freqs = res.frequencies_scaled
        cutoff = res.cutoff_cm1

        # Plot all vibrational frequencies as sticks
        low_freqs = [f for f in freqs if f <= cutoff]
        high_freqs = [f for f in freqs if f > cutoff]

        if low_freqs:
            self.ax.vlines(low_freqs, ymin=0, ymax=1.0, colors="#f43f5e", linewidth=2.0, label=f"Floppy Modes (≤ {cutoff:.0f} cm⁻¹)")
        if high_freqs:
            self.ax.vlines(high_freqs, ymin=0, ymax=0.6, colors="#38bdf8", linewidth=1.5, label="Harmonic Modes")

        # Shaded cutoff region
        self.ax.axvspan(0, cutoff, color="#f43f5e", alpha=0.15, label="Quasi-Harmonic Damping Zone")
        self.ax.axvline(cutoff, color="#fbbf24", linestyle="--", linewidth=1.5, label=f"Cutoff ω₀ = {cutoff:.0f} cm⁻¹")

        max_f = max(freqs) if freqs else 4000.0
        self.ax.set_xlim(-20.0, min(max_f + 100.0, 3600.0))
        self.ax.set_ylim(0.0, 1.15)

        self.ax.set_xlabel("Vibrational Frequency ν (cm⁻¹)", color="#f1f5f9", fontsize=11, fontweight="bold")
        self.ax.set_ylabel("Relative Mode Weighting", color="#f1f5f9", fontsize=11)
        self.ax.set_title(
            f"Vibrational Frequency Spectrum & Damping (ΔG_Grimme = {res.delta_g_grimme_kcal_mol:+.3f} kcal/mol)",
            color="#38bdf8",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        self.ax.legend(loc="upper right", facecolor="#18181b", edgecolor="#3f3f46", labelcolor="#f1f5f9")
        self.canvas.draw()

    def _on_export_csv(self):
        if not self.result:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Quasi-Harmonic Thermochemistry",
            "quasiharmonic_thermo.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            res = self.result
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Temperature (K),{res.temperature_k}\n")
                f.write(f"# Cutoff (cm^-1),{res.cutoff_cm1}\n")
                f.write(f"# Scaling Factor,{res.scaling_factor}\n")
                f.write(f"# ZPE (kcal/mol),{res.zpe_kcal_mol:.4f}\n")
                f.write("Model,S_vib_cal_mol_K,G_vib_kcal_mol,DeltaG_vs_RRHO_kcal_mol\n")
                f.write(f"Standard_RRHO,{res.s_vib_standard_cal_mol_k:.4f},{res.g_vib_standard_kcal_mol:.4f},0.0000\n")
                f.write(f"Grimme_mRRHO,{res.s_vib_grimme_cal_mol_k:.4f},{res.g_vib_grimme_kcal_mol:.4f},{res.delta_g_grimme_kcal_mol:.4f}\n")
                f.write(f"Truhlar_QuasiHarmonic,{res.s_vib_truhlar_cal_mol_k:.4f},{res.g_vib_truhlar_kcal_mol:.4f},{res.delta_g_truhlar_kcal_mol:.4f}\n")
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Successfully exported data to:\n{path}")
