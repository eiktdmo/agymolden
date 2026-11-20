"""
Interactive 1D NMR Chemical Shift Referencing & Spectrum Plotter Dialog.

Provides:
- Selection of active nucleus (1H, 13C, 19F, 31P)
- Standard DFT linear referencing models (TMS, Tantillo scaled, CFCl3, H3PO4) or custom slope/intercept
- Continuous Lorentzian line-broadened 1D NMR spectrum plot with reversed ppm scale (standard chemical shift axis)
- Peak table with atom index, element, isotropic shielding, and referenced chemical shift (ppm)
- Click on spectrum peak or table row to emit atom selection highlight to the 3D viewer
- Export spectrum to high-resolution PNG / SVG / PDF and CSV peak list
"""

from __future__ import annotations
import logging

import os
from typing import Dict, List, Optional
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from .nmr_engine import (
    NMR_REFERENCE_STANDARDS,
    NMRPeak,
    NMRSpectrum,
    compute_chemical_shifts,
    synthesize_nmr_spectrum,
)


class NMRDialog(QtWidgets.QDialog):
    """Dialog for interactive NMR chemical shift referencing and 1D spectrum visualization."""

    # Signal emitted when a peak or table row is clicked: transmits atom index (0-based)
    atom_selected = QtCore.pyqtSignal(int)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        shieldings: Optional[List[Dict[str, float]]] = None,
        log_path: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("NMR Chemical Shift Referencing & 1D Spectrum Plotter")
        self.resize(960, 680)

        self.shieldings: List[Dict[str, float]] = shieldings or []
        self.log_path = log_path
        self.current_peaks: List[NMRPeak] = []
        self.current_spectrum: Optional[NMRSpectrum] = None

        # Auto-parse log if path given and no shieldings provided
        if not self.shieldings and self.log_path and os.path.exists(self.log_path):
            self._load_from_log_path(self.log_path)

        self._init_ui()
        self._populate_standards()
        self._update_all()

    def _load_from_log_path(self, path: str):
        try:
            from .qm_log_parser import parse_qm_log
            res = parse_qm_log(path)
            if res.nmr_shieldings:
                self.shieldings = res.nmr_shieldings
        except Exception:
            logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Left Panel: Controls & Peak List
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(360)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Header Info Card
        hdr_frame = QtWidgets.QFrame()
        hdr_frame.setStyleSheet("background-color: #0f172a; border-radius: 6px; padding: 10px;")
        hdr_box = QtWidgets.QHBoxLayout(hdr_frame)
        hdr_icon = QtWidgets.QLabel("<span style='font-size: 26px;'>🧲</span>")
        hdr_box.addWidget(hdr_icon)
        hdr_text = QtWidgets.QVBoxLayout()
        hdr_text.addWidget(QtWidgets.QLabel("<b style='color: #38bdf8; font-size: 13px;'>NMR Shielding Referencer</b>"))
        hdr_text.addWidget(QtWidgets.QLabel("<span style='color: #94a3b8; font-size: 11px;'>Referenced shifts & continuous 1D spectra</span>"))
        hdr_box.addLayout(hdr_text)
        left_layout.addWidget(hdr_frame)

        # File load button
        btn_load = QtWidgets.QPushButton("📂 Load QM Log (ORCA / Gaussian)...")
        btn_load.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 5px;")
        btn_load.clicked.connect(self._on_load_log_clicked)
        left_layout.addWidget(btn_load)

        # Settings Group
        grp_settings = QtWidgets.QGroupBox("Nucleus & Reference Standard")
        form = QtWidgets.QFormLayout(grp_settings)
        form.setSpacing(8)

        self.cbo_nucleus = QtWidgets.QComboBox()
        self.cbo_nucleus.addItems(["1H", "13C", "19F", "31P"])
        self.cbo_nucleus.currentIndexChanged.connect(self._on_nucleus_changed)
        form.addRow("Active Nucleus:", self.cbo_nucleus)

        self.cbo_standard = QtWidgets.QComboBox()
        self.cbo_standard.currentIndexChanged.connect(self._on_standard_changed)
        form.addRow("Referencing Model:", self.cbo_standard)

        self.spin_intercept = QtWidgets.QDoubleSpinBox()
        self.spin_intercept.setRange(-1000.0, 1000.0)
        self.spin_intercept.setDecimals(2)
        self.spin_intercept.valueChanged.connect(self._update_all)
        form.addRow("Intercept (σ_ref):", self.spin_intercept)

        self.spin_slope = QtWidgets.QDoubleSpinBox()
        self.spin_slope.setRange(-5.0, 5.0)
        self.spin_slope.setDecimals(3)
        self.spin_slope.setSingleStep(0.01)
        self.spin_slope.setValue(-1.000)
        self.spin_slope.valueChanged.connect(self._update_all)
        form.addRow("Slope (-1.0 direct):", self.spin_slope)

        self.spin_fwhm = QtWidgets.QDoubleSpinBox()
        self.spin_fwhm.setRange(0.001, 10.0)
        self.spin_fwhm.setDecimals(3)
        self.spin_fwhm.setSingleStep(0.01)
        self.spin_fwhm.setValue(0.050)
        self.spin_fwhm.valueChanged.connect(self._replot_only)
        form.addRow("Peak Broadening FWHM (ppm):", self.spin_fwhm)

        left_layout.addWidget(grp_settings)

        # Table of Peaks
        tbl_box = QtWidgets.QGroupBox("Chemical Shifts Peak List (Click to View)")
        tbl_lay = QtWidgets.QVBoxLayout(tbl_box)
        self.tbl_peaks = QtWidgets.QTableWidget(0, 4)
        self.tbl_peaks.setHorizontalHeaderLabels(["Atom", "Sym", "σ_iso (ppm)", "δ (ppm)"])
        self.tbl_peaks.horizontalHeader().setStretchLastSection(True)
        self.tbl_peaks.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_peaks.itemSelectionChanged.connect(self._on_table_selection_changed)
        tbl_lay.addWidget(self.tbl_peaks)
        left_layout.addWidget(tbl_box)

        # Bottom buttons
        btn_csv = QtWidgets.QPushButton("💾 Export Peak Table (CSV)...")
        btn_csv.clicked.connect(self._on_export_csv)
        left_layout.addWidget(btn_csv)

        main_layout.addWidget(left_panel)

        # Right Panel: Matplotlib Figure
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(facecolor="#18181b", tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, right_panel)
        self.toolbar.setStyleSheet("background-color: #27272a; color: white;")

        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        # Connect canvas click event
        self.canvas.mpl_connect("button_press_event", self._on_plot_clicked)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        main_layout.addWidget(right_panel)

    def _style_axes(self):
        self.ax.set_facecolor("#09090b")
        self.ax.tick_params(colors="#94a3b8", labelsize=10)
        for spine in self.ax.spines.values():
            spine.set_color("#3f3f46")
        self.ax.grid(True, linestyle="--", alpha=0.25, color="#71717a")

    def _populate_standards(self):
        nuc = self.cbo_nucleus.currentText()
        stds = NMR_REFERENCE_STANDARDS.get(nuc, {})
        self.cbo_standard.blockSignals(True)
        self.cbo_standard.clear()
        for k in stds.keys():
            self.cbo_standard.addItem(k)
        self.cbo_standard.addItem("Custom Parameters...")
        self.cbo_standard.blockSignals(False)

        if stds:
            first_k = next(iter(stds))
            self.spin_intercept.blockSignals(True)
            self.spin_slope.blockSignals(True)
            self.spin_intercept.setValue(stds[first_k]["intercept"])
            self.spin_slope.setValue(stds[first_k]["slope"])
            self.spin_intercept.blockSignals(False)
            self.spin_slope.blockSignals(False)

    def _on_nucleus_changed(self):
        nuc = self.cbo_nucleus.currentText()
        if nuc == "1H":
            self.spin_fwhm.setValue(0.04)
        elif nuc == "13C":
            self.spin_fwhm.setValue(0.50)
        else:
            self.spin_fwhm.setValue(0.20)
        self._populate_standards()
        self._update_all()

    def _on_standard_changed(self):
        nuc = self.cbo_nucleus.currentText()
        std_name = self.cbo_standard.currentText()
        stds = NMR_REFERENCE_STANDARDS.get(nuc, {})
        if std_name in stds:
            self.spin_intercept.blockSignals(True)
            self.spin_slope.blockSignals(True)
            self.spin_intercept.setValue(stds[std_name]["intercept"])
            self.spin_slope.setValue(stds[std_name]["slope"])
            self.spin_intercept.blockSignals(False)
            self.spin_slope.blockSignals(False)
        self._update_all()

    def _on_load_log_clicked(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open QM NMR Output File",
            "",
            "QM Output (*.log *.out *.txt);;All Files (*)",
        )
        if path:
            from .qm_log_parser import parse_qm_log
            res = parse_qm_log(path)
            if not res.nmr_shieldings:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No NMR Shieldings Found",
                    f"No isotropic shieldings could be parsed from:\n{path}\n"
                    "Ensure this log was generated from an ORCA EPRNMR / NMR or Gaussian NMR calculation.",
                )
                return
            self.shieldings = res.nmr_shieldings
            self.log_path = path
            self._update_all()

    def _update_all(self):
        nuc = self.cbo_nucleus.currentText()
        intercept = self.spin_intercept.value()
        slope = self.spin_slope.value()

        # Compute referenced chemical shifts
        self.current_peaks = compute_chemical_shifts(
            self.shieldings,
            nucleus=nuc,
            custom_intercept=intercept,
            custom_slope=slope,
        )

        # Update table
        self.tbl_peaks.blockSignals(True)
        self.tbl_peaks.setRowCount(len(self.current_peaks))
        for row, p in enumerate(self.current_peaks):
            self.tbl_peaks.setItem(row, 0, QtWidgets.QTableWidgetItem(f"#{p.atom_index + 1}"))
            self.tbl_peaks.setItem(row, 1, QtWidgets.QTableWidgetItem(p.symbol))
            self.tbl_peaks.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{p.sigma_iso:.2f}"))
            self.tbl_peaks.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{p.chemical_shift_ppm:.2f}"))
        self.tbl_peaks.blockSignals(False)

        self._replot_only()

    def _replot_only(self):
        nuc = self.cbo_nucleus.currentText()
        fwhm = self.spin_fwhm.value()

        self.current_spectrum = synthesize_nmr_spectrum(
            self.current_peaks,
            nucleus=nuc,
            fwhm_ppm=fwhm,
        )

        self.ax.clear()
        self._style_axes()

        if not self.current_peaks:
            self.ax.text(
                0.5,
                0.5,
                f"No shieldings found for {nuc}.\nLoad a QM log containing NMR isotropic shieldings.",
                transform=self.ax.transAxes,
                color="#71717a",
                ha="center",
                va="center",
                fontsize=12,
            )
            self.canvas.draw()
            return

        spec = self.current_spectrum
        x = spec.ppm_grid
        y = spec.intensity_grid

        # Standard NMR plot: curve with blue fill
        self.ax.plot(x, y, color="#38bdf8", linewidth=1.6, label="Lorentzian Envelope")
        self.ax.fill_between(x, 0, y, color="#38bdf8", alpha=0.18)

        # Discrete peak markers
        peak_x = [p.chemical_shift_ppm for p in self.current_peaks]
        peak_y = [100.0] * len(peak_x)
        self.ax.vlines(peak_x, ymin=0, ymax=peak_y, colors="#f43f5e", linestyles="dashed", alpha=0.6, label="Peak Positions")

        # Label peaks with atom number
        for p in self.current_peaks:
            self.ax.annotate(
                f"{p.symbol}{p.atom_index + 1}\n{p.chemical_shift_ppm:.1f}",
                xy=(p.chemical_shift_ppm, 95.0),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#f8fafc",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e293b", edgecolor="#475569", alpha=0.85),
            )

        # Standard NMR spectrum axis: REVERSED (e.g. 12 -> 0 for 1H, 220 -> 0 for 13C)
        self.ax.set_xlim(np.max(x), np.min(x))
        self.ax.set_ylim(-2.0, 125.0)

        self.ax.set_xlabel(f"Chemical Shift δ (ppm) [{nuc}]", color="#f1f5f9", fontsize=11, fontweight="bold")
        self.ax.set_ylabel("Relative Intensity (%)", color="#f1f5f9", fontsize=11)
        self.ax.set_title(
            f"Synthetic 1D {nuc} NMR Spectrum ({len(self.current_peaks)} peaks, FWHM = {fwhm:.3f} ppm)",
            color="#38bdf8",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        self.canvas.draw()

    def _on_table_selection_changed(self):
        sel_rows = self.tbl_peaks.selectionModel().selectedRows()
        if not sel_rows:
            return
        row = sel_rows[0].row()
        if 0 <= row < len(self.current_peaks):
            peak = self.current_peaks[row]
            self.atom_selected.emit(peak.atom_index)

    def _on_plot_clicked(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        click_ppm = float(event.xdata)
        if not self.current_peaks:
            return

        # Find closest peak
        closest_p = min(self.current_peaks, key=lambda p: abs(p.chemical_shift_ppm - click_ppm))
        if abs(closest_p.chemical_shift_ppm - click_ppm) < (self.spin_fwhm.value() * 3.0 + 0.2):
            # Select table row
            idx = self.current_peaks.index(closest_p)
            self.tbl_peaks.selectRow(idx)
            self.atom_selected.emit(closest_p.atom_index)

    def _on_export_csv(self):
        if not self.current_peaks:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export NMR Peaks to CSV",
            f"nmr_peaks_{self.cbo_nucleus.currentText()}.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write("atom_index,symbol,sigma_iso_ppm,anisotropy_ppm,chemical_shift_ppm\n")
                for p in self.current_peaks:
                    f.write(f"{p.atom_index + 1},{p.symbol},{p.sigma_iso:.4f},{p.anisotropy:.4f},{p.chemical_shift_ppm:.4f}\n")
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Successfully exported peaks to:\n{path}")
