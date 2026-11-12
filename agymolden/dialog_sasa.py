"""
Interactive Solvent-Accessible Surface Area (SASA) & Polar Area Dialog.

Provides:
- Real-time calculation of total SASA (Å²), Polar SASA, and Hydrophobic SASA
- Adjustable probe sphere radius (default 1.4 Å for water) and sphere point density
- Interactive per-atom accessibility table
- 1-Click projection of atomic exposure heatmaps directly onto the 3D molecular viewer
- Export data to CSV
"""

from __future__ import annotations

import os
from typing import List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np

from .sasa import SASAResult, calculate_sasa


class SASADialog(QtWidgets.QDialog):
    """Dialog for Solvent-Accessible Surface Area and Polar Area analysis."""

    # Signal to project exposure ratios [0, 1] onto 3D atoms in main viewer
    project_exposure_to_viewer = QtCore.pyqtSignal(object)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        symbols: Optional[List[str]] = None,
        coords: Optional[np.ndarray] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Solvent-Accessible Surface Area (SASA) & Polar Area")
        self.resize(840, 580)

        self.symbols: List[str] = list(symbols) if symbols else []
        self.coords: np.ndarray = np.array(coords, copy=True) if coords is not None else np.empty((0, 3))
        self.result: Optional[SASAResult] = None

        self._init_ui()
        self._recompute()

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Header Banner
        hdr_frame = QtWidgets.QFrame()
        hdr_frame.setStyleSheet("background-color: #0f172a; border-radius: 6px; padding: 10px;")
        h_box = QtWidgets.QHBoxLayout(hdr_frame)
        h_icon = QtWidgets.QLabel("<span style='font-size: 26px;'>💧</span>")
        h_box.addWidget(h_icon)
        h_text = QtWidgets.QVBoxLayout()
        h_text.addWidget(QtWidgets.QLabel("<b style='color: #38bdf8; font-size: 14px;'>Solvent-Accessible Surface Area (SASA)</b>"))
        h_text.addWidget(QtWidgets.QLabel("<span style='color: #94a3b8; font-size: 11px;'>Shrake-Rupley numerical rolling probe sphere (1.4 Å water probe).</span>"))
        h_box.addLayout(h_text)
        h_box.addStretch()
        main_layout.addWidget(hdr_frame)

        # Controls & KPI Summary Bar
        top_bar = QtWidgets.QHBoxLayout()

        top_bar.addWidget(QtWidgets.QLabel("<b>Probe Radius (Å):</b>"))
        self.spin_probe = QtWidgets.QDoubleSpinBox()
        self.spin_probe.setRange(0.5, 4.0)
        self.spin_probe.setSingleStep(0.1)
        self.spin_probe.setValue(1.40)
        self.spin_probe.valueChanged.connect(self._recompute)
        top_bar.addWidget(self.spin_probe)

        top_bar.addWidget(QtWidgets.QLabel("Points/Atom:"))
        self.spin_pts = QtWidgets.QSpinBox()
        self.spin_pts.setRange(30, 500)
        self.spin_pts.setSingleStep(20)
        self.spin_pts.setValue(120)
        self.spin_pts.valueChanged.connect(self._recompute)
        top_bar.addWidget(self.spin_pts)

        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # Metrics Card Frame
        metrics_frame = QtWidgets.QFrame()
        metrics_frame.setStyleSheet("background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        m_lay = QtWidgets.QHBoxLayout(metrics_frame)

        self.lbl_total_sasa = QtWidgets.QLabel("<b>Total SASA:</b> - Å²")
        self.lbl_total_sasa.setStyleSheet("color: #38bdf8; font-size: 13px;")
        self.lbl_polar_sasa = QtWidgets.QLabel("<b>Polar SASA (PSA):</b> - Å²")
        self.lbl_polar_sasa.setStyleSheet("color: #4ade80; font-size: 13px;")
        self.lbl_hydrophobic_sasa = QtWidgets.QLabel("<b>Hydrophobic SASA:</b> - Å²")
        self.lbl_hydrophobic_sasa.setStyleSheet("color: #fbbf24; font-size: 13px;")

        m_lay.addWidget(self.lbl_total_sasa)
        m_lay.addWidget(self.lbl_polar_sasa)
        m_lay.addWidget(self.lbl_hydrophobic_sasa)
        m_lay.addStretch()
        main_layout.addWidget(metrics_frame)

        # Atomic Exposure Table
        self.tbl_atoms = QtWidgets.QTableWidget(0, 5)
        self.tbl_atoms.setHorizontalHeaderLabels(["Atom", "Symbol", "R_vdw (Å)", "Area (Å²)", "Relative Exposure (%)"])
        self.tbl_atoms.horizontalHeader().setStretchLastSection(True)
        self.tbl_atoms.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        main_layout.addWidget(self.tbl_atoms)

        # Action Buttons
        btn_bar = QtWidgets.QHBoxLayout()

        btn_project = QtWidgets.QPushButton("🎨 Project Exposure Heatmap to 3D Viewport")
        btn_project.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 12px;")
        btn_project.clicked.connect(self._on_project_clicked)
        btn_bar.addWidget(btn_project)

        btn_bar.addStretch()

        btn_csv = QtWidgets.QPushButton("💾 Export CSV...")
        btn_csv.clicked.connect(self._on_export_csv)
        btn_bar.addWidget(btn_csv)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_bar.addWidget(btn_close)

        main_layout.addLayout(btn_bar)

    def _recompute(self):
        if len(self.symbols) == 0 or len(self.coords) == 0:
            return

        probe_r = self.spin_probe.value()
        n_pts = self.spin_pts.value()

        self.result = calculate_sasa(
            symbols=self.symbols,
            coords=self.coords,
            probe_radius=probe_r,
            n_points_per_atom=n_pts,
        )

        res = self.result

        # Update KPI labels
        self.lbl_total_sasa.setText(f"<b>Total SASA:</b> {res.total_sasa:.1f} Å²")
        self.lbl_polar_sasa.setText(f"<b>Polar SASA (PSA):</b> {res.polar_sasa:.1f} Å²")
        self.lbl_hydrophobic_sasa.setText(f"<b>Hydrophobic SASA:</b> {res.hydrophobic_sasa:.1f} Å²")

        # Update Table
        self.tbl_atoms.blockSignals(True)
        self.tbl_atoms.setRowCount(len(res.atom_results))

        for r, at in enumerate(res.atom_results):
            self.tbl_atoms.setItem(r, 0, QtWidgets.QTableWidgetItem(f"#{at.atom_index + 1}"))

            sym_item = QtWidgets.QTableWidgetItem(at.symbol)
            if at.is_polar:
                sym_item.setForeground(QtGui.QBrush(QtGui.QColor("#4ade80")))
            self.tbl_atoms.setItem(r, 1, sym_item)

            self.tbl_atoms.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{at.vdw_radius:.2f}"))
            self.tbl_atoms.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{at.accessible_area:.2f}"))

            exp_pct = at.relative_exposure * 100.0
            pct_item = QtWidgets.QTableWidgetItem(f"{exp_pct:.1f}%")
            if exp_pct > 40.0:
                pct_item.setForeground(QtGui.QBrush(QtGui.QColor("#38bdf8")))
            self.tbl_atoms.setItem(r, 4, pct_item)

        self.tbl_atoms.blockSignals(False)

    def _on_project_clicked(self):
        if not self.result:
            return
        self.project_exposure_to_viewer.emit(self.result.exposure_vector)
        QtWidgets.QMessageBox.information(
            self,
            "SASA Projected",
            "Projected solvent accessibility exposure values to 3D atoms!\n"
            "Higher values (solvent-exposed) are highlighted in the viewport.",
        )

    def _on_export_csv(self):
        if not self.result:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export SASA Results",
            "sasa_results.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            res = self.result
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Total SASA (A^2),{res.total_sasa:.4f}\n")
                f.write(f"# Polar SASA (A^2),{res.polar_sasa:.4f}\n")
                f.write(f"# Hydrophobic SASA (A^2),{res.hydrophobic_sasa:.4f}\n")
                f.write(f"# Probe Radius (A),{res.probe_radius:.2f}\n")
                f.write("atom_index,symbol,r_vdw_ang,area_ang2,relative_exposure_pct,is_polar\n")
                for at in res.atom_results:
                    f.write(f"{at.atom_index + 1},{at.symbol},{at.vdw_radius:.4f},{at.accessible_area:.4f},{at.relative_exposure * 100.0:.2f},{at.is_polar}\n")
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Saved SASA table to:\n{path}")
