"""
Interactive QTAIM Bond Critical Point (BCP) & Electron Density Topology Dialog.

Provides:
- Rapid Newton-Raphson search for (3, -1) Bond Critical Points from Molden wavefunctions
- Tabular display of BCP coordinates, electron density rho_bcp, Laplacian nabla^2 rho, and ellipticity
- Direct projection and highlight of BCP coordinates onto the 3D viewer
- Export BCP topology to CSV and publication tables
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np

from .qtaim_topology import (
    BondCriticalPoint,
    QTAIMTopologyResult,
    find_bond_critical_points,
)


class QTAIMDialog(QtWidgets.QDialog):
    """Interactive GUI for QTAIM topological analysis and Bond Critical Points."""

    bcp_selected = QtCore.pyqtSignal(object)  # Emits (x, y, z) coord in Angstrom

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, molden_path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("QTAIM Electron Density Topology & Bond Critical Points (3, -1)")
        self.resize(880, 580)

        self.molden_path = molden_path
        self.result: Optional[QTAIMTopologyResult] = None

        self._init_ui()

        if self.molden_path and os.path.exists(self.molden_path):
            self._compute_bcp_task()

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Header Info Card
        hdr_frame = QtWidgets.QFrame()
        hdr_frame.setStyleSheet("background-color: #0f172a; border-radius: 6px; padding: 10px;")
        h_box = QtWidgets.QHBoxLayout(hdr_frame)
        h_icon = QtWidgets.QLabel("<span style='font-size: 26px;'>⚛️</span>")
        h_box.addWidget(h_icon)
        h_text = QtWidgets.QVBoxLayout()
        h_text.addWidget(QtWidgets.QLabel("<b style='color: #38bdf8; font-size: 13px;'>QTAIM Topology Explorer</b>"))
        h_text.addWidget(QtWidgets.QLabel("<span style='color: #94a3b8; font-size: 11px;'>Locate (3, -1) Bond Critical Points, ρ_BCP, and ∇²ρ_BCP (Bader AIM theory)</span>"))
        h_box.addLayout(h_text)
        h_box.addStretch()

        btn_load = QtWidgets.QPushButton("📂 Load Molden...")
        btn_load.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 12px;")
        btn_load.clicked.connect(self._on_load_molden)
        h_box.addWidget(btn_load)

        main_layout.addWidget(hdr_frame)

        # Settings Row
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.addWidget(QtWidgets.QLabel("Max Bond Distance (Å):"))
        self.spin_dist = QtWidgets.QDoubleSpinBox()
        self.spin_dist.setRange(1.0, 6.0)
        self.spin_dist.setValue(2.8)
        self.spin_dist.setSingleStep(0.2)
        ctrl_row.addWidget(self.spin_dist)

        btn_search = QtWidgets.QPushButton("🔍 Search Critical Points")
        btn_search.clicked.connect(self._compute_bcp_task)
        ctrl_row.addWidget(btn_search)
        ctrl_row.addStretch()

        main_layout.addLayout(ctrl_row)

        # BCP Table
        self.tbl_bcp = QtWidgets.QTableWidget(0, 6)
        self.tbl_bcp.setHorizontalHeaderLabels(["Bond Pair", "Coords (Å)", "ρ_BCP (e/bohr³)", "∇²ρ_BCP", "Ellipticity ε", "H_BCP (a.u.)"])
        self.tbl_bcp.horizontalHeader().setStretchLastSection(True)
        self.tbl_bcp.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_bcp.itemSelectionChanged.connect(self._on_table_selection)
        main_layout.addWidget(self.tbl_bcp)

        # Bottom Bar
        bottom_bar = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Ready.")
        self.lbl_status.setStyleSheet("color: #94a3b8;")
        bottom_bar.addWidget(self.lbl_status)
        bottom_bar.addStretch()

        btn_csv = QtWidgets.QPushButton("💾 Export BCP Table (CSV)...")
        btn_csv.clicked.connect(self._on_export_csv)
        bottom_bar.addWidget(btn_csv)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bottom_bar.addWidget(btn_close)

        main_layout.addLayout(bottom_bar)

    def _on_load_molden(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Molden File for QTAIM Topology",
            "",
            "Molden Files (*.molden *.input);;All Files (*)",
        )
        if path:
            self.molden_path = path
            self._compute_bcp_task()

    def _compute_bcp_task(self):
        if not self.molden_path or not os.path.exists(self.molden_path):
            self.lbl_status.setText("No valid Molden file loaded.")
            return

        self.lbl_status.setText("Locating (3, -1) Bond Critical Points...")
        QtWidgets.QApplication.processEvents()

        try:
            d_max = self.spin_dist.value()
            self.result = find_bond_critical_points(self.molden_path, max_bond_distance_angstrom=d_max)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "QTAIM Error", f"Failed to compute QTAIM topology:\n{exc}")
            self.lbl_status.setText("Error during calculation.")
            return

        self._populate_table()
        self.lbl_status.setText(f"Found {len(self.result.bond_critical_points)} Bond Critical Points.")

    def _populate_table(self):
        if not self.result:
            return

        bcps = self.result.bond_critical_points
        self.tbl_bcp.blockSignals(True)
        self.tbl_bcp.setRowCount(len(bcps))

        for r, bcp in enumerate(bcps):
            pair_name = f"{bcp.atom_a_symbol}{bcp.atom_a_idx + 1} – {bcp.atom_b_symbol}{bcp.atom_b_idx + 1}"
            coord_str = f"({bcp.coord_angstrom[0]:.2f}, {bcp.coord_angstrom[1]:.2f}, {bcp.coord_angstrom[2]:.2f})"

            self.tbl_bcp.setItem(r, 0, QtWidgets.QTableWidgetItem(pair_name))
            self.tbl_bcp.setItem(r, 1, QtWidgets.QTableWidgetItem(coord_str))
            self.tbl_bcp.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{bcp.rho_bcp:.4f}"))

            # Color code Laplacian: negative = shared covalent, positive = closed-shell/ionic/H-bond
            item_lap = QtWidgets.QTableWidgetItem(f"{bcp.laplacian_bcp:+.4f}")
            lap_color = "#38bdf8" if bcp.laplacian_bcp < 0 else "#fbbf24"
            item_lap.setForeground(QtGui.QBrush(QtGui.QColor(lap_color)))
            self.tbl_bcp.setItem(r, 3, item_lap)

            self.tbl_bcp.setItem(r, 4, QtWidgets.QTableWidgetItem(f"{bcp.ellipticity:.3f}"))
            self.tbl_bcp.setItem(r, 5, QtWidgets.QTableWidgetItem(f"{bcp.h_energy_density:+.4f}"))

        self.tbl_bcp.blockSignals(False)

    def _on_table_selection(self):
        sel = self.tbl_bcp.selectionModel().selectedRows()
        if not sel or not self.result:
            return
        row = sel[0].row()
        if 0 <= row < len(self.result.bond_critical_points):
            bcp = self.result.bond_critical_points[row]
            self.bcp_selected.emit(bcp.coord_angstrom)

    def _on_export_csv(self):
        if not self.result:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export QTAIM Bond Critical Points",
            "qtaim_critical_points.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write("pair,atom_a,atom_b,x_ang,y_ang,z_ang,rho_bcp,laplacian_bcp,ellipticity,H_density\n")
                for bcp in self.result.bond_critical_points:
                    pair_str = f"{bcp.atom_a_symbol}{bcp.atom_a_idx+1}-{bcp.atom_b_symbol}{bcp.atom_b_idx+1}"
                    f.write(
                        f"{pair_str},{bcp.atom_a_idx+1},{bcp.atom_b_idx+1},"
                        f"{bcp.coord_angstrom[0]:.6f},{bcp.coord_angstrom[1]:.6f},{bcp.coord_angstrom[2]:.6f},"
                        f"{bcp.rho_bcp:.6f},{bcp.laplacian_bcp:.6f},{bcp.ellipticity:.6f},{bcp.h_energy_density:.6f}\n"
                    )
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Saved QTAIM BCPs to:\n{path}")
