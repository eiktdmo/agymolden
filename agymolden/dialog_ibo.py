"""
Interactive Intrinsic Bond Orbitals (IBO / IAO) and Lewis Structure Dialog.

Provides:
- One-click localization of delocalized molecular orbitals into intuitive chemical bonds and lone pairs
- Automatic classification of bonds (σ-bonds, π-bonds, lone pairs, core)
- Interactive table showing atomic population percentages for each bond
- Instant 3D orbital visualization or transmission into AgyMolden's primary 3D viewport
- Export localized orbital Molden files or tabular summary
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from .ibo_tools import IBOResult, LocalizedOrbital, compute_ibos


class IBODialog(QtWidgets.QDialog):
    """Dialog for Intrinsic Bond Orbital analysis and visualization."""

    # Signal emitted when a localized orbital is chosen for 3D display (orbital_index, coeff_vector)
    ibo_selected = QtCore.pyqtSignal(int, object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, molden_path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Intrinsic Bond Orbitals (IBO / IAO) & Lewis Localization")
        self.resize(880, 600)

        self.molden_path = molden_path
        self.ibo_result: Optional[IBOResult] = None

        self._init_ui()

        if self.molden_path and os.path.exists(self.molden_path):
            self._compute_ibos_task()

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Header Card
        hdr_frame = QtWidgets.QFrame()
        hdr_frame.setStyleSheet("background-color: #0f172a; border-radius: 6px; padding: 10px;")
        hdr_box = QtWidgets.QHBoxLayout(hdr_frame)
        hdr_icon = QtWidgets.QLabel("<span style='font-size: 26px;'>🧬</span>")
        hdr_box.addWidget(hdr_icon)
        hdr_text = QtWidgets.QVBoxLayout()
        hdr_text.addWidget(QtWidgets.QLabel("<b style='color: #38bdf8; font-size: 14px;'>Intrinsic Bond Orbitals (IBO / IAO)</b>"))
        hdr_text.addWidget(QtWidgets.QLabel("<span style='color: #94a3b8; font-size: 11px;'>Exact transformation of canonical MOs into localized Lewis bonds (σ, π) and lone pairs (Knizia 2013).</span>"))
        hdr_box.addLayout(hdr_text)
        hdr_box.addStretch()

        btn_load = QtWidgets.QPushButton("📂 Load Molden...")
        btn_load.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 12px;")
        btn_load.clicked.connect(self._on_load_molden)
        hdr_box.addWidget(btn_load)

        main_layout.addWidget(hdr_frame)

        # Settings bar
        bar_layout = QtWidgets.QHBoxLayout()
        bar_layout.addWidget(QtWidgets.QLabel("Localization Exponent (P):"))
        self.spin_exp = QtWidgets.QSpinBox()
        self.spin_exp.setRange(2, 6)
        self.spin_exp.setValue(4)
        bar_layout.addWidget(self.spin_exp)

        btn_recalc = QtWidgets.QPushButton("🔄 Recompute IBOs")
        btn_recalc.clicked.connect(self._compute_ibos_task)
        bar_layout.addWidget(btn_recalc)
        bar_layout.addStretch()

        main_layout.addLayout(bar_layout)

        # Orbitals Table
        self.tbl_ibos = QtWidgets.QTableWidget(0, 5)
        self.tbl_ibos.setHorizontalHeaderLabels(["IBO #", "Type", "Classification / Label", "Major Atom Centers (% Pop)", "Actions"])
        self.tbl_ibos.horizontalHeader().setStretchLastSection(True)
        self.tbl_ibos.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_ibos.itemSelectionChanged.connect(self._on_table_selection)
        main_layout.addWidget(self.tbl_ibos)

        # Status & Action Bar
        bottom_bar = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Ready.")
        self.lbl_status.setStyleSheet("color: #94a3b8;")
        bottom_bar.addWidget(self.lbl_status)
        bottom_bar.addStretch()

        btn_export = QtWidgets.QPushButton("💾 Export Summary Table...")
        btn_export.clicked.connect(self._on_export_table)
        bottom_bar.addWidget(btn_export)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bottom_bar.addWidget(btn_close)

        main_layout.addLayout(bottom_bar)

    def _on_load_molden(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Molden File for IBO Localization",
            "",
            "Molden Files (*.molden *.input);;All Files (*)",
        )
        if path:
            self.molden_path = path
            self._compute_ibos_task()

    def _compute_ibos_task(self):
        if not self.molden_path or not os.path.exists(self.molden_path):
            self.lbl_status.setText("No Molden file loaded.")
            return

        self.lbl_status.setText("Computing IAO/IBO localization via PySCF...")
        QtWidgets.QApplication.processEvents()

        try:
            exp = self.spin_exp.value()
            self.ibo_result = compute_ibos(self.molden_path, exponent=exp)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "IBO Localization Error", f"Failed to compute IBOs:\n{exc}")
            self.lbl_status.setText("Error during localization.")
            return

        self._populate_table()
        self.lbl_status.setText(f"Successfully localized {len(self.ibo_result.orbitals)} occupied orbitals.")

    def _populate_table(self):
        if not self.ibo_result:
            return

        res = self.ibo_result
        self.tbl_ibos.blockSignals(True)
        self.tbl_ibos.setRowCount(len(res.orbitals))

        type_colors = {
            "bond": "#38bdf8",
            "lone_pair": "#4ade80",
            "core": "#94a3b8",
            "delocalized": "#fbbf24",
        }

        for r, orb in enumerate(res.orbitals):
            self.tbl_ibos.setItem(r, 0, QtWidgets.QTableWidgetItem(f"IBO #{orb.index + 1}"))

            item_type = QtWidgets.QTableWidgetItem(orb.orbital_type.upper())
            color = type_colors.get(orb.orbital_type, "#f1f5f9")
            item_type.setForeground(QtGui.QBrush(QtGui.QColor(color)))
            self.tbl_ibos.setItem(r, 1, item_type)

            self.tbl_ibos.setItem(r, 2, QtWidgets.QTableWidgetItem(orb.label))

            atoms_summary = ", ".join(f"{sym}{idx+1}: {p*100:.1f}%" for sym, idx, p in orb.major_atoms)
            self.tbl_ibos.setItem(r, 3, QtWidgets.QTableWidgetItem(atoms_summary))

            btn_view = QtWidgets.QPushButton("👁️ View 3D")
            btn_view.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold;")
            btn_view.clicked.connect(lambda _, idx=orb.index, coeff=orb.coeff: self._emit_view(idx, coeff))
            self.tbl_ibos.setCellWidget(r, 4, btn_view)

        self.tbl_ibos.blockSignals(False)

    def _emit_view(self, index: int, coeff):
        self.ibo_selected.emit(index, coeff)
        self.lbl_status.setText(f"Viewing IBO #{index + 1} in 3D viewer.")

    def _on_table_selection(self):
        sel = self.tbl_ibos.selectionModel().selectedRows()
        if not sel or not self.ibo_result:
            return
        row = sel[0].row()
        if 0 <= row < len(self.ibo_result.orbitals):
            orb = self.ibo_result.orbitals[row]
            self.ibo_selected.emit(orb.index, orb.coeff)

    def _on_export_table(self):
        if not self.ibo_result:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export IBO Classification Table",
            "ibo_classification.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write("ibo_index,type,label,atom_distribution\n")
                for orb in self.ibo_result.orbitals:
                    dist = "; ".join(f"{sym}{idx+1}:{p*100:.1f}%" for sym, idx, p in orb.major_atoms)
                    f.write(f"{orb.index + 1},{orb.orbital_type},{orb.label},\"{dist}\"\n")
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Saved IBO summary to:\n{path}")
