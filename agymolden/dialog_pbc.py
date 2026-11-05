"""
Periodic Boundary Conditions (PBC) & Supercell Generator Dialog for AgyMolden.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .elements import get_mass
from .pbc import UnitCell, create_box_from_molecule, replicate_supercell


class PBCDialog(QtWidgets.QDialog):
    """
    Dialog for configuring Periodic Boundary Conditions (unit cell parameters),
    viewing fractional coordinates, and generating Nx x Ny x Nz supercells.
    """

    cellChanged = QtCore.pyqtSignal(object)  # Emits UnitCell
    supercellApplied = QtCore.pyqtSignal(list, object, object)  # Emits (symbols, coords, new_unit_cell)

    def __init__(self, parent=None, symbols: Optional[List[str]] = None, coords: Optional[np.ndarray] = None, unit_cell: Optional[UnitCell] = None, viewer=None):
        super().__init__(parent)
        self.setWindowTitle("Periodic Boundary Conditions (PBC) & Supercell")
        self.resize(750, 600)

        self.symbols = list(symbols) if symbols else []
        self.coords = np.array(coords, copy=True) if coords is not None and len(coords) > 0 else np.empty((0, 3))
        self.viewer = viewer

        if unit_cell is not None:
            self.unit_cell = UnitCell(
                a=unit_cell.a, b=unit_cell.b, c=unit_cell.c,
                alpha=unit_cell.alpha, beta=unit_cell.beta, gamma=unit_cell.gamma,
                origin=unit_cell.origin
            )
        elif len(self.coords) > 0:
            self.unit_cell = create_box_from_molecule(self.coords, padding=5.0)
        else:
            self.unit_cell = UnitCell(a=15.0, b=15.0, c=15.0)

        self._build_ui()
        self._populate_fields()
        self._update_calculated_labels()
        self._populate_fractional_table()

    def _col(self, light_hex: str, dark_hex: str) -> str:
        dark = False
        if hasattr(self, "palette"):
            bg = self.palette().color(QtGui.QPalette.ColorRole.Window)
            dark = bg.lightnessF() < 0.4
        return dark_hex if dark else light_hex

    def _build_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Header Info Bar
        top_bar = QtWidgets.QHBoxLayout()
        self.lbl_summary = QtWidgets.QLabel(f"<b>Structure:</b> {len(self.symbols)} atoms loaded.")
        top_bar.addWidget(self.lbl_summary)
        top_bar.addStretch()

        self.chk_show_box = QtWidgets.QCheckBox("Show Unit Cell in 3D Viewport")
        self.chk_show_box.setChecked(True)
        if self.viewer is not None:
            self.chk_show_box.setChecked(getattr(self.viewer, "show_unit_cell", True))
        self.chk_show_box.toggled.connect(self._on_toggle_show_box)
        top_bar.addWidget(self.chk_show_box)

        main_layout.addLayout(top_bar)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_cell_tab(), "Unit Cell Parameters")
        self.tabs.addTab(self._build_supercell_tab(), "Supercell Replication")
        self.tabs.addTab(self._build_fractional_tab(), "Fractional Coordinates")
        main_layout.addWidget(self.tabs, stretch=1)

        # Bottom Button Box
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        main_layout.addLayout(btn_box)

    # ---------------------------------------------------------------------
    # Tab 1: Unit Cell Parameters
    # ---------------------------------------------------------------------
    def _build_cell_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(15)
        grid.setVerticalSpacing(8)

        # Cell lengths
        grid.addWidget(QtWidgets.QLabel("<b>a (Å):</b>"), 0, 0)
        self.spin_a = QtWidgets.QDoubleSpinBox()
        self.spin_a.setRange(0.5, 9999.0)
        self.spin_a.setDecimals(4)
        self.spin_a.valueChanged.connect(self._on_cell_param_changed)
        grid.addWidget(self.spin_a, 0, 1)

        grid.addWidget(QtWidgets.QLabel("<b>b (Å):</b>"), 0, 2)
        self.spin_b = QtWidgets.QDoubleSpinBox()
        self.spin_b.setRange(0.5, 9999.0)
        self.spin_b.setDecimals(4)
        self.spin_b.valueChanged.connect(self._on_cell_param_changed)
        grid.addWidget(self.spin_b, 0, 3)

        grid.addWidget(QtWidgets.QLabel("<b>c (Å):</b>"), 0, 4)
        self.spin_c = QtWidgets.QDoubleSpinBox()
        self.spin_c.setRange(0.5, 9999.0)
        self.spin_c.setDecimals(4)
        self.spin_c.valueChanged.connect(self._on_cell_param_changed)
        grid.addWidget(self.spin_c, 0, 5)

        # Cell angles
        grid.addWidget(QtWidgets.QLabel("<b>α (deg):</b>"), 1, 0)
        self.spin_alpha = QtWidgets.QDoubleSpinBox()
        self.spin_alpha.setRange(1.0, 179.0)
        self.spin_alpha.setDecimals(3)
        self.spin_alpha.valueChanged.connect(self._on_cell_param_changed)
        grid.addWidget(self.spin_alpha, 1, 1)

        grid.addWidget(QtWidgets.QLabel("<b>β (deg):</b>"), 1, 2)
        self.spin_beta = QtWidgets.QDoubleSpinBox()
        self.spin_beta.setRange(1.0, 179.0)
        self.spin_beta.setDecimals(3)
        self.spin_beta.valueChanged.connect(self._on_cell_param_changed)
        grid.addWidget(self.spin_beta, 1, 3)

        grid.addWidget(QtWidgets.QLabel("<b>γ (deg):</b>"), 1, 4)
        self.spin_gamma = QtWidgets.QDoubleSpinBox()
        self.spin_gamma.setRange(1.0, 179.0)
        self.spin_gamma.setDecimals(3)
        self.spin_gamma.valueChanged.connect(self._on_cell_param_changed)
        grid.addWidget(self.spin_gamma, 1, 5)

        layout.addLayout(grid)

        # Derived metrics box
        metrics_grp = QtWidgets.QGroupBox("Lattice Properties & Metrics")
        metrics_lay = QtWidgets.QVBoxLayout(metrics_grp)
        self.lbl_volume = QtWidgets.QLabel("Volume: —")
        self.lbl_density = QtWidgets.QLabel("Estimated Crystal Density: —")
        self.lbl_vectors = QtWidgets.QLabel("Lattice Vectors:\n—")
        self.lbl_vectors.setStyleSheet("font-family: monospace; font-size: 11px;")
        metrics_lay.addWidget(self.lbl_volume)
        metrics_lay.addWidget(self.lbl_density)
        metrics_lay.addWidget(self.lbl_vectors)
        layout.addWidget(metrics_grp)

        # Operations
        ops_grp = QtWidgets.QGroupBox("Geometry Alignment & Box Fitting")
        ops_lay = QtWidgets.QHBoxLayout(ops_grp)

        ops_lay.addWidget(QtWidgets.QLabel("Padding (Å):"))
        self.spin_pad = QtWidgets.QDoubleSpinBox()
        self.spin_pad.setRange(0.0, 50.0)
        self.spin_pad.setValue(5.0)
        self.spin_pad.setSingleStep(1.0)
        ops_lay.addWidget(self.spin_pad)

        btn_auto_box = QtWidgets.QPushButton("Fit Box to Atoms")
        btn_auto_box.setToolTip("Creates an orthorhombic unit cell enclosing the atoms with the given padding.")
        btn_auto_box.clicked.connect(self._on_auto_fit_box)
        ops_lay.addWidget(btn_auto_box)

        btn_wrap = QtWidgets.QPushButton("Wrap Atoms to [0, 1)")
        btn_wrap.setToolTip("Wraps all atomic coordinates into the periodic unit cell boundaries.")
        btn_wrap.clicked.connect(self._on_wrap_atoms)
        ops_lay.addWidget(btn_wrap)

        ops_lay.addStretch()
        layout.addWidget(ops_grp)

        layout.addStretch()
        return w

    # ---------------------------------------------------------------------
    # Tab 2: Supercell Replication
    # ---------------------------------------------------------------------
    def _build_supercell_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)

        rep_grp = QtWidgets.QGroupBox("Supercell Expansion Factors")
        rep_lay = QtWidgets.QGridLayout(rep_grp)
        rep_lay.setHorizontalSpacing(15)

        rep_lay.addWidget(QtWidgets.QLabel("<b>Na (along vector a):</b>"), 0, 0)
        self.spin_nx = QtWidgets.QSpinBox()
        self.spin_nx.setRange(1, 20)
        self.spin_nx.setValue(2)
        self.spin_nx.valueChanged.connect(self._update_supercell_preview)
        rep_lay.addWidget(self.spin_nx, 0, 1)

        rep_lay.addWidget(QtWidgets.QLabel("<b>Nb (along vector b):</b>"), 0, 2)
        self.spin_ny = QtWidgets.QSpinBox()
        self.spin_ny.setRange(1, 20)
        self.spin_ny.setValue(2)
        self.spin_ny.valueChanged.connect(self._update_supercell_preview)
        rep_lay.addWidget(self.spin_ny, 0, 3)

        rep_lay.addWidget(QtWidgets.QLabel("<b>Nc (along vector c):</b>"), 0, 4)
        self.spin_nz = QtWidgets.QSpinBox()
        self.spin_nz.setRange(1, 20)
        self.spin_nz.setValue(2)
        self.spin_nz.valueChanged.connect(self._update_supercell_preview)
        rep_lay.addWidget(self.spin_nz, 0, 5)

        self.chk_remove_dups = QtWidgets.QCheckBox("Remove duplicate boundary atoms (coincident at cell faces)")
        self.chk_remove_dups.setChecked(True)
        rep_lay.addWidget(self.chk_remove_dups, 1, 0, 1, 6)

        layout.addWidget(rep_grp)

        self.lbl_supercell_info = QtWidgets.QLabel("")
        self.lbl_supercell_info.setStyleSheet("font-size: 12px; margin: 8px 0;")
        layout.addWidget(self.lbl_supercell_info)

        action_bar = QtWidgets.QHBoxLayout()
        btn_apply = QtWidgets.QPushButton("Apply Supercell to Main Viewport")
        btn_apply.setStyleSheet("background-color: #0284c7; color: #fff; font-weight: bold; padding: 6px 16px;")
        btn_apply.clicked.connect(self._on_apply_supercell)
        action_bar.addWidget(btn_apply)

        btn_export = QtWidgets.QPushButton("💾 Export Supercell to XYZ...")
        btn_export.clicked.connect(self._on_export_supercell)
        action_bar.addWidget(btn_export)

        action_bar.addStretch()
        layout.addLayout(action_bar)

        self._update_supercell_preview()
        layout.addStretch()
        return w

    # ---------------------------------------------------------------------
    # Tab 3: Fractional Coordinates
    # ---------------------------------------------------------------------
    def _build_fractional_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)

        self.tbl_frac = QtWidgets.QTableWidget()
        self.tbl_frac.setColumnCount(8)
        self.tbl_frac.setHorizontalHeaderLabels([
            "Atom #", "Symbol", "Frac X", "Frac Y", "Frac Z", "Cart X (Å)", "Cart Y (Å)", "Cart Z (Å)"
        ])
        self.tbl_frac.horizontalHeader().setStretchLastSection(True)
        self.tbl_frac.setAlternatingRowColors(True)
        layout.addWidget(self.tbl_frac, stretch=1)
        return w

    # ---------------------------------------------------------------------
    # Logic & Events
    # ---------------------------------------------------------------------
    def _populate_fields(self):
        self.spin_a.blockSignals(True)
        self.spin_b.blockSignals(True)
        self.spin_c.blockSignals(True)
        self.spin_alpha.blockSignals(True)
        self.spin_beta.blockSignals(True)
        self.spin_gamma.blockSignals(True)

        self.spin_a.setValue(self.unit_cell.a)
        self.spin_b.setValue(self.unit_cell.b)
        self.spin_c.setValue(self.unit_cell.c)
        self.spin_alpha.setValue(self.unit_cell.alpha)
        self.spin_beta.setValue(self.unit_cell.beta)
        self.spin_gamma.setValue(self.unit_cell.gamma)

        self.spin_a.blockSignals(False)
        self.spin_b.blockSignals(False)
        self.spin_c.blockSignals(False)
        self.spin_alpha.blockSignals(False)
        self.spin_beta.blockSignals(False)
        self.spin_gamma.blockSignals(False)

    def _on_cell_param_changed(self):
        self.unit_cell.a = self.spin_a.value()
        self.unit_cell.b = self.spin_b.value()
        self.unit_cell.c = self.spin_c.value()
        self.unit_cell.alpha = self.spin_alpha.value()
        self.unit_cell.beta = self.spin_beta.value()
        self.unit_cell.gamma = self.spin_gamma.value()

        self._update_calculated_labels()
        self._populate_fractional_table()
        self._notify_viewer()

    def _update_calculated_labels(self):
        vol = self.unit_cell.volume
        self.lbl_volume.setText(f"<b>Volume:</b> {vol:,.3f} Å³")

        # Estimate density
        if len(self.symbols) > 0 and vol > 1e-4:
            total_mass = sum(get_mass(s) for s in self.symbols)
            # g/cm^3 = (amu / A^3) * 1.66053906660
            density = (total_mass / vol) * 1.660539
            self.lbl_density.setText(f"<b>Estimated Crystal Density:</b> {density:.3f} g/cm³ ({total_mass:.1f} g/mol in cell)")
        else:
            self.lbl_density.setText("<b>Estimated Crystal Density:</b> —")

        M = self.unit_cell.lattice_vectors
        self.lbl_vectors.setText(
            f"<b>Lattice Vectors:</b><br>"
            f"a = [{M[0, 0]:8.4f}, {M[0, 1]:8.4f}, {M[0, 2]:8.4f}] Å<br>"
            f"b = [{M[1, 0]:8.4f}, {M[1, 1]:8.4f}, {M[1, 2]:8.4f}] Å<br>"
            f"c = [{M[2, 0]:8.4f}, {M[2, 1]:8.4f}, {M[2, 2]:8.4f}] Å"
        )

    def _populate_fractional_table(self):
        if len(self.coords) == 0:
            self.tbl_frac.setRowCount(0)
            return

        frac = self.unit_cell.cartesian_to_fractional(self.coords)
        self.tbl_frac.setRowCount(len(self.symbols))
        for i, (sym, c, f) in enumerate(zip(self.symbols, self.coords, frac)):
            self.tbl_frac.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            self.tbl_frac.setItem(i, 1, QtWidgets.QTableWidgetItem(sym))
            self.tbl_frac.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{f[0]:.5f}"))
            self.tbl_frac.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{f[1]:.5f}"))
            self.tbl_frac.setItem(i, 4, QtWidgets.QTableWidgetItem(f"{f[2]:.5f}"))
            self.tbl_frac.setItem(i, 5, QtWidgets.QTableWidgetItem(f"{c[0]:.4f}"))
            self.tbl_frac.setItem(i, 6, QtWidgets.QTableWidgetItem(f"{c[1]:.4f}"))
            self.tbl_frac.setItem(i, 7, QtWidgets.QTableWidgetItem(f"{c[2]:.4f}"))

    def _on_toggle_show_box(self, checked: bool):
        if self.viewer is not None:
            self.viewer.show_unit_cell = checked
            self.viewer.unit_cell = self.unit_cell if checked else None
            self.viewer.update()

    def _notify_viewer(self):
        if self.viewer is not None and getattr(self.viewer, "show_unit_cell", True):
            self.viewer.unit_cell = self.unit_cell
            self.viewer.update()
        self.cellChanged.emit(self.unit_cell)

    def _on_auto_fit_box(self):
        if len(self.coords) == 0:
            return
        pad = self.spin_pad.value()
        self.unit_cell = create_box_from_molecule(self.coords, padding=pad)
        self._populate_fields()
        self._update_calculated_labels()
        self._populate_fractional_table()
        self._notify_viewer()

    def _on_wrap_atoms(self):
        if len(self.coords) == 0:
            return
        wrapped = self.unit_cell.wrap_coordinates(self.coords, to_range="[0,1)")
        self.coords = wrapped
        self._populate_fractional_table()
        if self.viewer is not None:
            self.viewer.coords = self.coords
            self.viewer.compute_bonds()
            self.viewer.update()

    def _update_supercell_preview(self):
        nx = self.spin_nx.value()
        ny = self.spin_ny.value()
        nz = self.spin_nz.value()
        total_mult = nx * ny * nz
        expected_atoms = len(self.symbols) * total_mult
        self.lbl_supercell_info.setText(
            f"<b>Replication:</b> {nx} × {ny} × {nz} = {total_mult}× multiplication.<br>"
            f"Resulting supercell will contain approx <b>{expected_atoms:,}</b> atoms "
            f"(unit cell dimensions: {self.unit_cell.a * nx:.2f} × {self.unit_cell.b * ny:.2f} × {self.unit_cell.c * nz:.2f} Å)."
        )

    def _on_apply_supercell(self):
        nx = self.spin_nx.value()
        ny = self.spin_ny.value()
        nz = self.spin_nz.value()
        rem_dup = self.chk_remove_dups.isChecked()

        rep_syms, rep_coords, new_cell = replicate_supercell(
            self.symbols, self.coords, self.unit_cell, nx=nx, ny=ny, nz=nz, remove_duplicates=rem_dup
        )

        self.symbols = rep_syms
        self.coords = rep_coords
        self.unit_cell = new_cell

        self._populate_fields()
        self._update_calculated_labels()
        self._populate_fractional_table()
        self._update_supercell_preview()

        self.lbl_summary.setText(f"<b>Structure:</b> {len(self.symbols)} atoms loaded.")
        if self.viewer is not None:
            self.viewer.symbols = list(self.symbols)
            self.viewer.coords = np.array(self.coords, copy=True)
            self.viewer.unit_cell = self.unit_cell
            self.viewer.compute_bonds()
            self.viewer.auto_center()
            self.viewer.update()

        self.supercellApplied.emit(self.symbols, self.coords, self.unit_cell)
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            QtWidgets.QMessageBox.information(
                self, "Supercell Applied", f"Successfully created {nx}×{ny}×{nz} supercell ({len(rep_syms)} atoms)."
            )

    def _on_export_supercell(self):
        nx = self.spin_nx.value()
        ny = self.spin_ny.value()
        nz = self.spin_nz.value()
        rem_dup = self.chk_remove_dups.isChecked()

        rep_syms, rep_coords, new_cell = replicate_supercell(
            self.symbols, self.coords, self.unit_cell, nx=nx, ny=ny, nz=nz, remove_duplicates=rem_dup
        )

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Supercell XYZ", f"supercell_{nx}x{ny}x{nz}.xyz", "XYZ Files (*.xyz);;All Files (*)"
        )
        if not out_path:
            return

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"{len(rep_syms)}\n")
            f.write(f"Supercell {nx}x{ny}x{nz} a={new_cell.a:.4f} b={new_cell.b:.4f} c={new_cell.c:.4f}\n")
            for s, c in zip(rep_syms, rep_coords):
                f.write(f"{s:<4} {c[0]:12.6f} {c[1]:12.6f} {c[2]:12.6f}\n")

        if not os.environ.get("PYTEST_CURRENT_TEST"):
            QtWidgets.QMessageBox.information(self, "Export Successful", f"Saved {len(rep_syms)} atoms to {out_path}.")
