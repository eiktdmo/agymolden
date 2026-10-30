"""
Multiwfn Electronic Structure & Wavefunction Analysis Dialog.

Provides four tabs of advanced quantum chemical analyses:
1. Mayer & Wiberg Lowdin Bond Orders & Atomic Valences
2. Conceptual DFT (CDFT): Condensed Fukui Functions (f+, f-, f0) & Dual Descriptor (Δf)
3. QTAIM / AIM Topology Analysis: Critical Points ((3,-3) Nuclear, (3,-1) Bond CPs)
4. Atomic Partial Charges: CM5 (Charge Model 5) & Hirshfeld charges
"""

import os
import logging
from typing import Optional, List, Dict, Any
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .multiwfn_analysis import (
    detect_multiwfn_executable,
    save_multiwfn_custom_path,
    friendly_error_text,
    compute_mayer_and_wiberg_orders,
    compute_condensed_fukui,
    compute_qtaim_topology,
    compute_hirshfeld_and_cm5_charges,
    BondOrderResult,
    FukuiResult,
    QTAIMResult,
    PartialChargesResult,
)
from .workers import run_in_background

log = logging.getLogger(__name__)


class MultiwfnAnalysisDialog(QtWidgets.QDialog):
    """Interactive multi-tab dialog for Multiwfn & Wavefunction analyses."""

    def __init__(self, parent=None, molden_path: Optional[str] = None, viewer=None):
        super().__init__(parent)
        self.setWindowTitle("Multiwfn Wavefunction & Electronic Structure Analysis")
        self.resize(780, 560)
        self.molden_path = molden_path
        self.viewer = viewer

        self._mayer_res: Optional[BondOrderResult] = None
        self._wiberg_res: Optional[BondOrderResult] = None
        self._fukui_res: Optional[FukuiResult] = None
        self._qtaim_res: Optional[QTAIMResult] = None
        self._charges_res: Optional[PartialChargesResult] = None

        self._multiwfn_only_buttons: List[QtWidgets.QPushButton] = []
        self._setup_ui()
        self._check_multiwfn_status()

    def _is_dark_theme(self) -> bool:
        return self.palette().color(QtGui.QPalette.ColorRole.Window).lightness() < 128

    def _col(self, light: str, dark: str) -> str:
        """Pick a text colour with good contrast for the current (light/dark) palette."""
        return dark if self._is_dark_theme() else light

    def _setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Header bar: Active Molden file and Multiwfn engine status
        header = QtWidgets.QFrame()
        header.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        h_layout = QtWidgets.QHBoxLayout(header)
        h_layout.setContentsMargins(8, 6, 8, 6)

        self.lbl_file = QtWidgets.QLabel(
            f"<b>Wavefunction:</b> {os.path.basename(self.molden_path) if self.molden_path else 'None loaded'}"
        )
        h_layout.addWidget(self.lbl_file, stretch=1)

        self.lbl_engine = QtWidgets.QLabel("Engine: Detecting...")
        self.lbl_engine.setStyleSheet(f"font-weight: bold; color: {self._col('#0369a1', '#7dd3fc')};")
        h_layout.addWidget(self.lbl_engine)

        btn_mwfn = QtWidgets.QPushButton("Set Multiwfn path...")
        btn_mwfn.setToolTip("Select the Multiwfn executable (needed for QTAIM and CM5/Hirshfeld). Free download: http://sobereva.com/multiwfn")
        btn_mwfn.clicked.connect(self._on_set_multiwfn_path)
        h_layout.addWidget(btn_mwfn)

        btn_browse = QtWidgets.QPushButton("Load .molden...")
        btn_browse.clicked.connect(self._on_browse_molden)
        h_layout.addWidget(btn_browse)

        main_layout.addWidget(header)

        # Tab widget for the 4 core tools
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._create_bond_order_tab(), "1. Mayer & Wiberg Orders")
        self.tabs.addTab(self._create_fukui_tab(), "2. Conceptual DFT / Fukui")
        self.tabs.addTab(self._create_qtaim_tab(), "3. QTAIM Topology (BCP)")
        self.tabs.addTab(self._create_charges_tab(), "4. CM5 & Hirshfeld Charges")
        main_layout.addWidget(self.tabs, stretch=1)

        # Bottom buttons
        btn_box = QtWidgets.QHBoxLayout()
        self.btn_run_all = QtWidgets.QPushButton("🚀 Run All Multiwfn Analyses")
        self.btn_run_all.setStyleSheet("background-color: #2563eb; color: #fff; font-weight: bold; padding: 6px 14px;")
        self.btn_run_all.clicked.connect(self._on_run_all)
        btn_box.addWidget(self.btn_run_all)

        self.btn_export_tsv = QtWidgets.QPushButton("📋 Copy Current Tab Table")
        self.btn_export_tsv.setToolTip("Copy tab results to clipboard (tab-separated for Excel/Word/Sheets)")
        self.btn_export_tsv.clicked.connect(self._on_copy_active_table)
        btn_box.addWidget(self.btn_export_tsv)

        self.btn_export_csv = QtWidgets.QPushButton("💾 Save CSV...")
        self.btn_export_csv.setToolTip("Export the current analysis table to a CSV file")
        self.btn_export_csv.clicked.connect(self._on_save_active_csv)
        btn_box.addWidget(self.btn_export_csv)

        btn_box.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)

        main_layout.addLayout(btn_box)

    def _check_multiwfn_status(self):
        bin_path = detect_multiwfn_executable()
        if bin_path:
            self.lbl_engine.setText(f"Engine: Multiwfn ({os.path.basename(bin_path)})")
            self.lbl_engine.setToolTip(f"Executable path:\n{bin_path}")
            self.lbl_engine.setStyleSheet(f"font-weight: bold; color: {self._col('#15803d', '#86efac')};")
        else:
            self.lbl_engine.setText("Multiwfn not found (QTAIM & CM5 unavailable)")
            self.lbl_engine.setToolTip(
                "Bond orders and Fukui use the built-in PySCF fallback.\n"
                "QTAIM and CM5/Hirshfeld need Multiwfn: http://sobereva.com/multiwfn\n"
                "Use 'Set Multiwfn path...' after installing it."
            )
            self.lbl_engine.setStyleSheet(f"font-weight: bold; color: {self._col('#b45309', '#fcd34d')};")
        # Multiwfn-only features are disabled with an explanation when it is missing
        for btn in getattr(self, "_multiwfn_only_buttons", []):
            btn.setEnabled(bool(bin_path))
            btn.setToolTip("" if bin_path else "Requires Multiwfn - click 'Set Multiwfn path...' at the top.")

    def _on_set_multiwfn_path(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Multiwfn executable", os.path.expanduser("~"))
        if not path:
            return
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            QtWidgets.QMessageBox.warning(self, "Not executable", f"'{path}' is not an executable file.")
            return
        save_multiwfn_custom_path(path)
        self._check_multiwfn_status()

    def _on_browse_molden(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Molden / Wavefunction File", "", "Molden Files (*.molden *.molden.input *.molf);;All Files (*)"
        )
        if path:
            self.molden_path = path
            self.lbl_file.setText(f"<b>Wavefunction:</b> {os.path.basename(path)}")

    # ---------------------------------------------------------------------
    # Tab 1: Mayer & Wiberg Bond Orders
    # ---------------------------------------------------------------------
    def _create_bond_order_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)

        ctrl_bar = QtWidgets.QHBoxLayout()
        btn_calc = QtWidgets.QPushButton("Calculate Bond Orders & Valences")
        btn_calc.setStyleSheet("background-color: #0f766e; color: #fff; font-weight: bold; padding: 5px 12px;")
        btn_calc.clicked.connect(self._on_calc_bond_orders)
        ctrl_bar.addWidget(btn_calc)

        ctrl_bar.addWidget(QtWidgets.QLabel("Min Bond Order:"))
        self.spin_bo_cutoff = QtWidgets.QDoubleSpinBox()
        self.spin_bo_cutoff.setRange(0.001, 1.0)
        self.spin_bo_cutoff.setValue(0.05)
        self.spin_bo_cutoff.setSingleStep(0.01)
        ctrl_bar.addWidget(self.spin_bo_cutoff)

        ctrl_bar.addStretch()

        self.btn_apply_bo = QtWidgets.QPushButton("Apply to 3D Viewport Bonds")
        self.btn_apply_bo.setToolTip("Sets bond multiplicity and visual display in the 3D viewer based on Mayer orders")
        self.btn_apply_bo.clicked.connect(self._on_apply_bond_orders)
        self.btn_apply_bo.setEnabled(False)
        ctrl_bar.addWidget(self.btn_apply_bo)

        layout.addLayout(ctrl_bar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Bond Orders Table
        bo_grp = QtWidgets.QGroupBox("Bond Orders (A – B)")
        bo_lay = QtWidgets.QVBoxLayout(bo_grp)
        self.tbl_bonds = QtWidgets.QTableWidget()
        self.tbl_bonds.setColumnCount(5)
        self.tbl_bonds.setHorizontalHeaderLabels(["Atom 1", "Atom 2", "Mayer Order", "Wiberg (Löwdin)", "Estimated Type"])
        self.tbl_bonds.horizontalHeader().setStretchLastSection(True)
        self.tbl_bonds.setAlternatingRowColors(True)
        bo_lay.addWidget(self.tbl_bonds)
        splitter.addWidget(bo_grp)

        # Atomic Valences Table
        val_grp = QtWidgets.QGroupBox("Total Atomic Valences")
        val_lay = QtWidgets.QVBoxLayout(val_grp)
        self.tbl_valences = QtWidgets.QTableWidget()
        self.tbl_valences.setColumnCount(4)
        self.tbl_valences.setHorizontalHeaderLabels(["Atom", "Symbol", "Mayer Valence", "Wiberg Valence"])
        self.tbl_valences.horizontalHeader().setStretchLastSection(True)
        self.tbl_valences.setAlternatingRowColors(True)
        val_lay.addWidget(self.tbl_valences)
        splitter.addWidget(val_grp)

        splitter.setSizes([450, 280])
        layout.addWidget(splitter, stretch=1)
        return w

    def _on_calc_bond_orders(self):
        if not self.molden_path or not os.path.isfile(self.molden_path):
            QtWidgets.QMessageBox.warning(self, "No Wavefunction", "Please select or load a valid .molden file first.")
            return

        cutoff = self.spin_bo_cutoff.value()

        def _worker():
            return compute_mayer_and_wiberg_orders(self.molden_path, min_threshold=cutoff)

        def _on_done(result):
            mayer, wiberg = result
            self._mayer_res = mayer
            self._wiberg_res = wiberg

            # Populate tables
            self.tbl_bonds.setRowCount(0)
            wiberg_map = {(b[0], b[1]): b[4] for b in wiberg.bonds}
            w_rev_map = {(b[1], b[0]): b[4] for b in wiberg.bonds}

            # Merge all pairs
            all_pairs = list(mayer.bonds)
            mayer_pairs = {(b[0], b[1]) for b in mayer.bonds} | {(b[1], b[0]) for b in mayer.bonds}
            for wb in wiberg.bonds:
                if (wb[0], wb[1]) not in mayer_pairs:
                    all_pairs.append((wb[0], wb[1], wb[2], wb[3], 0.0))

            self.tbl_bonds.setRowCount(len(all_pairs))
            for row, b in enumerate(all_pairs):
                idx1, idx2, s1, s2, m_val = b
                w_val = wiberg_map.get((idx1, idx2), w_rev_map.get((idx1, idx2), 0.0))

                btype = "Single (1)"
                ref_order = m_val if m_val > 0 else w_val
                if ref_order >= 2.5:
                    btype = "Triple (≡)"
                elif ref_order >= 1.5:
                    btype = "Double (=)"
                elif ref_order < 0.5:
                    btype = "Weak / Non-covalent"

                self.tbl_bonds.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{idx1} ({s1})"))
                self.tbl_bonds.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{idx2} ({s2})"))
                self.tbl_bonds.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{m_val:.4f}" if m_val > 0 else "—"))
                self.tbl_bonds.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{w_val:.4f}" if w_val > 0 else "—"))
                self.tbl_bonds.setItem(row, 4, QtWidgets.QTableWidgetItem(btype))

            # Valences
            all_atoms = sorted(set(mayer.valences.keys()) | set(wiberg.valences.keys()))
            self.tbl_valences.setRowCount(len(all_atoms))
            for row, a_idx in enumerate(all_atoms):
                m_v = mayer.valences.get(a_idx, 0.0)
                w_v = wiberg.valences.get(a_idx, 0.0)
                # find symbol from bonds or default
                sym = "X"
                for b in mayer.bonds:
                    if b[0] == a_idx:
                        sym = b[2]
                        break
                    elif b[1] == a_idx:
                        sym = b[3]
                        break
                self.tbl_valences.setItem(row, 0, QtWidgets.QTableWidgetItem(str(a_idx)))
                self.tbl_valences.setItem(row, 1, QtWidgets.QTableWidgetItem(sym))
                self.tbl_valences.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{m_v:.4f}"))
                self.tbl_valences.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{w_v:.4f}"))

            self.btn_apply_bo.setEnabled(True)

        def _on_err(e):
            QtWidgets.QMessageBox.critical(self, "Bond Order Calculation Failed", friendly_error_text(e))

        run_in_background(self, "Bond Orders", "Computing Mayer and Wiberg bond orders...", _worker, _on_done, _on_err)

    def _on_apply_bond_orders(self):
        if not self.viewer or not self._mayer_res:
            return
        # Set viewer.bond_orders mapping (0-based)
        bo_dict = {}
        for b in self._mayer_res.bonds:
            i = b[0] - 1
            j = b[1] - 1
            order = b[4]
            int_order = 1
            if order >= 2.5:
                int_order = 3
            elif order >= 1.5:
                int_order = 2
            bo_dict[(min(i, j), max(i, j))] = int_order

        self.viewer.bond_orders = bo_dict
        self.viewer.update()
        QtWidgets.QMessageBox.information(
            self, "Bond Orders Applied", f"Updated {len(bo_dict)} bonds in the 3D viewport based on Mayer bond orders!"
        )

    # ---------------------------------------------------------------------
    # Tab 2: Conceptual DFT / Fukui & Dual Descriptors
    # ---------------------------------------------------------------------
    def _create_fukui_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)

        ctrl_bar = QtWidgets.QHBoxLayout()
        btn_calc = QtWidgets.QPushButton("Calculate Condensed Fukui & Dual Descriptors")
        btn_calc.setStyleSheet("background-color: #7c3aed; color: #fff; font-weight: bold; padding: 5px 12px;")
        btn_calc.clicked.connect(self._on_calc_fukui)
        ctrl_bar.addWidget(btn_calc)

        ctrl_bar.addWidget(QtWidgets.QLabel("Population Scheme:"))
        self.combo_fukui_pop = QtWidgets.QComboBox()
        self.combo_fukui_pop.addItems(["Löwdin (Basis-set Stable)", "Mulliken"])
        ctrl_bar.addWidget(self.combo_fukui_pop)

        ctrl_bar.addStretch()

        layout.addLayout(ctrl_bar)

        desc_lbl = QtWidgets.QLabel(
            "<b>Conceptual DFT Interpretation:</b><br>"
            "• <b>f⁻ (HOMO donor):</b> Governs electrophilic attack (high f⁻ = easiest electron loss).<br>"
            "• <b>f⁺ (LUMO acceptor):</b> Governs nucleophilic attack (high f⁺ = easiest electron acceptance).<br>"
            "• <b>Dual Descriptor Δf = f⁺ − f⁻:</b> Δf > 0 indicates electrophilic attack site; Δf < 0 indicates nucleophilic attack site."
        )
        desc_lbl.setStyleSheet("font-size: 11px; margin: 4px 0;")
        layout.addWidget(desc_lbl)

        self.tbl_fukui = QtWidgets.QTableWidget()
        self.tbl_fukui.setColumnCount(7)
        self.tbl_fukui.setHorizontalHeaderLabels([
            "Atom", "Symbol", "f⁻ (Electrophile)", "f⁺ (Nucleophile)", "f⁰ (Radical)", "Dual Δf (f⁺−f⁻)", "Site Prediction"
        ])
        self.tbl_fukui.horizontalHeader().setStretchLastSection(True)
        self.tbl_fukui.setAlternatingRowColors(True)
        layout.addWidget(self.tbl_fukui, stretch=1)
        return w

    def _on_calc_fukui(self):
        if not self.molden_path or not os.path.isfile(self.molden_path):
            QtWidgets.QMessageBox.warning(self, "No Wavefunction", "Please select or load a valid .molden file first.")
            return

        scheme = "Lowdin" if "lowdin" in self.combo_fukui_pop.currentText().lower() else "Mulliken"

        def _worker():
            return compute_condensed_fukui(self.molden_path, population_scheme=scheme)

        def _on_done(res: FukuiResult):
            self._fukui_res = res
            self.tbl_fukui.setRowCount(len(res.atoms))
            for row, a in enumerate(res.atoms):
                self.tbl_fukui.setItem(row, 0, QtWidgets.QTableWidgetItem(str(a["index"])))
                self.tbl_fukui.setItem(row, 1, QtWidgets.QTableWidgetItem(str(a["symbol"])))
                self.tbl_fukui.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{a['f_minus']:.4f}"))
                self.tbl_fukui.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{a['f_plus']:.4f}"))
                self.tbl_fukui.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{a['f_zero']:.4f}"))

                # Color dual descriptor item
                item_dual = QtWidgets.QTableWidgetItem(f"{a['dual']:+.4f}")
                if a["dual"] > 0.05:
                    item_dual.setForeground(QtGui.QBrush(QtGui.QColor(self._col("#1d4ed8", "#93c5fd"))))  # Blue
                elif a["dual"] < -0.05:
                    item_dual.setForeground(QtGui.QBrush(QtGui.QColor(self._col("#b91c1c", "#fca5a5"))))  # Red
                self.tbl_fukui.setItem(row, 5, item_dual)

                self.tbl_fukui.setItem(row, 6, QtWidgets.QTableWidgetItem(a["site_pred"]))

        def _on_err(e):
            QtWidgets.QMessageBox.critical(self, "CDFT Calculation Failed", friendly_error_text(e))

        run_in_background(self, "Conceptual DFT", "Computing condensed Fukui functions...", _worker, _on_done, _on_err)

    # ---------------------------------------------------------------------
    # Tab 3: QTAIM Topology & Critical Points
    # ---------------------------------------------------------------------
    def _create_qtaim_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)

        ctrl_bar = QtWidgets.QHBoxLayout()
        btn_calc = QtWidgets.QPushButton("Search QTAIM Critical Points (AIM)")
        btn_calc.setStyleSheet("background-color: #0369a1; color: #fff; font-weight: bold; padding: 5px 12px;")
        btn_calc.clicked.connect(self._on_calc_qtaim)
        self._multiwfn_only_buttons.append(btn_calc)
        ctrl_bar.addWidget(btn_calc)

        self.btn_show_cps = QtWidgets.QPushButton("Display BCP Spheres in 3D Viewport")
        self.btn_show_cps.setToolTip("Renders (3,-1) Bond Critical Points as cyan spheres in the 3D molecular viewer")
        self.btn_show_cps.clicked.connect(self._on_toggle_qtaim_cps)
        self.btn_show_cps.setEnabled(False)
        ctrl_bar.addWidget(self.btn_show_cps)

        ctrl_bar.addStretch()

        self.lbl_qtaim_summary = QtWidgets.QLabel("No topology analysis performed yet.")
        ctrl_bar.addWidget(self.lbl_qtaim_summary)

        layout.addLayout(ctrl_bar)

        self.tbl_qtaim = QtWidgets.QTableWidget()
        self.tbl_qtaim.setColumnCount(6)
        self.tbl_qtaim.setHorizontalHeaderLabels([
            "CP #", "Type", "Classification", "X (Å)", "Y (Å)", "Z (Å)"
        ])
        self.tbl_qtaim.horizontalHeader().setStretchLastSection(True)
        self.tbl_qtaim.setAlternatingRowColors(True)
        layout.addWidget(self.tbl_qtaim, stretch=1)
        return w

    def _on_calc_qtaim(self):
        if not self.molden_path or not os.path.isfile(self.molden_path):
            QtWidgets.QMessageBox.warning(self, "No Wavefunction", "Please select or load a valid .molden file first.")
            return

        def _worker():
            return compute_qtaim_topology(self.molden_path)

        def _on_done(res: QTAIMResult):
            self._qtaim_res = res
            self.lbl_qtaim_summary.setText(
                f"Found {len(res.critical_points)} CPs: {res.num_bcp} Bond (BCP), {res.num_nuclear} Nuclear, "
                f"{res.num_rcp} Ring (RCP), {res.num_ccp} Cage (CCP)"
            )
            self.tbl_qtaim.setRowCount(len(res.critical_points))
            for row, cp in enumerate(res.critical_points):
                self.tbl_qtaim.setItem(row, 0, QtWidgets.QTableWidgetItem(str(cp.index)))
                self.tbl_qtaim.setItem(row, 1, QtWidgets.QTableWidgetItem(cp.cp_type))
                self.tbl_qtaim.setItem(row, 2, QtWidgets.QTableWidgetItem(cp.type_name))
                self.tbl_qtaim.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{cp.coords_angstrom[0]:.6f}"))
                self.tbl_qtaim.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{cp.coords_angstrom[1]:.6f}"))
                self.tbl_qtaim.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{cp.coords_angstrom[2]:.6f}"))

            self.btn_show_cps.setEnabled(True)

        def _on_err(e):
            QtWidgets.QMessageBox.critical(self, "QTAIM Analysis Failed", friendly_error_text(e))

        run_in_background(self, "QTAIM Topology", "Searching critical points via Multiwfn...", _worker, _on_done, _on_err)

    def _on_toggle_qtaim_cps(self):
        if not self.viewer or not self._qtaim_res:
            return
        current_state = getattr(self.viewer, "show_qtaim_cps", False)
        new_state = not current_state
        self.viewer.qtaim_critical_points = self._qtaim_res.critical_points
        self.viewer.show_qtaim_cps = new_state
        self.viewer.update()

        self.btn_show_cps.setText("Hide BCP Spheres in 3D Viewport" if new_state else "Display BCP Spheres in 3D Viewport")
        QtWidgets.QMessageBox.information(
            self,
            "QTAIM 3D Critical Points",
            f"{'Displaying' if new_state else 'Hidden'} {len(self._qtaim_res.critical_points)} critical points "
            "(cyan: BCP, amber: RCP, purple: CCP) in the 3D viewer.",
        )

    # ---------------------------------------------------------------------
    # Tab 4: CM5 & Hirshfeld Partial Charges
    # ---------------------------------------------------------------------
    def _create_charges_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)

        ctrl_bar = QtWidgets.QHBoxLayout()
        btn_calc = QtWidgets.QPushButton("Calculate CM5 & Hirshfeld Charges")
        btn_calc.setStyleSheet("background-color: #047857; color: #fff; font-weight: bold; padding: 5px 12px;")
        btn_calc.clicked.connect(self._on_calc_charges)
        self._multiwfn_only_buttons.append(btn_calc)
        ctrl_bar.addWidget(btn_calc)

        self.btn_apply_cm5 = QtWidgets.QPushButton("Apply CM5 Charges to 3D Viewer")
        self.btn_apply_cm5.setToolTip("Displays CM5 charges directly on atom labels in the 3D viewport")
        self.btn_apply_cm5.clicked.connect(self._on_apply_cm5)
        self.btn_apply_cm5.setEnabled(False)
        ctrl_bar.addWidget(self.btn_apply_cm5)

        ctrl_bar.addStretch()

        self.lbl_dipole = QtWidgets.QLabel("")
        ctrl_bar.addWidget(self.lbl_dipole)

        layout.addLayout(ctrl_bar)

        desc_lbl = QtWidgets.QLabel(
            "<b>Charge Model 5 (CM5)</b> (Marenich, Jerome, Cramer, Truhlar, <i>J. Chem. Theory Comput.</i> 2012):<br>"
            "CM5 yields physically accurate, basis-set insensitive partial atomic charges matching experimental dipole moments."
        )
        desc_lbl.setStyleSheet("font-size: 11px; margin: 4px 0;")
        layout.addWidget(desc_lbl)

        self.tbl_charges = QtWidgets.QTableWidget()
        self.tbl_charges.setColumnCount(4)
        self.tbl_charges.setHorizontalHeaderLabels([
            "Atom", "Symbol", "CM5 Charge (e)", "Hirshfeld Charge (e)"
        ])
        self.tbl_charges.horizontalHeader().setStretchLastSection(True)
        self.tbl_charges.setAlternatingRowColors(True)
        layout.addWidget(self.tbl_charges, stretch=1)
        return w

    def _on_calc_charges(self):
        if not self.molden_path or not os.path.isfile(self.molden_path):
            QtWidgets.QMessageBox.warning(self, "No Wavefunction", "Please select or load a valid .molden file first.")
            return

        def _worker():
            return compute_hirshfeld_and_cm5_charges(self.molden_path)

        def _on_done(res: PartialChargesResult):
            self._charges_res = res
            self.tbl_charges.setRowCount(len(res.atoms))
            for row, a in enumerate(res.atoms):
                self.tbl_charges.setItem(row, 0, QtWidgets.QTableWidgetItem(str(a["index"])))
                self.tbl_charges.setItem(row, 1, QtWidgets.QTableWidgetItem(str(a["symbol"])))

                # CM5
                item_cm5 = QtWidgets.QTableWidgetItem(f"{a['cm5']:+.4f}")
                if a["cm5"] < 0:
                    item_cm5.setForeground(QtGui.QBrush(QtGui.QColor(self._col("#b91c1c", "#fca5a5"))))  # Red for negative
                else:
                    item_cm5.setForeground(QtGui.QBrush(QtGui.QColor(self._col("#1d4ed8", "#93c5fd"))))  # Blue for positive
                self.tbl_charges.setItem(row, 2, item_cm5)

                # Hirshfeld
                item_h = QtWidgets.QTableWidgetItem(f"{a['hirshfeld']:+.4f}")
                self.tbl_charges.setItem(row, 3, item_h)

            if res.dipole_cm5:
                # 1 a.u. = 2.541746 Debye
                mu_au = float(np.linalg.norm(res.dipole_cm5))
                mu_debye = mu_au * 2.541746
                self.lbl_dipole.setText(f"CM5 Dipole: <b>{mu_debye:.3f} D</b> ({mu_au:.3f} a.u.)")

            self.btn_apply_cm5.setEnabled(True)

        def _on_err(e):
            QtWidgets.QMessageBox.critical(self, "Charges Calculation Failed", friendly_error_text(e))

        run_in_background(self, "CM5 Charges", "Calculating CM5 and Hirshfeld charges via Multiwfn...", _worker, _on_done, _on_err)

    def _on_apply_cm5(self):
        if not self.viewer or not self._charges_res:
            return
        charges_arr = np.array([a["cm5"] for a in self._charges_res.atoms], dtype=np.float64)
        self.viewer.custom_charges = charges_arr
        self.viewer.show_charges = True
        self.viewer.update()
        QtWidgets.QMessageBox.information(
            self, "CM5 Charges Applied", f"Applied CM5 partial charges to {len(charges_arr)} atoms in the 3D viewer!"
        )

    def _get_active_table(self) -> Optional[QtWidgets.QTableWidget]:
        curr = self.tabs.currentIndex()
        tables = [self.tbl_bonds, self.tbl_fukui, self.tbl_qtaim, self.tbl_charges]
        if 0 <= curr < len(tables):
            return tables[curr]
        return None

    def _extract_table_data(self, tbl: QtWidgets.QTableWidget) -> Tuple[List[str], List[List[str]]]:
        headers = []
        for c in range(tbl.columnCount()):
            item = tbl.horizontalHeaderItem(c)
            headers.append(item.text() if item else f"Col_{c+1}")

        rows = []
        for r in range(tbl.rowCount()):
            row_vals = []
            for c in range(tbl.columnCount()):
                item = tbl.item(r, c)
                row_vals.append(item.text() if item else "")
            rows.append(row_vals)
        return headers, rows

    def _on_copy_active_table(self):
        tbl = self._get_active_table()
        if not tbl or tbl.rowCount() == 0:
            QtWidgets.QMessageBox.information(self, "No Data", "The current table contains no calculated data yet.")
            return

        headers, rows = self._extract_table_data(tbl)
        tsv_lines = ["\t".join(headers)]
        for r in rows:
            tsv_lines.append("\t".join(r))
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText("\n".join(tsv_lines))
        QtWidgets.QMessageBox.information(
            self, "Copied to Clipboard", f"Copied {len(rows)} rows to the clipboard as tab-separated values!"
        )

    def _on_save_active_csv(self):
        tbl = self._get_active_table()
        if not tbl or tbl.rowCount() == 0:
            QtWidgets.QMessageBox.information(self, "No Data", "The current table contains no calculated data yet.")
            return

        curr_tab_name = self.tabs.tabText(self.tabs.currentIndex()).split(". ")[-1].replace(" ", "_").lower()
        default_file = f"multiwfn_{curr_tab_name}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Analysis Table as CSV", default_file, "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        import csv
        headers, rows = self._extract_table_data(tbl)
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            QtWidgets.QMessageBox.information(self, "File Saved", f"Exported {len(rows)} rows to:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Failed", f"Could not save CSV file:\n{e}")

    # ---------------------------------------------------------------------
    # Run All
    # ---------------------------------------------------------------------
    def _on_run_all(self):
        if not self.molden_path or not os.path.isfile(self.molden_path):
            QtWidgets.QMessageBox.warning(self, "No Wavefunction", "Please select or load a valid .molden file first.")
            return

        self._on_calc_bond_orders()
        self._on_calc_fukui()
        self._on_calc_qtaim()
        self._on_calc_charges()
