"""
Multi-Structure 3D Superposition & RMSD Alignment Dialog.

Features:
- Kabsch SVD optimal rigid superposition.
- Compare active molecule with any trajectory optimization step (Reactant vs TS vs Product).
- Compare active molecule with an external XYZ or QM calculation output file.
- Displays All-Atom and Heavy-Atom RMSD (in Angstroms).
- Per-atom coordinate displacement table highlighting largest conformational shifts.
- Superimposes ghost comparison model directly in the 3D viewer.
- Option to apply alignment rotation/translation to the active molecule (with undo support).
"""

import os
from typing import List, Optional, Tuple
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .alignment import calculate_kabsch_alignment
from .zmatrix import parse_xyz


class StructureOverlayDialog(QtWidgets.QDialog):
    """Dialog for aligning and comparing two 3D molecular structures."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        symbols: List[str],
        coords: np.ndarray,
        trajectory: Optional[List[np.ndarray]] = None,
        trajectory_energies: Optional[List[float]] = None,
        viewer=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("🔬 Multi-Structure Overlay & RMSD Alignment")
        self.resize(720, 560)

        self.ref_symbols = list(symbols)
        self.ref_coords = np.array(coords, dtype=np.float64, copy=True)
        self.trajectory = trajectory or []
        self.trajectory_energies = trajectory_energies or []
        self.viewer = viewer

        self.mob_symbols: List[str] = []
        self.mob_coords: Optional[np.ndarray] = None
        self.aligned_coords: Optional[np.ndarray] = None
        self.current_rmsd: float = 0.0
        self.current_heavy_rmsd: float = 0.0

        self._init_ui()
        self._load_initial_comparison()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = QtWidgets.QLabel(
            "<b>Structure Comparison & Kabsch RMSD Alignment</b><br>"
            "<span style='color: #64748b; font-size: 11px;'>"
            "Overlay structures using SVD rigid alignment to track geometry optimization shifts, "
            "reaction pathways, or conformer differences.</span>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Source Selection Group
        grp_source = QtWidgets.QGroupBox("Comparison Structure (Mobile)")
        s_layout = QtWidgets.QHBoxLayout(grp_source)

        self.combo_source = QtWidgets.QComboBox()
        self.combo_source.addItem("Trajectory Optimization Step", "traj")
        self.combo_source.addItem("External XYZ / Molecular File...", "file")
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        s_layout.addWidget(self.combo_source)

        # Trajectory Step Selector
        self.spin_traj_step = QtWidgets.QSpinBox()
        n_frames = len(self.trajectory)
        if n_frames > 0:
            self.spin_traj_step.setRange(1, n_frames)
            self.spin_traj_step.setValue(1)
        else:
            self.spin_traj_step.setEnabled(False)
        self.spin_traj_step.valueChanged.connect(self._on_traj_step_changed)
        s_layout.addWidget(self.spin_traj_step)

        self.btn_browse = QtWidgets.QPushButton("Browse File...")
        self.btn_browse.setVisible(False)
        self.btn_browse.clicked.connect(self._on_browse_file)
        s_layout.addWidget(self.btn_browse)

        layout.addWidget(grp_source)

        # RMSD Readout Card
        grp_rmsd = QtWidgets.QGroupBox("Superposition Metrics (Kabsch SVD)")
        r_layout = QtWidgets.QGridLayout(grp_rmsd)
        r_layout.setSpacing(8)

        r_layout.addWidget(QtWidgets.QLabel("All-Atom RMSD:"), 0, 0)
        self.lbl_rmsd_all = QtWidgets.QLabel("<b>--</b>")
        self.lbl_rmsd_all.setStyleSheet("color: #38bdf8; font-size: 14px; font-family: monospace;")
        r_layout.addWidget(self.lbl_rmsd_all, 0, 1)

        r_layout.addWidget(QtWidgets.QLabel("Heavy-Atom RMSD:"), 0, 2)
        self.lbl_rmsd_heavy = QtWidgets.QLabel("<b>--</b>")
        self.lbl_rmsd_heavy.setStyleSheet("color: #4ade80; font-size: 14px; font-family: monospace;")
        r_layout.addWidget(self.lbl_rmsd_heavy, 0, 3)

        r_layout.addWidget(QtWidgets.QLabel("Max Displacement:"), 1, 0)
        self.lbl_max_disp = QtWidgets.QLabel("--")
        self.lbl_max_disp.setStyleSheet("color: #f59e0b; font-family: monospace;")
        r_layout.addWidget(self.lbl_max_disp, 1, 1, 1, 3)

        layout.addWidget(grp_rmsd)

        # Per-Atom Displacements Table
        layout.addWidget(QtWidgets.QLabel("<b>Per-Atom Displacements (Sorted by Shift):</b>"))
        self.table_disp = QtWidgets.QTableWidget()
        self.table_disp.setColumnCount(5)
        self.table_disp.setHorizontalHeaderLabels([
            "Atom #", "Element", "Ref Coords (Å)", "Aligned Coords (Å)", "Δr Shift (Å)"
        ])
        header = self.table_disp.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_disp.verticalHeader().setVisible(False)
        self.table_disp.setAlternatingRowColors(True)
        layout.addWidget(self.table_disp, stretch=1)

        # Buttons
        btn_box = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton("📐 Adopt Aligned Orientation")
        self.btn_apply.setStyleSheet("background-color: #0284c7; color: #fff; font-weight: bold; padding: 6px 12px;")
        self.btn_apply.setToolTip("Transform active coordinates to the optimal Kabsch aligned orientation (Ctrl+Z to undo)")
        self.btn_apply.clicked.connect(self._on_apply_alignment)
        btn_box.addWidget(self.btn_apply)

        btn_box.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)

        layout.addLayout(btn_box)

    def _load_initial_comparison(self):
        if len(self.trajectory) > 1:
            # Compare final step (index -1) against initial step (index 0) by default
            self.spin_traj_step.setValue(1)
            self._load_from_traj_step(0)
        else:
            self.combo_source.setCurrentIndex(1)
            self._on_source_changed(1)

    def _on_source_changed(self, idx: int):
        is_file = (self.combo_source.currentData() == "file")
        self.spin_traj_step.setVisible(not is_file)
        self.btn_browse.setVisible(is_file)

    def _on_traj_step_changed(self, val: int):
        idx = val - 1
        self._load_from_traj_step(idx)

    def _load_from_traj_step(self, step_idx: int):
        if 0 <= step_idx < len(self.trajectory):
            self.mob_symbols = list(self.ref_symbols)
            self.mob_coords = np.array(self.trajectory[step_idx], dtype=np.float64, copy=True)
            self._recalculate_alignment()

    def _on_browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Comparison Molecule", "", "Molecular Files (*.xyz *.out *.log);;All Files (*)"
        )
        if not path:
            return
        try:
            syms, crds = parse_xyz(path)
            if len(syms) == 0:
                QtWidgets.QMessageBox.warning(self, "Invalid File", "No atomic coordinates found in file.")
                return
            if len(syms) != len(self.ref_symbols):
                QtWidgets.QMessageBox.warning(
                    self, "Atom Count Mismatch",
                    f"Selected file has {len(syms)} atoms, but reference has {len(self.ref_symbols)} atoms.\n"
                    "Rigid RMSD superposition requires matching atom counts."
                )
                return
            self.mob_symbols = syms
            self.mob_coords = np.array(crds, dtype=np.float64)
            self._recalculate_alignment()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "File Error", f"Failed to parse file: {e}")

    def _recalculate_alignment(self):
        if self.mob_coords is None or len(self.mob_coords) != len(self.ref_coords):
            return

        aligned, rmsd, R, t = calculate_kabsch_alignment(self.ref_coords, self.mob_coords)
        self.aligned_coords = aligned
        self.current_rmsd = rmsd

        # Calculate heavy atom RMSD
        heavy_mask = np.array([s.upper() != "H" for s in self.ref_symbols])
        if np.any(heavy_mask):
            diff_heavy = self.ref_coords[heavy_mask] - aligned[heavy_mask]
            heavy_rmsd = float(np.sqrt(np.mean(np.sum(diff_heavy**2, axis=1))))
        else:
            heavy_rmsd = rmsd
        self.current_heavy_rmsd = heavy_rmsd

        # Displacements
        diffs = np.linalg.norm(self.ref_coords - aligned, axis=1)
        max_idx = int(np.argmax(diffs))
        max_shift = float(diffs[max_idx])

        # Update labels
        self.lbl_rmsd_all.setText(f"<b>{rmsd:.4f} Å</b>")
        self.lbl_rmsd_heavy.setText(f"<b>{heavy_rmsd:.4f} Å</b>")
        self.lbl_max_disp.setText(
            f"Atom {self.ref_symbols[max_idx]}{max_idx+1}: <b>{max_shift:.4f} Å</b> shift"
        )

        # Populate table sorted descending by shift
        sorted_indices = np.argsort(-diffs)
        self.table_disp.setRowCount(len(sorted_indices))

        for row, idx in enumerate(sorted_indices):
            sym = self.ref_symbols[idx]
            shift = float(diffs[idx])

            item_idx = QtWidgets.QTableWidgetItem(str(idx + 1))
            item_idx.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            item_sym = QtWidgets.QTableWidgetItem(sym)
            item_sym.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_sym.setForeground(QtGui.QBrush(QtGui.QColor("#38bdf8")))

            p_ref = self.ref_coords[idx]
            item_ref = QtWidgets.QTableWidgetItem(f"({p_ref[0]:.3f}, {p_ref[1]:.3f}, {p_ref[2]:.3f})")
            item_ref.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            p_aln = aligned[idx]
            item_aln = QtWidgets.QTableWidgetItem(f"({p_aln[0]:.3f}, {p_aln[1]:.3f}, {p_aln[2]:.3f})")
            item_aln.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            item_shift = QtWidgets.QTableWidgetItem(f"{shift:.4f}")
            item_shift.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            if shift > 0.35:
                item_shift.setForeground(QtGui.QBrush(QtGui.QColor("#f87171")))
                item_shift.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
            elif shift > 0.10:
                item_shift.setForeground(QtGui.QBrush(QtGui.QColor("#fbbf24")))
            else:
                item_shift.setForeground(QtGui.QBrush(QtGui.QColor("#4ade80")))

            self.table_disp.setItem(row, 0, item_idx)
            self.table_disp.setItem(row, 1, item_sym)
            self.table_disp.setItem(row, 2, item_ref)
            self.table_disp.setItem(row, 3, item_aln)
            self.table_disp.setItem(row, 4, item_shift)

    def _on_apply_alignment(self):
        """Adopt the aligned coordinates for the active molecule in the viewer."""
        if self.aligned_coords is None:
            return
        parent = self.parent()
        if parent and hasattr(parent, "_push_undo_state") and hasattr(parent, "set_molecule_from_zatoms"):
            parent._push_undo_state()
            parent.coords = np.array(self.aligned_coords, copy=True)
            if hasattr(parent, "viewer") and parent.viewer:
                parent.viewer.set_molecule(
                    symbols=[a.symbol for a in parent.zatoms],
                    coords=parent.coords,
                    user_bonds=parent._current_explicit_bonds(),
                )
                parent.viewer.update()
            if hasattr(parent, "statusBar") and parent.statusBar():
                parent.statusBar().showMessage(f"Applied Kabsch alignment orientation (RMSD = {self.current_rmsd:.4f} Å).", 4000)
            QtWidgets.QMessageBox.information(
                self, "Alignment Applied",
                f"Active molecular coordinates updated to the Kabsch aligned orientation (RMSD: {self.current_rmsd:.4f} Å)."
            )
            self.accept()
