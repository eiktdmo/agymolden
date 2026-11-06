"""
Dialog for Reaction Path Interpolation and Transition State (TS) Guess Generation.
Supports Image-Dependent Pair Potential (IDPP), Kabsch alignment, live 3D animation,
and input generation for ORCA NEB-TS and Gaussian QST2/QST3.
"""

import os
from typing import List, Optional
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui

from .ts_interpolator import (
    kabsch_align,
    interpolate_linear,
    interpolate_idpp,
    format_path_xyz,
    generate_orca_neb_input,
    generate_gaussian_qst2_input,
    generate_gaussian_qst3_input,
)


class ReactionPathDialog(QtWidgets.QDialog):
    """Interactive Reaction Path & Transition State Guess Builder."""

    path_adopted = QtCore.pyqtSignal(list, np.ndarray)  # (symbols, coords) to adopt TS into editor
    trajectory_preview = QtCore.pyqtSignal(list, list)   # (symbols, list of coords arrays)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        initial_symbols: Optional[List[str]] = None,
        initial_coords: Optional[np.ndarray] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("🔄 Reaction Path & Transition State (TS) Guess Interpolator")
        self.resize(680, 560)

        self.r_symbols = list(initial_symbols) if initial_symbols else []
        self.r_coords = np.array(initial_coords, dtype=np.float64) if initial_coords is not None else np.empty((0, 3))
        self.p_symbols: List[str] = []
        self.p_coords: np.ndarray = np.empty((0, 3))

        self.path_images: Optional[np.ndarray] = None

        self._init_ui()
        self._update_status()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        header = QtWidgets.QLabel(
            "<b>Reaction Path & Transition State (TS) Guess Interpolator</b><br>"
            "<span style='color: #64748b; font-size: 11px;'>"
            "Aligns Reactant and Product geometries using Kabsch SVD alignment and generates smooth, "
            "collision-free reaction paths using Image-Dependent Pair Potential (IDPP). "
            "Instantly export ORCA NEB-TS or Gaussian QST2/QST3 calculation inputs.</span>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Structure Selection Box
        grp_structs = QtWidgets.QGroupBox("1. Reactant & Product Structures")
        s_layout = QtWidgets.QGridLayout(grp_structs)
        s_layout.setSpacing(8)

        # Reactant Row
        s_layout.addWidget(QtWidgets.QLabel("<b>Reactant:</b>"), 0, 0)
        self.lbl_reactant_info = QtWidgets.QLabel("None loaded")
        self.lbl_reactant_info.setStyleSheet("color: #0284c7; font-weight: bold;")
        s_layout.addWidget(self.lbl_reactant_info, 0, 1)

        btn_r_current = QtWidgets.QPushButton("Use Active Canvas")
        btn_r_current.setToolTip("Use the molecule currently open in AgyMolden as the Reactant")
        btn_r_current.clicked.connect(self._on_use_active_as_reactant)
        s_layout.addWidget(btn_r_current, 0, 2)

        btn_r_paste = QtWidgets.QPushButton("Paste Clipboard")
        btn_r_paste.setToolTip("Parse Cartesian XYZ coordinates from system clipboard as the Reactant")
        btn_r_paste.clicked.connect(self._on_paste_reactant)
        s_layout.addWidget(btn_r_paste, 0, 3)

        btn_r_load = QtWidgets.QPushButton("Load XYZ...")
        btn_r_load.clicked.connect(lambda: self._on_load_xyz("reactant"))
        s_layout.addWidget(btn_r_load, 0, 4)

        # Product Row
        s_layout.addWidget(QtWidgets.QLabel("<b>Product:</b>"), 1, 0)
        self.lbl_product_info = QtWidgets.QLabel("None loaded")
        self.lbl_product_info.setStyleSheet("color: #d97706; font-weight: bold;")
        s_layout.addWidget(self.lbl_product_info, 1, 1)

        btn_p_current = QtWidgets.QPushButton("Use Active Canvas")
        btn_p_current.setToolTip("Use the molecule currently open in AgyMolden as the Product")
        btn_p_current.clicked.connect(self._on_use_active_as_product)
        s_layout.addWidget(btn_p_current, 1, 2)

        btn_p_paste = QtWidgets.QPushButton("Paste Clipboard")
        btn_p_paste.setToolTip("Parse Cartesian XYZ coordinates from system clipboard as the Product")
        btn_p_paste.clicked.connect(self._on_paste_product)
        s_layout.addWidget(btn_p_paste, 1, 3)

        btn_p_load = QtWidgets.QPushButton("Load XYZ...")
        btn_p_load.clicked.connect(lambda: self._on_load_xyz("product"))
        s_layout.addWidget(btn_p_load, 1, 4)

        layout.addWidget(grp_structs)

        # Status / Compatibility Badge
        self.lbl_status = QtWidgets.QLabel("Load both Reactant and Product geometries to proceed.")
        self.lbl_status.setStyleSheet("background-color: #1e293b; color: #94a3b8; padding: 8px; border-radius: 4px;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        # Interpolation Parameters
        grp_params = QtWidgets.QGroupBox("2. Path Interpolation Method & Resolution")
        p_layout = QtWidgets.QFormLayout(grp_params)
        p_layout.setSpacing(8)

        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.addItems([
            "IDPP (Image-Dependent Pair Potential — Collision Free, Recommended)",
            "Linear Transit (Direct Cartesian Interpolation)"
        ])
        p_layout.addRow("Algorithm:", self.combo_method)

        img_row = QtWidgets.QHBoxLayout()
        self.spin_images = QtWidgets.QSpinBox()
        self.spin_images.setRange(3, 25)
        self.spin_images.setValue(7)
        self.spin_images.setSingleStep(2)  # Odd number preferred so exact middle is TS guess
        img_row.addWidget(self.spin_images)

        self.lbl_middle_hint = QtWidgets.QLabel("(Frame 4 will be the Transition State Guess)")
        self.lbl_middle_hint.setStyleSheet("color: #64748b; font-style: italic;")
        img_row.addWidget(self.lbl_middle_hint)
        img_row.addStretch()
        p_layout.addRow("Number of Images:", img_row)

        self.spin_images.valueChanged.connect(self._update_middle_hint)

        layout.addWidget(grp_params)

        # Generation Actions Box
        grp_actions = QtWidgets.QGroupBox("3. Actions & Exports")
        a_layout = QtWidgets.QVBoxLayout(grp_actions)
        a_layout.setSpacing(8)

        self.btn_interpolate = QtWidgets.QPushButton("⚡ Compute Smooth Reaction Path & TS Guess")
        self.btn_interpolate.setStyleSheet("font-weight: bold; background-color: #2563eb; color: #ffffff; padding: 6px;")
        self.btn_interpolate.clicked.connect(self._on_compute_path)
        a_layout.addWidget(self.btn_interpolate)

        btn_row1 = QtWidgets.QHBoxLayout()
        self.btn_preview = QtWidgets.QPushButton("▶️ Preview Animation in Viewport")
        self.btn_preview.setEnabled(False)
        self.btn_preview.setToolTip("Send reaction trajectory to AgyMolden's player slider for real-time inspection")
        self.btn_preview.clicked.connect(self._on_preview_trajectory)
        btn_row1.addWidget(self.btn_preview)

        self.btn_adopt_ts = QtWidgets.QPushButton("🎯 Adopt TS Guess in 3D Editor")
        self.btn_adopt_ts.setEnabled(False)
        self.btn_adopt_ts.setStyleSheet("color: #059669; font-weight: bold;")
        self.btn_adopt_ts.setToolTip("Load the peak/middle IDPP transition state guess directly into the active editor")
        self.btn_adopt_ts.clicked.connect(self._on_adopt_ts_guess)
        btn_row1.addWidget(self.btn_adopt_ts)

        self.btn_launch_ts_pipeline = QtWidgets.QPushButton("⚡ 1-Click TS Pipeline...")
        self.btn_launch_ts_pipeline.setEnabled(False)
        self.btn_launch_ts_pipeline.setStyleSheet("background-color: #7c3aed; color: #ffffff; font-weight: bold;")
        self.btn_launch_ts_pipeline.setToolTip("Launch the multi-stage TS Opt → Freq verification → IRC MEP pipeline on the TS guess")
        self.btn_launch_ts_pipeline.clicked.connect(self._on_launch_ts_pipeline)
        btn_row1.addWidget(self.btn_launch_ts_pipeline)

        a_layout.addLayout(btn_row1)

        btn_row2 = QtWidgets.QHBoxLayout()
        self.btn_export_xyz = QtWidgets.QPushButton("💾 Save Path XYZ...")
        self.btn_export_xyz.setEnabled(False)
        self.btn_export_xyz.clicked.connect(self._on_export_xyz)
        btn_row2.addWidget(self.btn_export_xyz)

        self.btn_orca_neb = QtWidgets.QPushButton("🚀 ORCA NEB-TS Input...")
        self.btn_orca_neb.setEnabled(False)
        self.btn_orca_neb.clicked.connect(self._on_generate_orca_neb)
        btn_row2.addWidget(self.btn_orca_neb)

        self.btn_gauss_qst = QtWidgets.QPushButton("⚛️ Gaussian QST2/3 Input...")
        self.btn_gauss_qst.setEnabled(False)
        self.btn_gauss_qst.clicked.connect(self._on_generate_gaussian_qst)
        btn_row2.addWidget(self.btn_gauss_qst)

        a_layout.addLayout(btn_row2)
        layout.addWidget(grp_actions)

        # Close button
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

    def _update_middle_hint(self, val: int):
        mid = (val // 2) + 1
        self.lbl_middle_hint.setText(f"(Frame {mid} will be the Transition State Guess)")

    def _get_main_window(self) -> Optional[QtWidgets.QMainWindow]:
        curr = self.parent()
        while curr is not None:
            if hasattr(curr, "coords") and (hasattr(curr, "zatoms") or hasattr(curr, "symbols")):
                return curr
            curr = curr.parent()
        return None

    def _on_use_active_as_reactant(self):
        mw = self._get_main_window()
        if mw is not None and hasattr(mw, "coords") and len(mw.coords) > 0:
            syms = [a.symbol for a in getattr(mw, "zatoms", [])] if getattr(mw, "zatoms", None) else list(getattr(mw, "symbols", []))
            if syms and len(syms) == len(mw.coords):
                self.r_symbols = list(syms)
                self.r_coords = np.array(mw.coords, dtype=np.float64)
                self._update_status()
                return
        QtWidgets.QMessageBox.information(self, "No Active Molecule", "The active canvas in AgyMolden does not currently contain a molecule.")

    def _on_use_active_as_product(self):
        mw = self._get_main_window()
        if mw is not None and hasattr(mw, "coords") and len(mw.coords) > 0:
            syms = [a.symbol for a in getattr(mw, "zatoms", [])] if getattr(mw, "zatoms", None) else list(getattr(mw, "symbols", []))
            if syms and len(syms) == len(mw.coords):
                self.p_symbols = list(syms)
                self.p_coords = np.array(mw.coords, dtype=np.float64)
                self._update_status()
                return
        QtWidgets.QMessageBox.information(self, "No Active Molecule", "The active canvas in AgyMolden does not currently contain a molecule.")

    def _on_paste_reactant(self):
        clipboard = QtWidgets.QApplication.clipboard()
        text = clipboard.text().strip() if clipboard else ""
        syms, coords = self._parse_xyz_string(text)
        if not syms:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Clipboard Text", "Clipboard does not contain recognizable Cartesian XYZ format."
            )
            return
        self.r_symbols, self.r_coords = syms, coords
        self._update_status()

    def _on_paste_product(self):
        clipboard = QtWidgets.QApplication.clipboard()
        text = clipboard.text().strip() if clipboard else ""
        syms, coords = self._parse_xyz_string(text)
        if not syms:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Clipboard Text", "Clipboard does not contain recognizable Cartesian XYZ format."
            )
            return
        self.p_symbols, self.p_coords = syms, coords
        self._update_status()

    def _on_load_xyz(self, target: str):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Open {target.title()} XYZ File", "", "XYZ Coordinates (*.xyz);;All Files (*)"
        )
        if not path or not os.path.isfile(path):
            return
        syms, coords = self._parse_xyz_file(path)
        if not syms:
            QtWidgets.QMessageBox.warning(self, "Parse Error", "Could not parse valid Cartesian XYZ coordinates.")
            return
        if target == "reactant":
            self.r_symbols, self.r_coords = syms, coords
        else:
            self.p_symbols, self.p_coords = syms, coords
        self._update_status()

    @staticmethod
    def _parse_xyz_file(path: str) -> Tuple[List[str], np.ndarray]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return ReactionPathDialog._parse_xyz_string(f.read())
        except Exception:
            return [], np.empty((0, 3))

    @staticmethod
    def _parse_xyz_string(text: str) -> Tuple[List[str], np.ndarray]:
        from .zmatrix import parse_xyz
        try:
            syms, coords = parse_xyz(text)
            if syms and len(coords) == len(syms):
                return syms, coords
        except Exception:
            pass
        return [], np.empty((0, 3))

    def _update_status(self):
        # Update labels
        if len(self.r_symbols) > 0:
            self.lbl_reactant_info.setText(f"{len(self.r_symbols)} atoms ({''.join(sorted(set(self.r_symbols)))})")
        else:
            self.lbl_reactant_info.setText("None loaded")

        if len(self.p_symbols) > 0:
            self.lbl_product_info.setText(f"{len(self.p_symbols)} atoms ({''.join(sorted(set(self.p_symbols)))})")
        else:
            self.lbl_product_info.setText("None loaded")

        if len(self.r_symbols) == 0 or len(self.p_symbols) == 0:
            self.lbl_status.setText("Load both Reactant and Product geometries to proceed.")
            self.lbl_status.setStyleSheet("background-color: #1e293b; color: #94a3b8; padding: 8px; border-radius: 4px;")
            self.btn_interpolate.setEnabled(False)
            return

        if len(self.r_symbols) != len(self.p_symbols):
            self.lbl_status.setText(f"❌ Atom count mismatch: Reactant has {len(self.r_symbols)} atoms, Product has {len(self.p_symbols)} atoms.")
            self.lbl_status.setStyleSheet("background-color: #450a0a; color: #fca5a5; padding: 8px; border-radius: 4px; font-weight: bold;")
            self.btn_interpolate.setEnabled(False)
            return

        # Check atom element ordering
        r_upper = [s.upper() for s in self.r_symbols]
        p_upper = [s.upper() for s in self.p_symbols]
        if r_upper != p_upper:
            self.lbl_status.setText(
                "⚠️ Atom ordering warning: Element sequences do not match 1-to-1. "
                "Ensure identical atom indexing between reactant and product."
            )
            self.lbl_status.setStyleSheet("background-color: #451a03; color: #fdba74; padding: 8px; border-radius: 4px;")
            self.btn_interpolate.setEnabled(True)
            return

        # Compute Kabsch alignment RMSD
        p_aligned = kabsch_align(self.r_coords, self.p_coords)
        rmsd = float(np.sqrt(np.mean(np.sum((self.r_coords - p_aligned) ** 2, axis=1))))

        self.lbl_status.setText(
            f"✓ Ready: Reactant and Product match ({len(self.r_symbols)} atoms). "
            f"Kabsch rotational/translational RMSD = {rmsd:.3f} Å."
        )
        self.lbl_status.setStyleSheet("background-color: #064e3b; color: #6ee7b7; padding: 8px; border-radius: 4px; font-weight: bold;")
        self.btn_interpolate.setEnabled(True)

    def _on_compute_path(self):
        n_imgs = self.spin_images.value()
        is_idpp = "IDPP" in self.combo_method.currentText()

        self.lbl_status.setText("⏳ Interpolating path images and optimizing pair potential...")
        QtWidgets.QApplication.processEvents()

        try:
            if is_idpp:
                self.path_images = interpolate_idpp(self.r_coords, self.p_coords, n_images=n_imgs)
            else:
                self.path_images = interpolate_linear(self.r_coords, self.p_coords, n_images=n_imgs)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Interpolation Failed", f"Error during path generation:\n{e}")
            return

        mid = len(self.path_images) // 2
        self.lbl_status.setText(f"✓ Path generated successfully ({len(self.path_images)} frames). Frame {mid + 1} is TS Guess.")
        self.lbl_status.setStyleSheet("background-color: #064e3b; color: #6ee7b7; padding: 8px; border-radius: 4px; font-weight: bold;")

        self.btn_preview.setEnabled(True)
        self.btn_adopt_ts.setEnabled(True)
        self.btn_launch_ts_pipeline.setEnabled(True)
        self.btn_export_xyz.setEnabled(True)
        self.btn_orca_neb.setEnabled(True)
        self.btn_gauss_qst.setEnabled(True)

    def _on_preview_trajectory(self):
        if self.path_images is None:
            return
        frames = [self.path_images[k] for k in range(len(self.path_images))]
        self.trajectory_preview.emit(self.r_symbols, frames)

    def _on_adopt_ts_guess(self):
        if self.path_images is None:
            return
        mid = len(self.path_images) // 2
        ts_coords = self.path_images[mid]
        self.path_adopted.emit(self.r_symbols, ts_coords)
        QtWidgets.QMessageBox.information(
            self,
            "TS Guess Adopted",
            f"Successfully adopted Frame {mid + 1}/{len(self.path_images)} as the Transition State geometry in the 3D editor!"
        )

    def _on_launch_ts_pipeline(self):
        if self.path_images is None or len(self.path_images) == 0:
            QtWidgets.QMessageBox.warning(self, "No Path", "Please compute a reaction path first.")
            return

        mid = len(self.path_images) // 2
        ts_coords = self.path_images[mid]

        try:
            from .calc_runner.dialog_pipeline import WorkflowPipelineDialog
            dlg = WorkflowPipelineDialog(
                parent=self.parent(),
                code_key="orca",
                symbols=self.r_symbols,
                coords=ts_coords,
            )
            for idx in range(dlg.combo_recipe.count()):
                if "TS Opt" in dlg.combo_recipe.itemText(idx):
                    dlg.combo_recipe.setCurrentIndex(idx)
                    break
            dlg.exec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Pipeline Error", f"Failed to launch TS Workflow Pipeline:\n{e}")

    def _on_export_xyz(self):
        if self.path_images is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Multi-Frame Reaction Path XYZ", "reaction_path.xyz", "XYZ Coordinates (*.xyz);;All Files (*)"
        )
        if not path:
            return
        content = format_path_xyz(self.r_symbols, self.path_images)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        QtWidgets.QMessageBox.information(self, "Exported", f"Saved {len(self.path_images)} reaction frames to:\n{path}")

    def _on_generate_orca_neb(self):
        if self.path_images is None:
            return
        content = generate_orca_neb_input(self.r_symbols, self.path_images)
        self._display_generated_input("ORCA NEB-TS Input File", content, default_name="orca_neb.inp")

    def _on_generate_gaussian_qst(self):
        if self.path_images is None:
            return
        mid = len(self.path_images) // 2
        ts_coords = self.path_images[mid]
        content = generate_gaussian_qst3_input(self.r_symbols, self.r_coords, self.p_coords, ts_coords)
        self._display_generated_input("Gaussian QST3 Transition State Input File", content, default_name="gauss_qst3.gjf")

    def _display_generated_input(self, title: str, content: str, default_name: str):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(600, 480)
        vbox = QtWidgets.QVBoxLayout(dlg)

        txt = QtWidgets.QPlainTextEdit()
        txt.setFont(QtGui.QFont("Monospace", 10))
        txt.setPlainText(content)
        vbox.addWidget(txt)

        btn_row = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("📋 Copy to Clipboard")
        btn_copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(txt.toPlainText()))
        btn_row.addWidget(btn_copy)

        btn_save = QtWidgets.QPushButton("💾 Save File...")
        def _save():
            p, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save Input File", default_name)
            if p:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(txt.toPlainText())
                QtWidgets.QMessageBox.information(dlg, "Saved", f"Saved to {p}")
        btn_save.clicked.connect(_save)
        btn_row.addWidget(btn_save)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)

        vbox.addLayout(btn_row)
        dlg.exec()
