"""
Quantum Chemistry Calculation Setup Dialog (GaussView-style).

Provides an interactive GUI to configure calculations (Job Type, Method, Basis,
Solvation, Charge, Multiplicity, Hardware resources) and preview/export the
generated input file for any supported QM package (Gaussian, ORCA, PySCF,
openLOWDIN, xTB, Q-Chem, etc.).
"""

import os
from typing import List, Optional, Dict
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .zmatrix import ZAtom
from .qm_input_generator import (
    QM_GENERATORS,
    GaussianGenerator,
    QMCalculationParameters,
    COMMON_METHODS,
    COMMON_BASIS_SETS,
    COMMON_SOLVENTS,
)


class QMRunConfirmDialog(QtWidgets.QDialog):
    """
    Launch confirmation screen before starting calculation.
    Shows current file name and folder tree with options to rename or browse destination.
    """
    def __init__(self, parent: QtWidgets.QWidget, default_filename: str, default_dir: str, content: str, qm_name: str):
        super().__init__(parent)
        self.setWindowTitle(f"Launch {qm_name} Calculation")
        self.resize(580, 260)
        self.content = content
        self.qm_name = qm_name

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        info_lbl = QtWidgets.QLabel(
            f"<b>Ready to launch {qm_name} calculation.</b><br>"
            "<span style='color: #64748b;'>Confirm or change the file name and working directory below:</span>"
        )
        layout.addWidget(info_lbl)

        grp = QtWidgets.QGroupBox("Target File & Directory")
        form = QtWidgets.QFormLayout(grp)
        form.setSpacing(10)

        self.txt_filename = QtWidgets.QLineEdit(default_filename)
        form.addRow("File Name:", self.txt_filename)

        dir_row = QtWidgets.QHBoxLayout()
        self.txt_dir = QtWidgets.QLineEdit(default_dir)
        dir_row.addWidget(self.txt_dir, stretch=1)
        btn_browse = QtWidgets.QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse_dir)
        dir_row.addWidget(btn_browse)
        form.addRow("Folder Tree:", dir_row)

        self.lbl_full_path = QtWidgets.QLabel("")
        self.lbl_full_path.setStyleSheet("color: #0284c7; font-family: monospace; font-size: 11px;")
        self.lbl_full_path.setWordWrap(True)
        form.addRow("Full Path:", self.lbl_full_path)

        layout.addWidget(grp)

        self.txt_filename.textChanged.connect(self._update_full_path)
        self.txt_dir.textChanged.connect(self._update_full_path)
        self._update_full_path()

        # Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch()

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        self.btn_launch = QtWidgets.QPushButton("▶ Launch Calculation")
        self.btn_launch.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 6px 16px;")
        self.btn_launch.clicked.connect(self._on_confirm_launch)
        btn_box.addWidget(self.btn_launch)

        layout.addLayout(btn_box)

    def _update_full_path(self):
        full = os.path.join(self.txt_dir.text().strip(), self.txt_filename.text().strip())
        self.lbl_full_path.setText(full)

    def _on_browse_dir(self):
        cur = self.txt_dir.text().strip()
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Working Folder", cur or os.getcwd())
        if chosen:
            self.txt_dir.setText(chosen)

    def _on_confirm_launch(self):
        dest_dir = self.txt_dir.text().strip()
        fname = self.txt_filename.text().strip()
        if not fname:
            QtWidgets.QMessageBox.warning(self, "Missing Name", "Please provide a valid file name.")
            return
        if not dest_dir:
            dest_dir = os.getcwd()

        os.makedirs(dest_dir, exist_ok=True)
        full_path = os.path.join(dest_dir, fname)

        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(self.content)
            self.saved_path = full_path
            self.accept()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Write Error", f"Could not write input file:\n{e}")

    def get_saved_path(self) -> Optional[str]:
        return getattr(self, "saved_path", None)


class QMCalculationDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        symbols: List[str],
        coords: np.ndarray,
        zatoms: Optional[List[ZAtom]] = None,
        variables: Optional[Dict[str, float]] = None,
        frozen_atoms: Optional[List[int]] = None,
        selected_atoms: Optional[List[int]] = None,
        auto_enable_scan: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("Quantum Chemistry Calculation Setup")
        self.resize(780, 720)

        self.symbols = symbols
        self.coords = coords
        self.zatoms = zatoms or []
        self.variables = variables or {}
        self.frozen_atoms = frozen_atoms or []
        self.selected_atoms = [i for i in (selected_atoms or []) if 0 <= i < len(symbols)]
        self.auto_enable_scan = bool(auto_enable_scan)

        self._init_ui()
        # Initialize default preview and labels for the first selected QM code
        first_code = self.combo_qm_code.currentText()
        first_gen = QM_GENERATORS.get(first_code)
        if first_gen:
            self.lbl_ext.setText(first_gen.file_extension)
            self.chk_zmatrix.setEnabled(first_gen.supports_zmatrix)
            if not first_gen.supports_zmatrix:
                self.chk_zmatrix.setChecked(False)
            if "orca" in first_code.lower():
                self.spin_procs.setValue(1)
                if self.combo_dispersion.currentText() == "D4":
                    self.combo_dispersion.setCurrentText("D3BJ")
        self._update_route_ui()
        self._update_preview()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # Top banner: QM Code selector
        top_group = QtWidgets.QGroupBox("Target Quantum Chemistry Software")
        top_layout = QtWidgets.QHBoxLayout(top_group)

        top_layout.addWidget(QtWidgets.QLabel("<b>QM Package:</b>"))
        self.combo_qm_code = QtWidgets.QComboBox()
        for name in QM_GENERATORS.keys():
            self.combo_qm_code.addItem(name)
        self.combo_qm_code.currentTextChanged.connect(self._on_qm_code_changed)
        top_layout.addWidget(self.combo_qm_code, stretch=2)

        self.lbl_ext = QtWidgets.QLabel(".gjf")
        self.lbl_ext.setStyleSheet("font-weight: bold; color: #0284c7;")
        top_layout.addWidget(QtWidgets.QLabel("Format:"))
        top_layout.addWidget(self.lbl_ext)

        self.btn_format_info = QtWidgets.QToolButton()
        self.btn_format_info.setText("ℹ️")
        self.btn_format_info.setToolTip("View detailed format documentation & syntax for this QM package")
        self.btn_format_info.setStyleSheet("font-weight: bold; font-size: 13px; color: #0284c7; border: none; padding: 2px 6px;")
        self.btn_format_info.clicked.connect(self._on_show_format_info)
        top_layout.addWidget(self.btn_format_info)
        layout.addWidget(top_group)

        # Tabbed settings: Job/Method, System/Solvent, Hardware
        tabs = QtWidgets.QTabWidget()

        # Tab 1: Method & Job Type
        tab1 = QtWidgets.QWidget()
        t1_layout = QtWidgets.QFormLayout(tab1)
        t1_layout.setContentsMargins(10, 10, 10, 10)
        t1_layout.setSpacing(8)

        self.combo_job_type = QtWidgets.QComboBox()
        self.combo_job_type.addItems(["Opt + Freq", "Optimization (Opt)", "Single Point Energy (SP)", "Frequencies (Freq)", "Transition State (TS)"])
        self.combo_job_type.currentTextChanged.connect(self._update_preview)
        t1_layout.addRow("Job Type:", self.combo_job_type)

        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.setEditable(True)
        self.combo_method.addItems(COMMON_METHODS)
        self.combo_method.currentTextChanged.connect(self._update_preview)
        t1_layout.addRow("Method / Functional:", self.combo_method)

        self.combo_basis = QtWidgets.QComboBox()
        self.combo_basis.setEditable(True)
        self.combo_basis.addItems(COMMON_BASIS_SETS)
        self.combo_basis.currentTextChanged.connect(self._update_preview)
        t1_layout.addRow("Basis Set:", self.combo_basis)

        self.combo_dispersion = QtWidgets.QComboBox()
        self.combo_dispersion.addItems(["D3BJ", "D4", "D3Zero", "None"])
        self.combo_dispersion.currentTextChanged.connect(self._update_preview)
        t1_layout.addRow("Empirical Dispersion:", self.combo_dispersion)

        # Relaxed PES Scan Group (GaussView style: auto-populated from 2/3/4 selected atoms)
        self.grp_pes_scan = QtWidgets.QGroupBox("Relaxed Potential Energy Surface (PES) Scan (Selected 2/3/4 Atoms)")
        self.grp_pes_scan.setCheckable(True)
        self.grp_pes_scan.setChecked(self.auto_enable_scan and len(self.selected_atoms) in (2, 3, 4))
        self.grp_pes_scan.toggled.connect(self._on_scan_toggled)
        scan_row = QtWidgets.QHBoxLayout(self.grp_pes_scan)
        scan_row.setSpacing(6)

        self.lbl_scan_coord = QtWidgets.QLabel("Select 2, 3, or 4 atoms in 3D")
        self.lbl_scan_coord.setStyleSheet("color: #0284c7; font-weight: bold;")
        scan_row.addWidget(self.lbl_scan_coord)

        scan_row.addWidget(QtWidgets.QLabel("From:"))
        self.spin_scan_start = QtWidgets.QDoubleSpinBox()
        self.spin_scan_start.setRange(-360.0, 360.0)
        self.spin_scan_start.setDecimals(2)
        self.spin_scan_start.valueChanged.connect(self._update_preview)
        scan_row.addWidget(self.spin_scan_start)

        scan_row.addWidget(QtWidgets.QLabel("To:"))
        self.spin_scan_end = QtWidgets.QDoubleSpinBox()
        self.spin_scan_end.setRange(-360.0, 360.0)
        self.spin_scan_end.setDecimals(2)
        self.spin_scan_end.valueChanged.connect(self._update_preview)
        scan_row.addWidget(self.spin_scan_end)

        scan_row.addWidget(QtWidgets.QLabel("Steps:"))
        self.spin_scan_steps = QtWidgets.QSpinBox()
        self.spin_scan_steps.setRange(2, 180)
        self.spin_scan_steps.setValue(12)
        self.spin_scan_steps.valueChanged.connect(self._update_preview)
        scan_row.addWidget(self.spin_scan_steps)

        self._populate_scan_defaults_from_selection()
        t1_layout.addRow(self.grp_pes_scan)

        tabs.addTab(tab1, "Job & Method")

        # Tab 2: System & Solvation
        tab2 = QtWidgets.QWidget()
        t2_layout = QtWidgets.QFormLayout(tab2)
        t2_layout.setContentsMargins(10, 10, 10, 10)
        t2_layout.setSpacing(8)

        row_cm = QtWidgets.QHBoxLayout()
        self.spin_charge = QtWidgets.QSpinBox()
        self.spin_charge.setRange(-10, 10)
        self.spin_charge.setValue(0)
        self.spin_charge.valueChanged.connect(self._update_preview)
        row_cm.addWidget(QtWidgets.QLabel("Charge:"))
        row_cm.addWidget(self.spin_charge)

        self.spin_mult = QtWidgets.QSpinBox()
        self.spin_mult.setRange(1, 10)
        self.spin_mult.setValue(1)
        self.spin_mult.valueChanged.connect(self._update_preview)
        row_cm.addWidget(QtWidgets.QLabel("Multiplicity (2S+1):"))
        row_cm.addWidget(self.spin_mult)
        t2_layout.addRow("State:", row_cm)

        self.combo_solvent = QtWidgets.QComboBox()
        self.combo_solvent.addItems(COMMON_SOLVENTS)
        self.combo_solvent.currentTextChanged.connect(self._update_preview)
        t2_layout.addRow("Solvent (Implicit):", self.combo_solvent)

        self.chk_zmatrix = QtWidgets.QCheckBox("Use Z-Matrix Internal Coordinates (if supported)")
        self.chk_zmatrix.setChecked(bool(self.zatoms))
        self.chk_zmatrix.toggled.connect(self._update_preview)
        t2_layout.addRow("Geometry Representation:", self.chk_zmatrix)

        self.txt_title = QtWidgets.QLineEdit("AgyMolden generated calculation")
        self.txt_title.textChanged.connect(self._update_preview)
        t2_layout.addRow("Title / Comment:", self.txt_title)

        tabs.addTab(tab2, "Charge & Solvation")

        # Tab 3: Hardware & Extra Options
        tab3 = QtWidgets.QWidget()
        t3_layout = QtWidgets.QFormLayout(tab3)
        t3_layout.setContentsMargins(10, 10, 10, 10)
        t3_layout.setSpacing(8)

        self.spin_procs = QtWidgets.QSpinBox()
        self.spin_procs.setRange(1, 128)
        self.spin_procs.setValue(8)
        self.spin_procs.valueChanged.connect(self._update_preview)
        t3_layout.addRow("Processors / Threads:", self.spin_procs)

        self.spin_mem = QtWidgets.QSpinBox()
        self.spin_mem.setRange(1, 1024)
        self.spin_mem.setValue(16)
        self.spin_mem.setSuffix(" GB")
        self.spin_mem.valueChanged.connect(self._update_preview)
        t3_layout.addRow("RAM Memory:", self.spin_mem)

        self.txt_extra = QtWidgets.QLineEdit("")
        self.txt_extra.setPlaceholderText("e.g. Scf(MaxCycle=200,Tight) or TightSCF")
        self.txt_extra.textChanged.connect(self._update_preview)
        t3_layout.addRow("Extra Keywords:", self.txt_extra)

        tabs.addTab(tab3, "Resources & Advanced")

        # Tab 4: Executable Path & Custom Launch Route
        tab4 = QtWidgets.QWidget()
        t4_layout = QtWidgets.QVBoxLayout(tab4)
        t4_layout.setContentsMargins(10, 10, 10, 10)
        t4_layout.setSpacing(10)

        self.lbl_exec_status = QtWidgets.QLabel("")
        self.lbl_exec_status.setWordWrap(True)
        self.lbl_exec_status.setStyleSheet("padding: 8px; border-radius: 4px; font-size: 11px;")
        t4_layout.addWidget(self.lbl_exec_status)

        form_exec = QtWidgets.QFormLayout()
        form_exec.setSpacing(8)

        row_route = QtWidgets.QHBoxLayout()
        self.txt_custom_route = QtWidgets.QLineEdit()
        self.txt_custom_route.setPlaceholderText("Default detected path or custom command (e.g. /opt/orca/orca or mpirun -np 4 /opt/orca/orca)")
        self.txt_custom_route.textChanged.connect(self._on_custom_route_changed)
        row_route.addWidget(self.txt_custom_route, stretch=1)

        btn_browse_bin = QtWidgets.QPushButton("Browse Executable...")
        btn_browse_bin.clicked.connect(self._on_browse_executable)
        row_route.addWidget(btn_browse_bin)
        form_exec.addRow("Executable / Command Route:", row_route)

        t4_layout.addLayout(form_exec)

        lbl_route_hint = QtWidgets.QLabel(
            "<span style='color: #64748b; font-size: 11px;'>"
            "• Leave blank to use the automatically discovered system/local binary.<br>"
            "• You can provide a custom binary path or special launch wrapper (e.g., specific MPI path or wrapper script).<br>"
            "• The configured route is automatically tested and remembered for subsequent calculations."
            "</span>"
        )
        lbl_route_hint.setWordWrap(True)
        t4_layout.addWidget(lbl_route_hint)
        t4_layout.addStretch()

        tabs.addTab(tab4, "Executable & Route")

        layout.addWidget(tabs)

        # Preview Section
        grp_preview = QtWidgets.QGroupBox("Generated Input File Preview")
        prev_layout = QtWidgets.QVBoxLayout(grp_preview)

        self.text_preview = QtWidgets.QPlainTextEdit()
        self.text_preview.setFont(QtGui.QFont("Monospace", 10))
        self.text_preview.setReadOnly(False)  # User can make manual last-second edits
        prev_layout.addWidget(self.text_preview)
        layout.addWidget(grp_preview, stretch=1)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("📋 Copy to Clipboard")
        btn_copy.clicked.connect(self._on_copy_clipboard)
        btn_layout.addWidget(btn_copy)

        btn_save = QtWidgets.QPushButton("💾 Save Input File...")
        btn_save.setStyleSheet("background-color: #2563eb; font-weight: bold;")
        btn_save.clicked.connect(self._on_save_file)
        btn_layout.addWidget(btn_save)

        btn_run = QtWidgets.QPushButton("▶ Run Calculation...")
        btn_run.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 4px 10px;")
        btn_run.setToolTip("Run this calculation and monitor output live")
        btn_run.clicked.connect(self._on_run_calculation)
        btn_layout.addWidget(btn_run)

        btn_pipeline = QtWidgets.QPushButton("⚡ Workflow Pipeline...")
        btn_pipeline.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 4px 10px;")
        btn_pipeline.setToolTip("Launch chained multi-stage QM recipe (e.g., Opt → Freq True Minimum, TS → IRC, Opt → TD-DFT)")
        btn_pipeline.clicked.connect(self._on_launch_workflow_pipeline)
        btn_layout.addWidget(btn_pipeline)

        # Multi-Package Pipeline Stage Chaining (Option A)
        self.btn_chain_stage = QtWidgets.QPushButton("🔗 + Chain Stage (newBRORSEN)...")
        self.btn_chain_stage.setStyleSheet("background-color: #0d9488; color: white; font-weight: bold; padding: 4px 10px;")
        self.btn_chain_stage.setToolTip(
            "Multi-Package Calculation Pipeline: Save this calculation (Stage 1) and immediately\n"
            "chain into a Stage 2 newBRORSEN NEO / Positron calculation on the resulting geometry!"
        )
        self.btn_chain_stage.clicked.connect(self._on_chain_brorsen_stage)
        btn_layout.addWidget(self.btn_chain_stage)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    def _on_show_format_info(self):
        """Displays informative explanation for the selected QM package file format."""
        qm_name = self.combo_qm_code.currentText()
        gen_cls = QM_GENERATORS.get(qm_name)
        ext = gen_cls.file_extension if gen_cls else ".inp"
        supports_zmat = gen_cls.supports_zmatrix if gen_cls else False

        info_dict = {
            "ORCA": (
                "<b>ORCA 5.x / 6.x Input File (<code>.inp</code>)</b><br>"
                "<ul>"
                "  <li><b>Route Line:</b> Starts with exclamation mark <code>!</code> specifying method, basis set, and job flags (e.g. <code>! B3LYP def2-TZVP Opt Freq</code>).</li>"
                "  <li><b>Parallelization:</b> Configured via <code>%pal nprocs N end</code>.</li>"
                "  <li><b>Memory:</b> Configured via <code>%maxcore M</code> (memory in MB per core).</li>"
                "  <li><b>Coordinates:</b> Stored in <code>* xyz charge mult</code> or <code>* gzmat</code> blocks terminated by <code>*</code>.</li>"
                "</ul>"
            ),
            "PySCF": (
                "<b>PySCF Python Driver Script (<code>.py</code>)</b><br>"
                "<ul>"
                "  <li><b>Self-contained Python script:</b> Uses <code>pyscf.gto.Mole</code> and <code>pyscf.scf / dft / cc</code>.</li>"
                "  <li><b>Geometry:</b> Embedded directly as Python multiline string (Cartesian Ångströms).</li>"
                "  <li><b>Execution:</b> Can be run directly via <code>python script.py</code> without compilation.</li>"
                "</ul>"
            ),
            "Gaussian": (
                "<b>Gaussian 09 / 16 Input File (<code>.gjf / .com</code>)</b><br>"
                "<ul>"
                "  <li><b>Link 0 Commands:</b> <code>%chk=...</code>, <code>%nprocshared=...</code>, <code>%mem=...</code>.</li>"
                "  <li><b>Route Section:</b> Begins with <code>#p</code> specifying model chemistry (e.g. <code>#p B3LYP/6-31G(d) Opt Freq</code>).</li>"
                "  <li><b>Title & Charge:</b> Blank line separator, title card, blank line, charge and multiplicity, followed by coordinates and a mandatory trailing blank line.</li>"
                "  <li>Supports both Cartesian and full symbolic Z-matrices.</li>"
                "</ul>"
            ),
            "openLOWDIN": (
                "<b>openLOWDIN APMO Input File (<code>.dat / .inp</code>)</b><br>"
                "<ul>"
                "  <li><b>APMO Methodology:</b> Any Particle Molecular Orbital calculation package for nuclear quantum effects and positronic chemistry.</li>"
                "  <li><b>Structure:</b> Distinct electronic and nuclear coordinate blocks with custom basis assignments per quantum particle.</li>"
                "</ul>"
            ),
            "xTB": (
                "<b>xTB Semiempirical Input (<code>.coord / .inp</code>)</b><br>"
                "<ul>"
                "  <li><b>Engine:</b> Grimme's extended Tight-Binding quantum chemistry engine (GFN0-xTB, GFN1-xTB, GFN2-xTB).</li>"
                "  <li><b>Format:</b> Ultra-fast semiempirical screening, geometry optimizations, and Hessian frequency evaluations.</li>"
                "</ul>"
            ),
            "Q-Chem": (
                "<b>Q-Chem Input Deck (<code>.in</code>)</b><br>"
                "<ul>"
                "  <li><b>Structure:</b> Structured <code>$rem ... $end</code> directive block specifying <code>jobtype</code>, <code>method</code>, and <code>basis</code>.</li>"
                "  <li><b>Coordinates:</b> Contained within <code>$molecule ... $end</code> block with charge and multiplicity on the first line.</li>"
                "</ul>"
            ),
        }

        # Match key in info_dict
        matched_info = None
        for k, v in info_dict.items():
            if k.lower() in qm_name.lower():
                matched_info = v
                break

        if not matched_info:
            matched_info = (
                f"<b>{qm_name} Input Format (<code>{ext}</code>)</b><br>"
                "<p>Standard input deck for quantum chemistry calculation.</p>"
            )

        zmat_note = "<p>✅ <b>Z-Matrix Supported:</b> You can toggle between Cartesian and Internal Z-Matrix formats.</p>" if supports_zmat else "<p style='color: #64748b;'>ℹ️ <b>Cartesian Only:</b> This package natively uses Cartesian coordinates.</p>"

        QtWidgets.QMessageBox.information(
            self,
            f"Format Options — {qm_name}",
            f"{matched_info}<hr>{zmat_note}"
        )

    def _on_qm_code_changed(self, name: str):
        gen_cls = QM_GENERATORS.get(name)
        if gen_cls:
            self.lbl_ext.setText(gen_cls.file_extension)
            self.chk_zmatrix.setEnabled(gen_cls.supports_zmatrix)
            if not gen_cls.supports_zmatrix:
                self.chk_zmatrix.setChecked(False)

            # Update method dropdown if specific list defined
            cur_meth = self.combo_method.currentText()
            supp_meth = getattr(gen_cls, "supported_methods", None)
            if supp_meth:
                self.combo_method.blockSignals(True)
                self.combo_method.clear()
                self.combo_method.addItems(supp_meth)
                idx_m = self.combo_method.findText(cur_meth)
                if idx_m >= 0:
                    self.combo_method.setCurrentIndex(idx_m)
                elif "gaussian" in name.lower() and cur_meth == "wB97X-D":
                    idx_w = self.combo_method.findText("wB97XD")
                    if idx_w >= 0:
                        self.combo_method.setCurrentIndex(idx_w)
                self.combo_method.blockSignals(False)

            # Update basis set dropdown if specific list defined
            cur_bas = self.combo_basis.currentText()
            supp_bas = getattr(gen_cls, "supported_basis_sets", None)
            if supp_bas:
                self.combo_basis.blockSignals(True)
                self.combo_basis.clear()
                self.combo_basis.addItems(supp_bas)
                idx_b = self.combo_basis.findText(cur_bas)
                if idx_b >= 0:
                    self.combo_basis.setCurrentIndex(idx_b)
                elif "gaussian" in name.lower() and cur_bas == "def2-SVP":
                    idx_svp = self.combo_basis.findText("def2SVP")
                    if idx_svp >= 0:
                        self.combo_basis.setCurrentIndex(idx_svp)
                self.combo_basis.blockSignals(False)

        if "orca" in name.lower():
            # Default to 1 processor for ORCA so serial runs work out-of-the-box without OpenMPI ABI issues
            self.spin_procs.setValue(1)
            if self.combo_dispersion.currentText() == "D4":
                self.combo_dispersion.setCurrentText("D3BJ")
        elif "gaussian" in name.lower():
            if hasattr(self, "spin_mem") and self.spin_mem.value() < 2:
                self.spin_mem.setValue(2)
        self._update_route_ui()
        self._update_preview()

    def _update_route_ui(self):
        qm_name = self.combo_qm_code.currentText()
        from .qm_detect import detect_qm_executable, get_qm_route
        stored = get_qm_route(qm_name)
        self.txt_custom_route.blockSignals(True)
        self.txt_custom_route.setText(stored)
        self.txt_custom_route.blockSignals(False)

        info = detect_qm_executable(qm_name, custom_route=stored)
        self._apply_info_badge(info)

    def _on_custom_route_changed(self, text: str):
        qm_name = self.combo_qm_code.currentText()
        from .qm_detect import detect_qm_executable, save_qm_route
        save_qm_route(qm_name, text)
        info = detect_qm_executable(qm_name, custom_route=text)
        self._apply_info_badge(info)

    def _apply_info_badge(self, info):
        if info.found:
            self.lbl_exec_status.setText(info.message)
            self.lbl_exec_status.setStyleSheet(
                "background-color: #064e3b; color: #6ee7b7; border: 1px solid #059669; padding: 8px; border-radius: 4px; font-weight: bold;"
            )
        else:
            self.lbl_exec_status.setText(info.message)
            self.lbl_exec_status.setStyleSheet(
                "background-color: #450a0a; color: #fca5a5; border: 1px solid #dc2626; padding: 8px; border-radius: 4px;"
            )

    def apply_qc_doctor_remedy(self, remedy_keywords: Dict[str, str], title: str = ""):
        """Apply recommended keywords and settings from QC Doctor failure diagnostics."""
        if not remedy_keywords:
            return

        extra_items = []
        curr_extra = self.txt_extra.text().strip()
        if curr_extra:
            extra_items.append(curr_extra)

        for k, v in remedy_keywords.items():
            if k in ("extra", "scf", "opt", "solver", "lindep"):
                if v not in extra_items:
                    extra_items.append(v)
            elif k == "mem" and "GB" in v:
                m_num = "".join(ch for ch in v if ch.isdigit())
                if m_num:
                    self.spin_mem.setValue(int(m_num))

        self.txt_extra.setText(" ".join(extra_items))
        self._update_preview()
        QtWidgets.QToolTip.showText(
            self.mapToGlobal(QtCore.QPoint(50, 50)),
            f"Applied QC Doctor recovery parameters for: {title}",
            self,
        )

    def _on_browse_executable(self):
        qm_name = self.combo_qm_code.currentText()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Select Executable for {qm_name}", "", "Executable Files (*);;All Files (*)"
        )
        if path:
            self.txt_custom_route.setText(path)

    def _populate_scan_defaults_from_selection(self):
        from .zmatrix import calculate_bond_distance, calculate_angle, calculate_dihedral
        sel = self.selected_atoms
        k = len(sel)
        self.spin_scan_start.blockSignals(True)
        self.spin_scan_end.blockSignals(True)
        self.spin_scan_steps.blockSignals(True)
        if k == 2:
            a, b = sel
            d = calculate_bond_distance(self.coords[a], self.coords[b])
            self.lbl_scan_coord.setText(f"Bond {self.symbols[a]}{a+1}-{self.symbols[b]}{b+1} (Å):")
            self.spin_scan_start.setValue(round(d, 2))
            self.spin_scan_end.setValue(round(d + 1.0, 2))
            self.spin_scan_steps.setValue(11)
        elif k == 3:
            a, b, c = sel
            ang = calculate_angle(self.coords[a], self.coords[b], self.coords[c])
            self.lbl_scan_coord.setText(f"Angle {self.symbols[a]}{a+1}-{self.symbols[b]}{b+1}-{self.symbols[c]}{c+1} (°):")
            self.spin_scan_start.setValue(round(ang, 1))
            self.spin_scan_end.setValue(round(min(179.0, ang + 30.0), 1))
            self.spin_scan_steps.setValue(7)
        elif k == 4:
            a, b, c, d_idx = sel
            dih = calculate_dihedral(self.coords[a], self.coords[b], self.coords[c], self.coords[d_idx])
            self.lbl_scan_coord.setText(f"Dihedral {self.symbols[a]}{a+1}-{self.symbols[b]}{b+1}-{self.symbols[c]}{c+1}-{self.symbols[d_idx]}{d_idx+1} (°):")
            self.spin_scan_start.setValue(round(dih, 1))
            self.spin_scan_end.setValue(round(dih + 360.0, 1))
            self.spin_scan_steps.setValue(13)
        else:
            self.lbl_scan_coord.setText("Tip: Select 2, 3, or 4 atoms in 3D before opening to scan Bond/Angle/Dihedral")
            self.grp_pes_scan.setChecked(False)
            self.grp_pes_scan.setEnabled(False)
        self.spin_scan_start.blockSignals(False)
        self.spin_scan_end.blockSignals(False)
        self.spin_scan_steps.blockSignals(False)

    def _on_scan_toggled(self, checked: bool):
        if checked and "Opt" not in self.combo_job_type.currentText():
            self.combo_job_type.setCurrentText("Optimization (Opt)")
        self._update_preview()

    def _get_params(self) -> QMCalculationParameters:
        scan_spec = None
        if hasattr(self, "grp_pes_scan") and self.grp_pes_scan.isChecked() and len(self.selected_atoms) in (2, 3, 4):
            k = len(self.selected_atoms)
            stype = {2: "B", 3: "A", 4: "D"}[k]
            scan_spec = (
                stype,
                list(self.selected_atoms[:k]),
                float(self.spin_scan_start.value()),
                float(self.spin_scan_end.value()),
                int(self.spin_scan_steps.value()),
            )
        return QMCalculationParameters(
            job_type=self.combo_job_type.currentText(),
            method=self.combo_method.currentText(),
            basis=self.combo_basis.currentText(),
            charge=self.spin_charge.value(),
            multiplicity=self.spin_mult.value(),
            solvent=self.combo_solvent.currentText(),
            n_procs=self.spin_procs.value(),
            memory_gb=self.spin_mem.value(),
            dispersion=self.combo_dispersion.currentText(),
            title=self.txt_title.text().strip() or "AgyMolden calculation",
            use_zmatrix=self.chk_zmatrix.isChecked(),
            extra_keywords=self.txt_extra.text().strip(),
            frozen_atoms=self.frozen_atoms,
            scan_spec=scan_spec,
        )

    def _update_preview(self):
        qm_name = self.combo_qm_code.currentText()
        gen_cls = QM_GENERATORS.get(qm_name)
        if not gen_cls:
            return
        generator = gen_cls()
        params = self._get_params()
        try:
            content = generator.generate_input(
                self.symbols, self.coords, params, self.zatoms, self.variables
            )
            self.text_preview.setPlainText(content)
        except Exception as e:
            self.text_preview.setPlainText(f"Error generating input: {str(e)}")

    def _on_copy_clipboard(self):
        QtWidgets.QApplication.clipboard().setText(self.text_preview.toPlainText())
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Copied to clipboard!", self)

    def _on_save_file(self):
        qm_name = self.combo_qm_code.currentText()
        gen_cls = QM_GENERATORS.get(qm_name, GaussianGenerator)
        ext = gen_cls.file_extension
        filt = f"{qm_name} Input (*{ext});;All Files (*)"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, f"Save {qm_name} Input", f"calculation{ext}", filt)
        if path:
            with open(path, "w") as f:
                f.write(self.text_preview.toPlainText())
            QtWidgets.QMessageBox.information(self, "File Saved", f"Successfully saved input to:\n{path}")
            return path
        return None

    def _on_run_calculation(self):
        qm_name = self.combo_qm_code.currentText()
        gen_cls = QM_GENERATORS.get(qm_name, GaussianGenerator)
        ext = gen_cls.file_extension
        content = self.text_preview.toPlainText()

        from .qm_detect import detect_qm_executable, get_qm_route
        info = detect_qm_executable(qm_name)

        # Pre-Flight Calculation Validation (Prevents crashes before job execution)
        from .calc_runner.preflight import validate_qm_calculation
        from .calc_runner.dialog_preflight import PreflightValidationDialog
        params = self._get_params()
        report = validate_qm_calculation(
            code_key=qm_name,
            symbols=self.symbols,
            coords=self.coords,
            params=params,
            executable_info=info,
        )

        if report.issues:
            # If there are errors or warnings, present interactive inspection dialog
            pref_dlg = PreflightValidationDialog(self, report, qm_name=qm_name)
            if pref_dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return  # Blocked by user or fatal error

        if not info.found:
            msg = (
                f"<b>{info.display_name} executable was not detected on this system.</b><br><br>"
                f"{info.message}<br><br>"
                f"Would you like to review/set a custom executable path or launch command under "
                f"the <b>Executable & Route</b> tab before proceeding?"
            )
            reply = QtWidgets.QMessageBox.warning(
                self,
                f"{info.display_name} Not Found",
                msg,
                QtWidgets.QMessageBox.StandardButton.Open | QtWidgets.QMessageBox.StandardButton.Ignore | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Open,
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Open:
                # Switch to Tab 4 (Executable & Route)
                tabs = self.findChild(QtWidgets.QTabWidget)
                if tabs and tabs.count() >= 4:
                    tabs.setCurrentIndex(3)
                return
            elif reply == QtWidgets.QMessageBox.StandardButton.Cancel:
                return

        # Determine default working directory from main window / settings
        main_win = self.parent()
        default_dir = os.getcwd()
        if hasattr(main_win, "_last_dir") and main_win._last_dir():
            default_dir = main_win._last_dir()

        title_slug = self.txt_title.text().strip().replace(" ", "_") or "calculation"
        default_filename = f"{title_slug}{ext}"

        # Present the next confirmation screen with current name, folder tree, and rename option
        confirm_dlg = QMRunConfirmDialog(
            self,
            default_filename=default_filename,
            default_dir=default_dir,
            content=content,
            qm_name=qm_name,
        )
        if confirm_dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return  # Cancelled by user

        saved_path = confirm_dlg.get_saved_path()
        if not saved_path:
            return

        if hasattr(main_win, "_remember_file"):
            main_win._remember_file(saved_path)

        runner_type = info.code_key
        if hasattr(main_win, "_last_qm_method"):
            main_win._last_qm_method = self.combo_method.currentText()
            main_win._last_qm_basis = self.combo_basis.currentText()
        elif main_win is not None:
            setattr(main_win, "_last_qm_method", self.combo_method.currentText())
            setattr(main_win, "_last_qm_basis", self.combo_basis.currentText())

        if hasattr(main_win, "launch_calculation_job"):
            self.accept()
            main_win.launch_calculation_job(saved_path, runner_type=runner_type)
        else:
            from agymolden.calc_runner import JobMonitorDialog
            monitor = JobMonitorDialog(self)
            monitor.run_calculation(saved_path, runner_type=runner_type)

    def _on_chain_brorsen_stage(self):
        """Save this calculation input and chain into Stage 2 newBRORSEN setup dialog."""
        saved_path = self._on_save_file()
        if not saved_path:
            return
        main_win = self.parent()
        self.accept()
        if main_win and hasattr(main_win, "_on_open_brorsen_dialog"):
            main_win._on_open_brorsen_dialog()

    def _on_launch_workflow_pipeline(self):
        """Launch multi-stage chained calculation pipeline dialog (Opt → Freq, TS → IRC, Opt → TD-DFT)."""
        qm_name = self.combo_qm_code.currentText()
        params = self._get_params()
        from .calc_runner.dialog_pipeline import WorkflowPipelineDialog
        dlg = WorkflowPipelineDialog(
            parent=self,
            code_key=qm_name,
            symbols=self.symbols,
            coords=self.coords,
            base_params=params,
        )
        def _on_geom(syms, coords, label):
            main_win = self.parent()
            if main_win and hasattr(main_win, "set_molecule_from_zatoms"):
                from .zmatrix import cartesian_to_zmatrix
                zat = cartesian_to_zmatrix(syms, coords)
                main_win.set_molecule_from_zatoms(zat)
        dlg.pipeline_completed_with_geom.connect(_on_geom)
        dlg.exec()

