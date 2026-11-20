"""
Quick ORCA Calculation Dialog and Input Studio.
Provides 1-click execution for standard calculations (Optimization, IR/Vibrations, Orbitals, ESP)
with automatic coordinate generation, method/basis set selection, charge/multiplicity defaults,
and live 3D monitoring.
"""

import os
from typing import Optional, List, Tuple, Dict, Any
from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np

from agymolden.zmatrix import ZAtom


COURSE_PRESETS = [
    {
        "name": "⚡ Geometry Optimization (B3LYP / def2-SVP)",
        "route": "! B3LYP def2-SVP Opt",
        "method": "B3LYP",
        "basis": "def2-SVP",
        "job": "Opt",
        "desc": "Standard ground-state geometry optimization. Finds the minimum energy structure.",
        "icon": "📐"
    },
    {
        "name": "🎵 Vibrations & IR Spectrum (B3LYP / def2-SVP Freq)",
        "route": "! B3LYP def2-SVP Opt Freq",
        "method": "B3LYP",
        "basis": "def2-SVP",
        "job": "Opt Freq",
        "desc": "Optimizes geometry and computes harmonic vibrational frequencies and IR intensities.",
        "icon": "🎵"
    },
    {
        "name": "⚡ Transition State Search (OptTS + NumFreq)",
        "route": "! B3LYP def2-SVP OptTS NumFreq",
        "method": "B3LYP",
        "basis": "def2-SVP",
        "job": "OptTS NumFreq",
        "desc": "Optimizes to a first-order saddle point transition state and calculates frequencies to confirm 1 imaginary mode.",
        "icon": "⚡"
    },
    {
        "name": "🌐 Molecular Orbitals / HOMO-LUMO (B3LYP / def2-SVP SP)",
        "route": "! B3LYP def2-SVP SP",
        "method": "B3LYP",
        "basis": "def2-SVP",
        "job": "SP",
        "desc": "Single point energy to calculate molecular orbital energies, HOMO-LUMO gap, and dipole moment.",
        "icon": "🌐"
    },
    {
        "name": "🌌 Anions & Diffuse Orbitals (ma-def2-SVP)",
        "route": "! B3LYP ma-def2-SVP Opt",
        "method": "B3LYP",
        "basis": "ma-def2-SVP",
        "job": "Opt",
        "desc": "Minimally augmented diffuse basis set for anions, Rydberg states, and bound positron species.",
        "icon": "🌌"
    },
    {
        "name": "⚡ Fast Semi-empirical Tight-Binding (GFN2-xTB)",
        "route": "! GFN2-xTB Opt",
        "method": "GFN2-xTB",
        "basis": "",
        "job": "Opt",
        "desc": "Extremely fast geometry optimization for large molecules using Grimme's xTB method.",
        "icon": "⚡"
    },
    {
        "name": "🛠️ Custom Route Line",
        "route": "! B3LYP 6-31G* Opt",
        "method": "B3LYP",
        "basis": "6-31G*",
        "job": "Opt",
        "desc": "Specify your own custom ORCA keywords (e.g. ! wB97X-D3 def2-TZVP).",
        "icon": "✏️"
    }
]

ORCA_METHODS = [
    ("B3LYP (Hybrid GGA - Standard general purpose)", "B3LYP"),
    ("PBE0 (Hybrid GGA - Accurate structures & gaps)", "PBE0"),
    ("wB97X-D3 (Range-separated hybrid - Best non-covalent)", "wB97X-D3"),
    ("r2SCAN-3c (Composite meta-GGA - Fast, highly accurate)", "r2SCAN-3c"),
    ("B97-3c (Composite GGA - Excellent speed/accuracy)", "B97-3c"),
    ("M06-2X (Minnesota meta-hybrid - Thermochemistry & kinetics)", "M06-2X"),
    ("CAM-B3LYP (Coulomb-attenuated hybrid)", "CAM-B3LYP"),
    ("BP86 (Pure GGA - Organometallics & rapid opt)", "BP86"),
    ("HF (Hartree-Fock - Wavefunction baseline)", "HF"),
    ("MP2 (Second-order Møller-Plesset perturbation)", "MP2"),
    ("DLPNO-CCSD(T) (Coupled Cluster - Gold standard energetics)", "DLPNO-CCSD(T)"),
    ("GFN2-xTB (Semi-empirical Tight-Binding - Ultrafast)", "GFN2-xTB"),
]

ORCA_BASIS_SETS = [
    ("def2-SVP (Split valence polarized - Fast & standard)", "def2-SVP"),
    ("def2-TZVP (Triple-zeta polarized - Publication quality)", "def2-TZVP"),
    ("def2-TZVPP (Triple-zeta valence double-polarization)", "def2-TZVPP"),
    ("def2-QZVP (Quadruple-zeta - Benchmark accuracy)", "def2-QZVP"),
    ("ma-def2-SVP (Minimally augmented diffuse - Anions/Rydberg)", "ma-def2-SVP"),
    ("ma-def2-TZVP (Augmented triple-zeta diffuse)", "ma-def2-TZVP"),
    ("6-31G* (Pople double-zeta polarized)", "6-31G*"),
    ("6-31+G(d,p) (Pople diffuse & polarized)", "6-31+G(d,p)"),
    ("6-311++G(d,p) (Pople triple-zeta diffuse)", "6-311++G(d,p)"),
    ("cc-pVDZ (Dunning correlation consistent double-zeta)", "cc-pVDZ"),
    ("cc-pVTZ (Dunning correlation consistent triple-zeta)", "cc-pVTZ"),
]

ORCA_JOBS = [
    ("⚡ Geometry Optimization (! Opt)", "Opt"),
    ("🎵 Optimization + Frequencies (! Opt Freq)", "Opt Freq"),
    ("🌐 Single Point Energy (! SP)", "SP"),
    ("⚡ Transition State Search (! OptTS NumFreq)", "OptTS NumFreq"),
    ("🎵 Harmonic Frequencies Only (! Freq)", "Freq"),
]


class QuickOrcaDialog(QtWidgets.QDialog):
    """Rich interactive modal dialog for setting up and running ORCA calculations."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], zatoms: List[ZAtom], coords: Optional[np.ndarray] = None):
        super().__init__(parent if isinstance(parent, QtWidgets.QWidget) else None)
        self.setWindowTitle("🚀 ORCA Quantum Chemistry Studio & Input Generator")
        self.resize(680, 620)
        self.zatoms = zatoms
        self.coords = coords
        self._syncing_route = False

        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header bar with switch to full studio
        hdr_layout = QtWidgets.QHBoxLayout()
        info_box = QtWidgets.QLabel(
            "<b>ORCA Quantum Chemistry Calculation Studio</b><br>"
            "<span style='color: #64748b; font-size: 11px;'>"
            "Configure theory, basis set, solvation, properties (ESP/NMR), and resources. Live 3D monitoring included.</span>"
        )
        info_box.setWordWrap(True)
        hdr_layout.addWidget(info_box, 1)

        btn_full_studio = QtWidgets.QPushButton("🎛️ Full Multi-Code QM Studio...")
        btn_full_studio.setStyleSheet("background-color: #334155; color: white; padding: 5px 10px; font-size: 11px;")
        btn_full_studio.setToolTip("Switch to the advanced multi-tab GaussView-style QM generator (Gaussian, ORCA, PySCF, xTB)")
        btn_full_studio.clicked.connect(self._on_switch_to_full_studio)
        hdr_layout.addWidget(btn_full_studio, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addLayout(hdr_layout)

        # Tabs for Options vs Deck Preview vs Course Presets
        self.tabs = QtWidgets.QTabWidget()

        # -------------------------------------------------------------
        # Tab 1: Comprehensive Options
        # -------------------------------------------------------------
        tab_opts = QtWidgets.QWidget()
        opts_layout = QtWidgets.QVBoxLayout(tab_opts)
        opts_layout.setSpacing(10)

        # 1. Theory & Basis Section
        grp_theory = QtWidgets.QGroupBox("1. Method, Basis Set & Job Type")
        theory_grid = QtWidgets.QGridLayout(grp_theory)
        theory_grid.setVerticalSpacing(6)
        theory_grid.setHorizontalSpacing(10)

        theory_grid.addWidget(QtWidgets.QLabel("Job Type:"), 0, 0)
        self.combo_job = QtWidgets.QComboBox()
        for label, val in ORCA_JOBS:
            self.combo_job.addItem(label, val)
        self.combo_job.currentIndexChanged.connect(self._on_combo_options_changed)
        theory_grid.addWidget(self.combo_job, 0, 1)

        theory_grid.addWidget(QtWidgets.QLabel("Functional / Method:"), 1, 0)
        self.combo_method = QtWidgets.QComboBox()
        for label, val in ORCA_METHODS:
            self.combo_method.addItem(label, val)
        self.combo_method.currentIndexChanged.connect(self._on_combo_options_changed)
        theory_grid.addWidget(self.combo_method, 1, 1)

        theory_grid.addWidget(QtWidgets.QLabel("Basis Set:"), 2, 0)
        self.combo_basis = QtWidgets.QComboBox()
        for label, val in ORCA_BASIS_SETS:
            self.combo_basis.addItem(label, val)
        self.combo_basis.currentIndexChanged.connect(self._on_combo_options_changed)
        theory_grid.addWidget(self.combo_basis, 2, 1)

        theory_grid.addWidget(QtWidgets.QLabel("Grid Quality:"), 3, 0)
        self.combo_grid = QtWidgets.QComboBox()
        self.combo_grid.addItem("DefGrid2 (Standard Default)", "DefGrid2")
        self.combo_grid.addItem("DefGrid3 (Tight Grid for Freq/Opt)", "DefGrid3")
        self.combo_grid.currentIndexChanged.connect(self._on_combo_options_changed)
        theory_grid.addWidget(self.combo_grid, 3, 1)

        self.lbl_compat_info = QtWidgets.QLabel()
        self.lbl_compat_info.setStyleSheet("color: #38bdf8; font-size: 11px; padding: 2px 4px;")
        self.lbl_compat_info.setVisible(False)
        self.lbl_compat_info.setWordWrap(True)
        theory_grid.addWidget(self.lbl_compat_info, 4, 0, 1, 2)

        opts_layout.addWidget(grp_theory)

        # 2. Environment (Solvent & Dispersion)
        grp_env = QtWidgets.QGroupBox("2. Environment & Dispersion")
        env_grid = QtWidgets.QGridLayout(grp_env)
        env_grid.setVerticalSpacing(6)

        env_grid.addWidget(QtWidgets.QLabel("Solvent (CPCM):"), 0, 0)
        self.combo_solvent = QtWidgets.QComboBox()
        self.combo_solvent.addItems([
            "None (Gas Phase)", "Water (CPCM)", "Dichloromethane (CPCM)",
            "Acetonitrile (CPCM)", "Chloroform (CPCM)", "DMSO (CPCM)",
            "THF (CPCM)", "Toluene (CPCM)", "Ethanol (CPCM)", "Methanol (CPCM)"
        ])
        self.combo_solvent.currentIndexChanged.connect(self._on_combo_options_changed)
        env_grid.addWidget(self.combo_solvent, 0, 1)

        env_grid.addWidget(QtWidgets.QLabel("Dispersion Correction:"), 1, 0)
        self.combo_dispersion = QtWidgets.QComboBox()
        self.combo_dispersion.addItems(["None / In Route", "D3BJ", "D4"])
        self.combo_dispersion.currentIndexChanged.connect(self._on_combo_options_changed)
        env_grid.addWidget(self.combo_dispersion, 1, 1)

        opts_layout.addWidget(grp_env)

        # 3. Properties to Calculate
        grp_props = QtWidgets.QGroupBox("3. Properties & Wavefunction Output")
        props_layout = QtWidgets.QGridLayout(grp_props)
        props_layout.setVerticalSpacing(4)

        self.chk_molden = QtWidgets.QCheckBox("📁 Auto-Generate Molden File (.molden.input / .molden)")
        self.chk_molden.setToolTip("Automatically run orca_2mkl upon completion to generate .molden.input / .molden file for 3D orbital, ESP, and property analysis")
        self.chk_molden.setChecked(True)
        props_layout.addWidget(self.chk_molden, 0, 0)

        self.chk_nmr = QtWidgets.QCheckBox("🧲 NMR Chemical Shielding Tensors (! NMR)")
        self.chk_nmr.toggled.connect(self._on_combo_options_changed)
        props_layout.addWidget(self.chk_nmr, 0, 1)

        self.chk_polar = QtWidgets.QCheckBox("⚡ Molecular Polarizability (! Polar)")
        self.chk_polar.toggled.connect(self._on_combo_options_changed)
        props_layout.addWidget(self.chk_polar, 1, 0)

        self.chk_mo_cubes = QtWidgets.QCheckBox("🌐 Export Frontier Orbitals (HOMO & LUMO Cube Files)")
        self.chk_mo_cubes.setToolTip("Appends %plots Format Cube; mo(\"homo.cube\", ...); mo(\"lumo.cube\", ...); end to evaluate 3D MO cube grids")
        self.chk_mo_cubes.setChecked(False)
        self.chk_mo_cubes.toggled.connect(self._update_deck_preview)
        props_layout.addWidget(self.chk_mo_cubes, 1, 1)

        self.chk_esp = QtWidgets.QCheckBox("⚡ Calculate Electrostatic Potential (ESP)")
        self.chk_esp.setToolTip("Request ESP analysis via Molden orca_2mkl conversion and companion grid generation")
        self.chk_esp.setChecked(False)
        self.chk_esp.toggled.connect(self._update_deck_preview)
        props_layout.addWidget(self.chk_esp, 2, 0)

        opts_layout.addWidget(grp_props)

        # 4. Molecule State & Resources
        grp_state = QtWidgets.QGroupBox("4. Electronic State & Hardware Resources")
        state_grid = QtWidgets.QGridLayout(grp_state)
        state_grid.setVerticalSpacing(6)

        from .chemistry_checker import detect_cpu_cores, predict_default_multiplicity
        from .calc_runner.orca_runner import OrcaRunner

        init_symbols = [getattr(a, "symbol", "C") for a in self.zatoms]
        n_cpus = detect_cpu_cores()
        has_mpi = bool(OrcaRunner._resolve_orca_mpi_env()) or bool(os.environ.get("OMPI_COMM_WORLD_SIZE"))
        if not has_mpi:
            import shutil
            has_mpi = bool(shutil.which("mpirun"))

        row_charge = QtWidgets.QHBoxLayout()
        self.spn_charge = QtWidgets.QSpinBox()
        self.spn_charge.setRange(-10, 10)
        self.spn_charge.setValue(0)
        row_charge.addWidget(QtWidgets.QLabel("Net Charge:"))
        row_charge.addWidget(self.spn_charge)

        self.spn_mult = QtWidgets.QSpinBox()
        self.spn_mult.setRange(1, 10)
        self.spn_mult.setValue(predict_default_multiplicity(init_symbols, 0))
        row_charge.addWidget(QtWidgets.QLabel("Multiplicity:"))
        row_charge.addWidget(self.spn_mult)
        row_charge.addStretch()
        state_grid.addLayout(row_charge, 0, 0, 1, 2)

        self.lbl_parity_warning = QtWidgets.QLabel("")
        self.lbl_parity_warning.setWordWrap(True)
        state_grid.addWidget(self.lbl_parity_warning, 1, 0, 1, 2)

        self.spn_charge.valueChanged.connect(self._on_charge_changed)
        self.spn_mult.valueChanged.connect(self._validate_charge_multiplicity)

        row_hw = QtWidgets.QHBoxLayout()
        self.spn_cores = QtWidgets.QSpinBox()
        self.spn_cores.setRange(1, max(64, n_cpus))
        self.spn_cores.setValue(min(4, n_cpus) if has_mpi else 1)
        row_hw.addWidget(QtWidgets.QLabel("Cores (%pal):"))
        row_hw.addWidget(self.spn_cores)

        self.btn_use_all_cores = QtWidgets.QPushButton(f"All ({n_cpus})")
        self.btn_use_all_cores.setToolTip(f"Set Cores (%pal) to all {n_cpus} detected CPU cores")
        self.btn_use_all_cores.setStyleSheet("padding: 2px 6px; font-size: 11px;")
        self.btn_use_all_cores.clicked.connect(lambda: self.spn_cores.setValue(n_cpus))
        row_hw.addWidget(self.btn_use_all_cores)

        self.lbl_detected_cores = QtWidgets.QLabel(f"({n_cpus} detected)")
        self.lbl_detected_cores.setStyleSheet("color: #64748b; font-size: 11px;")
        row_hw.addWidget(self.lbl_detected_cores)

        self.spn_mem = QtWidgets.QSpinBox()
        self.spn_mem.setRange(500, 32000)
        self.spn_mem.setSingleStep(500)
        self.spn_mem.setValue(2000)
        self.spn_mem.setSuffix(" MB")
        row_hw.addWidget(QtWidgets.QLabel("RAM per core:"))
        row_hw.addWidget(self.spn_mem)
        row_hw.addStretch()
        state_grid.addLayout(row_hw, 2, 0, 1, 2)

        self.spn_cores.valueChanged.connect(self._update_deck_preview)
        self.spn_mem.valueChanged.connect(self._update_deck_preview)

        opts_layout.addWidget(grp_state)

        # Route Line
        row_route = QtWidgets.QHBoxLayout()
        row_route.addWidget(QtWidgets.QLabel("<b>Active Route:</b>"))
        self.txt_route = QtWidgets.QLineEdit("! B3LYP def2-SVP Opt")
        self.txt_route.textChanged.connect(self._on_route_text_edited)
        row_route.addWidget(self.txt_route, 1)
        opts_layout.addLayout(row_route)

        opts_layout.addStretch()
        self.tabs.addTab(tab_opts, "⚡ Calculation Setup")

        # -------------------------------------------------------------
        # Tab 2: Deck Preview
        # -------------------------------------------------------------
        tab_preview = QtWidgets.QWidget()
        prev_layout = QtWidgets.QVBoxLayout(tab_preview)
        prev_layout.setSpacing(8)

        lbl_prev_info = QtWidgets.QLabel("Full ORCA input deck (*.inp) generated from your current configuration:")
        lbl_prev_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        prev_layout.addWidget(lbl_prev_info)

        self.txt_deck_preview = QtWidgets.QPlainTextEdit()
        self.txt_deck_preview.setFont(QtGui.QFont("Monospace", 10))
        self.txt_deck_preview.setStyleSheet("background-color: #0f172a; color: #f8fafc;")
        prev_layout.addWidget(self.txt_deck_preview, 1)

        row_prev_btns = QtWidgets.QHBoxLayout()
        btn_copy_deck = QtWidgets.QPushButton("📋 Copy Deck to Clipboard")
        btn_copy_deck.clicked.connect(self._on_copy_deck)
        row_prev_btns.addWidget(btn_copy_deck)
        row_prev_btns.addStretch()
        prev_layout.addLayout(row_prev_btns)

        self.tabs.addTab(tab_preview, "📄 Live Deck Preview")

        # -------------------------------------------------------------
        # Tab 3: Quick Presets
        # -------------------------------------------------------------
        tab_presets = QtWidgets.QWidget()
        presets_layout = QtWidgets.QVBoxLayout(tab_presets)
        presets_layout.setSpacing(8)

        lbl_pr_info = QtWidgets.QLabel("1-Click Course Lab Presets (standard student exercises):")
        lbl_pr_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        presets_layout.addWidget(lbl_pr_info)

        self.combo_preset = QtWidgets.QComboBox()
        for p in COURSE_PRESETS:
            self.combo_preset.addItem(f"{p['icon']} {p['name']}", p)
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        presets_layout.addWidget(self.combo_preset)

        self.lbl_preset_desc = QtWidgets.QLabel(COURSE_PRESETS[0]["desc"])
        self.lbl_preset_desc.setStyleSheet("color: #475569; font-style: italic; font-size: 12px; padding: 6px;")
        self.lbl_preset_desc.setWordWrap(True)
        presets_layout.addWidget(self.lbl_preset_desc)
        presets_layout.addStretch()

        self.tabs.addTab(tab_presets, "📚 Course Presets")

        layout.addWidget(self.tabs, 1)

        # Bottom: Working Directory, File Name & Action Buttons
        bottom_box = QtWidgets.QGroupBox("Target File & Launch")
        bot_layout = QtWidgets.QFormLayout(bottom_box)
        bot_layout.setSpacing(6)

        self.txt_filename = QtWidgets.QLineEdit("orca_calc.inp")
        bot_layout.addRow("Input File:", self.txt_filename)

        dir_row = QtWidgets.QHBoxLayout()
        self.txt_dir = QtWidgets.QLineEdit(os.path.abspath("."))
        dir_row.addWidget(self.txt_dir, stretch=1)
        btn_browse = QtWidgets.QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse_dir)
        dir_row.addWidget(btn_browse)
        bot_layout.addRow("Directory:", dir_row)

        # ORCA Executable / Route row (lets users paste, browse, or search anywhere on Linux/macOS/Win)
        from agymolden.qm_detect import detect_qm_executable, get_qm_route
        orca_init_info = detect_qm_executable("orca")
        orca_row = QtWidgets.QHBoxLayout()
        self.txt_orca_path = QtWidgets.QLineEdit(get_qm_route("orca"))
        self.txt_orca_path.setPlaceholderText(
            orca_init_info.resolved_path if orca_init_info.found else "Auto-detect failed — paste path, Browse..., or click Search"
        )
        self.txt_orca_path.setToolTip(
            "Path to the 'orca' executable (or its folder, or full launch command). Saved automatically."
        )
        self.txt_orca_path.textChanged.connect(self._on_orca_path_changed)
        orca_row.addWidget(self.txt_orca_path, stretch=1)

        btn_orca_browse = QtWidgets.QPushButton("Browse...")
        btn_orca_browse.setToolTip("Browse for the ORCA executable binary")
        btn_orca_browse.clicked.connect(self._on_browse_orca)
        orca_row.addWidget(btn_orca_browse)

        btn_orca_search = QtWidgets.QPushButton("🔍 Search")
        btn_orca_search.setToolTip("Search home directory, /opt, /usr/local, and system folders for ORCA")
        btn_orca_search.clicked.connect(self._on_search_orca)
        orca_row.addWidget(btn_orca_search)
        bot_layout.addRow("ORCA Executable:", orca_row)

        self.lbl_orca_status = QtWidgets.QLabel("")
        self.lbl_orca_status.setWordWrap(True)
        self.lbl_orca_status.setStyleSheet("font-size: 11px;")
        bot_layout.addRow("", self.lbl_orca_status)
        self._refresh_orca_status(orca_init_info)

        layout.addWidget(bottom_box)

        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch()

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        self.btn_run = QtWidgets.QPushButton("🚀 Run ORCA Now")
        self.btn_run.setStyleSheet(
            "background-color: #2563eb; color: white; font-weight: bold; font-size: 13px; padding: 8px 20px; border-radius: 4px;"
        )
        self.btn_run.clicked.connect(self._on_run_clicked)
        btn_box.addWidget(self.btn_run)

        layout.addLayout(btn_box)

        # Initial composition
        self._compose_route_from_controls()
        self._validate_charge_multiplicity()
        self._update_deck_preview()

    def _on_switch_to_full_studio(self):
        """Switch to full multi-code QM Calculation Setup dialog."""
        self.reject()
        parent = self.parent()
        if parent and hasattr(parent, "_on_open_qm_dialog"):
            parent._on_open_qm_dialog(target_code="orca")

    def _compose_route_from_controls(self):
        """Construct the ORCA route line string from current combo and checkbox controls."""
        if self._syncing_route:
            return
        self._syncing_route = True

        tokens = ["!"]
        method = self.combo_method.currentData() or "B3LYP"
        basis = self.combo_basis.currentData() or "def2-SVP"
        job = self.combo_job.currentData() or "Opt"
        grid = self.combo_grid.currentData() or "DefGrid2"
        disp = self.combo_dispersion.currentText()
        solv = self.combo_solvent.currentText()

        # xTB and composite methods have fixed basis & dispersion
        is_composite = method in ("GFN2-xTB", "r2SCAN-3c", "B97-3c")
        self.combo_basis.setEnabled(not is_composite)
        self.combo_dispersion.setEnabled(not is_composite)
        if hasattr(self, "lbl_compat_info"):
            if is_composite:
                self.lbl_compat_info.setText(f"ℹ️ {method} is a composite/tight-binding method with intrinsic basis set & dispersion.")
                self.lbl_compat_info.setVisible(True)
            elif method == "DLPNO-CCSD(T)" and not str(basis).startswith("def2"):
                self.lbl_compat_info.setText("💡 Tip: DLPNO-CCSD(T) performs best with Karlsruhe def2 basis sets (def2-TZVP / def2-QZVP).")
                self.lbl_compat_info.setVisible(True)
            else:
                self.lbl_compat_info.setVisible(False)

        if is_composite:
            tokens.append(method)
        else:
            tokens.append(method)
            if basis:
                tokens.append(basis)
            if disp in ("D3BJ", "D4"):
                tokens.append(disp)

        if job:
            tokens.append(job)

        if grid and grid != "DefGrid2":
            tokens.append(grid)

        if self.chk_nmr.isChecked():
            tokens.append("NMR")

        if self.chk_polar.isChecked():
            tokens.append("Polar")

        if solv != "None (Gas Phase)":
            solv_name = solv.split()[0]
            if solv_name == "Dichloromethane":
                solv_name = "CH2Cl2"
            tokens.append(f"CPCM({solv_name})")

        route_str = " ".join(tokens)
        self.txt_route.setText(route_str)
        self._syncing_route = False

        self._update_deck_preview()

    def _on_combo_options_changed(self, _idx: int = 0):
        self._compose_route_from_controls()

    def _on_route_text_edited(self, text: str):
        self._validate_charge_multiplicity()
        self._update_deck_preview()

    def _generate_deck_content(self) -> str:
        return self.get_orca_input_text()

    def get_orca_input_text(self) -> str:
        """Assemble the complete ORCA input deck as a string."""
        route_line = self.txt_route.text().strip()
        if not route_line.startswith("!"):
            route_line = "! " + route_line

        lines = [route_line]

        nprocs = self.spn_cores.value()
        if nprocs > 1:
            lines.append(f"%pal nprocs {nprocs} end")

        lines.append(f"%maxcore {self.spn_mem.value()}")

        # Check for MO cubes (valid ORCA %plots block syntax)
        if hasattr(self, "chk_mo_cubes") and self.chk_mo_cubes.isChecked():
            from .chemistry_checker import check_calculation_sanity
            symbols = [getattr(a, "symbol", "C") for a in self.zatoms]
            c_check = self.coords if self.coords is not None and len(self.coords) == len(self.zatoms) else np.zeros((len(symbols), 3))
            res_s = check_calculation_sanity(symbols, coords=c_check, charge=self.spn_charge.value(), multiplicity=self.spn_mult.value())
            n_elec = res_s.get("info", {}).get("n_electrons", 10)
            homo_idx = max(0, (n_elec // 2) - 1)
            lumo_idx = homo_idx + 1

            lines.append("%plots")
            lines.append("  Format Cube")
            lines.append("  dim1 40")
            lines.append("  dim2 40")
            lines.append("  dim3 40")
            lines.append(f'  mo("homo.cube", {homo_idx}, 0);')
            lines.append(f'  mo("lumo.cube", {lumo_idx}, 0);')
            lines.append("end")

        lines.append("")
        lines.append(f"* xyz {self.spn_charge.value()} {self.spn_mult.value()}")

        if self.coords is not None and len(self.coords) == len(self.zatoms):
            c_use = self.coords
        elif self.zatoms:
            from .zmatrix import zmatrix_to_cartesian
            try:
                c_use = zmatrix_to_cartesian(self.zatoms)
            except Exception:
                c_use = np.zeros((len(self.zatoms), 3))
        else:
            c_use = np.zeros((0, 3))

        for atom, pos in zip(self.zatoms, c_use):
            lines.append(f"  {atom.symbol:<2}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}")
        lines.append("*\n")

        return "\n".join(lines)

    def _update_deck_preview(self):
        """Update live text in Tab 2."""
        if hasattr(self, "txt_deck_preview"):
            self.txt_deck_preview.setPlainText(self._generate_deck_content())

    def _on_copy_deck(self):
        QtWidgets.QApplication.clipboard().setText(self._generate_deck_content())
        self.statusBar = getattr(self, "statusBar", None)

    def _on_preset_changed(self, idx: int):
        data = self.combo_preset.currentData()
        if data:
            self._syncing_route = True
            self.txt_route.setText(data["route"])
            self.lbl_preset_desc.setText(data["desc"])

            # Sync combo controls if known
            method = data.get("method")
            if method:
                for i in range(self.combo_method.count()):
                    if self.combo_method.itemData(i) == method:
                        self.combo_method.blockSignals(True)
                        self.combo_method.setCurrentIndex(i)
                        self.combo_method.blockSignals(False)
                        break

            basis = data.get("basis")
            if basis is not None:
                for i in range(self.combo_basis.count()):
                    if self.combo_basis.itemData(i) == basis:
                        self.combo_basis.blockSignals(True)
                        self.combo_basis.setCurrentIndex(i)
                        self.combo_basis.blockSignals(False)
                        break

            job = data.get("job")
            if job:
                for i in range(self.combo_job.count()):
                    if self.combo_job.itemData(i) == job:
                        self.combo_job.blockSignals(True)
                        self.combo_job.setCurrentIndex(i)
                        self.combo_job.blockSignals(False)
                        break

            self._syncing_route = False
            self._validate_charge_multiplicity()
            self._update_deck_preview()

    def _run_preflight_sanity(self) -> dict:
        """Run chemistry_checker pre-flight sanity checks."""
        from .chemistry_checker import check_calculation_sanity
        symbols = [getattr(a, "symbol", "C") for a in self.zatoms]
        coords_arr = self.coords
        if coords_arr is None or len(coords_arr) != len(symbols):
            from .zmatrix import zmatrix_to_cartesian
            try:
                coords_arr = zmatrix_to_cartesian(self.zatoms)
            except Exception:
                coords_arr = np.zeros((len(symbols), 3))
        route_str = self.txt_route.text().strip() if hasattr(self, "txt_route") else ""
        return check_calculation_sanity(
            symbols=symbols,
            coords=coords_arr,
            charge=self.spn_charge.value(),
            multiplicity=self.spn_mult.value(),
            basis_set=route_str,
            qm_package="ORCA",
        )

    def _on_charge_changed(self):
        """Auto-adjust multiplicity parity when net charge changes, then re-validate."""
        if self.zatoms:
            from .chemistry_checker import predict_default_multiplicity
            symbols = [getattr(a, "symbol", "C") for a in self.zatoms]
            pred_mult = predict_default_multiplicity(symbols, self.spn_charge.value())
            if ((self.spn_mult.value() - 1) % 2) != ((pred_mult - 1) % 2):
                self.spn_mult.blockSignals(True)
                self.spn_mult.setValue(pred_mult)
                self.spn_mult.blockSignals(False)
        self._validate_charge_multiplicity()

    def _validate_charge_multiplicity(self):
        """Validates electron count parity, interatomic clashes, and basis suitability."""
        if not self.zatoms:
            self.lbl_parity_warning.setText("<span style='color: #64748b;'>No molecule loaded</span>")
            return

        res = self._run_preflight_sanity()
        info = res["info"]
        num_electrons = info["n_electrons"]
        mult = info["multiplicity"]

        if res["errors"]:
            err_msg = res["errors"][0]
            prefix = "⚠️ Spin mismatch / Pre-flight Error:" if "Spin state" in err_msg else "⚠️ Pre-flight Error:"
            self.lbl_parity_warning.setText(
                f"<span style='color: #dc2626; font-weight: bold;'>{prefix}</span> "
                f"<span style='color: #dc2626;'>{err_msg}</span>"
            )
        elif res["warnings"]:
            warn_msg = res["warnings"][0]
            self.lbl_parity_warning.setText(
                f"<span style='color: #d97706; font-weight: bold;'>⚠️ Advisory ({num_electrons} e⁻, mult={mult}):</span> "
                f"<span style='color: #d97706;'>{warn_msg}</span>"
            )
        else:
            self.lbl_parity_warning.setText(
                f"<span style='color: #16a34a; font-weight: bold;'>✓ Valid state</span> "
                f"<span style='color: #16a34a;'>({info['formula']}, {num_electrons} e⁻, mult={mult})</span>"
            )

    def _on_browse_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Working Directory", self.txt_dir.text())
        if d:
            self.txt_dir.setText(d)

    def _refresh_orca_status(self, info=None):
        if info is None:
            from agymolden.qm_detect import detect_qm_executable
            info = detect_qm_executable("orca", custom_route=self.txt_orca_path.text().strip() or None)
        if info.found:
            self.lbl_orca_status.setText(f"<span style='color: #16a34a; font-weight: bold;'>{info.message}</span>")
        else:
            self.lbl_orca_status.setText(f"<span style='color: #d97706;'>{info.message}</span>")

    def _on_orca_path_changed(self, text: str):
        from agymolden.qm_detect import detect_qm_executable, save_qm_route
        save_qm_route("orca", text.strip())
        info = detect_qm_executable("orca", custom_route=text.strip() or None)
        self._refresh_orca_status(info)
        if self.parent() is not None and hasattr(self.parent(), "_update_env_badge"):
            self.parent()._update_env_badge()

    def _on_browse_orca(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ORCA Executable", os.path.expanduser("~"), "All Files (*)"
        )
        if p:
            self.txt_orca_path.setText(p)

    def _on_search_orca(self):
        from agymolden.qm_detect import search_qm_executable_anywhere
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            matches = search_qm_executable_anywhere("orca", timeout_sec=5.0)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if matches:
            if len(matches) == 1:
                self.txt_orca_path.setText(matches[0])
            else:
                chosen, ok = QtWidgets.QInputDialog.getItem(
                    self,
                    "ORCA Executables Found",
                    "Multiple ORCA executables were found. Select one:",
                    matches,
                    0,
                    False,
                )
                if ok and chosen:
                    self.txt_orca_path.setText(chosen)
        else:
            QtWidgets.QMessageBox.information(
                self,
                "ORCA Search",
                "Could not automatically locate an 'orca' executable in your home directory, /opt, or /usr/local.\n\n"
                "Please paste the full path to your ORCA binary (or its folder) into the ORCA Executable field, or click 'Browse...'.",
            )

    def _on_run_clicked(self):
        if not self.zatoms:
            QtWidgets.QMessageBox.warning(self, "No Molecule", "Please build or load a molecule first before running ORCA.")
            return

        from agymolden.qm_detect import detect_qm_executable, get_qm_route
        info = detect_qm_executable("orca", custom_route=self.txt_orca_path.text().strip() or None)
        if not info.found:
            msg = (
                "<b>ORCA executable was not detected on your system.</b><br><br>"
                "To run this calculation locally, enter or browse to your ORCA executable in the "
                "<b>ORCA Executable</b> field, click <b>🔍 Search</b>, or configure it in QM Engine settings.<br><br>"
                "<b>Options:</b><br>"
                "• Click <b>Configure Route</b> to browse/search for your ORCA executable.<br>"
                "• Click <b>Save Input Only</b> to write the .inp deck to disk without running locally (e.g. to transfer to a supercomputer/cluster)."
            )
            dlg_box = QtWidgets.QMessageBox(self)
            dlg_box.setWindowTitle("ORCA Not Detected")
            dlg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            dlg_box.setText(msg)
            btn_cfg = dlg_box.addButton("⚙️ Configure Route...", QtWidgets.QMessageBox.ButtonRole.ActionRole)
            btn_save = dlg_box.addButton("💾 Save Input Only", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            btn_cancel = dlg_box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
            dlg_box.exec()

            if dlg_box.clickedButton() == btn_cfg:
                from agymolden.qm_detect import QMEnginesStatusDialog
                QMEnginesStatusDialog.show_dialog(self)
                self.txt_orca_path.setText(get_qm_route("orca"))
                info = detect_qm_executable("orca", custom_route=self.txt_orca_path.text().strip() or None)
                self._refresh_orca_status(info)
                if not info.found:
                    return
            elif dlg_box.clickedButton() == btn_cancel:
                return

        sanity = self._run_preflight_sanity()
        issues = list(sanity.get("errors", [])) + list(sanity.get("warnings", []))
        if issues:
            ans = QtWidgets.QMessageBox.warning(
                self,
                "Pre-Flight Chemistry Sanity Check",
                "The following potential issues were detected in your ORCA setup:\n\n• "
                + "\n• ".join(issues)
                + "\n\nDo you want to launch the ORCA calculation anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        fname = self.txt_filename.text().strip()
        if not fname.endswith(".inp"):
            fname += ".inp"

        target_dir = self.txt_dir.text().strip() or os.path.abspath(".")
        target_path = os.path.join(target_dir, fname)

        content = self._generate_deck_content()
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Could not write ORCA input file:\n{e}")
            return

        self.saved_input_path = target_path
        self.accept()
