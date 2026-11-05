"""
UI Builder Mixin for the Vibrations, Point Group Symmetry, and Molecular Orbitals Inspector Tabs.

Keeps all widget attributes directly on `self` (MainWindow) so existing mixins
(`mw_calculations.py`, `mw_symmetry.py`, `mw_file_io.py`) work with zero breaking changes.
"""

from PyQt6 import QtCore, QtWidgets
from .spectrum_widget import IRSpectrumWidget, UVVisSpectrumWidget, NMRSpectrumWidget, DOSSpectrumWidget
from .mo_ladder_widget import MOEnergyLadderWidget


class UIInspectorTabsMixin:
    def _init_vibrations_tab(self):
        tab_vibrations = QtWidgets.QWidget()
        tab_vib_layout = QtWidgets.QVBoxLayout(tab_vibrations)
        tab_vib_layout.setSpacing(6)

        # Header with calculation triggers
        vib_ctrl_top = QtWidgets.QHBoxLayout()
        self.btn_calc_vibrations = QtWidgets.QPushButton("⚡ Calculate Frequencies (Force Field)")
        self.btn_calc_vibrations.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; padding: 6px;")
        self.btn_calc_vibrations.setToolTip("Calculate Hessian and normal vibrational modes using universal force field")
        self.btn_calc_vibrations.clicked.connect(self._on_calculate_vibrations)
        vib_ctrl_top.addWidget(self.btn_calc_vibrations, stretch=1)

        self.btn_isotope_shifts = QtWidgets.QPushButton("⚗️ Isotopologue && KIE...")
        self.btn_isotope_shifts.setStyleSheet("background-color: #7c3aed; color: #ffffff; font-weight: bold; padding: 6px 10px;")
        self.btn_isotope_shifts.setToolTip(
            "GaussView / Chemcraft style Isotopologue Calculator:\n"
            "Substitute isotopes (H→D, ¹²C→¹³C, ¹⁶O→¹⁸O) and instantly inspect frequency shifts Δν,\n"
            "Zero-Point Energy shift ΔZPE, and Kinetic Isotope Effects (k_H / k_D) without re-running DFT!"
        )
        self.btn_isotope_shifts.clicked.connect(self._on_open_isotopologue_dialog)
        vib_ctrl_top.addWidget(self.btn_isotope_shifts)
        tab_vib_layout.addLayout(vib_ctrl_top)

        # Frequency Filter Bar
        row_filter = QtWidgets.QHBoxLayout()
        row_filter.addWidget(QtWidgets.QLabel("Filter Modes:"))
        self.combo_vib_filter = QtWidgets.QComboBox()
        self.combo_vib_filter.addItem("All Modes", "all")
        self.combo_vib_filter.addItem("Genuine Vibrations (>15 cm⁻¹)", "genuine")
        self.combo_vib_filter.addItem("Imaginary / TS Frequencies (<0 cm⁻¹)", "imag")
        self.combo_vib_filter.addItem("Fingerprint Region (500–1500 cm⁻¹)", "fingerprint")
        self.combo_vib_filter.addItem("Functional Groups (1500–4000 cm⁻¹)", "functional")
        self.combo_vib_filter.currentIndexChanged.connect(self._on_filter_vibrations)
        row_filter.addWidget(self.combo_vib_filter, stretch=1)
        tab_vib_layout.addLayout(row_filter)

        # Frequency Table
        self.table_vibrations = QtWidgets.QTableWidget()
        self.table_vibrations.setColumnCount(5)
        self.table_vibrations.setHorizontalHeaderLabels([
            "Mode #", "Harmonic (cm⁻¹)", "Scaled (cm⁻¹)", "Symmetry (Irrep)", "Type / Status"
        ])
        vib_header = self.table_vibrations.horizontalHeader()
        vib_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        vib_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        vib_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        vib_header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        vib_header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_vibrations.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_vibrations.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table_vibrations.verticalHeader().setVisible(False)
        self.table_vibrations.setAlternatingRowColors(True)
        self.table_vibrations.itemSelectionChanged.connect(self._on_vibration_mode_selected)
        tab_vib_layout.addWidget(self.table_vibrations, stretch=1)

        # Vibration Animation Controls (Avogadro style)
        grp_vib_anim = QtWidgets.QGroupBox("Vibrational Mode Animation Controls")
        vib_anim_layout = QtWidgets.QVBoxLayout(grp_vib_anim)
        vib_anim_layout.setSpacing(6)

        row_play = QtWidgets.QHBoxLayout()
        self.btn_play_vibration = QtWidgets.QPushButton("▶ Start Simulation")
        self.btn_play_vibration.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold;")
        self.btn_play_vibration.setEnabled(False)
        self.btn_play_vibration.clicked.connect(self._on_toggle_vibration_play)
        row_play.addWidget(self.btn_play_vibration)

        self.btn_stop_vibration = QtWidgets.QPushButton("⏹ Stop")
        self.btn_stop_vibration.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold;")
        self.btn_stop_vibration.setEnabled(False)
        self.btn_stop_vibration.clicked.connect(self._on_stop_vibration)
        row_play.addWidget(self.btn_stop_vibration)

        self.btn_export_vib_gif = QtWidgets.QPushButton("🎬 Export GIF...")
        self.btn_export_vib_gif.setStyleSheet("background-color: #7c3aed; color: #ffffff; font-weight: bold;")
        self.btn_export_vib_gif.setToolTip("Export one full oscillation cycle of the selected vibrational mode as an animated GIF")
        self.btn_export_vib_gif.clicked.connect(self._on_export_vibration_gif)
        row_play.addWidget(self.btn_export_vib_gif)
        vib_anim_layout.addLayout(row_play)

        # Follow / Adopt Displaced Mode Geometry (GaussView style for breaking imaginary TS modes)
        row_adopt = QtWidgets.QHBoxLayout()
        row_adopt.addWidget(QtWidgets.QLabel("Follow Mode:"))
        self.btn_adopt_vib_pos = QtWidgets.QPushButton("📐 Adopt +Displaced Geometry")
        self.btn_adopt_vib_pos.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold;")
        self.btn_adopt_vib_pos.setToolTip(
            "GaussView style: Displace the molecular coordinates forward (+Amplitude × Normal Mode Vector)\n"
            "to follow an imaginary transition-state mode or break symmetry, and adopt as the new geometry (Ctrl+Z to undo)."
        )
        self.btn_adopt_vib_pos.clicked.connect(lambda: self._on_adopt_vibrational_displacement(sign=1.0))
        row_adopt.addWidget(self.btn_adopt_vib_pos, stretch=1)

        self.btn_adopt_vib_neg = QtWidgets.QPushButton("📐 Adopt −Displaced Geometry")
        self.btn_adopt_vib_neg.setStyleSheet("background-color: #0d9488; color: #ffffff; font-weight: bold;")
        self.btn_adopt_vib_neg.setToolTip(
            "GaussView style: Displace the molecular coordinates backward (−Amplitude × Normal Mode Vector)\n"
            "to follow the opposite side of an imaginary transition-state mode, and adopt as the new geometry (Ctrl+Z to undo)."
        )
        self.btn_adopt_vib_neg.clicked.connect(lambda: self._on_adopt_vibrational_displacement(sign=-1.0))
        row_adopt.addWidget(self.btn_adopt_vib_neg, stretch=1)
        vib_anim_layout.addLayout(row_adopt)

        # Interactive Mode Coordinate Scrubber (Smooth manual deformation along normal mode)
        row_scrub = QtWidgets.QHBoxLayout()
        row_scrub.addWidget(QtWidgets.QLabel("Manual Scrub:"))
        self.slider_vib_scrub = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_vib_scrub.setRange(-100, 100)  # -1.0x to +1.0x of amplitude
        self.slider_vib_scrub.setValue(0)
        self.slider_vib_scrub.setToolTip("Interactively deform the molecule back and forth along the selected normal coordinate vector.")
        self.slider_vib_scrub.valueChanged.connect(self._on_vib_scrub_changed)
        row_scrub.addWidget(self.slider_vib_scrub, stretch=1)
        self.lbl_vib_scrub = QtWidgets.QLabel("0.00×")
        self.lbl_vib_scrub.setFixedWidth(40)
        row_scrub.addWidget(self.lbl_vib_scrub)
        btn_reset_scrub = QtWidgets.QPushButton("0")
        btn_reset_scrub.setFixedWidth(24)
        btn_reset_scrub.setToolTip("Reset manual displacement to equilibrium (0.00×)")
        btn_reset_scrub.clicked.connect(lambda: self.slider_vib_scrub.setValue(0))
        row_scrub.addWidget(btn_reset_scrub)
        vib_anim_layout.addLayout(row_scrub)

        # Amplitude Slider
        row_amp = QtWidgets.QHBoxLayout()
        row_amp.addWidget(QtWidgets.QLabel("Amplitude:"))
        self.slider_vib_amp = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_vib_amp.setRange(5, 200)  # 0.05 to 2.00
        self.slider_vib_amp.setValue(50)      # 0.50
        self.slider_vib_amp.valueChanged.connect(self._on_vib_amplitude_changed)
        row_amp.addWidget(self.slider_vib_amp, stretch=1)
        self.lbl_vib_amp = QtWidgets.QLabel("0.50")
        self.lbl_vib_amp.setFixedWidth(36)
        row_amp.addWidget(self.lbl_vib_amp)
        vib_anim_layout.addLayout(row_amp)

        # Vector Arrow Scale Slider
        row_vec = QtWidgets.QHBoxLayout()
        row_vec.addWidget(QtWidgets.QLabel("Arrow Scale:"))
        self.slider_vib_vec_scale = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_vib_vec_scale.setRange(20, 300)  # 0.2x to 3.0x
        self.slider_vib_vec_scale.setValue(100)      # 1.0x
        self.slider_vib_vec_scale.valueChanged.connect(self._on_vib_vec_scale_changed)
        row_vec.addWidget(self.slider_vib_vec_scale, stretch=1)
        self.lbl_vib_vec_scale = QtWidgets.QLabel("1.0×")
        self.lbl_vib_vec_scale.setFixedWidth(36)
        row_vec.addWidget(self.lbl_vib_vec_scale)
        vib_anim_layout.addLayout(row_vec)

        # Vectors toggle & Details
        row_opt = QtWidgets.QHBoxLayout()
        self.chk_vib_vectors = QtWidgets.QCheckBox("Show 3D Displacement Vectors (Arrows)")
        self.chk_vib_vectors.setChecked(True)
        self.chk_vib_vectors.toggled.connect(self._on_toggle_vib_vectors)
        row_opt.addWidget(self.chk_vib_vectors)
        row_opt.addStretch()
        vib_anim_layout.addLayout(row_opt)

        self.lbl_vib_info = QtWidgets.QLabel("Select a mode from the list above to view or animate.")
        self.lbl_vib_info.setStyleSheet("color: #94a3b8; font-style: italic;")
        vib_anim_layout.addWidget(self.lbl_vib_info)

        tab_vib_layout.addWidget(grp_vib_anim)

        # Interactive Synthetic Spectra Plots (IR and UV-Vis tabs)
        self.tabs_spectra = QtWidgets.QTabWidget()

        grp_spectrum = QtWidgets.QWidget()
        spectrum_layout = QtWidgets.QVBoxLayout(grp_spectrum)
        spectrum_layout.setContentsMargins(4, 4, 4, 4)
        self.spectrum_widget = IRSpectrumWidget()
        self.spectrum_widget.mode_clicked.connect(self._on_spectrum_mode_clicked)
        self.spectrum_widget.scale_factor_changed.connect(self._on_spectrum_scale_changed)
        spectrum_layout.addWidget(self.spectrum_widget)
        self.tabs_spectra.addTab(grp_spectrum, "IR Spectrum")

        grp_uvvis = QtWidgets.QWidget()
        uvvis_layout = QtWidgets.QVBoxLayout(grp_uvvis)
        uvvis_layout.setContentsMargins(4, 4, 4, 4)
        self.uvvis_widget = UVVisSpectrumWidget()
        uvvis_layout.addWidget(self.uvvis_widget)
        self.tabs_spectra.addTab(grp_uvvis, "UV-Vis Spectrum")

        grp_nmr = QtWidgets.QWidget()
        nmr_layout = QtWidgets.QVBoxLayout(grp_nmr)
        nmr_layout.setContentsMargins(4, 4, 4, 4)
        self.nmr_widget = NMRSpectrumWidget()
        self.nmr_widget.atom_clicked.connect(self._on_nmr_atom_clicked)
        nmr_layout.addWidget(self.nmr_widget)
        self.tabs_spectra.addTab(grp_nmr, "NMR Spectrum (ppm)")

        grp_dos = QtWidgets.QWidget()
        dos_layout = QtWidgets.QVBoxLayout(grp_dos)
        dos_layout.setContentsMargins(4, 4, 4, 4)
        self.dos_widget = DOSSpectrumWidget()
        self.dos_widget.orbital_clicked.connect(self._on_dos_orbital_clicked)
        dos_layout.addWidget(self.dos_widget)
        self.tabs_spectra.addTab(grp_dos, "DOS Spectrum (eV)")

        tab_vib_layout.addWidget(self.tabs_spectra, stretch=1)
        self.tab_vibrations_widget = tab_vibrations
        scroll_vib = QtWidgets.QScrollArea()
        scroll_vib.setWidgetResizable(True)
        scroll_vib.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_vib.setWidget(tab_vibrations)
        self.tab_vibrations = scroll_vib
        self.tabs.addTab(scroll_vib, "Vibrations")

    def _init_symmetry_tab(self):
        tab_symmetry = QtWidgets.QWidget()
        tab_sym_layout = QtWidgets.QVBoxLayout(tab_symmetry)
        tab_sym_layout.setContentsMargins(8, 8, 8, 8)
        tab_sym_layout.setSpacing(8)

        # Symmetry controls group
        grp_sym_ctrl = QtWidgets.QGroupBox("Point Group Symmetry Detection")
        sym_ctrl_layout = QtWidgets.QVBoxLayout(grp_sym_ctrl)

        row_sym_btn = QtWidgets.QHBoxLayout()
        self.btn_detect_symmetry = QtWidgets.QPushButton("🔮 Detect Point Group Symmetry")
        self.btn_detect_symmetry.setStyleSheet("background-color: #7c3aed; color: #ffffff; font-weight: bold; padding: 6px;")
        self.btn_detect_symmetry.clicked.connect(lambda _checked=False: self._on_detect_symmetry(switch_tab=True))
        row_sym_btn.addWidget(self.btn_detect_symmetry)

        row_sym_btn.addWidget(QtWidgets.QLabel("Tolerance:"))

        self.slider_sym_tol = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_sym_tol.setRange(1, 100)  # 0.01 Å to 1.00 Å
        self.slider_sym_tol.setValue(15)      # 0.15 Å
        self.slider_sym_tol.setToolTip("Slide to dynamically tune coordinate mapping tolerance for detecting pseudo-symmetry")
        self.slider_sym_tol.valueChanged.connect(self._on_sym_slider_tol_changed)
        row_sym_btn.addWidget(self.slider_sym_tol, stretch=1)

        self.spin_sym_tol = QtWidgets.QDoubleSpinBox()
        self.spin_sym_tol.setRange(0.01, 1.0)
        self.spin_sym_tol.setSingleStep(0.05)
        self.spin_sym_tol.setValue(0.15)
        self.spin_sym_tol.setSuffix(" Å")
        self.spin_sym_tol.setToolTip("Coordinate mapping tolerance for symmetry operations")
        self.spin_sym_tol.valueChanged.connect(self._on_sym_spin_tol_changed)
        row_sym_btn.addWidget(self.spin_sym_tol)
        sym_ctrl_layout.addLayout(row_sym_btn)

        # Row: Choose Higher / Detected / Lower Point Group Symmetry (Subgroups & Supergroups)
        row_sym_select = QtWidgets.QHBoxLayout()
        row_sym_select.setSpacing(6)
        self.lbl_choose_sym = QtWidgets.QLabel("Select Symmetry:")
        self.lbl_choose_sym.setStyleSheet("font-weight: bold;")
        row_sym_select.addWidget(self.lbl_choose_sym)

        self.combo_sym_group = QtWidgets.QComboBox()
        self.combo_sym_group.setToolTip(
            "Choose from Higher (approximate supergroups at looser tolerance),\n"
            "Detected (exact point group), or Lower (subgroups down to C1) symmetries.\n"
            "Selecting a group updates symmetry operations, irreps, and Impose/Enforce Symmetry."
        )
        self.combo_sym_group.currentIndexChanged.connect(self._on_sym_group_combo_changed)
        row_sym_select.addWidget(self.combo_sym_group, stretch=1)

        self.btn_sym_higher = QtWidgets.QPushButton("⬆ Higher")
        self.btn_sym_higher.setToolTip("Step up to the next higher-order symmetry group (supergroup)")
        self.btn_sym_higher.setStyleSheet("background-color: #334155; color: #ffffff; font-weight: bold; padding: 4px 8px;")
        self.btn_sym_higher.clicked.connect(self._on_step_higher_symmetry)
        self.btn_sym_higher.setEnabled(False)
        row_sym_select.addWidget(self.btn_sym_higher)

        self.btn_sym_lower = QtWidgets.QPushButton("⬇ Lower")
        self.btn_sym_lower.setToolTip("Step down to the next lower-order symmetry subgroup")
        self.btn_sym_lower.setStyleSheet("background-color: #334155; color: #ffffff; font-weight: bold; padding: 4px 8px;")
        self.btn_sym_lower.clicked.connect(self._on_step_lower_symmetry)
        self.btn_sym_lower.setEnabled(False)
        row_sym_select.addWidget(self.btn_sym_lower)
        sym_ctrl_layout.addLayout(row_sym_select)

        # Action buttons row: Align to standard orientation, Enforce symmetry & Impose symmetry
        row_sym_actions = QtWidgets.QHBoxLayout()
        self.btn_sym_align = QtWidgets.QPushButton("📐 Align to Standard Orientation")
        self.btn_sym_align.setToolTip("Align principal symmetry axis (highest Cn) along Z and plane in XZ/XY")
        self.btn_sym_align.setStyleSheet("background-color: #0d9488; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_sym_align.clicked.connect(self._on_align_symmetry)
        self.btn_sym_align.setEnabled(False)
        row_sym_actions.addWidget(self.btn_sym_align)

        self.btn_sym_enforce = QtWidgets.QPushButton("⚖️ Enforce Symmetry")
        self.btn_sym_enforce.setToolTip("Project coordinates into exact representation subspace if tolerance is not fulfilled")
        self.btn_sym_enforce.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_sym_enforce.clicked.connect(self._on_enforce_symmetry)
        self.btn_sym_enforce.setEnabled(False)
        row_sym_actions.addWidget(self.btn_sym_enforce)

        self.btn_impose_symmetry = QtWidgets.QPushButton("🔒 Impose Symmetry: OFF")
        self.btn_impose_symmetry.setCheckable(True)
        self.btn_impose_symmetry.setChecked(False)
        self.btn_impose_symmetry.setToolTip(
            "GaussView style: Constrain/lock point group symmetry during editing.\n"
            "Replacing an atom or attaching a ligand (e.g. replacing H by CO in Oh CrH6)\n"
            "automatically applies the edit to all symmetry-equivalent atoms!"
        )
        self.btn_impose_symmetry.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_impose_symmetry.toggled.connect(self._on_toggle_impose_symmetry)
        row_sym_actions.addWidget(self.btn_impose_symmetry)
        sym_ctrl_layout.addLayout(row_sym_actions)

        # Advanced options row: Matrices & Representations
        row_sym_opts = QtWidgets.QHBoxLayout()
        self.btn_sym_matrices = QtWidgets.QPushButton("🔢 Show Symmetry Matrices")
        self.btn_sym_matrices.setToolTip("Display 3x3 transformation matrix for selected symmetry operation")
        self.btn_sym_matrices.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_sym_matrices.clicked.connect(self._on_show_symmetry_matrices)
        self.btn_sym_matrices.setEnabled(False)
        row_sym_opts.addWidget(self.btn_sym_matrices)

        self.btn_calc_reps = QtWidgets.QPushButton("📊 Reducible & Irreps Analysis")
        self.btn_calc_reps.setToolTip("Calculate reducible representations (Γ_tot, Γ_trans, Γ_rot, Γ_vib) and decompose into direct sum of irreps")
        self.btn_calc_reps.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_calc_reps.clicked.connect(self._on_calculate_symmetry_representations)
        self.btn_calc_reps.setEnabled(False)
        row_sym_opts.addWidget(self.btn_calc_reps)
        sym_ctrl_layout.addLayout(row_sym_opts)

        # Point group result banner
        self.lbl_sym_group = QtWidgets.QLabel("Point Group: Not calculated")
        self.lbl_sym_group.setStyleSheet("font-size: 15px; font-weight: bold; color: #7c3aed; padding: 4px;")
        sym_ctrl_layout.addWidget(self.lbl_sym_group)

        self.lbl_sym_details = QtWidgets.QLabel("Click 'Detect Point Group Symmetry' to analyze current molecular geometry.")
        self.lbl_sym_details.setStyleSheet("color: #64748b; font-style: italic;")
        self.lbl_sym_details.setWordWrap(True)
        sym_ctrl_layout.addWidget(self.lbl_sym_details)

        # Irreducible representations display box
        self.box_sym_reps = QtWidgets.QGroupBox("Rotational & Vibrational Irreps Decomposition")
        box_reps_layout = QtWidgets.QVBoxLayout(self.box_sym_reps)
        self.lbl_sym_reps = QtWidgets.QLabel("Click 'Reducible & Irreps Analysis' above to decompose Γ into direct sum of irreps.")
        self.lbl_sym_reps.setStyleSheet("font-family: monospace; font-size: 11px; color: #e2e8f0; background: #0f172a; padding: 6px; border-radius: 4px;")
        self.lbl_sym_reps.setWordWrap(True)
        self.lbl_sym_reps.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        box_reps_layout.addWidget(self.lbl_sym_reps)
        self.box_sym_reps.setVisible(False)
        sym_ctrl_layout.addWidget(self.box_sym_reps)

        tab_sym_layout.addWidget(grp_sym_ctrl)

        # Coordinate Centering & Axis / Plane Alignment for Symmetry Matrices
        grp_sym_orient = QtWidgets.QGroupBox("Coordinate Centering & Axis / Plane Alignment (for 3×3 Matrices)")
        sym_orient_layout = QtWidgets.QVBoxLayout(grp_sym_orient)
        sym_orient_layout.setSpacing(5)

        row_so_center = QtWidgets.QHBoxLayout()
        row_so_center.setSpacing(5)
        row_so_center.addWidget(QtWidgets.QLabel("Origin:"))
        btn_so_com = QtWidgets.QPushButton("🎯 Center COM")
        btn_so_com.setStyleSheet("background-color: #334155; color: #ffffff; font-weight: bold; padding: 4px 8px;")
        btn_so_com.setToolTip("Place Center of Mass at (0, 0, 0)")
        btn_so_com.clicked.connect(lambda: self._on_center_molecule_mode("com"))
        btn_so_cent = QtWidgets.QPushButton("🎯 Centroid")
        btn_so_cent.setStyleSheet("background-color: #334155; color: #ffffff; font-weight: bold; padding: 4px 8px;")
        btn_so_cent.setToolTip("Place Geometric Centroid at (0, 0, 0)")
        btn_so_cent.clicked.connect(lambda: self._on_center_molecule_mode("centroid"))
        btn_so_atom = QtWidgets.QPushButton("🎯 Selected Atom → (0,0,0)")
        btn_so_atom.setStyleSheet("background-color: #334155; color: #ffffff; font-weight: bold; padding: 4px 8px;")
        btn_so_atom.setToolTip("Place selected atom at (0, 0, 0)")
        btn_so_atom.clicked.connect(lambda: self._on_center_molecule_mode("atom"))
        row_so_center.addWidget(btn_so_com)
        row_so_center.addWidget(btn_so_cent)
        row_so_center.addWidget(btn_so_atom)
        sym_orient_layout.addLayout(row_so_center)

        row_so_axis = QtWidgets.QHBoxLayout()
        row_so_axis.setSpacing(5)
        row_so_axis.addWidget(QtWidgets.QLabel("Align Axis/Bond:"))
        for ax_name in ("+X", "+Y", "+Z", "-X", "-Y", "-Z"):
            b_sax = QtWidgets.QPushButton(ax_name)
            b_sax.setStyleSheet("background-color: #0d9488; color: #ffffff; font-weight: bold; padding: 4px 6px;")
            b_sax.setToolTip(
                f"Align selected 2-atom vector (or 1-atom vector / principal symmetry axis) along {ax_name} and update 3×3 symmetry matrices"
            )
            b_sax.clicked.connect(lambda checked=False, a=ax_name: self._on_align_to_axis(a))
            row_so_axis.addWidget(b_sax)
        sym_orient_layout.addLayout(row_so_axis)

        row_so_plane = QtWidgets.QHBoxLayout()
        row_so_plane.setSpacing(5)
        row_so_plane.addWidget(QtWidgets.QLabel("Align Plane:"))
        for pl_name in ("XY", "XZ", "YZ"):
            b_spl = QtWidgets.QPushButton(f"{pl_name} Plane")
            b_spl.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; padding: 4px 8px;")
            b_spl.setToolTip(
                f"Align plane defined by 3 selected atoms (or molecular plane) into {pl_name} and update 3×3 symmetry matrices"
            )
            b_spl.clicked.connect(lambda checked=False, p=pl_name: self._on_align_to_plane(p))
            row_so_plane.addWidget(b_spl)
        sym_orient_layout.addLayout(row_so_plane)

        tab_sym_layout.addWidget(grp_sym_orient)

        # Table of Symmetry Operations
        grp_ops = QtWidgets.QGroupBox("Symmetry Elements & Operations (Double-click or click 'Show Symmetry Matrices' for 3x3 matrix)")
        ops_layout = QtWidgets.QVBoxLayout(grp_ops)

        self.table_symmetry_ops = QtWidgets.QTableWidget(0, 3)
        self.table_symmetry_ops.setHorizontalHeaderLabels(["#", "Operation", "Type / Details"])
        self.table_symmetry_ops.horizontalHeader().setStretchLastSection(True)
        self.table_symmetry_ops.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_symmetry_ops.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_symmetry_ops.verticalHeader().setVisible(False)
        self.table_symmetry_ops.setColumnWidth(0, 40)
        self.table_symmetry_ops.setColumnWidth(1, 100)
        self.table_symmetry_ops.itemDoubleClicked.connect(lambda item: self._on_show_symmetry_matrices())
        ops_layout.addWidget(self.table_symmetry_ops)

        tab_sym_layout.addWidget(grp_ops, stretch=1)
        self.tab_symmetry_widget = tab_symmetry
        scroll_sym = QtWidgets.QScrollArea()
        scroll_sym.setWidgetResizable(True)
        scroll_sym.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_sym.setWidget(tab_symmetry)
        self.tab_symmetry = scroll_sym
        self.tabs.addTab(scroll_sym, "Symmetry")

    def _init_orbitals_tab(self):
        tab_orbitals = QtWidgets.QWidget()
        tab_mo_layout = QtWidgets.QVBoxLayout(tab_orbitals)
        tab_mo_layout.setSpacing(6)

        # Row 1: File loading, Snapshot gallery, Clear (never crushed on small screens)
        mo_top_bar1 = QtWidgets.QHBoxLayout()
        mo_top_bar1.setSpacing(5)
        btn_open_molden_tab = QtWidgets.QPushButton("📂 Load Molden / Log...")
        btn_open_molden_tab.setStyleSheet("background-color: #0d9488; color: #ffffff; font-weight: bold; font-size: 11px; padding: 4px 6px;")
        btn_open_molden_tab.setMinimumWidth(110)
        btn_open_molden_tab.setToolTip("Open any past calculation (.molden, .out, .log, .cube) to view its molecular orbital list and energy diagram")
        btn_open_molden_tab.clicked.connect(self._on_load_orbitals_prompt)
        mo_top_bar1.addWidget(btn_open_molden_tab, stretch=2)

        self.combo_mo_file_source = QtWidgets.QComboBox()
        self.combo_mo_file_source.setMinimumWidth(120)
        self.combo_mo_file_source.setStyleSheet("font-size: 11px;")
        self.combo_mo_file_source.setToolTip(
            "Switch the orbital table & energy ladder between Electronic MOs and Positronic MOs "
            "(or multiple stages from the same calculation)"
        )
        self.combo_mo_file_source.setVisible(False)
        self.combo_mo_file_source.currentIndexChanged.connect(self._on_mo_source_combo_changed)
        mo_top_bar1.addWidget(self.combo_mo_file_source, stretch=1)

        self.btn_orbital_snapshot_gallery = QtWidgets.QPushButton("📸 Gallery")
        self.btn_orbital_snapshot_gallery.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; font-size: 11px; padding: 4px 6px;")
        self.btn_orbital_snapshot_gallery.setMinimumWidth(75)
        self.btn_orbital_snapshot_gallery.setToolTip("Capture all or selected orbitals in a batch snapshot contact sheet or image gallery (like Gabedit)")
        self.btn_orbital_snapshot_gallery.clicked.connect(self._on_open_orbital_snapshot_gallery)
        mo_top_bar1.addWidget(self.btn_orbital_snapshot_gallery, stretch=1)

        btn_clear_mo = QtWidgets.QPushButton("✖ Clear")
        btn_clear_mo.setStyleSheet("font-size: 11px; padding: 4px 6px;")
        btn_clear_mo.setMinimumWidth(55)
        btn_clear_mo.setToolTip("Clear 3D orbital / isosurface (Shortcut: Esc)")
        btn_clear_mo.clicked.connect(self._on_clear_orbital_surface)
        mo_top_bar1.addWidget(btn_clear_mo, stretch=1)
        tab_mo_layout.addLayout(mo_top_bar1)

        # Row 2: 1-Click Quick Surface Buttons
        mo_top_bar2 = QtWidgets.QHBoxLayout()
        mo_top_bar2.setSpacing(4)

        self.btn_quick_homo = QtWidgets.QPushButton("⚡ HOMO")
        self.btn_quick_homo.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: bold; font-size: 11px; padding: 4px 6px;")
        self.btn_quick_homo.setToolTip("Quickly evaluate and display the Highest Occupied Molecular Orbital (HOMO)")
        self.btn_quick_homo.clicked.connect(self._on_quick_view_homo)
        mo_top_bar2.addWidget(self.btn_quick_homo, stretch=1)

        self.btn_quick_lumo = QtWidgets.QPushButton("⚡ LUMO")
        self.btn_quick_lumo.setStyleSheet("background-color: #7c3aed; color: #ffffff; font-weight: bold; font-size: 11px; padding: 4px 6px;")
        self.btn_quick_lumo.setToolTip("Quickly evaluate and display the Lowest Unoccupied Molecular Orbital (LUMO)")
        self.btn_quick_lumo.clicked.connect(self._on_quick_view_lumo)
        mo_top_bar2.addWidget(self.btn_quick_lumo, stretch=1)

        self.btn_quick_spin_dens = QtWidgets.QPushButton("🌀 Spin")
        self.btn_quick_spin_dens.setStyleSheet("background-color: #0d9488; color: #ffffff; font-weight: bold; font-size: 11px; padding: 4px 6px;")
        self.btn_quick_spin_dens.setToolTip(
            "1-Click Open-Shell Spin Density (Δρ = ρ_α − ρ_β):\n"
            "Evaluate and render excess α/β spin density lobes directly in the 3D viewport"
        )
        self.btn_quick_spin_dens.clicked.connect(self._on_quick_view_spin_density)
        mo_top_bar2.addWidget(self.btn_quick_spin_dens, stretch=1)

        self.btn_quick_esp_map = QtWidgets.QPushButton("🌈 ESP")
        self.btn_quick_esp_map.setStyleSheet("background-color: #db2777; color: #ffffff; font-weight: bold; font-size: 11px; padding: 4px 6px;")
        self.btn_quick_esp_map.setToolTip(
            "1-Click Electrostatic Potential (ESP) Mapped Electron Density Surface:\n"
            "Colors the electron density isosurface from Red (δ⁻ nucleophilic) to Blue (δ⁺ electrophilic) (Ctrl+Alt+P)"
        )
        self.btn_quick_esp_map.clicked.connect(self._on_quick_view_esp_map)
        mo_top_bar2.addWidget(self.btn_quick_esp_map, stretch=1)

        self.btn_localize_mos = QtWidgets.QPushButton("🧲 LMO ▾")
        self.btn_localize_mos.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold; font-size: 11px; padding: 4px 6px;")
        self.btn_localize_mos.setToolTip(
            "1-Click PySCF Unitary Orbital Localization (Pipek-Mezey or Foster-Boys):\n"
            "Transforms canonical delocalized MOs into Localized Molecular Orbitals (LMOs: σ bonds, π bonds, lone pairs)"
        )
        menu_loc = QtWidgets.QMenu(self.btn_localize_mos)
        act_loc_pm = menu_loc.addAction("🧲 Pipek-Mezey (PM — Preserves σ / π Separation)")
        act_loc_pm.triggered.connect(lambda: self._on_localize_orbitals("pipek-mezey"))
        act_loc_boys = menu_loc.addAction("🎯 Foster-Boys (Minimum Spatial Variance)")
        act_loc_boys.triggered.connect(lambda: self._on_localize_orbitals("boys"))
        menu_loc.addSeparator()
        act_loc_canon = menu_loc.addAction("↩️ Restore Canonical MOs")
        act_loc_canon.triggered.connect(self._on_restore_canonical_orbitals)
        self.btn_localize_mos.setMenu(menu_loc)
        mo_top_bar2.addWidget(self.btn_localize_mos, stretch=1)

        self.btn_positron_overlay = QtWidgets.QPushButton("⚛️ +Positron")
        self.btn_positron_overlay.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold; font-size: 11px; padding: 4px 6px;")
        self.btn_positron_overlay.setToolTip(
            "Overlay companion Positron ground-state orbital surface (gold/green) onto the current Electron MO (blue/red)"
        )
        self.btn_positron_overlay.setVisible(False)
        self.btn_positron_overlay.clicked.connect(self._on_overlay_positron_orbital)
        mo_top_bar2.addWidget(self.btn_positron_overlay, stretch=1)

        tab_mo_layout.addLayout(mo_top_bar2)

        self.lbl_mo_status = QtWidgets.QLabel("No orbital/wavefunction data loaded. Run a calculation or open a .molden / .cube file.")
        self.lbl_mo_status.setStyleSheet("color: #64748b; font-style: italic; font-size: 11px;")
        self.lbl_mo_status.setWordWrap(True)
        tab_mo_layout.addWidget(self.lbl_mo_status)

        # Chemcraft-style Quantum Chemistry Summary Pill Bar (HOMO, LUMO, Gap, Hardness)
        self.lbl_mo_gap_badge = QtWidgets.QLabel("")
        self.lbl_mo_gap_badge.setStyleSheet(
            "QLabel { background-color: #0284c7; color: #ffffff; font-weight: bold; font-size: 11px; border-radius: 4px; padding: 4px 8px; }"
        )
        self.lbl_mo_gap_badge.setVisible(False)
        self.lbl_mo_gap_badge.setWordWrap(True)
        tab_mo_layout.addWidget(self.lbl_mo_gap_badge)

        # Quick Orbital Energy Level Table + Interactive Energy Ladder Diagram
        mo_split_layout = QtWidgets.QHBoxLayout()
        mo_split_layout.setSpacing(6)

        self.table_orbitals = QtWidgets.QTableWidget()
        self.table_orbitals.setColumnCount(5)
        self.table_orbitals.setHorizontalHeaderLabels(["MO #", "Type", "Energy (eV)", "Energy (a.u.)", "Occ."])
        mo_header = self.table_orbitals.horizontalHeader()
        mo_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        mo_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        mo_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_orbitals.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_orbitals.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table_orbitals.verticalHeader().setVisible(False)
        self.table_orbitals.setAlternatingRowColors(True)
        self.table_orbitals.itemSelectionChanged.connect(self._on_mo_table_selected)
        mo_split_layout.addWidget(self.table_orbitals, stretch=3)

        self.mo_ladder_widget = MOEnergyLadderWidget(self)
        self.mo_ladder_widget.orbital_clicked.connect(lambda idx: self.table_orbitals.selectRow(idx))
        mo_split_layout.addWidget(self.mo_ladder_widget, stretch=2)

        tab_mo_layout.addLayout(mo_split_layout, stretch=1)

        # Quick Surface Parameter Bar (Isovalue & Opacity)
        mo_params_group = QtWidgets.QGroupBox("Orbital Surface Controls")
        mo_params_layout = QtWidgets.QHBoxLayout(mo_params_group)
        mo_params_layout.setContentsMargins(6, 6, 6, 6)

        mo_params_layout.addWidget(QtWidgets.QLabel("Isovalue:"))
        self.slider_quick_iso = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_quick_iso.setRange(1, 1000)  # 0.0001 to 0.5000 (scaled by mapping)
        self.slider_quick_iso.setValue(60)
        self.slider_quick_iso.setToolTip("Adjust orbital isosurface boundary level")
        self.slider_quick_iso.valueChanged.connect(self._on_quick_slider_iso_changed)
        mo_params_layout.addWidget(self.slider_quick_iso, stretch=1)

        self.spin_quick_iso = QtWidgets.QDoubleSpinBox()
        self.spin_quick_iso.setRange(0.0001, 2.0)
        self.spin_quick_iso.setDecimals(4)
        self.spin_quick_iso.setSingleStep(0.005)
        self.spin_quick_iso.setValue(0.0300)
        self.spin_quick_iso.valueChanged.connect(self._on_quick_spin_iso_changed)
        mo_params_layout.addWidget(self.spin_quick_iso)

        mo_params_layout.addWidget(QtWidgets.QLabel("Grid:"))
        self.combo_quick_grid = QtWidgets.QComboBox()
        self.combo_quick_grid.addItem("Low (30³)", 30)
        self.combo_quick_grid.addItem("Med (40³)", 40)
        self.combo_quick_grid.addItem("High (60³)", 60)
        self.combo_quick_grid.setCurrentIndex(1)
        self.combo_quick_grid.setToolTip("3D voxel grid resolution for Molecular Orbital evaluation")
        self.combo_quick_grid.currentIndexChanged.connect(self._on_quick_iso_changed)
        mo_params_layout.addWidget(self.combo_quick_grid)

        mo_params_layout.addWidget(QtWidgets.QLabel("Box Margin:"))
        self.spin_quick_margin = QtWidgets.QDoubleSpinBox()
        self.spin_quick_margin.setRange(1.0, 15.0)
        self.spin_quick_margin.setSingleStep(0.5)
        self.spin_quick_margin.setValue(3.0)
        self.spin_quick_margin.setSuffix(" a₀")
        self.spin_quick_margin.setToolTip("Cartesian evaluation box padding/margin beyond outer atoms (Bohr). Increase if diffuse orbital lobes are clipped.")
        self.spin_quick_margin.valueChanged.connect(self._on_quick_iso_changed)
        mo_params_layout.addWidget(self.spin_quick_margin)

        mo_params_layout.addWidget(QtWidgets.QLabel("Opacity:"))
        self.slider_quick_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_quick_opacity.setRange(10, 100)
        self.slider_quick_opacity.setValue(80)
        self.slider_quick_opacity.valueChanged.connect(self._on_quick_opacity_changed)
        mo_params_layout.addWidget(self.slider_quick_opacity, stretch=1)

        mo_params_layout.addWidget(QtWidgets.QLabel("Colors:"))
        self.combo_mo_phase_colors = QtWidgets.QComboBox()
        self.combo_mo_phase_colors.addItem("Blue (+) / Red (-)", "blue_red")
        self.combo_mo_phase_colors.addItem("Cyan (+) / Orange (-)", "cyan_orange")
        self.combo_mo_phase_colors.addItem("Green (+) / Purple (-)", "green_purple")
        self.combo_mo_phase_colors.addItem("Gold (+) / Emerald (-)", "gold_emerald")
        self.combo_mo_phase_colors.setToolTip("Customize 3D orbital positive/negative phase lobe color scheme (Avogadro / VMD style)")
        self.combo_mo_phase_colors.currentIndexChanged.connect(lambda: self._remesh_current_mo(self.spin_quick_iso.value()))
        mo_params_layout.addWidget(self.combo_mo_phase_colors)

        tab_mo_layout.addWidget(mo_params_group)
        self.tab_orbitals_widget = tab_orbitals
        scroll_mo = QtWidgets.QScrollArea()
        scroll_mo.setWidgetResizable(True)
        scroll_mo.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_mo.setWidget(tab_orbitals)
        self.tab_orbitals = scroll_mo
        self.tabs.addTab(scroll_mo, "Molecular Orbitals")
