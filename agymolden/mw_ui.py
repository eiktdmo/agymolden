"""Mixin modules extracted from MainWindow for clean separation of concerns."""

import os
import math
import copy
import logging
from typing import List, Optional, Dict, Tuple
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets

from .workers import run_in_background
from .elements import ELEMENT_DATA, normalize_symbol, estimate_bond_length
from .zmatrix import (
    ZAtom,
    zmatrix_to_cartesian,
    cartesian_to_zmatrix,
    parse_zmatrix_text,
    format_zmatrix_text,
    parse_xyz,
    write_xyz,
    calculate_bond_distance,
    calculate_angle,
    calculate_dihedral,
    format_zmatrix_numeric,
    format_zmatrix_variables,
    apply_variables,
    sync_variables_from_atoms,
    assign_symmetric_variables,
    clear_variables,
    _split_var,
)
from .fragments import FRAGMENTS, SAMPLE_MOLECULES
from .viewer import MoleculeViewer
from .table_view import ZMatrixTableWidget
from .periodic_table import PeriodicTableDialog
from .topology import (
    norm_bond, build_adj, remove_atoms, fix_valence_explicit, bfs_reorder,
    rotation_between, fragment_outward_axis, form_single_bond, BondFormationError,
)
from .forcefield import (
    DEFAULT_VALENCES,
    add_hydrogens_to_valence,
    adjust_hydrogens_to_valence,
    optimize_geometry_force_field,
    detect_connectivity,
    detect_bond_orders,
    get_cov_radius,
    _calculate_hydrogen_directions,
)
from .vibrations import VibrationalAnalysis, compute_force_field_vibrations
from .symmetry import (
    find_symmetry_operations,
    PointGroupResult,
    align_molecule_to_symmetry,
    enforce_point_group_symmetry,
    calculate_symmetry_representations,
)
from .fragment_library import (
    FRAGMENT_CATEGORIES,
    initialize_fragment_library,
    get_fragments_by_category,
)
from .spectrum_widget import IRSpectrumWidget, UVVisSpectrumWidget

log = logging.getLogger(__name__)

from .theme import (
    RESOURCES_DIR,
    ARROW_DARK_PATH,
    ARROW_LIGHT_PATH,
    DARK_THEME_QSS,
    LIGHT_THEME_QSS,
)
from .mw_ui_tabs import UIInspectorTabsMixin


class UIMixin(UIInspectorTabsMixin):
    def _init_ui(self):
        # Central splitter: Left is 3D Viewport, Right is Control Panel & ZMAT Table
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left Widget: 3D Viewport + View controls bar
        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # 3D Viewport
        self.viewer = MoleculeViewer()
        self.viewer.atom_clicked.connect(self._on_viewport_atom_clicked)
        self.viewer.atom_right_clicked.connect(self._on_viewport_atom_context_menu)
        self.viewer.bond_right_clicked.connect(self._on_viewport_bond_context_menu)
        self.viewer.canvas_right_clicked.connect(self._on_viewport_canvas_context_menu)
        self.viewer.atom_right_clicked_delete.connect(self._on_viewport_atom_right_clicked_delete)
        self.viewer.atom_picked_for_connectivity.connect(self._on_atom_picked_for_connectivity)
        self.viewer.atom_picked_for_reorder.connect(self._on_atom_picked_for_reorder)
        self.viewer.atom_picked_for_partial_reorder.connect(self._on_atom_picked_for_partial_reorder)
        self.viewer.quick_add_atom_dragged.connect(self._on_quick_add_atom_dragged)
        self.viewer.cross_bond_drawn.connect(self._on_cross_bond_drawn)
        self.viewer.atom_drag_started.connect(self._on_atom_drag_started)
        self.viewer.atom_moved_by_drag.connect(self._on_atom_moved_by_drag)
        self.viewer.atom_drag_finished.connect(self._on_atom_drag_finished)
        self.viewer.region_rotation_finished.connect(self._on_region_rotation_finished)
        self.viewer.atom_double_clicked.connect(self._on_atom_double_clicked)
        self.viewer.bond_double_clicked.connect(self._on_bond_double_clicked)
        self.viewer.bond_torsion_dragged.connect(self._on_bond_torsion_dragged)
        self.viewer.canvas_double_clicked.connect(self._on_canvas_double_clicked)
        self.viewer.canvas_dragged_or_clicked.connect(self._on_canvas_dragged_or_clicked)
        self.viewer.surface_cleared.connect(self._on_clear_orbital_surface)
        self.viewer.measurement_updated.connect(self._on_measurement_updated)
        left_layout.addWidget(self.viewer, stretch=1)

        # Viewer quick controls ribbon (compact so essential builder controls are always visible without scrolling)
        view_ctrls = QtWidgets.QHBoxLayout()
        view_ctrls.setSpacing(6)

        self.btn_mode_toggle = QtWidgets.QPushButton("✏️ Build")
        self.btn_mode_toggle.setCheckable(True)
        self.btn_mode_toggle.setChecked(False)
        self.btn_mode_toggle.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 2px 8px;")
        self.btn_mode_toggle.setToolTip(
            "Toggle between Build/Draw Mode (✏️) and Inspect/Measure Mode (🔍) (Shortcut: Tab).\n"
            "• ✏️ Build: Dragging from an atom sprouts a new bond/atom; double-clicking mutates/attaches.\n"
            "• 🔍 Inspect: Clicking atoms only selects/measures; dragging anywhere orbits the 3D camera."
        )
        self.btn_mode_toggle.toggled.connect(self._on_toggle_interaction_mode)
        view_ctrls.addWidget(self.btn_mode_toggle)

        self.chk_auto_h = QtWidgets.QCheckBox("Adjust H")
        self.chk_auto_h.setChecked(True)
        self.chk_auto_h.setToolTip("Avogadro style: automatically complete new atom valence and prune hydrogens on anchor atom to maintain correct valency")
        self.chk_auto_h.toggled.connect(lambda: self._update_viewer_drag_label())
        view_ctrls.addWidget(self.chk_auto_h)

        self.chk_impose_symmetry = QtWidgets.QCheckBox("🔒 Symm.")
        self.chk_impose_symmetry.setChecked(False)
        self.chk_impose_symmetry.setToolTip(
            "GaussView style: Lock/constrain molecular point group symmetry so structural edits\n"
            "(fragment replacement, atom substitution, bond changes) automatically replicate across\n"
            "all symmetry-equivalent atoms to preserve the imposed point group."
        )
        self.chk_impose_symmetry.toggled.connect(self._on_toggle_impose_symmetry)
        view_ctrls.addWidget(self.chk_impose_symmetry)

        self.combo_view_sym_group = QtWidgets.QComboBox()
        self.combo_view_sym_group.setMaximumWidth(120)
        self.combo_view_sym_group.setToolTip(
            "Select higher (approximate supergroup), detected, or lower (subgroup) point group symmetry.\n"
            "Choosing a lower symmetry (e.g. D4h or C4v on Oh CrH6) restricts symmetry-equivalent orbits\n"
            "so edits only affect that subgroup's equivalent atoms."
        )
        self.combo_view_sym_group.addItem("Auto (Symm)", "")
        self.combo_view_sym_group.currentIndexChanged.connect(self._on_view_sym_group_changed)
        view_ctrls.addWidget(self.combo_view_sym_group)

        self.btn_reset_view = QtWidgets.QPushButton("Reset")
        self.btn_reset_view.setToolTip("Center and fit molecule in 3D viewport (Shortcut: Space)")
        self.btn_reset_view.clicked.connect(self.viewer.auto_center)
        view_ctrls.addWidget(self.btn_reset_view)

        for pl_lbl in ("XY", "XZ", "YZ"):
            btn_pl = QtWidgets.QPushButton(pl_lbl)
            btn_pl.setFixedWidth(28)
            btn_pl.setToolTip(f"Snap 3D camera to view directly onto the {pl_lbl} Cartesian plane")
            btn_pl.clicked.connect(lambda _c=False, p=pl_lbl: self.viewer.set_camera_plane_view(p))
            view_ctrls.addWidget(btn_pl)

        btn_axes = QtWidgets.QPushButton("Axes")
        btn_axes.setFixedWidth(36)
        btn_axes.setToolTip("Align camera view to look along the molecule's Principal Inertial Axes (Chemcraft / VMD)")
        btn_axes.clicked.connect(lambda: self.viewer.align_to_principal_axes())
        view_ctrls.addWidget(btn_axes)

        self.btn_camera_proj = QtWidgets.QPushButton("Persp")
        self.btn_camera_proj.setFixedWidth(42)
        self.btn_camera_proj.setToolTip("Toggle between Perspective and Orthographic parallel projection (VMD / GaussView style) (F5)")
        self.btn_camera_proj.clicked.connect(self._on_toggle_camera_projection)
        view_ctrls.addWidget(self.btn_camera_proj)

        # Compact popup menu for display styles, theme, and 3D overlays so the main bar never overflows
        self.btn_display_menu = QtWidgets.QToolButton()
        self.btn_display_menu.setText("👁️ Display ▾")
        self.btn_display_menu.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_display_menu.setToolTip("Configure 3D theme, rendering style, atom labels, and property overlays (Dipole, Charges, H-Bonds, Forces, 3D Gizmo)")
        self.menu_display_overlays = QtWidgets.QMenu(self)

        display_panel = QtWidgets.QWidget()
        display_layout = QtWidgets.QVBoxLayout(display_panel)
        display_layout.setContentsMargins(10, 8, 10, 8)
        display_layout.setSpacing(6)

        row_theme = QtWidgets.QHBoxLayout()
        row_theme.addWidget(QtWidgets.QLabel("Theme:"))
        self.combo_theme = QtWidgets.QComboBox()
        self.combo_theme.addItem("☀️ White Theme", "white")
        self.combo_theme.addItem("🌙 Dark Theme", "dark")
        self.combo_theme.setToolTip("Toggle between White and Dark viewport & application themes")
        self.combo_theme.currentIndexChanged.connect(self._on_theme_changed)
        row_theme.addWidget(self.combo_theme, stretch=1)
        display_layout.addLayout(row_theme)

        row_style = QtWidgets.QHBoxLayout()
        row_style.addWidget(QtWidgets.QLabel("Style:"))
        self.combo_style = QtWidgets.QComboBox()
        self.combo_style.addItems(["Ball & Stick", "Spacefill", "Licorice / Tube", "Wireframe"])
        self.combo_style.currentTextChanged.connect(self._on_style_changed)
        row_style.addWidget(self.combo_style, stretch=1)
        display_layout.addLayout(row_style)

        self.chk_labels = QtWidgets.QCheckBox("Atom Labels")
        self.chk_labels.setChecked(True)
        self.chk_labels.toggled.connect(self._on_toggle_labels)
        display_layout.addWidget(self.chk_labels)

        self.chk_indices = QtWidgets.QCheckBox("Atom Indices")
        self.chk_indices.setChecked(True)
        self.chk_indices.toggled.connect(self._on_toggle_indices)
        display_layout.addWidget(self.chk_indices)

        self.chk_dipole = QtWidgets.QCheckBox("Dipole Vector (μ)")
        self.chk_dipole.setToolTip("Display 3D molecular electric dipole moment vector (Debye)")
        self.chk_dipole.toggled.connect(self._on_toggle_dipole)
        display_layout.addWidget(self.chk_dipole)

        self.chk_charges = QtWidgets.QCheckBox("Partial Charges (δ)")
        self.chk_charges.setToolTip("Color atoms by partial atomic charge (Red δ- to Blue δ+) and show charge labels")
        self.chk_charges.toggled.connect(self._on_toggle_charges)
        display_layout.addWidget(self.chk_charges)

        self.chk_hbonds = QtWidgets.QCheckBox("Hydrogen Bonds (⋯)")
        self.chk_hbonds.setToolTip("Avogadro style: Automatically detect and render hydrogen bond networks as dashed cyan cylinders")
        self.chk_hbonds.toggled.connect(self._on_toggle_hbonds)
        display_layout.addWidget(self.chk_hbonds)

        self.chk_forces = QtWidgets.QCheckBox("Nuclear Forces (F)")
        self.chk_forces.setToolTip("GaussView / newBRORSEN style: Display 3D nuclear force / Cartesian gradient vectors (-dE/dx) for geometry optimization diagnostics")
        self.chk_forces.toggled.connect(self._on_toggle_forces)
        display_layout.addWidget(self.chk_forces)

        self.chk_gizmo = QtWidgets.QCheckBox("3D Axis Gizmo")
        self.chk_gizmo.setChecked(False)
        self.chk_gizmo.setToolTip(
            "Show interactive 3D Cartesian axis gizmo (X=Red, Y=Green, Z=Blue) on selected atom(s)\n"
            "to slide/translate atoms strictly along the X, Y, or Z axis."
        )
        self.chk_gizmo.toggled.connect(lambda chk: setattr(self.viewer, "show_manipulator_gizmo", chk) or self.viewer.update())
        display_layout.addWidget(self.chk_gizmo)

        self.btn_clear_surface_ribbon = QtWidgets.QPushButton("✖ Clear Surface (Esc)")
        self.btn_clear_surface_ribbon.setToolTip("Clear any displayed 3D Molecular Orbital / ESP / Density surface (Shortcut: Esc)")
        self.btn_clear_surface_ribbon.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; padding: 2px 8px;")
        self.btn_clear_surface_ribbon.clicked.connect(self._on_clear_orbital_surface)
        display_layout.addWidget(self.btn_clear_surface_ribbon)

        act_display_panel = QtWidgets.QWidgetAction(self)
        act_display_panel.setDefaultWidget(display_panel)
        self.menu_display_overlays.addAction(act_display_panel)
        self.btn_display_menu.setMenu(self.menu_display_overlays)
        view_ctrls.addWidget(self.btn_display_menu)

        view_ctrls.addStretch()

        # Ribbon lives in a horizontal scroll area so its width never forces the window wider
        ribbon_widget = QtWidgets.QWidget()
        ribbon_widget.setLayout(view_ctrls)
        view_ctrls.setContentsMargins(0, 0, 0, 0)
        ribbon_scroll = QtWidgets.QScrollArea()
        ribbon_scroll.setWidget(ribbon_widget)
        ribbon_scroll.setWidgetResizable(True)
        ribbon_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        ribbon_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ribbon_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        ribbon_scroll.setMinimumWidth(100)
        ribbon_scroll.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        ribbon_scroll.setFixedHeight(ribbon_widget.sizeHint().height() + 18)
        left_layout.addWidget(ribbon_scroll)

        # Measurement readout + inline live geometry spinbox (GaussView style)
        meas_row = QtWidgets.QHBoxLayout()
        meas_row.setContentsMargins(0, 0, 0, 0)
        meas_row.setSpacing(8)

        self.lbl_measurement = QtWidgets.QLabel("Shift+Click atoms to measure distances, angles, dihedrals")
        self.lbl_measurement.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.lbl_measurement.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.lbl_measurement.setMinimumWidth(50)
        meas_row.addWidget(self.lbl_measurement, stretch=1)

        self.lbl_live_geom = QtWidgets.QLabel("")
        self.lbl_live_geom.setStyleSheet("color: #e2e8f0; font-weight: bold;")
        self.lbl_live_geom.setVisible(False)
        meas_row.addWidget(self.lbl_live_geom)

        self.spin_live_geom = QtWidgets.QDoubleSpinBox()
        self.spin_live_geom.setFixedWidth(92)
        self.spin_live_geom.setDecimals(3)
        self.spin_live_geom.setKeyboardTracking(False)
        self.spin_live_geom.setToolTip("Directly type or adjust the selected 2-atom Distance (Å), 3-atom Angle (°), or 4-atom Dihedral (°)")
        self.spin_live_geom.setVisible(False)
        self.spin_live_geom.valueChanged.connect(self._on_live_geom_spin_changed)
        meas_row.addWidget(self.spin_live_geom)

        self.btn_scan_selected_coord = QtWidgets.QPushButton("📈 Scan...")
        self.btn_scan_selected_coord.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 2px 8px;")
        self.btn_scan_selected_coord.setToolTip(
            "GaussView style: Open QM Calculation Setup with a Relaxed PES Scan pre-configured\n"
            "for the currently selected 2-atom Bond, 3-atom Angle, or 4-atom Dihedral!"
        )
        self.btn_scan_selected_coord.setVisible(False)
        self.btn_scan_selected_coord.clicked.connect(lambda: self._on_open_qm_dialog(auto_enable_scan=True))
        meas_row.addWidget(self.btn_scan_selected_coord)

        left_layout.addLayout(meas_row)

        # Compact GaussView-style Calculation Summary Banner (shown when QM log / Molden is loaded)
        self.grp_qm_summary = QtWidgets.QFrame()
        self.grp_qm_summary.setStyleSheet(
            "QFrame { background-color: #0f172a; border: 1px solid #334155; border-radius: 5px; padding: 3px 6px; }"
        )
        qm_sum_layout = QtWidgets.QHBoxLayout(self.grp_qm_summary)
        qm_sum_layout.setContentsMargins(6, 3, 6, 3)
        qm_sum_layout.setSpacing(8)

        self.lbl_qm_summary = QtWidgets.QLabel("")
        self.lbl_qm_summary.setStyleSheet("color: #f8fafc; font-size: 11px; border: none;")
        self.lbl_qm_summary.setWordWrap(True)
        self.lbl_qm_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        qm_sum_layout.addWidget(self.lbl_qm_summary, stretch=1)

        self.btn_view_qm_log = QtWidgets.QPushButton("📄 View Log...")
        self.btn_view_qm_log.setStyleSheet("background-color: #0d9488; color: white; font-weight: bold; padding: 2px 8px; font-size: 11px;")
        self.btn_view_qm_log.setToolTip("Open full calculation output log viewer (F2)")
        self.btn_view_qm_log.clicked.connect(self._on_view_qm_output_file)
        qm_sum_layout.addWidget(self.btn_view_qm_log)

        self.btn_copy_qm_summary = QtWidgets.QPushButton("📋 Copy Summary")
        self.btn_copy_qm_summary.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 2px 8px; font-size: 11px;")
        self.btn_copy_qm_summary.setToolTip("Copy calculation summary (Energy, HOMO-LUMO, Dipole, ZPE, H, G, Imaginary Modes) to clipboard")
        self.btn_copy_qm_summary.clicked.connect(self._on_copy_qm_summary)
        qm_sum_layout.addWidget(self.btn_copy_qm_summary)

        btn_hide_qm_sum = QtWidgets.QPushButton("✖")
        btn_hide_qm_sum.setFixedWidth(22)
        btn_hide_qm_sum.setToolTip("Hide Calculation Summary banner")
        btn_hide_qm_sum.clicked.connect(lambda: self.grp_qm_summary.setVisible(False))
        qm_sum_layout.addWidget(btn_hide_qm_sum)

        self.grp_qm_summary.setVisible(False)
        left_layout.addWidget(self.grp_qm_summary)

        # Floating Interactive Quick Bond Stretch Bar (GaussView Style)
        self.grp_bond_tweak = QtWidgets.QGroupBox("📏 Interactive Bond Length Tweaker (GaussView Style)")
        tweak_layout = QtWidgets.QHBoxLayout(self.grp_bond_tweak)
        tweak_layout.setContentsMargins(6, 4, 6, 4)
        tweak_layout.setSpacing(6)
        self.lbl_tweak_bond = QtWidgets.QLabel("Bond:")
        self.lbl_tweak_bond.setStyleSheet("font-weight: bold; color: #38bdf8;")
        tweak_layout.addWidget(self.lbl_tweak_bond)

        self.slider_bond_tweak = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_bond_tweak.setRange(50, 400)  # 0.50 Å to 4.00 Å
        self.slider_bond_tweak.setValue(154)
        self.slider_bond_tweak.valueChanged.connect(self._on_bond_tweak_slider_changed)
        self.slider_bond_tweak.sliderReleased.connect(self._on_bond_tweak_slider_released)
        tweak_layout.addWidget(self.slider_bond_tweak, stretch=1)

        self.lbl_tweak_val = QtWidgets.QLabel("1.540 Å")
        self.lbl_tweak_val.setStyleSheet("font-weight: bold; font-family: monospace; min-width: 65px;")
        tweak_layout.addWidget(self.lbl_tweak_val)

        btn_tweak_close = QtWidgets.QPushButton("✕")
        btn_tweak_close.setFixedWidth(24)
        btn_tweak_close.setToolTip("Close Bond Tweaker Bar")
        btn_tweak_close.clicked.connect(lambda: self.grp_bond_tweak.setVisible(False))
        tweak_layout.addWidget(btn_tweak_close)

        self.grp_bond_tweak.setVisible(False)
        left_layout.addWidget(self.grp_bond_tweak)

        # Trajectory Player Bar (for QM geometry optimizations & IRC paths)
        self.grp_trajectory = QtWidgets.QGroupBox("Trajectory Player (Optimization Steps / IRC)")
        traj_box_layout = QtWidgets.QVBoxLayout(self.grp_trajectory)
        traj_box_layout.setContentsMargins(6, 4, 6, 4)
        traj_box_layout.setSpacing(4)

        # Row 1: Movie Playback Controls & Scrubber Slider
        row1_layout = QtWidgets.QHBoxLayout()
        row1_layout.setSpacing(4)

        self.btn_traj_first = QtWidgets.QPushButton("⏮")
        self.btn_traj_first.setFixedWidth(28)
        self.btn_traj_first.setToolTip("Jump to first / initial geometry (Home)")
        self.btn_traj_first.clicked.connect(self._on_traj_first_step)
        row1_layout.addWidget(self.btn_traj_first)

        self.btn_traj_prev = QtWidgets.QPushButton("◀")
        self.btn_traj_prev.setFixedWidth(28)
        self.btn_traj_prev.setToolTip("Previous optimization/IRC step (Left)")
        self.btn_traj_prev.clicked.connect(self._on_traj_prev_step)
        row1_layout.addWidget(self.btn_traj_prev)

        self.btn_traj_play = QtWidgets.QPushButton("▶ Play")
        self.btn_traj_play.setFixedWidth(62)
        self.btn_traj_play.setStyleSheet("font-weight: bold;")
        self.btn_traj_play.setToolTip("Play/Pause trajectory movie animation (Space)")
        self.btn_traj_play.clicked.connect(self._on_traj_toggle_play)
        row1_layout.addWidget(self.btn_traj_play)

        self.btn_traj_next = QtWidgets.QPushButton("▶")
        self.btn_traj_next.setFixedWidth(28)
        self.btn_traj_next.setToolTip("Next optimization/IRC step (Right)")
        self.btn_traj_next.clicked.connect(self._on_traj_next_step)
        row1_layout.addWidget(self.btn_traj_next)

        self.btn_traj_last = QtWidgets.QPushButton("⏭")
        self.btn_traj_last.setFixedWidth(28)
        self.btn_traj_last.setToolTip("Jump to final optimized geometry (End)")
        self.btn_traj_last.clicked.connect(self._on_traj_last_step)
        row1_layout.addWidget(self.btn_traj_last)

        self.slider_trajectory = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_trajectory.setRange(0, 0)
        self.slider_trajectory.valueChanged.connect(self._on_trajectory_slider_changed)
        row1_layout.addWidget(self.slider_trajectory, stretch=1)

        self.lbl_trajectory_step = QtWidgets.QLabel("Step 0/0")
        self.lbl_trajectory_step.setStyleSheet("font-weight: bold; color: #38bdf8; min-width: 65px;")
        row1_layout.addWidget(self.lbl_trajectory_step)

        self.combo_traj_mode = QtWidgets.QComboBox()
        self.combo_traj_mode.addItem("Loop 🔁", "loop")
        self.combo_traj_mode.addItem("Bounce ⮂", "bounce")
        self.combo_traj_mode.addItem("Once ➔", "once")
        self.combo_traj_mode.setToolTip("Trajectory animation repeat behavior (Bounce is ideal for IRC pathways)")
        self.combo_traj_mode.setFixedWidth(92)
        row1_layout.addWidget(self.combo_traj_mode)

        self.combo_traj_speed = QtWidgets.QComboBox()
        self.combo_traj_speed.addItem("0.5x", 250)
        self.combo_traj_speed.addItem("1.0x", 120)
        self.combo_traj_speed.addItem("2.0x", 60)
        self.combo_traj_speed.addItem("4.0x", 30)
        self.combo_traj_speed.setCurrentIndex(1)
        self.combo_traj_speed.setToolTip("Animation playback speed")
        self.combo_traj_speed.setFixedWidth(64)
        self.combo_traj_speed.currentIndexChanged.connect(self._on_traj_speed_changed)
        row1_layout.addWidget(self.combo_traj_speed)

        traj_box_layout.addLayout(row1_layout)

        # Row 2: Step Energy & Analysis Tools
        row2_layout = QtWidgets.QHBoxLayout()
        row2_layout.setSpacing(6)

        self.lbl_trajectory_energy = QtWidgets.QLabel("")
        self.lbl_trajectory_energy.setStyleSheet("color: #a7f3d0; font-family: monospace;")
        row2_layout.addWidget(self.lbl_trajectory_energy)

        row2_layout.addStretch()

        self.btn_export_traj_movie = QtWidgets.QPushButton("🎬 Export Movie...")
        self.btn_export_traj_movie.setStyleSheet("background-color: #7c3aed; color: #ffffff; padding: 2px 8px; font-weight: bold;")
        self.btn_export_traj_movie.setToolTip("Export molecular optimization or MD trajectory as an animated GIF or MP4/WebM video")
        self.btn_export_traj_movie.clicked.connect(self._on_export_trajectory_movie)
        row2_layout.addWidget(self.btn_export_traj_movie)

        self.btn_pes_plot = QtWidgets.QPushButton("📈 Energy Curve...")
        self.btn_pes_plot.setStyleSheet("background-color: #0284c7; color: #ffffff; padding: 2px 8px; font-weight: bold;")
        self.btn_pes_plot.setToolTip("GaussView / newBRORSEN style: Open interactive Potential Energy Surface (PES) Scan & Morse Fit plot")
        self.btn_pes_plot.clicked.connect(self._on_open_pes_plot)
        row2_layout.addWidget(self.btn_pes_plot)

        self.btn_view_loaded_log = QtWidgets.QPushButton("📄 View Log...")
        self.btn_view_loaded_log.setStyleSheet("background-color: #334155; color: #ffffff; padding: 2px 8px; font-weight: bold;")
        self.btn_view_loaded_log.setToolTip("Open full calculation output log with search and section indexing (F2)")
        self.btn_view_loaded_log.clicked.connect(self._on_view_qm_output_file)
        row2_layout.addWidget(self.btn_view_loaded_log)

        traj_box_layout.addLayout(row2_layout)

        self.grp_trajectory.setVisible(False)
        left_layout.addWidget(self.grp_trajectory)

        left_container.setMinimumWidth(250)
        splitter.addWidget(left_container)

        # Right Widget: Tabbed Control Panel (Z-Matrix Table, Raw Text ZMAT, Molden Builder Tools)
        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(QtCore.Qt.TextElideMode.ElideNone)

        # Tab 1: Interactive Z-Matrix Table & Parameter Sliders
        tab_table = QtWidgets.QWidget()
        tab_table_layout = QtWidgets.QVBoxLayout(tab_table)
        tab_table_layout.setSpacing(8)

        # Table
        self.table_widget = ZMatrixTableWidget()
        self.table_widget.before_data_changed.connect(self._push_undo_state)
        self.table_widget.data_changed.connect(self._on_table_data_changed)
        self.table_widget.atom_selected.connect(self._on_table_atom_selected)
        self.table_widget.cell_selected.connect(self._on_table_cell_selected)
        tab_table_layout.addWidget(self.table_widget, stretch=1)

        # Compact 2-row Z-Matrix action bars (gives the Z-Matrix table much more vertical room)
        row_action_layout1 = QtWidgets.QHBoxLayout()
        row_action_layout1.setSpacing(5)
        self.btn_add_atom = QtWidgets.QPushButton("➕ Add Atom")
        self.btn_add_atom.setToolTip("Add new atom line to Z-Matrix (opens periodic table)")
        self.btn_add_atom.clicked.connect(self._on_add_line)

        self.btn_del_atom = QtWidgets.QPushButton("🗑️ Delete")
        self.btn_del_atom.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold;")
        self.btn_del_atom.setToolTip("Delete currently selected atom(s) from Z-Matrix (Shortcut: Del)")
        self.btn_del_atom.clicked.connect(self._on_delete_selected_atom)

        self.btn_add_hydrogens = QtWidgets.QPushButton("💧 Add H")
        self.btn_add_hydrogens.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold;")
        self.btn_add_hydrogens.setToolTip("Complete valence of all atoms with standard geometric hydrogens (Ctrl+H)")
        self.btn_add_hydrogens.clicked.connect(self._on_add_hydrogens)

        self.btn_close_ring = QtWidgets.QPushButton("🔗 Form Bond")
        self.btn_close_ring.setStyleSheet("background-color: #0d9488; color: #ffffff; font-weight: bold;")
        self.btn_close_ring.setToolTip("Select 2 atoms and click to bond them (ring closure or any new bond). Keeps all existing bonds, removes the H that is replaced, and verifies valence everywhere")
        self.btn_close_ring.clicked.connect(self._on_close_ring_clicked)

        self.btn_reorder_zmat = QtWidgets.QPushButton("🔄 Reorder")
        self.btn_reorder_zmat.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold;")
        self.btn_reorder_zmat.setToolTip("Click atoms sequentially in 3D viewport to reorder Z-matrix connectivity (Molden ZMEORD)")
        self.btn_reorder_zmat.clicked.connect(self._on_reorder_zmat_clicked)

        self.btn_read_zmat = QtWidgets.QPushButton("📂 Open ZMAT")
        self.btn_read_zmat.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold;")
        self.btn_read_zmat.setToolTip("Open and load a Z-matrix file (*.zmat, *.dat, *.inp, *.txt)")
        self.btn_read_zmat.clicked.connect(self._on_open_zmat)
        self.btn_read_zmat.setVisible(False)  # Accessible via File -> Open File (Ctrl+O) and Raw ZMAT tab

        row_action_layout1.addWidget(self.btn_add_atom)
        row_action_layout1.addWidget(self.btn_del_atom)
        row_action_layout1.addWidget(self.btn_add_hydrogens)
        row_action_layout1.addWidget(self.btn_close_ring)
        row_action_layout1.addWidget(self.btn_reorder_zmat)
        tab_table_layout.addLayout(row_action_layout1)

        row_action_layout_opt = QtWidgets.QHBoxLayout()
        row_action_layout_opt.setSpacing(5)
        self.combo_opt_method = QtWidgets.QComboBox()
        self.combo_opt_method.addItem("UFF / MMFF94s (Analytical)", "uff_mmff")
        self.combo_opt_method.addItem("Semiempirical QM (GFN0-xTB)", "semiempirical")
        self.combo_opt_method.setToolTip(
            "Select geometry optimization & vibrational engine:\n"
            "• UFF / MMFF94s (Analytical): Fast analytical force field with periodic torsions & Gasteiger electrostatics\n"
            "• Semiempirical QM (GFN0-xTB): Self-Consistent Tight-Binding QM with Wiberg bond orders, HOMO-LUMO gap & SCC charges"
        )
        self.combo_opt_method.currentIndexChanged.connect(self._on_opt_method_changed)

        self.btn_optimize_ff = QtWidgets.QPushButton("⚡ Optimize")
        self.btn_optimize_ff.setStyleSheet("background-color: #7c3aed; color: #ffffff; font-weight: bold;")
        self.btn_optimize_ff.setToolTip("Optimize molecular geometry using the selected force field or semiempirical QM method (Ctrl+M)")
        self.btn_optimize_ff.clicked.connect(self._on_optimize_geometry)

        self.btn_conformer_search = QtWidgets.QPushButton("🔄 Conformers")
        self.btn_conformer_search.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold;")
        self.btn_conformer_search.setToolTip(
            "Avogadro / Gabedit style: Systematic & stochastic rotor conformational search.\n"
            "Finds the lowest-energy conformer and loads the ranked conformer ensemble into the Trajectory Player!"
        )
        self.btn_conformer_search.clicked.connect(self._on_conformer_search)

        # Compact Variables Menu Button (keeps individual buttons alive on self for tests & programmatic access)
        self.btn_vars_menu = QtWidgets.QToolButton()
        self.btn_vars_menu.setText("🔤 Variables ▾")
        self.btn_vars_menu.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_vars_menu.setStyleSheet("background-color: #0369a1; color: #ffffff; font-weight: bold; padding: 4px 8px; border-radius: 4px;")
        self.btn_vars_menu.setToolTip("Assign or clear symbolic Z-Matrix variables (Symm. Bonds, Symm. All, All Variables, Numbers Only)")
        menu_zmat_vars = QtWidgets.QMenu(self)

        vars_panel = QtWidgets.QWidget()
        row_vars_layout = QtWidgets.QVBoxLayout(vars_panel)
        row_vars_layout.setContentsMargins(8, 6, 8, 6)
        row_vars_layout.setSpacing(5)

        self.btn_sym_bonds = QtWidgets.QPushButton("🔗 Symm. Bonds Only")
        self.btn_sym_bonds.setStyleSheet("background-color: #0369a1; color: #ffffff; font-weight: bold;")
        self.btn_sym_bonds.setToolTip(
            "Assign symmetric variables ONLY to bond distances (e.g. R1 for C-C, R2 for C-H in benzene)\n"
            "while keeping all angles and dihedrals as fixed numerical constants."
        )
        self.btn_sym_bonds.clicked.connect(lambda: (menu_zmat_vars.close(), self._on_assign_symmetric_bond_variables()))

        self.btn_sym_all = QtWidgets.QPushButton("🔗 Symm. All (Bonds/Angles/Dihedrals)")
        self.btn_sym_all.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold;")
        self.btn_sym_all.setToolTip("Assign shared variables to all symmetry-equivalent bonds, angles, and dihedrals")
        self.btn_sym_all.clicked.connect(lambda: (menu_zmat_vars.close(), self._on_assign_symmetric_variables()))

        self.btn_indiv_vars = QtWidgets.QPushButton("🔤 All Individual Variables")
        self.btn_indiv_vars.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold;")
        self.btn_indiv_vars.setToolTip("Assign an individual variable (R1..Rn, A1..An, D1..Dn) to every coordinate")
        self.btn_indiv_vars.clicked.connect(lambda: (menu_zmat_vars.close(), self._on_assign_individual_variables()))

        self.btn_clear_vars = QtWidgets.QPushButton("🔢 Numbers Only (Clear Variables)")
        self.btn_clear_vars.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold;")
        self.btn_clear_vars.setToolTip("Clear all symbolic variables and use explicit numbers")
        self.btn_clear_vars.clicked.connect(lambda: (menu_zmat_vars.close(), self._on_clear_variables()))

        row_vars_layout.addWidget(self.btn_sym_bonds)
        row_vars_layout.addWidget(self.btn_sym_all)
        row_vars_layout.addWidget(self.btn_indiv_vars)
        row_vars_layout.addWidget(self.btn_clear_vars)

        act_vars_panel = QtWidgets.QWidgetAction(self)
        act_vars_panel.setDefaultWidget(vars_panel)
        menu_zmat_vars.addAction(act_vars_panel)
        self.btn_vars_menu.setMenu(menu_zmat_vars)

        row_action_layout_opt.addWidget(self.combo_opt_method, stretch=2)
        row_action_layout_opt.addWidget(self.btn_optimize_ff, stretch=1)
        row_action_layout_opt.addWidget(self.btn_conformer_search, stretch=1)
        row_action_layout_opt.addWidget(self.btn_vars_menu)
        tab_table_layout.addLayout(row_action_layout_opt)

        # Selected Atom Fine Adjustment Group (Molden Style Slider Adjustment + GaussView SmartSlides)
        self.grp_adjust = QtWidgets.QGroupBox("Interactive Parameter Tuning / SmartSlides (Selected Atom or 2/3/4-Atom Selection)")
        adjust_layout = QtWidgets.QGridLayout(self.grp_adjust)
        adjust_layout.setHorizontalSpacing(10)
        adjust_layout.setVerticalSpacing(6)

        # SmartSlides Displacement Mode & Stereocenter Inversion bar
        row_smart_mode = QtWidgets.QHBoxLayout()
        row_smart_mode.setSpacing(6)
        row_smart_mode.addWidget(QtWidgets.QLabel("Displacement Mode:"))
        self.combo_smart_mode = QtWidgets.QComboBox()
        self.combo_smart_mode.addItem("Move Substituent Group (Keep Atom 1 Fixed)", "group")
        self.combo_smart_mode.addItem("Move Both Groups Symmetrically (50/50)", "symmetric")
        self.combo_smart_mode.addItem("Move Single Atom Only", "atom")
        self.combo_smart_mode.setToolTip(
            "GaussView SmartSlides mode when 2, 3, or 4 atoms are selected in the 3D viewport:\n"
            "• Move Substituent Group: Translates/rotates the connected substituent subgraph\n"
            "• Move Both Groups Symmetrically: Splits displacement 50/50 across both sides\n"
            "• Move Single Atom Only: Moves only the terminal selected atom"
        )
        row_smart_mode.addWidget(self.combo_smart_mode, stretch=1)

        self.btn_invert_center = QtWidgets.QPushButton("🔄 Invert Chiral Center (R/S)")
        self.btn_invert_center.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold; padding: 4px 8px;")
        self.btn_invert_center.setToolTip(
            "Invert local tetrahedral/pyramidal stereocenter (select chiral center atom or center + 2 substituents)\n"
            "without flipping the rest of the molecule!"
        )
        self.btn_invert_center.clicked.connect(self._on_invert_stereocenter)
        row_smart_mode.addWidget(self.btn_invert_center)
        adjust_layout.addLayout(row_smart_mode, 0, 0, 1, 3)

        # Bond length slider
        adjust_layout.addWidget(QtWidgets.QLabel("Bond Length:"), 1, 0)
        self.spin_bond = QtWidgets.QDoubleSpinBox()
        self.spin_bond.setRange(0.1, 10.0)
        self.spin_bond.setSingleStep(0.01)
        self.spin_bond.valueChanged.connect(self._on_spin_bond_changed)
        adjust_layout.addWidget(self.spin_bond, 1, 1)

        self.slider_bond = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_bond.setRange(50, 400) # 0.5 to 4.0 Å (step 0.01)
        self.slider_bond.sliderPressed.connect(self._push_undo_state)
        self.slider_bond.valueChanged.connect(self._on_slider_bond_changed)
        adjust_layout.addWidget(self.slider_bond, 1, 2)

        # Angle slider
        adjust_layout.addWidget(QtWidgets.QLabel("Angle:"), 2, 0)
        self.spin_angle = QtWidgets.QDoubleSpinBox()
        self.spin_angle.setRange(0.0, 180.0)
        self.spin_angle.setSingleStep(0.5)
        self.spin_angle.valueChanged.connect(self._on_spin_angle_changed)
        adjust_layout.addWidget(self.spin_angle, 2, 1)

        self.slider_angle = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_angle.setRange(0, 1800) # 0 to 180 deg (step 0.1)
        self.slider_angle.sliderPressed.connect(self._push_undo_state)
        self.slider_angle.valueChanged.connect(self._on_slider_angle_changed)
        adjust_layout.addWidget(self.slider_angle, 2, 2)

        # Dihedral slider
        adjust_layout.addWidget(QtWidgets.QLabel("Dihedral:"), 3, 0)
        self.spin_dihedral = QtWidgets.QDoubleSpinBox()
        self.spin_dihedral.setRange(-180.0, 180.0)
        self.spin_dihedral.setSingleStep(1.0)
        self.spin_dihedral.valueChanged.connect(self._on_spin_dihedral_changed)
        adjust_layout.addWidget(self.spin_dihedral, 3, 1)

        self.slider_dihedral = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_dihedral.setRange(-1800, 1800) # -180 to 180 deg (step 0.1)
        self.slider_dihedral.sliderPressed.connect(self._push_undo_state)
        self.slider_dihedral.valueChanged.connect(self._on_slider_dihedral_changed)
        adjust_layout.addWidget(self.slider_dihedral, 3, 2)

        tab_table_layout.addWidget(self.grp_adjust)

        # Z-Matrix Symbolic Variable Sliders (ZOPT Coordinate Inspection)
        self.grp_var_sliders = QtWidgets.QGroupBox("Z-Matrix Symbolic Variables (ZOPT Sliders)")
        self.var_sliders_layout = QtWidgets.QGridLayout(self.grp_var_sliders)
        self.var_sliders_layout.setSpacing(6)
        self.grp_var_sliders.setVisible(False)
        tab_table_layout.addWidget(self.grp_var_sliders)

        self.tabs.addTab(tab_table, "Z-Matrix")

        # Tab 2: Raw Text Z-Matrix / Gaussian Format
        tab_raw = QtWidgets.QWidget()
        tab_raw_layout = QtWidgets.QVBoxLayout(tab_raw)
        self.text_raw_zmat = QtWidgets.QPlainTextEdit()
        tab_raw_layout.addWidget(QtWidgets.QLabel("Direct Z-Matrix Text (Molden / Gaussian format):"))
        tab_raw_layout.addWidget(self.text_raw_zmat, stretch=1)

        raw_vars_layout = QtWidgets.QHBoxLayout()
        raw_vars_layout.setSpacing(6)
        btn_raw_sym_bonds = QtWidgets.QPushButton("🔗 Symm. Bonds")
        btn_raw_sym_bonds.setStyleSheet("background-color: #0369a1; color: #ffffff; font-weight: bold;")
        btn_raw_sym_bonds.setToolTip("Assign symmetric variables ONLY to bond distances (keep angles & dihedrals fixed)")
        btn_raw_sym_bonds.clicked.connect(self._on_assign_symmetric_bond_variables)
        btn_raw_sym_all = QtWidgets.QPushButton("🔗 Symm. All")
        btn_raw_sym_all.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold;")
        btn_raw_sym_all.clicked.connect(self._on_assign_symmetric_variables)
        btn_raw_indiv = QtWidgets.QPushButton("🔤 All Variables")
        btn_raw_indiv.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold;")
        btn_raw_indiv.clicked.connect(self._on_assign_individual_variables)
        btn_raw_clear = QtWidgets.QPushButton("🔢 Numbers Only")
        btn_raw_clear.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold;")
        btn_raw_clear.clicked.connect(self._on_clear_variables)
        raw_vars_layout.addWidget(btn_raw_sym_bonds)
        raw_vars_layout.addWidget(btn_raw_sym_all)
        raw_vars_layout.addWidget(btn_raw_indiv)
        raw_vars_layout.addWidget(btn_raw_clear)
        tab_raw_layout.addLayout(raw_vars_layout)

        raw_btn_layout = QtWidgets.QHBoxLayout()
        btn_open_raw = QtWidgets.QPushButton("📂 Open ZMAT File...")
        btn_open_raw.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold;")
        btn_open_raw.clicked.connect(self._on_open_zmat)
        btn_save_raw = QtWidgets.QPushButton("💾 Save ZMAT File...")
        btn_save_raw.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold;")
        btn_save_raw.clicked.connect(self._on_save_zmat)
        btn_apply_raw = QtWidgets.QPushButton("Apply Text Z-Matrix to 3D Model")
        btn_apply_raw.clicked.connect(self._on_apply_text_zmat)
        raw_btn_layout.addWidget(btn_open_raw)
        raw_btn_layout.addWidget(btn_save_raw)
        raw_btn_layout.addWidget(btn_apply_raw)
        tab_raw_layout.addLayout(raw_btn_layout)
        self.tabs.addTab(tab_raw, "Raw ZMAT Text")

        # Tab 3: Chemical Builders & Fragment Substitution
        tab_builder = QtWidgets.QWidget()
        tab_builder_layout = QtWidgets.QVBoxLayout(tab_builder)

        # Presets Group
        grp_preset = QtWidgets.QGroupBox("Load Preset Molecule")
        preset_layout = QtWidgets.QHBoxLayout(grp_preset)
        self.combo_presets = QtWidgets.QComboBox()
        self.combo_presets.addItems(list(SAMPLE_MOLECULES.keys()))
        btn_load_preset = QtWidgets.QPushButton("Load Preset")
        btn_load_preset.clicked.connect(lambda: self.load_preset_molecule(self.combo_presets.currentText()))
        preset_layout.addWidget(self.combo_presets, stretch=1)
        preset_layout.addWidget(btn_load_preset)
        tab_builder_layout.addWidget(grp_preset)

        # Substituent / Fragment Attachment Group (Molden + GaussView Categorized Library)
        grp_fragment = QtWidgets.QGroupBox("Chemical Fragments, Rings, Solvents & Custom Library")
        frag_layout = QtWidgets.QVBoxLayout(grp_fragment)
        frag_layout.setSpacing(5)

        row_frag_cat = QtWidgets.QHBoxLayout()
        row_frag_cat.setSpacing(6)
        row_frag_cat.addWidget(QtWidgets.QLabel("Category:"))
        self.combo_frag_category = QtWidgets.QComboBox()
        self.combo_frag_category.addItems(FRAGMENT_CATEGORIES)
        self.combo_frag_category.currentTextChanged.connect(self._on_fragment_category_changed)
        row_frag_cat.addWidget(self.combo_frag_category, stretch=1)

        btn_save_custom_frag = QtWidgets.QPushButton("💾 Save Selection as Custom Fragment...")
        btn_save_custom_frag.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold; padding: 4px 8px;")
        btn_save_custom_frag.setToolTip(
            "Save the currently selected atoms (or whole molecule) to your persistent Custom User Library\n"
            "(~/.config/AgyMolden/custom_fragments.json) for reuse across sessions."
        )
        btn_save_custom_frag.clicked.connect(self._on_save_custom_fragment)
        row_frag_cat.addWidget(btn_save_custom_frag)
        frag_layout.addLayout(row_frag_cat)

        self.combo_fragment = QtWidgets.QComboBox()
        self.combo_fragment.addItems(list(FRAGMENTS.keys()))
        self.combo_fragment.currentTextChanged.connect(self._on_fragment_combo_changed)
        frag_layout.addWidget(QtWidgets.QLabel("Select Fragment / Molecule / Solvent:"))
        frag_layout.addWidget(self.combo_fragment)

        frag_layout.addWidget(QtWidgets.QLabel("Fragment Attachment Atom (binds to molecule):"))
        self.combo_frag_connect = QtWidgets.QComboBox()
        self.combo_frag_connect.setToolTip("Select which atom in the fragment binds to the selected molecule atom")
        self.combo_frag_connect.currentIndexChanged.connect(self._on_frag_connect_atom_changed)
        frag_layout.addWidget(self.combo_frag_connect)

        row_frag_btns = QtWidgets.QHBoxLayout()
        row_frag_btns.setSpacing(6)
        btn_attach_frag = QtWidgets.QPushButton("🔗 Attach at Selected Atom")
        btn_attach_frag.clicked.connect(self._on_attach_fragment)
        row_frag_btns.addWidget(btn_attach_frag)

        btn_add_free_mol = QtWidgets.QPushButton("💧 Add to Free Space (Canvas)")
        btn_add_free_mol.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold;")
        btn_add_free_mol.setToolTip(
            "Add selected molecule (Water, Solvent, or Fragment) into free space on the canvas without bonding to existing atoms.\n"
            "Perfect for building water clusters, solvent cages, and multi-component systems!"
        )
        btn_add_free_mol.clicked.connect(lambda: self._on_add_molecule_free_space())
        row_frag_btns.addWidget(btn_add_free_mol)
        frag_layout.addLayout(row_frag_btns)

        tab_builder_layout.addWidget(grp_fragment)

        # Optimization / Clean-up / Constraints / 1D Dihedral Scan
        grp_tools = QtWidgets.QGroupBox("Molecular Geometry, Frozen Constraints & Conformer Scan")
        tools_layout = QtWidgets.QVBoxLayout(grp_tools)
        tools_layout.setSpacing(5)

        row_tools_top = QtWidgets.QHBoxLayout()
        row_tools_top.setSpacing(6)
        btn_add_h = QtWidgets.QPushButton("💧 Complete Valence (Add H)")
        btn_add_h.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold;")
        btn_add_h.clicked.connect(self._on_add_hydrogens)
        row_tools_top.addWidget(btn_add_h)

        btn_opt = QtWidgets.QPushButton("⚡ Optimize Geometry")
        btn_opt.setStyleSheet("background-color: #7c3aed; color: #ffffff; font-weight: bold;")
        btn_opt.clicked.connect(self._on_optimize_geometry)
        row_tools_top.addWidget(btn_opt)
        tools_layout.addLayout(row_tools_top)

        row_freeze = QtWidgets.QHBoxLayout()
        row_freeze.setSpacing(6)
        self.btn_freeze_sel = QtWidgets.QPushButton("❄️ Freeze Selected Atoms")
        self.btn_freeze_sel.setStyleSheet("background-color: #0369a1; color: #ffffff; font-weight: bold;")
        self.btn_freeze_sel.setToolTip("Lock selected atoms at their exact 3D coordinates during geometry optimization")
        self.btn_freeze_sel.clicked.connect(self._on_freeze_selected_atoms)
        row_freeze.addWidget(self.btn_freeze_sel)

        self.btn_unfreeze_all = QtWidgets.QPushButton("🔥 Unfreeze All (0 frozen)")
        self.btn_unfreeze_all.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold;")
        self.btn_unfreeze_all.setToolTip("Clear all frozen atom coordinate constraints")
        self.btn_unfreeze_all.clicked.connect(self._on_unfreeze_all_atoms)
        row_freeze.addWidget(self.btn_unfreeze_all)
        tools_layout.addLayout(row_freeze)

        row_invert_scan = QtWidgets.QHBoxLayout()
        row_invert_scan.setSpacing(6)
        btn_invert_local = QtWidgets.QPushButton("🔄 Invert Chiral Center (R/S)")
        btn_invert_local.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold;")
        btn_invert_local.setToolTip("Invert local tetrahedral/pyramidal stereocenter (or whole molecule if 0 atoms selected)")
        btn_invert_local.clicked.connect(self._on_invert_stereocenter)
        row_invert_scan.addWidget(btn_invert_local)

        btn_dih_scan = QtWidgets.QPushButton("📈 1D Dihedral Rotor Scan...")
        btn_dih_scan.setStyleSheet("background-color: #0d9488; color: #ffffff; font-weight: bold;")
        btn_dih_scan.setToolTip(
            "Select 4 atoms defining a dihedral torsion angle and run a 1D Potential Energy Surface (PES)\n"
            "rotor scan to explore conformers and rotational barriers."
        )
        btn_dih_scan.clicked.connect(self._on_open_dihedral_scan_dialog)
        row_invert_scan.addWidget(btn_dih_scan)
        tools_layout.addLayout(row_invert_scan)

        tab_builder_layout.addWidget(grp_tools)

        # Precision 3D Coordinate Centering & Axis/Plane Alignment Group
        grp_orient = QtWidgets.QGroupBox("Precision 3D Centering & Axis / Plane Alignment")
        orient_layout = QtWidgets.QVBoxLayout(grp_orient)
        orient_layout.setSpacing(6)

        lbl_orient_hint = QtWidgets.QLabel(
            "Center at (0,0,0) or align selected atom(s) / bond / plane to Cartesian axes:"
        )
        lbl_orient_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        lbl_orient_hint.setWordWrap(True)
        orient_layout.addWidget(lbl_orient_hint)

        row_center_btns = QtWidgets.QHBoxLayout()
        row_center_btns.setSpacing(4)
        btn_c_com = QtWidgets.QPushButton("🎯 Center COM")
        btn_c_com.setToolTip("Translate molecule so Center of Mass (COM) is at (0, 0, 0)")
        btn_c_com.clicked.connect(lambda: self._on_center_molecule_mode("com"))
        btn_c_cent = QtWidgets.QPushButton("🎯 Centroid")
        btn_c_cent.setToolTip("Translate molecule so Geometric Centroid is at (0, 0, 0)")
        btn_c_cent.clicked.connect(lambda: self._on_center_molecule_mode("centroid"))
        btn_c_atom = QtWidgets.QPushButton("🎯 Selected Atom → (0,0,0)")
        btn_c_atom.setToolTip("Translate molecule so the currently selected atom is placed at (0, 0, 0)")
        btn_c_atom.clicked.connect(lambda: self._on_center_molecule_mode("atom"))
        row_center_btns.addWidget(btn_c_com)
        row_center_btns.addWidget(btn_c_cent)
        row_center_btns.addWidget(btn_c_atom)
        orient_layout.addLayout(row_center_btns)

        row_axis_btns = QtWidgets.QHBoxLayout()
        row_axis_btns.setSpacing(4)
        row_axis_btns.addWidget(QtWidgets.QLabel("Align Axis/Bond:"))
        for ax_name in ("+X", "+Y", "+Z"):
            b_ax = QtWidgets.QPushButton(f"→ {ax_name}")
            b_ax.setStyleSheet("background-color: #0d9488; color: #ffffff; font-weight: bold; padding: 4px 8px;")
            b_ax.setToolTip(
                f"Rotate molecule around origin so the selected 2-atom vector (or 1-atom vector / principal axis) points along {ax_name}"
            )
            b_ax.clicked.connect(lambda checked=False, a=ax_name: self._on_align_to_axis(a))
            row_axis_btns.addWidget(b_ax)
        orient_layout.addLayout(row_axis_btns)

        row_plane_btns = QtWidgets.QHBoxLayout()
        row_plane_btns.setSpacing(4)
        row_plane_btns.addWidget(QtWidgets.QLabel("Align Plane:"))
        for pl_name in ("XY", "XZ", "YZ"):
            b_pl = QtWidgets.QPushButton(f"→ {pl_name} Plane")
            b_pl.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; padding: 4px 8px;")
            b_pl.setToolTip(
                f"Rotate molecule around origin so the plane of 3 selected atoms (or molecular plane) lies in the {pl_name} plane"
            )
            b_pl.clicked.connect(lambda checked=False, p=pl_name: self._on_align_to_plane(p))
            row_plane_btns.addWidget(b_pl)
        orient_layout.addLayout(row_plane_btns)

        tab_builder_layout.addWidget(grp_orient)

        tab_builder_layout.addStretch()
        self.tab_builder = tab_builder
        self.tabs.addTab(tab_builder, "Presets & Fragments")

        # Tabs 4, 5, 6 (Vibrations, Symmetry, Molecular Orbitals) via UIInspectorTabsMixin
        self._init_vibrations_tab()
        self._init_symmetry_tab()
        self._init_orbitals_tab()

        right_layout.addWidget(self.tabs)
        # Tab contents must never dictate the window width
        self.tabs.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Expanding)
        right_container.setMinimumWidth(420)
        splitter.addWidget(right_container)

        # Splitter dynamic sizing: responsive in full window view
        splitter.setStretchFactor(0, 5) # 3D viewport
        splitter.setStretchFactor(1, 6) # ZMAT editor controls
        splitter.setSizes([600, 680])

        # Populate fragment connection atom options for default fragment without setting clipboard memory
        frag = FRAGMENTS.get(self.combo_fragment.currentText())
        if frag and hasattr(self, "combo_frag_connect"):
            self.combo_frag_connect.blockSignals(True)
            self.combo_frag_connect.clear()
            for i, sym in enumerate(frag["symbols"]):
                self.combo_frag_connect.addItem(f"Atom #{i + 1}: {sym}", i)
            default_idx = int(frag.get("connect_idx", 0))
            if 0 <= default_idx < self.combo_frag_connect.count():
                self.combo_frag_connect.setCurrentIndex(default_idx)
            self.combo_frag_connect.blockSignals(False)

    def _create_menus(self):
        menubar = self.menuBar()

        # File Menu
        menu_file = menubar.addMenu("&File")

        act_new = QtGui.QAction("&New Molecule", self)
        act_new.setShortcut(QtGui.QKeySequence.StandardKey.New)
        act_new.triggered.connect(self._on_new_molecule)
        menu_file.addAction(act_new)

        menu_file.addSeparator()

        act_open_any = QtGui.QAction("📂 &Open File (All Formats: QM Log, XYZ, Molden, ZMAT)...", self)
        act_open_any.setShortcut("Ctrl+O")
        act_open_any.setToolTip("Universal file opener sorted newest-first (.out, .log, .xyz, .molden.input, .molden, .gbw, .cube, .zmat, .inp)")
        act_open_any.triggered.connect(self._on_open_universal_file)
        menu_file.addAction(act_open_any)

        act_open_zmat = QtGui.QAction("Open &Z-Matrix File...", self)
        act_open_zmat.triggered.connect(self._on_open_zmat)
        menu_file.addAction(act_open_zmat)

        act_open_xyz = QtGui.QAction("Import &XYZ File...", self)
        act_open_xyz.triggered.connect(self._on_import_xyz)
        menu_file.addAction(act_open_xyz)

        act_open_inp = QtGui.QAction("Open newBRORSEN &Input File (*.inp)...", self)
        act_open_inp.setShortcut("Ctrl+I")
        act_open_inp.setToolTip("Open a newBRORSEN input file (.inp) into AGYMolden and the Input Builder")
        act_open_inp.triggered.connect(self._on_open_brorsen_input_file)
        menu_file.addAction(act_open_inp)

        act_open_smiles = QtGui.QAction("Open from &SMILES / InChI...", self)
        act_open_smiles.setToolTip("Input a SMILES string (e.g. c1ccccc1) or InChI string to build a 3D structure automatically")
        act_open_smiles.triggered.connect(self._on_prompt_smiles_or_inchi)
        menu_file.addAction(act_open_smiles)

        act_open_qm_log = QtGui.QAction("Import &QM Log / Output File (newBRORSEN, Gaussian, ORCA)...", self)
        act_open_qm_log.setToolTip("Open output file from newBRORSEN (.out), Gaussian, ORCA, Q-Chem, or xTB to view structures, optimization steps and frequencies")
        act_open_qm_log.triggered.connect(self._on_import_qm_log)
        menu_file.addAction(act_open_qm_log)

        act_view_out = QtGui.QAction("📄 View &Calculation Output Log (*.out, *.log)...", self)
        act_view_out.setToolTip("Inspect full output log with clickable section indexing and search (F2)")
        act_view_out.setShortcut("F2")
        act_view_out.triggered.connect(self._on_view_qm_output_file)
        menu_file.addAction(act_view_out)

        act_open_cube = QtGui.QAction("Open Volumetric &Grid / Orbitals (.cube, .molden)...", self)
        act_open_cube.setToolTip("Open Gaussian .cube file or Molden file to extract and view 3D Molecular Orbitals (HOMO/LUMO) and ESP isosurfaces")
        act_open_cube.triggered.connect(self._on_open_volumetric_dialog)
        menu_file.addAction(act_open_cube)

        act_orb_gallery = QtGui.QAction("📸 All Molecular Orbitals Snapshot &Gallery...", self)
        act_orb_gallery.setToolTip("Batch capture all or selected molecular orbitals into a contact sheet collage or folder of images (like Gabedit)")
        act_orb_gallery.triggered.connect(self._on_open_orbital_snapshot_gallery)
        menu_file.addAction(act_orb_gallery)

        act_esp_2d = QtGui.QAction("2D Electrostatic &Potential (ESP) Slicer...", self)
        act_esp_2d.setToolTip("Open Molden file and plot 2D ESP slice contour maps on XY, XZ, or YZ planes")
        act_esp_2d.triggered.connect(self._on_open_esp_2d_dialog)
        menu_file.addAction(act_esp_2d)

        act_multiwfn = QtGui.QAction("Multiwfn &Wavefunction Suite (Bonds / Fukui / QTAIM / CM5)...", self)
        act_multiwfn.setToolTip("Run Multiwfn Mayer/Wiberg bond orders, Conceptual DFT Fukui indices, QTAIM critical points, and CM5 charges")
        act_multiwfn.triggered.connect(self._on_open_multiwfn_dialog)
        menu_file.addAction(act_multiwfn)

        self.menu_recent = menu_file.addMenu("Open &Recent")
        self._update_recent_files_menu()

        menu_file.addSeparator()

        act_save_zmat = QtGui.QAction("Save Z-Matrix (&ZMAT)...", self)
        act_save_zmat.setShortcut("Ctrl+S")
        act_save_zmat.triggered.connect(self._on_save_zmat)
        menu_file.addAction(act_save_zmat)

        act_save_num = QtGui.QAction("Save Z-Matrix with &Numbers...", self)
        act_save_num.setToolTip("Every bond, angle and dihedral written as a number")
        act_save_num.triggered.connect(lambda: self._on_save_zmat_format("numeric"))
        menu_file.addAction(act_save_num)

        act_save_var = QtGui.QAction("Save Z-Matrix with &Variables...", self)
        act_save_var.setToolTip("Every coordinate written as a variable, with a Variables: block of values "
                                "(shared/symmetric variables are preserved)")
        act_save_var.triggered.connect(lambda: self._on_save_zmat_format("variables"))
        menu_file.addAction(act_save_var)

        menu_vars = menubar.addMenu("&Variables")
        act_sym_bonds = QtGui.QAction("Assign Symmetric &Bond Variables Only (Keep Angles/Dihedrals Fixed)", self)
        act_sym_bonds.setToolTip("Equivalent bond distances (e.g. C-C and C-H in benzene) share variables (R1, R2) "
                                 "while angles and dihedrals remain fixed constants")
        act_sym_bonds.triggered.connect(self._on_assign_symmetric_bond_variables)
        menu_vars.addAction(act_sym_bonds)

        act_sym_vars = QtGui.QAction("Assign &Symmetric Variables (All: Bonds, Angles, Dihedrals)", self)
        act_sym_vars.setToolTip("Equivalent coordinates (e.g. all C-C and all C-H bonds in benzene) share one "
                                "variable, reducing the number of optimization parameters")
        act_sym_vars.triggered.connect(self._on_assign_symmetric_variables)
        menu_vars.addAction(act_sym_vars)
        act_all_vars = QtGui.QAction("Assign &Individual Variables", self)
        act_all_vars.triggered.connect(self._on_assign_individual_variables)
        menu_vars.addAction(act_all_vars)
        act_clear_vars = QtGui.QAction("&Clear Variables (use numbers)", self)
        act_clear_vars.triggered.connect(self._on_clear_variables)
        menu_vars.addAction(act_clear_vars)

        act_save_xyz = QtGui.QAction("Export Coordinates (&XYZ)...", self)
        act_save_xyz.triggered.connect(self._on_export_xyz)
        menu_file.addAction(act_save_xyz)

        act_quick_orca = QtGui.QAction("🚀 Quick Run &ORCA Calculation...", self)
        act_quick_orca.setShortcut("Ctrl+R")
        act_quick_orca.setToolTip("Quick 1-Click setup and execution of ORCA calculations for coursework")
        act_quick_orca.triggered.connect(self._on_quick_run_orca)
        menu_file.addAction(act_quick_orca)

        act_generate_qm_input = QtGui.QAction("⚡ Generate &QM Calculation Input...", self)
        act_generate_qm_input.setShortcut("Ctrl+G")
        act_generate_qm_input.setToolTip("Set up and generate input files for Gaussian, ORCA, PySCF, openLOWDIN, or xTB")
        act_generate_qm_input.triggered.connect(self._on_open_qm_dialog)
        menu_file.addAction(act_generate_qm_input)

        act_generate_brorsen_input = QtGui.QAction("✨ new&BRORSEN Input Builder (Positrons, NEO)...", self)
        act_generate_brorsen_input.setShortcut("Ctrl+B")
        act_generate_brorsen_input.setToolTip("Set up multi-stage input files for newBRORSEN (Positrons, NEO, and CCSD(T))")
        act_generate_brorsen_input.triggered.connect(self._on_open_brorsen_dialog)
        menu_file.addAction(act_generate_brorsen_input)

        act_job_monitor = QtGui.QAction("🖥️ Calculation &Job Monitor...", self)
        act_job_monitor.setShortcut("Ctrl+J")
        act_job_monitor.setToolTip("View live calculation progress, streaming output logs, and real-time 3D updates")
        act_job_monitor.triggered.connect(self._on_open_job_monitor)
        menu_file.addAction(act_job_monitor)

        act_remote_transfer = QtGui.QAction("📁 Remote &File Transfer (SSH / Cluster)...", self)
        act_remote_transfer.setToolTip("Browse and transfer input and output files between local machine and remote cluster")
        act_remote_transfer.triggered.connect(self._on_open_remote_file_manager)
        menu_file.addAction(act_remote_transfer)

        act_qm_engines = QtGui.QAction("🔧 Configure &QM Engines & Routes...", self)
        act_qm_engines.setToolTip("Detect installed QM packages (ORCA, PySCF, Gaussian, openLOWDIN, xTB, Q-Chem) and configure custom executable routes")
        act_qm_engines.triggered.connect(self._on_configure_qm_engines)
        menu_file.addAction(act_qm_engines)

        act_summary_card = QtGui.QAction("📊 &Calculation Summary Card...", self)
        act_summary_card.setShortcut("F4")
        act_summary_card.setToolTip("Open publication-ready 1-page summary card of current QM calculation results (F4)")
        act_summary_card.triggered.connect(self._on_open_summary_card)
        menu_file.addAction(act_summary_card)

        act_export_png = QtGui.QAction("Export &Image (PNG)...", self)
        act_export_png.setShortcuts([
            QtGui.QKeySequence("Ctrl+E"),
            QtGui.QKeySequence("Ctrl+P"),
        ])
        act_export_png.setToolTip("Save a high-resolution PNG image of the 3D viewport (Ctrl+E)")
        act_export_png.triggered.connect(self._on_export_image)
        menu_file.addAction(act_export_png)

        act_clip_img = QtGui.QAction("📷 Copy Viewport Image to Clipboard", self)
        act_clip_img.setShortcut("Ctrl+Shift+C")
        act_clip_img.setToolTip("Copy publication-quality 3D image with transparent background directly to system clipboard (Ctrl+Shift+C)")
        act_clip_img.triggered.connect(lambda: self.copy_image_to_clipboard(transparent=True, scale=2, clean_only=True))
        menu_file.addAction(act_clip_img)

        act_export_html = QtGui.QAction("🌐 Export Interactive 3D &HTML...", self)
        act_export_html.setShortcut("Ctrl+Shift+H")
        act_export_html.setToolTip("Export molecule, properties, and trajectory to a standalone, zero-install 3D HTML webpage")
        act_export_html.triggered.connect(self._on_export_html)
        menu_file.addAction(act_export_html)

        menu_file.addSeparator()

        act_prefs = QtGui.QAction("⚙️ &Preferences...", self)
        act_prefs.setShortcut("Ctrl+,")
        act_prefs.setToolTip("Configure default colors, background mode, hardware resources, and symmetry defaults")
        act_prefs.triggered.connect(self._on_open_preferences)
        menu_file.addAction(act_prefs)

        menu_file.addSeparator()

        act_exit = QtGui.QAction("E&xit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

        # Edit Menu
        menu_edit = menubar.addMenu("&Edit")

        self.act_undo = QtGui.QAction("↶ &Undo", self)
        self.act_undo.setShortcut(QtGui.QKeySequence("Ctrl+Z"))
        self.act_undo.setToolTip("Undo last action (Ctrl+Z)")
        self.act_undo.setEnabled(False)
        self.act_undo.triggered.connect(self.undo)
        menu_edit.addAction(self.act_undo)

        self.act_redo = QtGui.QAction("↷ &Redo", self)
        self.act_redo.setShortcuts([
            QtGui.QKeySequence("Ctrl+Y"),
            QtGui.QKeySequence("Ctrl+Shift+Z")
        ])
        self.act_redo.setToolTip("Redo last undone action (Ctrl+Y)")
        self.act_redo.setEnabled(False)
        self.act_redo.triggered.connect(self.redo)
        menu_edit.addAction(self.act_redo)

        menu_edit.addSeparator()

        self.act_copy_mol = QtGui.QAction("📋 &Copy Molecule / Selection", self)
        self.act_copy_mol.setShortcut(QtGui.QKeySequence("Ctrl+C"))
        self.act_copy_mol.setToolTip("Copy selected atoms (or entire molecule if none selected) to clipboard / memory (Ctrl+C)")
        self.act_copy_mol.triggered.connect(self._copy_selected_fragment)
        menu_edit.addAction(self.act_copy_mol)

        self.act_paste_mol = QtGui.QAction("📋 &Paste Molecule / Fragment", self)
        self.act_paste_mol.setShortcut(QtGui.QKeySequence("Ctrl+V"))
        self.act_paste_mol.setToolTip("Paste copied fragment or clipboard XYZ into the 3D viewport (Ctrl+V)")
        self.act_paste_mol.triggered.connect(self._on_paste)
        menu_edit.addAction(self.act_paste_mol)

        menu_edit.addSeparator()

        # Copy Identifiers submenu
        menu_copy = menu_edit.addMenu("📋 Copy &Identifier...")
        act_cp_xyz = QtGui.QAction("Copy as &Cartesian XYZ", self)
        act_cp_xyz.setShortcut("Ctrl+Shift+C")
        act_cp_xyz.setToolTip("Copy molecule in standard Cartesian XYZ format to clipboard")
        act_cp_xyz.triggered.connect(self._on_copy_as_xyz)
        menu_copy.addAction(act_cp_xyz)

        act_cp_smiles = QtGui.QAction("Copy as &SMILES", self)
        act_cp_smiles.setToolTip("Generate and copy canonical/isomeric SMILES string to clipboard")
        act_cp_smiles.triggered.connect(self._on_copy_as_smiles)
        menu_copy.addAction(act_cp_smiles)

        act_cp_inchi = QtGui.QAction("Copy as &InChI", self)
        act_cp_inchi.setToolTip("Generate and copy IUPAC InChI string to clipboard")
        act_cp_inchi.triggered.connect(self._on_copy_as_inchi)
        menu_copy.addAction(act_cp_inchi)

        act_cp_inchikey = QtGui.QAction("Copy as InChI&Key", self)
        act_cp_inchikey.setToolTip("Generate and copy 27-character InChIKey to clipboard")
        act_cp_inchikey.triggered.connect(self._on_copy_as_inchikey)
        menu_copy.addAction(act_cp_inchikey)

        menu_edit.addSeparator()

        act_add_line = QtGui.QAction("&Add Atom Line", self)
        act_add_line.triggered.connect(self._on_add_line)
        menu_edit.addAction(act_add_line)

        act_del = QtGui.QAction("&Delete Selected Atom", self)
        act_del.triggered.connect(self._on_delete_selected_atom)
        menu_edit.addAction(act_del)

        act_reorder = QtGui.QAction("&Reorder Z-Matrix (Full)", self)
        act_reorder.setShortcut("Ctrl+R")
        act_reorder.setToolTip("Reorder entire Z-Matrix sequentially (Molden ZMEORD)")
        act_reorder.triggered.connect(self._on_reorder_zmat_clicked)
        menu_edit.addAction(act_reorder)

        act_adj_b = QtGui.QAction("Adjust &Bond (Alt+B)", self)
        act_adj_b.setShortcut("Alt+B")
        act_adj_b.setToolTip("Click 2 atoms to redefine bond partner of the 1st atom (Alt+B)")
        act_adj_b.triggered.connect(lambda: self._start_partial_reorder("bond"))
        menu_edit.addAction(act_adj_b)

        act_adj_a = QtGui.QAction("Adjust &Angle (Alt+A)", self)
        act_adj_a.setShortcut("Alt+A")
        act_adj_a.setToolTip("Click 3 atoms to redefine bond & angle partners of the 1st atom (Alt+A)")
        act_adj_a.triggered.connect(lambda: self._start_partial_reorder("angle"))
        menu_edit.addAction(act_adj_a)

        act_adj_d = QtGui.QAction("Adjust &Dihedral (Alt+D)", self)
        act_adj_d.setShortcut("Alt+D")
        act_adj_d.setToolTip("Click 4 atoms to redefine bond, angle & dihedral partners of the 1st atom (Alt+D)")
        act_adj_d.triggered.connect(lambda: self._start_partial_reorder("dihedral"))
        menu_edit.addAction(act_adj_d)

        menu_edit.addSeparator()

        act_add_h = QtGui.QAction("&Complete Valence with Hydrogens", self)
        act_add_h.setShortcut("Ctrl+H")
        act_add_h.triggered.connect(self._on_add_hydrogens)
        menu_edit.addAction(act_add_h)

        menu_origin = menu_edit.addMenu("🎯 &Translate to Origin (0, 0, 0)...")
        act_orig_com = QtGui.QAction("Center of Mass to Origin", self)
        act_orig_com.setToolTip("Translate molecule so its center-of-mass is at Cartesian (0, 0, 0)")
        act_orig_com.triggered.connect(lambda: self._center_molecule_to_origin("com"))
        menu_origin.addAction(act_orig_com)

        act_orig_anchor = QtGui.QAction("Selected Atom to Origin", self)
        act_orig_anchor.setToolTip("Translate molecule so the currently selected atom is at (0, 0, 0)")
        act_orig_anchor.triggered.connect(lambda: self._center_molecule_to_origin("anchor"))
        menu_origin.addAction(act_orig_anchor)

        menu_invert = menu_edit.addMenu("🔄 &Chirality & Mirror Inversion...")
        act_inv_all = QtGui.QAction("Invert All Coordinates (Enantiomer, -r)", self)
        act_inv_all.setToolTip("Invert all atomic coordinates through the center of inversion (x,y,z ➔ -x,-y,-z)")
        act_inv_all.triggered.connect(lambda: self._invert_molecule_coordinates("inversion"))
        menu_invert.addAction(act_inv_all)

        act_mir_xy = QtGui.QAction("Mirror Across XY Plane (z ➔ -z)", self)
        act_mir_xy.triggered.connect(lambda: self._invert_molecule_coordinates("xy"))
        menu_invert.addAction(act_mir_xy)

        act_mir_xz = QtGui.QAction("Mirror Across XZ Plane (y ➔ -y)", self)
        act_mir_xz.triggered.connect(lambda: self._invert_molecule_coordinates("xz"))
        menu_invert.addAction(act_mir_xz)

        act_mir_yz = QtGui.QAction("Mirror Across YZ Plane (x ➔ -x)", self)
        act_mir_yz.triggered.connect(lambda: self._invert_molecule_coordinates("yz"))
        menu_invert.addAction(act_mir_yz)

        act_flush_mem = QtGui.QAction("Flush &Memory Clipboard", self)
        act_flush_mem.setShortcuts([
            QtGui.QKeySequence("Ctrl+K"),
            QtGui.QKeySequence("Ctrl+Shift+X"),
        ])
        act_flush_mem.setToolTip("Flush atom and fragment clipboard memory (Ctrl+K or Ctrl+Shift+X). Double-clicking canvas will prompt Periodic Table / Fragment picker.")
        act_flush_mem.triggered.connect(self._on_flush_memory)
        menu_edit.addAction(act_flush_mem)

        act_opt = QtGui.QAction("&Optimize Geometry (Force Field)", self)
        act_opt.setShortcuts([
            QtGui.QKeySequence("Ctrl+M"),
            QtGui.QKeySequence("Ctrl+Shift+O"),
        ])
        act_opt.setToolTip("Optimize molecular geometry using force field / xTB (Ctrl+M or Ctrl+Shift+O)")
        act_opt.triggered.connect(self._on_optimize_geometry)
        menu_edit.addAction(act_opt)

        act_conf = QtGui.QAction("🔄 &Conformational Search (Find Lowest-Energy Rotamer)...", self)
        act_conf.setToolTip("Sample rotatable bonds, minimize candidate rotamers, and rank conformers by energy")
        act_conf.triggered.connect(self._on_conformer_search)
        menu_edit.addAction(act_conf)

        act_vib = QtGui.QAction("🎵 &Vibrational Frequencies (Normal Modes)", self)
        act_vib.setToolTip("Compute vibrational frequencies and normal modes simulation")
        act_vib.triggered.connect(self._on_calculate_vibrations)
        menu_edit.addAction(act_vib)

        act_sym = QtGui.QAction("🔮 &Detect Point Group Symmetry", self)
        act_sym.setToolTip("Analyze molecular point group symmetry (Schoenflies notation)")
        act_sym.triggered.connect(lambda _checked=False: self._on_detect_symmetry(switch_tab=True))
        menu_edit.addAction(act_sym)

        act_overlay = QtGui.QAction("🔬 &Compare & Align Structures (RMSD Overlay)...", self)
        act_overlay.setShortcut("Ctrl+Shift+A")
        act_overlay.setToolTip("Superimpose two structures with Kabsch SVD alignment and calculate all-atom/heavy-atom RMSD")
        act_overlay.triggered.connect(self._on_open_structure_overlay)
        menu_edit.addAction(act_overlay)

        # View Menu
        menu_view = menubar.addMenu("&View")

        self.act_theme_dark = QtGui.QAction("🌙 &Dark Theme", self)
        self.act_theme_dark.setCheckable(True)
        self.act_theme_dark.setChecked(True)
        self.act_theme_dark.triggered.connect(lambda: self.set_color_theme("dark"))
        menu_view.addAction(self.act_theme_dark)

        self.act_theme_white = QtGui.QAction("☀️ &White Theme", self)
        self.act_theme_white.setCheckable(True)
        self.act_theme_white.setChecked(False)
        self.act_theme_white.triggered.connect(lambda: self.set_color_theme("white"))
        menu_view.addAction(self.act_theme_white)

        theme_group = QtGui.QActionGroup(self)
        theme_group.addAction(self.act_theme_dark)
        theme_group.addAction(self.act_theme_white)

        menu_view.addSeparator()

        act_toggle_measure = QtGui.QAction("📏 Show 3D &Measurement Overlays", self)
        act_toggle_measure.setCheckable(True)
        act_toggle_measure.setChecked(True)
        act_toggle_measure.setShortcut("Ctrl+M")
        act_toggle_measure.setToolTip("Toggle 3D floating distance, angle, and dihedral measurements over selected atoms")
        def _toggle_meas(checked):
            self.viewer.show_measurements = checked
            self.viewer.update()
        act_toggle_measure.triggered.connect(_toggle_meas)
        menu_view.addAction(act_toggle_measure)

        act_reset_cam = QtGui.QAction("Reset Camera View", self)
        act_reset_cam.setShortcut("Space")
        act_reset_cam.triggered.connect(self.viewer.auto_center)
        menu_view.addAction(act_reset_cam)

        # Help Menu
        menu_help = menubar.addMenu("&Help")
        act_about = QtGui.QAction("&About AgyMolden", self)
        act_about.triggered.connect(self._on_about)
        menu_help.addAction(act_about)

        act_shortcuts = QtGui.QAction("&Keyboard && Mouse Shortcuts", self)
        act_shortcuts.triggered.connect(self._on_show_shortcuts)
        menu_help.addAction(act_shortcuts)

        act_manual = QtGui.QAction("Open User &Manual", self)
        act_manual.setShortcut("F1")
        act_manual.triggered.connect(self._on_open_manual)
        menu_help.addAction(act_manual)

    def _create_toolbar(self):
        toolbar = QtWidgets.QToolBar("Main Controls")
        toolbar.setIconSize(QtCore.QSize(18, 18))
        self.addToolBar(toolbar)

        btn_new = QtWidgets.QToolButton()
        btn_new.setText("📄 New")
        btn_new.clicked.connect(self._on_new_molecule)
        toolbar.addWidget(btn_new)

        btn_open = QtWidgets.QToolButton()
        btn_open.setText("📂 Open File")
        btn_open.setToolTip("Open any supported file sorted newest-first: .out, .log, .xyz, .molden.input, .molden, .gbw, .cube, .zmat (Ctrl+O)")
        btn_open.clicked.connect(self._on_open_universal_file)
        toolbar.addWidget(btn_open)

        btn_save = QtWidgets.QToolButton()
        btn_save.setText("💾 Save ZMAT")
        btn_save.clicked.connect(self._on_save_zmat)
        toolbar.addWidget(btn_save)

        btn_xyz = QtWidgets.QToolButton()
        btn_xyz.setText("📤 Export XYZ")
        btn_xyz.clicked.connect(self._on_export_xyz)
        toolbar.addWidget(btn_xyz)

        btn_png = QtWidgets.QToolButton()
        btn_png.setText("📷 Save PNG")
        btn_png.setToolTip("Export high-resolution PNG image of the 3D view (Ctrl+P)")
        btn_png.clicked.connect(self._on_export_image)
        toolbar.addWidget(btn_png)

        btn_html = QtWidgets.QToolButton()
        btn_html.setText("🌐 3D HTML")
        btn_html.setToolTip("Export interactive standalone 3D WebGL HTML webpage (Ctrl+Shift+H)")
        btn_html.clicked.connect(self._on_export_html)
        toolbar.addWidget(btn_html)

        toolbar.addSeparator()

        self.btn_undo = QtWidgets.QToolButton()
        self.btn_undo.setText("↶ Undo")
        self.btn_undo.setToolTip("Undo last action (Ctrl+Z)")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.undo)
        toolbar.addWidget(self.btn_undo)

        self.btn_redo = QtWidgets.QToolButton()
        self.btn_redo.setText("↷ Redo")
        self.btn_redo.setToolTip("Redo last undone action (Ctrl+Y)")
        self.btn_redo.setEnabled(False)
        self.btn_redo.clicked.connect(self.redo)
        toolbar.addWidget(self.btn_redo)

        toolbar.addSeparator()

        btn_h = QtWidgets.QToolButton()
        btn_h.setText("💧 Add H")
        btn_h.setToolTip("Complete valence with hydrogens (Ctrl+H)")
        btn_h.clicked.connect(self._on_add_hydrogens)
        toolbar.addWidget(btn_h)

        btn_opt = QtWidgets.QToolButton()
        btn_opt.setText("⚡ Optimize")
        btn_opt.setToolTip("Optimize geometry using force field / xTB (Ctrl+M)")
        btn_opt.clicked.connect(self._on_optimize_geometry)
        toolbar.addWidget(btn_opt)

        toolbar.addSeparator()

        btn_quick_orca = QtWidgets.QToolButton()
        btn_quick_orca.setText("🚀 Run ORCA")
        btn_quick_orca.setToolTip("Quick 1-Click ORCA calculation for course labs (Opt, Freq, Orbitals)")
        btn_quick_orca.setStyleSheet("color: #2563eb; font-weight: bold;")
        btn_quick_orca.clicked.connect(self._on_quick_run_orca)
        toolbar.addWidget(btn_quick_orca)

        btn_qm = QtWidgets.QToolButton()
        btn_qm.setText("⚛️ QM Setup")
        btn_qm.setToolTip("Generate calculation input for Gaussian, ORCA, PySCF, openLOWDIN, xTB (Ctrl+G)")
        btn_qm.clicked.connect(self._on_open_qm_dialog)
        toolbar.addWidget(btn_qm)

        btn_brorsen = QtWidgets.QToolButton()
        btn_brorsen.setText("✨ newBRORSEN")
        btn_brorsen.setToolTip("Generate input files for newBRORSEN (Positrons, NEO, CCSD(T)) (Ctrl+B)")
        btn_brorsen.clicked.connect(self._on_open_brorsen_dialog)
        toolbar.addWidget(btn_brorsen)

        btn_orbitals = QtWidgets.QToolButton()
        btn_orbitals.setText("🌐 Orbitals/Grid")
        btn_orbitals.setToolTip("Open Volumetric Cube or Molden file to view Molecular Orbitals / ESP isosurfaces")
        btn_orbitals.clicked.connect(self._on_open_volumetric_dialog)
        toolbar.addWidget(btn_orbitals)

        btn_view_log_tb = QtWidgets.QToolButton()
        btn_view_log_tb.setText("📄 View Log")
        btn_view_log_tb.setToolTip("Open Calculation Output Log & Results Viewer (F2)")
        btn_view_log_tb.clicked.connect(self._on_view_qm_output_file)
        toolbar.addWidget(btn_view_log_tb)

        btn_vib = QtWidgets.QToolButton()
        btn_vib.setText("🎵 Vibrations")
        btn_vib.setToolTip("Compute vibrational frequencies and simulate normal modes")
        btn_vib.clicked.connect(self._on_calculate_vibrations)
        toolbar.addWidget(btn_vib)

        self.btn_toolbar_sym = QtWidgets.QToolButton()
        self.btn_toolbar_sym.setText("🔮 Symmetry")
        self.btn_toolbar_sym.setToolTip("Detect Schoenflies point group symmetry")
        self.btn_toolbar_sym.clicked.connect(lambda _checked=False: self._on_detect_symmetry(switch_tab=True))
        toolbar.addWidget(self.btn_toolbar_sym)

        # Permanent Chemical Summary Label on Status Bar
        self.lbl_chem_summary = QtWidgets.QLabel("Empty Molecule")
        self.lbl_chem_summary.setStyleSheet("color: #38bdf8; font-weight: bold; padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.lbl_chem_summary)

    # Core logic
