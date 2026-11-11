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

        # Molecular Tweezers toggle button right in the viewport quick controls ribbon!
        self.btn_ribbon_tweezers = QtWidgets.QPushButton("🧲 Tweezers")
        self.btn_ribbon_tweezers.setCheckable(True)
        self.btn_ribbon_tweezers.setChecked(False)
        self.btn_ribbon_tweezers.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #e2e8f0;
                font-weight: bold;
                padding: 2px 8px;
                border: 1px solid #475569;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #0284c7;
                color: #ffffff;
                border: 1.5px solid #38bdf8;
            }
        """)
        self.btn_ribbon_tweezers.setToolTip("🧲 Molecular Tweezers (Live Drag & Relax): Click to activate. Once active, simply Left-Click and drag any atom in the 3D viewport to stretch and relax connected bonds in real time!")
        self.btn_ribbon_tweezers.toggled.connect(self._on_toggle_tweezers)
        view_ctrls.addWidget(self.btn_ribbon_tweezers)

        # Quick Element Brush Palette (Avogadro / GaussView style)
        self.btn_group_elements = QtWidgets.QButtonGroup(self)
        self.btn_group_elements.setExclusive(True)
        common_elements = [("C", "#475569"), ("H", "#94a3b8"), ("O", "#ef4444"), ("N", "#3b82f6"), ("P", "#f97316"), ("S", "#eab308"), ("F", "#10b981"), ("Cl", "#22c55e")]
        for sym, color in common_elements:
            b_el = QtWidgets.QPushButton(sym)
            b_el.setCheckable(True)
            b_el.setFixedWidth(24)
            b_el.setToolTip(f"Select {sym} as active drawing & mutation element brush")
            if sym == "C":
                b_el.setChecked(True)
            b_el.setStyleSheet(f"QPushButton:checked {{ background-color: {color}; color: white; font-weight: bold; border: 1px solid #ffffff; }}")
            b_el.clicked.connect(lambda _c=False, s=sym: self._on_select_brush_element(s))
            self.btn_group_elements.addButton(b_el)
            view_ctrls.addWidget(b_el)

        # Quick Stamp Menu (Benzene, Cyclohexane, Methyl, etc.)
        self.btn_quick_stamp = QtWidgets.QToolButton()
        self.btn_quick_stamp.setText("📌 Stamp ▾")
        self.btn_quick_stamp.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_quick_stamp.setToolTip("GaussView / Avogadro style: Stamp functional groups or rings onto selected atom in 1 click")
        self.btn_quick_stamp.setStyleSheet("background-color: #334155; color: white; font-weight: 600; padding: 2px 6px;")
        menu_stamp = QtWidgets.QMenu(self)
        for st_name in ["Phenyl (-C6H5)", "Methyl (-CH3)", "Hydroxyl (-OH)", "Amino (-NH2)", "Carbonyl (=O)", "Nitro (-NO2)", "Carboxyl (-COOH)", "Water (H2O)"]:
            act = menu_stamp.addAction(st_name)
            act.triggered.connect(lambda _c=False, n=st_name: self._on_quick_stamp_fragment(n))
        self.btn_quick_stamp.setMenu(menu_stamp)
        view_ctrls.addWidget(self.btn_quick_stamp)

        self.chk_auto_h = QtWidgets.QCheckBox("Adjust H")
        self.chk_auto_h.setChecked(True)
        self.chk_auto_h.setStyleSheet("font-weight: 600; padding: 2px 4px;")
        self.chk_auto_h.setToolTip("Avogadro style: automatically complete new atom valence and prune hydrogens on anchor atom to maintain correct valency")
        self.chk_auto_h.toggled.connect(lambda: self._update_viewer_drag_label())
        view_ctrls.addWidget(self.chk_auto_h)

        self.chk_impose_symmetry = QtWidgets.QCheckBox("🔒 Impose Symm.")
        self.chk_impose_symmetry.setChecked(False)
        self.chk_impose_symmetry.setStyleSheet("font-weight: 600; padding: 2px 4px;")
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

        self.btn_orient_sym = QtWidgets.QPushButton("📐 Orient")
        self.btn_orient_sym.setStyleSheet("font-weight: 600; padding: 2px 6px; background-color: #0d9488; color: #ffffff;")
        self.btn_orient_sym.setToolTip(
            "Universal Symmetry Orientation (Alt+S):\n"
            "Align molecule coordinates to standard Schoenflies/IUPAC convention for ANY point group\n"
            "(e.g. C2v water in XZ plane with C2 along Z, D2h, C3v, etc.) and center COM at (0,0,0)."
        )
        self.btn_orient_sym.clicked.connect(self._on_align_symmetry)
        view_ctrls.addWidget(self.btn_orient_sym)

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

        self.chk_bond_lengths = QtWidgets.QCheckBox("Bond Lengths (Å)")
        self.chk_bond_lengths.setChecked(False)
        self.chk_bond_lengths.setToolTip("Chemcraft style: Overlay exact bond lengths (Å) at the midpoint of every covalent bond")
        def _on_chk_bond_lengths(checked: bool):
            self.viewer.show_bond_lengths = bool(checked)
            if hasattr(self, "act_all_bond_lengths"):
                self.act_all_bond_lengths.blockSignals(True)
                self.act_all_bond_lengths.setChecked(bool(checked))
                self.act_all_bond_lengths.blockSignals(False)
            self.viewer.update()
        self.chk_bond_lengths.toggled.connect(_on_chk_bond_lengths)
        display_layout.addWidget(self.chk_bond_lengths)

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

        self.chk_nci = QtWidgets.QCheckBox("Non-Covalent (NCI / π-π)")
        self.chk_nci.setToolTip("Real-time Non-Covalent Interactions: Halogen bonds (orange) and aromatic π-π stacking (purple)")
        self.chk_nci.toggled.connect(self._on_toggle_nci)
        display_layout.addWidget(self.chk_nci)

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

        self.chk_pub_outlines = QtWidgets.QCheckBox("🎨 Publication Outlines")
        self.chk_pub_outlines.setChecked(False)
        self.chk_pub_outlines.setToolTip(
            "IboView / QuteMol style: Render crisp dark silhouette ink outlines and soft hemisphere lighting\n"
            "for high-contrast, journal-cover ready molecular figures."
        )
        self.chk_pub_outlines.toggled.connect(
            lambda chk: setattr(self.viewer, "show_publication_outlines", bool(chk)) or self.viewer.update()
        )
        display_layout.addWidget(self.chk_pub_outlines)

        self.chk_axis_triad = QtWidgets.QCheckBox("🧭 Corner Axis Triad")
        self.chk_axis_triad.setChecked(True)
        self.chk_axis_triad.setToolTip("Show 3D X/Y/Z orientation triad in the bottom-right corner of the viewport")
        self.chk_axis_triad.toggled.connect(
            lambda chk: setattr(self.viewer, "show_axis_triad", bool(chk)) or self.viewer.update()
        )
        display_layout.addWidget(self.chk_axis_triad)

        self.chk_scale_bar = QtWidgets.QCheckBox("📏 Scale Bar (Å)")
        self.chk_scale_bar.setChecked(False)
        self.chk_scale_bar.setToolTip("Show calibrated physical length scale bar (Å) in the bottom-right corner")
        self.chk_scale_bar.toggled.connect(
            lambda chk: setattr(self.viewer, "show_scale_bar", bool(chk)) or self.viewer.update()
        )
        display_layout.addWidget(self.chk_scale_bar)

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

        # Interactive Mini Energy Trajectory Profile Curve (Sparkline)
        from .widget_sparkline import SparklinePlot
        self.sparkline_trajectory = SparklinePlot(self)
        self.sparkline_trajectory.step_clicked.connect(lambda s: self.slider_trajectory.setValue(s))
        traj_box_layout.addWidget(self.sparkline_trajectory)

        # Row 2: Step Energy & Analysis Tools (Scrollable so buttons are always clearly readable)
        row2_scroll = QtWidgets.QScrollArea()
        row2_scroll.setWidgetResizable(True)
        row2_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        row2_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        row2_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        row2_scroll.setFixedHeight(36)

        row2_container = QtWidgets.QWidget()
        row2_layout = QtWidgets.QHBoxLayout(row2_container)
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(6)

        self.lbl_trajectory_energy = QtWidgets.QLabel("")
        self.lbl_trajectory_energy.setStyleSheet("color: #a7f3d0; font-family: monospace; font-size: 11px; white-space: nowrap;")
        row2_layout.addWidget(self.lbl_trajectory_energy)

        row2_layout.addStretch()

        self.btn_adopt_traj_frame = QtWidgets.QPushButton("📌 Adopt Frame")
        self.btn_adopt_traj_frame.setStyleSheet("background-color: #059669; color: #ffffff; padding: 3px 10px; font-weight: bold; font-size: 11px; white-space: nowrap;")
        self.btn_adopt_traj_frame.setToolTip(
            "Chemcraft style: Adopt the currently displayed trajectory/IRC/conformer frame into the editable Z-Matrix table (with Undo)"
        )
        self.btn_adopt_traj_frame.clicked.connect(self._on_adopt_trajectory_frame)
        row2_layout.addWidget(self.btn_adopt_traj_frame)

        self.btn_export_traj_movie = QtWidgets.QPushButton("🎬 Movie...")
        self.btn_export_traj_movie.setStyleSheet("background-color: #7c3aed; color: #ffffff; padding: 3px 10px; font-weight: bold; font-size: 11px; white-space: nowrap;")
        self.btn_export_traj_movie.setToolTip("Export molecular optimization or MD trajectory as an animated GIF or MP4/WebM video")
        self.btn_export_traj_movie.clicked.connect(self._on_export_trajectory_movie)
        row2_layout.addWidget(self.btn_export_traj_movie)

        self.btn_pes_plot = QtWidgets.QPushButton("📈 Energy...")
        self.btn_pes_plot.setStyleSheet("background-color: #0284c7; color: #ffffff; padding: 3px 10px; font-weight: bold; font-size: 11px; white-space: nowrap;")
        self.btn_pes_plot.setToolTip("GaussView / newBRORSEN style: Open interactive Potential Energy Surface (PES) Scan & Morse Fit plot")
        self.btn_pes_plot.clicked.connect(self._on_open_pes_plot)
        row2_layout.addWidget(self.btn_pes_plot)

        self.btn_crest_explorer = QtWidgets.QPushButton("🧬 Ensemble...")
        self.btn_crest_explorer.setStyleSheet("background-color: #0d9488; color: #ffffff; padding: 3px 10px; font-weight: bold; font-size: 11px; white-space: nowrap;")
        self.btn_crest_explorer.setToolTip("Open CREST & Conformer Ensemble Explorer (Boltzmann populations, RMSDs, duplicate pruning) (Ctrl+Shift+E)")
        self.btn_crest_explorer.clicked.connect(self._on_open_crest_explorer)
        row2_layout.addWidget(self.btn_crest_explorer)

        self.btn_irc_sweeper = QtWidgets.QPushButton("⚛️ IRC...")
        self.btn_irc_sweeper.setStyleSheet("background-color: #6366f1; color: #ffffff; padding: 3px 10px; font-weight: bold; font-size: 11px; white-space: nowrap;")
        self.btn_irc_sweeper.setToolTip("Open Interactive IRC Reaction Sweeper & Energy Profile (MEP, activation barriers, landmark jumps) (Ctrl+Shift+I)")
        self.btn_irc_sweeper.clicked.connect(self._on_open_irc_sweeper)
        row2_layout.addWidget(self.btn_irc_sweeper)

        self.btn_view_loaded_log = QtWidgets.QPushButton("📄 Log...")
        self.btn_view_loaded_log.setStyleSheet("background-color: #334155; color: #ffffff; padding: 3px 10px; font-weight: bold; font-size: 11px; white-space: nowrap;")
        self.btn_view_loaded_log.setToolTip("Open full calculation output log with search and section indexing (F2)")
        self.btn_view_loaded_log.clicked.connect(self._on_view_qm_output_file)
        row2_layout.addWidget(self.btn_view_loaded_log)

        row2_scroll.setWidget(row2_container)
        traj_box_layout.addWidget(row2_scroll)

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
        tab_table_layout.setContentsMargins(4, 4, 4, 4)
        tab_table_layout.setSpacing(4)

        zmat_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.zmat_splitter = zmat_splitter

        # Upper Container: Filter bar + Expansive Z-Matrix Table
        top_container = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.setSpacing(4)
        self.txt_zmat_filter = QtWidgets.QLineEdit()
        self.txt_zmat_filter.setPlaceholderText("🔍 Filter atoms (e.g. C, H, 1-20, >10)...")
        self.txt_zmat_filter.setClearButtonEnabled(True)
        self.txt_zmat_filter.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        filter_row.addWidget(self.txt_zmat_filter, stretch=1)

        btn_show_all = QtWidgets.QPushButton("Show All")
        btn_show_all.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        btn_show_all.clicked.connect(lambda: self.txt_zmat_filter.clear())
        filter_row.addWidget(btn_show_all)
        top_layout.addLayout(filter_row)

        # Table
        self.table_widget = ZMatrixTableWidget()
        self.table_widget.setMinimumHeight(150)
        self.table_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.txt_zmat_filter.textChanged.connect(self.table_widget.filter_rows)
        self.table_widget.before_data_changed.connect(self._push_undo_state)
        self.table_widget.data_changed.connect(self._on_table_data_changed)
        self.table_widget.atom_selected.connect(self._on_table_atom_selected)
        self.table_widget.cell_selected.connect(self._on_table_cell_selected)
        top_layout.addWidget(self.table_widget, stretch=1)
        zmat_splitter.addWidget(top_container)

        # Lower Container: Action bars & Tuning Sliders (scrollable so controls never clip on small screens)
        bottom_scroll = QtWidgets.QScrollArea()
        bottom_scroll.setWidgetResizable(True)
        bottom_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        bottom_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        bottom_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        bottom_container = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 4, 0, 0)
        bottom_layout.setSpacing(6)

        # Compact & Responsive Z-Matrix Action Bars (Never shrink or clip text on small screens)
        # Row 1: Add Atom, Add H, Delete (3 items)
        row_action_layout1 = QtWidgets.QHBoxLayout()
        row_action_layout1.setSpacing(5)
        self.btn_add_atom = QtWidgets.QPushButton("➕ Add Atom")
        self.btn_add_atom.setStyleSheet("font-size: 11px; font-weight: bold; padding: 4px 6px;")
        self.btn_add_atom.setMinimumWidth(75)
        self.btn_add_atom.setToolTip("Add new atom line to Z-Matrix (opens periodic table)")
        self.btn_add_atom.clicked.connect(self._on_add_line)

        self.btn_add_hydrogens = QtWidgets.QPushButton("💧 Add H")
        self.btn_add_hydrogens.setStyleSheet("background-color: #0284c7; color: #ffffff; font-size: 11px; font-weight: bold; padding: 4px 6px;")
        self.btn_add_hydrogens.setMinimumWidth(65)
        self.btn_add_hydrogens.setToolTip("Complete valence of all atoms with standard geometric hydrogens (Ctrl+H)")
        self.btn_add_hydrogens.clicked.connect(self._on_add_hydrogens)

        self.btn_del_atom = QtWidgets.QPushButton("🗑️ Delete")
        self.btn_del_atom.setStyleSheet("background-color: #dc2626; color: #ffffff; font-size: 11px; font-weight: bold; padding: 4px 6px;")
        self.btn_del_atom.setMinimumWidth(65)
        self.btn_del_atom.setToolTip("Delete currently selected atom(s) from Z-Matrix (Shortcut: Del)")
        self.btn_del_atom.clicked.connect(self._on_delete_selected_atom)

        row_action_layout1.addWidget(self.btn_add_atom)
        row_action_layout1.addWidget(self.btn_add_hydrogens)
        row_action_layout1.addWidget(self.btn_del_atom)
        bottom_layout.addLayout(row_action_layout1)

        # Row 2: Form Bond, Reorder ZMAT (2 items)
        row_action_layout2 = QtWidgets.QHBoxLayout()
        row_action_layout2.setSpacing(5)
        self.btn_close_ring = QtWidgets.QPushButton("🔗 Form Bond")
        self.btn_close_ring.setStyleSheet("background-color: #0d9488; color: #ffffff; font-size: 11px; font-weight: bold; padding: 4px 6px;")
        self.btn_close_ring.setMinimumWidth(85)
        self.btn_close_ring.setToolTip("Select 2 atoms and click to bond them (ring closure or any new bond). Keeps all existing bonds, removes the H that is replaced, and verifies valence everywhere")
        self.btn_close_ring.clicked.connect(self._on_close_ring_clicked)

        self.btn_reorder_zmat = QtWidgets.QPushButton("🔄 Reorder ZMAT")
        self.btn_reorder_zmat.setStyleSheet("background-color: #059669; color: #ffffff; font-size: 11px; font-weight: bold; padding: 4px 6px;")
        self.btn_reorder_zmat.setMinimumWidth(85)
        self.btn_reorder_zmat.setToolTip("Click atoms sequentially in 3D viewport to reorder Z-matrix connectivity (Molden ZMEORD)")
        self.btn_reorder_zmat.clicked.connect(self._on_reorder_zmat_clicked)

        self.btn_read_zmat = QtWidgets.QPushButton("📂 Open ZMAT")
        self.btn_read_zmat.setStyleSheet("background-color: #475569; color: #ffffff; font-size: 11px; font-weight: bold; padding: 4px 6px;")
        self.btn_read_zmat.setToolTip("Open and load a Z-matrix file (*.zmat, *.dat, *.inp, *.txt)")
        self.btn_read_zmat.clicked.connect(self._on_open_zmat)
        self.btn_read_zmat.setVisible(False)  # Accessible via File -> Open File (Ctrl+O) and Raw ZMAT tab

        row_action_layout2.addWidget(self.btn_close_ring)
        row_action_layout2.addWidget(self.btn_reorder_zmat)
        bottom_layout.addLayout(row_action_layout2)

        # Row 3: Optimization Method Selector & Optimize Button
        row_action_layout_opt = QtWidgets.QHBoxLayout()
        row_action_layout_opt.setSpacing(5)
        self.combo_opt_method = QtWidgets.QComboBox()
        self.combo_opt_method.addItem("Analytical UFF / MMFF94s", "uff_mmff")
        self.combo_opt_method.addItem("Auto (Hybrid MMFF / UFF)", "auto_hybrid")
        self.combo_opt_method.addItem("GFN0-xTB (Semiempirical QM)", "semiempirical")
        self.combo_opt_method.addItem("🧠 MLIP (ANI-2x / MACE / Hybrid QM)", "mlip")
        self.combo_opt_method.setStyleSheet("font-size: 11px; padding: 3px 6px;")
        self.combo_opt_method.setToolTip(
            "Select geometry optimization & vibrational engine:\n"
            "• Analytical UFF / MMFF94s: Built-in pure analytical force field (universal coverage, periodic torsions)\n"
            "• Auto (Hybrid MMFF / UFF): Tries RDKit MMFF94 for organic accuracy, auto-falling back to Analytical UFF\n"
            "• GFN0-xTB (Semiempirical QM): Self-Consistent Tight-Binding QM with Wiberg bond orders & charges\n"
            "• MLIP (ANI-2x / MACE / Hybrid QM): Universal Neural Network Interatomic Potential with SCC-TB + D3 fallback"
        )
        self.combo_opt_method.currentIndexChanged.connect(self._on_opt_method_changed)

        self.btn_optimize_ff = QtWidgets.QPushButton("⚡ Optimize")
        self.btn_optimize_ff.setStyleSheet("background-color: #7c3aed; color: #ffffff; font-size: 11px; font-weight: bold; padding: 4px 8px;")
        self.btn_optimize_ff.setMinimumWidth(75)
        self.btn_optimize_ff.setToolTip("Optimize molecular geometry using the selected force field or semiempirical QM method (Ctrl+M)")
        self.btn_optimize_ff.clicked.connect(self._on_optimize_geometry)

        row_action_layout_opt.addWidget(self.combo_opt_method, stretch=3)
        row_action_layout_opt.addWidget(self.btn_optimize_ff, stretch=2)
        bottom_layout.addLayout(row_action_layout_opt)

        # Row 4: Conformers & Variables Dropdown Button
        row_action_layout_adv = QtWidgets.QHBoxLayout()
        row_action_layout_adv.setSpacing(5)

        self.btn_conformer_search = QtWidgets.QPushButton("🔄 Conformers")
        self.btn_conformer_search.setStyleSheet("background-color: #b45309; color: #ffffff; font-size: 11px; font-weight: bold; padding: 4px 6px;")
        self.btn_conformer_search.setMinimumWidth(85)
        self.btn_conformer_search.setToolTip(
            "Avogadro / Gabedit style: Systematic & stochastic rotor conformational search.\n"
            "Finds the lowest-energy conformer and loads the ranked conformer ensemble into the Trajectory Player!"
        )
        self.btn_conformer_search.clicked.connect(self._on_conformer_search)

        # Compact Variables Menu Button (keeps individual buttons alive on self for tests & programmatic access)
        self.btn_vars_menu = QtWidgets.QToolButton()
        self.btn_vars_menu.setText("🔤 Variables ▾")
        self.btn_vars_menu.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_vars_menu.setStyleSheet("background-color: #0369a1; color: #ffffff; font-size: 11px; font-weight: bold; padding: 4px 8px; border-radius: 4px;")
        self.btn_vars_menu.setMinimumWidth(85)
        self.btn_vars_menu.setToolTip("Assign or clear symbolic Z-Matrix variables (Symm. Bonds, Symm. All, All Variables, Numbers Only)")
        menu_zmat_vars = QtWidgets.QMenu(self)

        vars_panel = QtWidgets.QWidget()
        row_vars_layout = QtWidgets.QVBoxLayout(vars_panel)
        row_vars_layout.setContentsMargins(8, 6, 8, 6)
        row_vars_layout.setSpacing(5)

        self.btn_sym_bonds = QtWidgets.QPushButton("🔗 Symm. Bonds Only")
        self.btn_sym_bonds.setStyleSheet("background-color: #0369a1; color: #ffffff; font-size: 11px; font-weight: bold;")
        self.btn_sym_bonds.setToolTip(
            "Assign symmetric variables ONLY to bond distances (e.g. R1 for C-C, R2 for C-H in benzene)\n"
            "while keeping all angles and dihedrals as fixed numerical constants."
        )
        self.btn_sym_bonds.clicked.connect(lambda: (menu_zmat_vars.close(), self._on_assign_symmetric_bond_variables()))

        self.btn_sym_all = QtWidgets.QPushButton("🔗 Symm. All (Bonds/Angles/Dihedrals)")
        self.btn_sym_all.setStyleSheet("background-color: #0284c7; color: #ffffff; font-size: 11px; font-weight: bold;")
        self.btn_sym_all.setToolTip("Assign shared variables to all symmetry-equivalent bonds, angles, and dihedrals")
        self.btn_sym_all.clicked.connect(lambda: (menu_zmat_vars.close(), self._on_assign_symmetric_variables()))

        self.btn_indiv_vars = QtWidgets.QPushButton("🔤 All Individual Variables")
        self.btn_indiv_vars.setStyleSheet("background-color: #475569; color: #ffffff; font-size: 11px; font-weight: bold;")
        self.btn_indiv_vars.setToolTip("Assign an individual variable (R1..Rn, A1..An, D1..Dn) to every coordinate")
        self.btn_indiv_vars.clicked.connect(lambda: (menu_zmat_vars.close(), self._on_assign_individual_variables()))

        self.btn_clear_vars = QtWidgets.QPushButton("🔢 Numbers Only (Clear Variables)")
        self.btn_clear_vars.setStyleSheet("background-color: #475569; color: #ffffff; font-size: 11px; font-weight: bold;")
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

        row_action_layout_adv.addWidget(self.btn_conformer_search, stretch=1)
        row_action_layout_adv.addWidget(self.btn_vars_menu, stretch=1)
        bottom_layout.addLayout(row_action_layout_adv)

        # Selected Atom Fine Adjustment Group (Molden Style Slider Adjustment + GaussView SmartSlides)
        self.grp_adjust = QtWidgets.QGroupBox("Interactive Parameter Tuning / SmartSlides (Selected Atom or 2/3/4-Atom Selection)")
        self.grp_adjust.setCheckable(True)
        self.grp_adjust.setChecked(True)
        self.grp_adjust.setToolTip("Toggle to expand or collapse Interactive Parameter Tuning / SmartSlides")
        adjust_layout = QtWidgets.QGridLayout(self.grp_adjust)
        adjust_layout.setHorizontalSpacing(8)
        adjust_layout.setVerticalSpacing(5)

        # SmartSlides Displacement Mode & Stereocenter Inversion bar
        row_smart_mode = QtWidgets.QHBoxLayout()
        row_smart_mode.setSpacing(6)
        lbl_disp = QtWidgets.QLabel("Displacement:")
        lbl_disp.setStyleSheet("font-size: 11px; font-weight: 600;")
        row_smart_mode.addWidget(lbl_disp)
        self.combo_smart_mode = QtWidgets.QComboBox()
        self.combo_smart_mode.addItem("Move Substituent Group", "group")
        self.combo_smart_mode.addItem("Move Both Symmetrically", "symmetric")
        self.combo_smart_mode.addItem("Move Single Atom Only", "atom")
        self.combo_smart_mode.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        self.combo_smart_mode.setToolTip(
            "GaussView SmartSlides mode when 2, 3, or 4 atoms are selected in the 3D viewport:\n"
            "• Move Substituent Group: Translates/rotates the connected substituent subgraph\n"
            "• Move Both Groups Symmetrically: Splits displacement 50/50 across both sides\n"
            "• Move Single Atom Only: Moves only the terminal selected atom"
        )
        row_smart_mode.addWidget(self.combo_smart_mode, stretch=1)
        adjust_layout.addLayout(row_smart_mode, 0, 0, 1, 3)

        self.btn_invert_center = QtWidgets.QPushButton("🔄 Invert Chiral Center (R/S)")
        self.btn_invert_center.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold; font-size: 11px; padding: 4px 8px;")
        self.btn_invert_center.setToolTip(
            "Invert local tetrahedral/pyramidal stereocenter (select chiral center atom or center + 2 substituents)\n"
            "without flipping the rest of the molecule!"
        )
        self.btn_invert_center.clicked.connect(self._on_invert_stereocenter)
        adjust_layout.addWidget(self.btn_invert_center, 1, 0, 1, 3)

        # Bond length slider
        lbl_bond = QtWidgets.QLabel("Bond (Å):")
        lbl_bond.setStyleSheet("font-size: 11px; font-weight: 600;")
        adjust_layout.addWidget(lbl_bond, 2, 0)
        self.spin_bond = QtWidgets.QDoubleSpinBox()
        self.spin_bond.setRange(0.1, 10.0)
        self.spin_bond.setSingleStep(0.01)
        self.spin_bond.setStyleSheet("font-size: 11px;")
        self.spin_bond.valueChanged.connect(self._on_spin_bond_changed)
        adjust_layout.addWidget(self.spin_bond, 2, 1)

        self.slider_bond = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_bond.setRange(50, 400) # 0.5 to 4.0 Å (step 0.01)
        self.slider_bond.sliderPressed.connect(self._push_undo_state)
        self.slider_bond.valueChanged.connect(self._on_slider_bond_changed)
        adjust_layout.addWidget(self.slider_bond, 2, 2)

        # Angle slider
        lbl_ang = QtWidgets.QLabel("Angle (°):")
        lbl_ang.setStyleSheet("font-size: 11px; font-weight: 600;")
        adjust_layout.addWidget(lbl_ang, 3, 0)
        self.spin_angle = QtWidgets.QDoubleSpinBox()
        self.spin_angle.setRange(0.0, 180.0)
        self.spin_angle.setSingleStep(0.5)
        self.spin_angle.setStyleSheet("font-size: 11px;")
        self.spin_angle.valueChanged.connect(self._on_spin_angle_changed)
        adjust_layout.addWidget(self.spin_angle, 3, 1)

        self.slider_angle = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_angle.setRange(0, 1800) # 0 to 180 deg (step 0.1)
        self.slider_angle.sliderPressed.connect(self._push_undo_state)
        self.slider_angle.valueChanged.connect(self._on_slider_angle_changed)
        adjust_layout.addWidget(self.slider_angle, 3, 2)

        # Dihedral slider
        lbl_dih = QtWidgets.QLabel("Dihed (°):")
        lbl_dih.setStyleSheet("font-size: 11px; font-weight: 600;")
        adjust_layout.addWidget(lbl_dih, 4, 0)
        self.spin_dihedral = QtWidgets.QDoubleSpinBox()
        self.spin_dihedral.setRange(-180.0, 180.0)
        self.spin_dihedral.setSingleStep(1.0)
        self.spin_dihedral.setStyleSheet("font-size: 11px;")
        self.spin_dihedral.valueChanged.connect(self._on_spin_dihedral_changed)
        adjust_layout.addWidget(self.spin_dihedral, 4, 1)

        self.slider_dihedral = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_dihedral.setRange(-1800, 1800) # -180 to 180 deg (step 0.1)
        self.slider_dihedral.sliderPressed.connect(self._push_undo_state)
        self.slider_dihedral.valueChanged.connect(self._on_slider_dihedral_changed)
        adjust_layout.addWidget(self.slider_dihedral, 4, 2)

        bottom_layout.addWidget(self.grp_adjust)

        # Z-Matrix Symbolic Variable Sliders (ZOPT Coordinate Inspection)
        self.grp_var_sliders = QtWidgets.QGroupBox("Z-Matrix Symbolic Variables (ZOPT Sliders)")
        self.var_sliders_layout = QtWidgets.QGridLayout(self.grp_var_sliders)
        self.var_sliders_layout.setSpacing(6)
        self.grp_var_sliders.setVisible(False)
        bottom_layout.addWidget(self.grp_var_sliders)

        bottom_scroll.setWidget(bottom_container)
        zmat_splitter.addWidget(bottom_scroll)
        zmat_splitter.setStretchFactor(0, 5) # Table gets 70-80% height!
        zmat_splitter.setStretchFactor(1, 2) # Sliders get remaining height
        zmat_splitter.setSizes([450, 220])
        tab_table_layout.addWidget(zmat_splitter)

        self.tab_table = tab_table
        self.scroll_tab_table = bottom_scroll
        self.tabs.addTab(tab_table, "Z-Matrix")

        # Tab 2: Raw Text Z-Matrix / Gaussian Format
        tab_raw = QtWidgets.QWidget()
        tab_raw_layout = QtWidgets.QVBoxLayout(tab_raw)
        
        # Custom subclass or event filter can handle Ctrl+Enter / Shift+Enter
        class RawZMatPlainTextEdit(QtWidgets.QPlainTextEdit):
            def __init__(self, main_win, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.main_win = main_win

            def keyPressEvent(self, event: QtGui.QKeyEvent):
                if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                    # If Ctrl+Enter, Shift+Enter, or Alt+Enter is pressed, trigger immediate apply!
                    if event.modifiers() & (
                        QtCore.Qt.KeyboardModifier.ControlModifier
                        | QtCore.Qt.KeyboardModifier.ShiftModifier
                        | QtCore.Qt.KeyboardModifier.AltModifier
                    ):
                        self.main_win._on_apply_text_zmat()
                        event.accept()
                        return
                super().keyPressEvent(event)

        self.text_raw_zmat = RawZMatPlainTextEdit(self)
        self.text_raw_zmat.setPlaceholderText("Enter Z-matrix coordinates or variables...\nPress Ctrl+Enter or Shift+Enter to immediately apply changes.")
        
        raw_header_layout = QtWidgets.QHBoxLayout()
        raw_header_layout.addWidget(QtWidgets.QLabel("Direct Z-Matrix Text (Molden / Gaussian format):"))
        self.chk_auto_apply_raw = QtWidgets.QCheckBox("Auto-apply changes as you type")
        self.chk_auto_apply_raw.setChecked(True)
        self.chk_auto_apply_raw.setToolTip("When checked, valid Z-matrix text changes update the 3D model automatically without having to press any buttons.")
        raw_header_layout.addWidget(self.chk_auto_apply_raw, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        tab_raw_layout.addLayout(raw_header_layout)
        tab_raw_layout.addWidget(self.text_raw_zmat, stretch=1)

        # Debounced live update timer (e.g. 350ms after user pauses typing)
        self._raw_zmat_timer = QtCore.QTimer(self)
        self._raw_zmat_timer.setSingleShot(True)
        self._raw_zmat_timer.setInterval(350)
        self._raw_zmat_timer.timeout.connect(self._on_auto_apply_text_zmat)
        self.text_raw_zmat.textChanged.connect(self._on_raw_zmat_text_changed)

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
        btn_apply_raw = QtWidgets.QPushButton("⚡ Apply Text Z-Matrix (Ctrl+Enter)")
        btn_apply_raw.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold;")
        btn_apply_raw.setToolTip("Apply the raw Z-matrix text to the 3D model and table (Shortcut: Ctrl+Enter or Shift+Enter)")
        btn_apply_raw.clicked.connect(self._on_apply_text_zmat)
        raw_btn_layout.addWidget(btn_open_raw)
        raw_btn_layout.addWidget(btn_save_raw)
        raw_btn_layout.addWidget(btn_apply_raw)
        tab_raw_layout.addLayout(raw_btn_layout)
        self.tabs.addTab(tab_raw, "Raw ZMAT Text")

        # Tab 3: Cartesian Coordinates (XYZ) Text Editor
        tab_xyz = QtWidgets.QWidget()
        tab_xyz_layout = QtWidgets.QVBoxLayout(tab_xyz)

        class RawXYZPlainTextEdit(QtWidgets.QPlainTextEdit):
            def __init__(self, main_win, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.main_win = main_win

            def keyPressEvent(self, event: QtGui.QKeyEvent):
                if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                    if event.modifiers() & (
                        QtCore.Qt.KeyboardModifier.ControlModifier
                        | QtCore.Qt.KeyboardModifier.ShiftModifier
                        | QtCore.Qt.KeyboardModifier.AltModifier
                    ):
                        self.main_win._on_apply_text_xyz()
                        event.accept()
                        return
                super().keyPressEvent(event)

        self.text_raw_xyz = RawXYZPlainTextEdit(self)
        self.text_raw_xyz.setPlaceholderText(
            "Paste or type Cartesian XYZ coordinates here (standard or headerless)...\n"
            "Example:\n"
            "  O   0.000000   0.000000   0.117400\n"
            "  H   0.000000   0.757000  -0.469600\n"
            "  H   0.000000  -0.757000  -0.469600\n\n"
            "Press Ctrl+Enter to apply immediately."
        )
        self.text_raw_xyz.setFont(QtGui.QFont("Monospace", 10))

        xyz_header_layout = QtWidgets.QHBoxLayout()
        xyz_header_layout.addWidget(QtWidgets.QLabel("Direct Cartesian XYZ Text:"))
        self.chk_auto_apply_xyz = QtWidgets.QCheckBox("Auto-apply changes as you type")
        self.chk_auto_apply_xyz.setChecked(True)
        self.chk_auto_apply_xyz.setToolTip("When checked, valid XYZ text updates the 3D model automatically without having to press any buttons.")
        xyz_header_layout.addWidget(self.chk_auto_apply_xyz, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        tab_xyz_layout.addLayout(xyz_header_layout)
        tab_xyz_layout.addWidget(self.text_raw_xyz, stretch=1)

        # Debounced live update timer for XYZ text editor
        self._raw_xyz_timer = QtCore.QTimer(self)
        self._raw_xyz_timer.setSingleShot(True)
        self._raw_xyz_timer.setInterval(350)
        self._raw_xyz_timer.timeout.connect(self._on_auto_apply_text_xyz)
        self.text_raw_xyz.textChanged.connect(self._on_raw_xyz_text_changed)

        xyz_quick_actions = QtWidgets.QHBoxLayout()
        xyz_quick_actions.setSpacing(6)
        btn_paste_clip = QtWidgets.QPushButton("📋 Paste Clipboard")
        btn_paste_clip.setToolTip("Paste whatever is currently in the system clipboard into this XYZ editor")
        btn_paste_clip.setStyleSheet("background-color: #0369a1; color: #ffffff; font-weight: bold;")
        btn_paste_clip.clicked.connect(self._on_xyz_tab_paste_clipboard)
        btn_copy_xyz_tab = QtWidgets.QPushButton("📋 Copy XYZ")
        btn_copy_xyz_tab.setToolTip("Copy the current XYZ text to system clipboard")
        btn_copy_xyz_tab.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold;")
        btn_copy_xyz_tab.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.text_raw_xyz.toPlainText()))
        btn_clear_xyz = QtWidgets.QPushButton("🧹 Clear")
        btn_clear_xyz.setStyleSheet("background-color: #475569; color: #ffffff;")
        btn_clear_xyz.clicked.connect(lambda: self.text_raw_xyz.clear())
        xyz_quick_actions.addWidget(btn_paste_clip)
        xyz_quick_actions.addWidget(btn_copy_xyz_tab)
        xyz_quick_actions.addWidget(btn_clear_xyz)
        tab_xyz_layout.addLayout(xyz_quick_actions)

        xyz_btn_layout = QtWidgets.QHBoxLayout()
        btn_import_xyz_file = QtWidgets.QPushButton("📂 Open XYZ File...")
        btn_import_xyz_file.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold;")
        btn_import_xyz_file.clicked.connect(self._on_import_xyz)
        btn_export_xyz_file = QtWidgets.QPushButton("💾 Save XYZ File...")
        btn_export_xyz_file.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold;")
        btn_export_xyz_file.clicked.connect(self._on_export_xyz)
        btn_apply_xyz = QtWidgets.QPushButton("⚡ Apply Cartesian XYZ (Ctrl+Enter)")
        btn_apply_xyz.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold;")
        btn_apply_xyz.setToolTip("Apply the Cartesian coordinates to the 3D model, table, and Z-Matrix (Ctrl+Enter)")
        btn_apply_xyz.clicked.connect(self._on_apply_text_xyz)
        xyz_btn_layout.addWidget(btn_import_xyz_file)
        xyz_btn_layout.addWidget(btn_export_xyz_file)
        xyz_btn_layout.addWidget(btn_apply_xyz)
        tab_xyz_layout.addLayout(xyz_btn_layout)
        self.tabs.addTab(tab_xyz, "Cartesian XYZ")

        # Tab 4: Chemical Builders & Fragment Substitution
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
        scroll_builder = QtWidgets.QScrollArea()
        scroll_builder.setWidgetResizable(True)
        scroll_builder.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_builder.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_builder.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_builder.setWidget(tab_builder)
        self.tab_builder = scroll_builder
        self.tab_builder_widget = tab_builder
        self.tabs.addTab(scroll_builder, "Presets & Fragments")

        # Tabs 4, 5, 6 (Vibrations, Symmetry, Molecular Orbitals) via UIInspectorTabsMixin
        self._init_vibrations_tab()
        self._init_symmetry_tab()
        self._init_orbitals_tab()

        right_layout.addWidget(self.tabs)
        # Tab contents must never dictate the window width
        self.tabs.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Expanding)
        right_container.setMinimumWidth(320)
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

        # Initialize Dockable Workspace System
        from .dock_panels import WorkspaceDockManager
        self.dock_manager = WorkspaceDockManager(self)

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

        act_gallery = QtGui.QAction("📚 Chemistry &Sample & Benchmark Gallery...", self)
        act_gallery.setToolTip("Open curated benchmark structures (Ferrocene, C60, Diels-Alder TS, Water Prism) in 1 click")
        act_gallery.triggered.connect(self._on_open_sample_gallery)
        menu_file.addAction(act_gallery)

        act_open_xyz = QtGui.QAction("Import &XYZ File...", self)
        act_open_xyz.triggered.connect(self._on_import_xyz)
        menu_file.addAction(act_open_xyz)

        act_paste_xyz_file = QtGui.QAction("📥 Paste &Cartesian Coordinates from Clipboard...", self)
        act_paste_xyz_file.setToolTip("Parse and load Cartesian XYZ coordinates from clipboard directly into the 3D viewer (Ctrl+Shift+V)")
        act_paste_xyz_file.triggered.connect(self._on_paste_as_new_molecule)
        menu_file.addAction(act_paste_xyz_file)

        act_open_inp = QtGui.QAction("Open newBRORSEN &Input File (*.inp)...", self)
        act_open_inp.setShortcut("Ctrl+I")
        act_open_inp.setToolTip("Open a newBRORSEN input file (.inp) into AGYMolden and the Input Builder")
        act_open_inp.triggered.connect(self._on_open_brorsen_input_file)
        menu_file.addAction(act_open_inp)

        act_open_smiles = QtGui.QAction("Build from &Name / SMILES / InChI...", self)
        act_open_smiles.setShortcut("Ctrl+Shift+B")
        act_open_smiles.setToolTip("Type a chemical name (e.g. aspirin, caffeine), SMILES (e.g. c1ccccc1), or InChI to build a 3D structure automatically (Ctrl+Shift+B)")
        act_open_smiles.triggered.connect(self._on_prompt_smiles_or_inchi)
        menu_file.addAction(act_open_smiles)

        act_2d_sketcher = QtGui.QAction("✏️ 2D &Molecular Bond-Line Sketcher (ChemDraw / Marvin Style)...", self)
        act_2d_sketcher.setShortcut("Ctrl+Shift+K")
        act_2d_sketcher.setToolTip("Sketch 2D chemical bond-line formulas (rings, heteroatoms, multiple bonds) and convert automatically to 3D (Ctrl+Shift+K)")
        act_2d_sketcher.triggered.connect(self._on_open_2d_sketcher)
        menu_file.addAction(act_2d_sketcher)

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

        self.menu_recent = menu_file.addMenu("Open &Recent")
        self._update_recent_files_menu()

        menu_file.addSeparator()

        act_save_proj = QtGui.QAction("Save Project Bundle (.agyproj)...", self)
        act_save_proj.setToolTip("Save full molecular model, Z-matrix, calculation history, and view settings into a portable project bundle")
        act_save_proj.triggered.connect(self._on_save_project_bundle)
        menu_file.addAction(act_save_proj)

        act_save_zmat = QtGui.QAction("Save Z-Matrix (&ZMAT)...", self)
        act_save_zmat.setShortcut("Ctrl+S")
        act_save_zmat.triggered.connect(self._on_save_zmat)
        menu_file.addAction(act_save_zmat)

        act_save_num = QtGui.QAction("Save Z-Matrix with &Numbers...", self)
        act_save_num.setToolTip("Every bond, angle and dihedral written as a number")
        act_save_num.triggered.connect(lambda: self._on_save_zmat_format("numeric"))
        menu_file.addAction(act_save_num)

        act_save_var = QtGui.QAction("Save Z-Matrix with &Variables...", self)
        act_save_var.setToolTip("Every coordinate written as a variable, with a Variables: block of values (shared/symmetric variables are preserved)")
        act_save_var.triggered.connect(lambda: self._on_save_zmat_format("variables"))
        menu_file.addAction(act_save_var)

        act_save_xyz = QtGui.QAction("Export Coordinates (&XYZ)...", self)
        act_save_xyz.triggered.connect(self._on_export_xyz)
        menu_file.addAction(act_save_xyz)

        act_save_pdb = QtGui.QAction("Export Protein Data Bank (&PDB)...", self)
        act_save_pdb.triggered.connect(self._on_export_pdb)
        menu_file.addAction(act_save_pdb)

        act_save_sdf = QtGui.QAction("Export Structure-Data File (&SDF / MOL)...", self)
        act_save_sdf.triggered.connect(self._on_export_sdf)
        menu_file.addAction(act_save_sdf)

        act_save_mol2 = QtGui.QAction("Export Tripos (&Mol2)...", self)
        act_save_mol2.setToolTip("Export molecular coordinates and bonds to Tripos Mol2 format")
        act_save_mol2.triggered.connect(self._on_export_mol2)
        menu_file.addAction(act_save_mol2)

        act_save_cif = QtGui.QAction("Export Crystallographic File (&CIF)...", self)
        act_save_cif.setToolTip("Export molecular coordinates and unit cell parameters to standard CIF syntax")
        act_save_cif.triggered.connect(self._on_export_cif)
        menu_file.addAction(act_save_cif)

        menu_file.addSeparator()

        act_export_png = QtGui.QAction("Export &Image (PNG)...", self)
        act_export_png.setShortcuts([
            QtGui.QKeySequence("Ctrl+E"),
            QtGui.QKeySequence("Ctrl+P"),
        ])
        act_export_png.setToolTip("Save a high-resolution PNG image of the 3D viewport (Ctrl+E)")
        act_export_png.triggered.connect(self._on_export_image)
        menu_file.addAction(act_export_png)

        act_export_svg = QtGui.QAction("📐 Export &Scalable Vector Graphics (SVG)...", self)
        act_export_svg.setToolTip("Export infinite-resolution 3D vector graphics for Illustrator, Inkscape, or LaTeX")
        act_export_svg.triggered.connect(self._on_export_svg)
        menu_file.addAction(act_export_svg)

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

        act_export_bundle = QtGui.QAction("📦 Export Reproducibility Bundle (&ZIP)...", self)
        act_export_bundle.setShortcut("Ctrl+Shift+S")
        act_export_bundle.setToolTip("Export complete Supporting Information (SI) package: XYZ, ZMAT, Summary JSON, Methods Markdown, and calculation logs (Ctrl+Shift+S)")
        act_export_bundle.triggered.connect(self._on_export_reproducibility_bundle)
        menu_file.addAction(act_export_bundle)

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
        self.act_paste_mol.setToolTip("Paste copied fragment, Cartesian XYZ coordinates, or SMILES/InChI into the 3D viewport (Ctrl+V)")
        self.act_paste_mol.triggered.connect(self._on_paste)
        menu_edit.addAction(self.act_paste_mol)

        act_paste_xyz = QtGui.QAction("📥 Paste &XYZ Coordinates as Molecule", self)
        act_paste_xyz.setShortcut(QtGui.QKeySequence("Ctrl+Shift+V"))
        act_paste_xyz.setToolTip("Directly load XYZ coordinates from clipboard as the active molecule (replaces or resets canvas) (Ctrl+Shift+V)")
        act_paste_xyz.triggered.connect(self._on_paste_as_new_molecule)
        menu_edit.addAction(act_paste_xyz)

        menu_edit.addSeparator()

        # Copy Identifiers submenu
        menu_copy = menu_edit.addMenu("📋 Copy &Identifier...")
        act_cp_xyz = QtGui.QAction("Copy as &Cartesian XYZ (with header)", self)
        act_cp_xyz.setShortcut("Ctrl+Alt+C")
        act_cp_xyz.setToolTip("Copy molecule in standard Cartesian XYZ format to clipboard (Ctrl+Alt+C)")
        act_cp_xyz.triggered.connect(self._on_copy_as_xyz)
        menu_copy.addAction(act_cp_xyz)

        act_cp_block = QtGui.QAction("Copy &Coordinates Block (No Header — ORCA/Gaussian)", self)
        act_cp_block.setShortcut("Ctrl+Alt+X")
        act_cp_block.setToolTip("Copy headerless 'Element X Y Z' coordinates block ready to paste directly into ORCA or Gaussian input files (Ctrl+Alt+X)")
        act_cp_block.triggered.connect(self._on_copy_coords_block)
        menu_copy.addAction(act_cp_block)

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

        menu_copy.addSeparator()

        act_cp_si_acs = QtGui.QAction("📑 Copy &Supporting Information (SI — ACS/RSC Block)", self)
        act_cp_si_acs.setShortcut("Ctrl+Alt+S")
        act_cp_si_acs.setToolTip("Copy publication-ready Supporting Information block (Formula, Point Group, Energies, N_imag, Coordinates) in ACS/RSC format (Ctrl+Alt+S)")
        act_cp_si_acs.triggered.connect(lambda: self._on_copy_supporting_info("acs"))
        menu_copy.addAction(act_cp_si_acs)

        act_cp_si_ltx = QtGui.QAction("📑 Copy Supporting Information (&LaTeX Table)", self)
        act_cp_si_ltx.setToolTip("Copy publication-ready Supporting Information table in LaTeX format")
        act_cp_si_ltx.triggered.connect(lambda: self._on_copy_supporting_info("latex"))
        menu_copy.addAction(act_cp_si_ltx)

        act_cp_si_md = QtGui.QAction("📑 Copy Supporting Information (&Markdown Block)", self)
        act_cp_si_md.setToolTip("Copy publication-ready Supporting Information entry in Markdown format")
        act_cp_si_md.triggered.connect(lambda: self._on_copy_supporting_info("markdown"))
        menu_copy.addAction(act_cp_si_md)

        menu_edit.addSeparator()

        act_add_line = QtGui.QAction("&Add Atom Line", self)
        act_add_line.triggered.connect(self._on_add_line)
        menu_edit.addAction(act_add_line)

        act_del = QtGui.QAction("&Delete Selected Atom", self)
        act_del.triggered.connect(self._on_delete_selected_atom)
        menu_edit.addAction(act_del)

        act_reorder = QtGui.QAction("&Reorder Z-Matrix (Full)", self)
        act_reorder.setShortcut("Ctrl+Shift+R")
        act_reorder.setToolTip("Reorder entire Z-Matrix sequentially (Molden ZMEORD) (Ctrl+Shift+R)")
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

        act_cycle_welder = QtGui.QAction("💍 &Weld / Fuse Molecular Rings...", self)
        act_cycle_welder.setShortcut("Ctrl+Shift+W")
        act_cycle_welder.setToolTip("Fuse aromatic or saturated rings onto selected bonds, or create bicyclic bridgeheads (Ctrl+Shift+W)")
        act_cycle_welder.triggered.connect(self._on_open_cycle_welder)
        menu_edit.addAction(act_cycle_welder)

        menu_origin = menu_edit.addMenu("🎯 &Translate to Origin (0, 0, 0)...")
        act_orig_com = QtGui.QAction("Center of Mass to Origin", self)
        act_orig_com.setToolTip("Translate molecule so its center-of-mass is at Cartesian (0, 0, 0)")
        act_orig_com.triggered.connect(lambda: self._center_molecule_to_origin("com"))
        menu_origin.addAction(act_orig_com)

        act_orig_anchor = QtGui.QAction("Selected Atom to Origin", self)
        act_orig_anchor.setToolTip("Translate molecule so the currently selected atom is at (0, 0, 0)")
        act_orig_anchor.triggered.connect(lambda: self._center_molecule_to_origin("anchor"))
        menu_origin.addAction(act_orig_anchor)

        act_orient_sym = QtGui.QAction("📐 Orient to &Standard Symmetry Orientation (All Groups)", self)
        act_orient_sym.setShortcut(QtGui.QKeySequence("Alt+S"))
        act_orient_sym.setToolTip(
            "Rigidly rotate and center molecule to standard IUPAC / Schoenflies orientation (Alt+S).\n"
            "Aligns principal axis (highest Cn) along Z, mirror plane in XZ/XY, and centers COM at (0,0,0)."
        )
        act_orient_sym.triggered.connect(self._on_align_symmetry)
        menu_edit.addAction(act_orient_sym)

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
        act_flush_mem.setShortcut(QtGui.QKeySequence("Ctrl+Shift+X"))
        act_flush_mem.setToolTip("Flush atom and fragment clipboard memory (Ctrl+Shift+X). Double-clicking canvas will prompt Periodic Table / Fragment picker.")
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

        self.act_tweezers = QtGui.QAction("🧲 Interactive Molecular Tweezers (Live UFF Drag)", self)
        self.act_tweezers.setCheckable(True)
        self.act_tweezers.setChecked(False)
        self.act_tweezers.setToolTip("When checked, simply Left-Click and drag any atom in the 3D viewport to dynamically relax connected atoms in real-time via UFF force field")
        self.act_tweezers.toggled.connect(self._on_toggle_tweezers)
        if hasattr(self, "btn_ribbon_tweezers"):
            self.act_tweezers.toggled.connect(self.btn_ribbon_tweezers.setChecked)
            self.btn_ribbon_tweezers.toggled.connect(self.act_tweezers.setChecked)
        menu_edit.addAction(self.act_tweezers)

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

        # Variables Menu
        menu_vars = menubar.addMenu("&Variables")

        act_sym_bonds = QtGui.QAction("🔗 Assign &Symmetric Bond Variables Only (Keep Angles/Dihedrals Fixed)", self)
        act_sym_bonds.setToolTip("Equivalent bond distances (e.g. C-C and C-H in benzene) share variables (R1, R2) "
                                 "while angles and dihedrals remain fixed numerical constants")
        act_sym_bonds.triggered.connect(self._on_assign_symmetric_bond_variables)
        menu_vars.addAction(act_sym_bonds)

        act_sym_vars = QtGui.QAction("🔗 Assign &Symmetric Variables (All: Bonds, Angles, Dihedrals)", self)
        act_sym_vars.setToolTip("Equivalent coordinates (e.g. all C-C and all C-H bonds in benzene) share one "
                                "variable, reducing the number of optimization parameters")
        act_sym_vars.triggered.connect(self._on_assign_symmetric_variables)
        menu_vars.addAction(act_sym_vars)

        act_all_vars = QtGui.QAction("🔤 Assign &Individual Variables", self)
        act_all_vars.setToolTip("Assign an individual variable name (R1..Rn, A1..An, D1..Dn) to every internal coordinate")
        act_all_vars.triggered.connect(self._on_assign_individual_variables)
        menu_vars.addAction(act_all_vars)

        menu_vars.addSeparator()

        act_var_save = QtGui.QAction("💾 Save Z-Matrix with &Variables...", self)
        act_var_save.setToolTip("Save Z-Matrix with symbolic variable names and a 'Variables:' block of values")
        act_var_save.triggered.connect(lambda: self._on_save_zmat_format("variables"))
        menu_vars.addAction(act_var_save)

        act_clear_vars = QtGui.QAction("🔢 &Clear Variables (Use Numbers Only)", self)
        act_clear_vars.setToolTip("Clear all symbolic variables and restore explicit numeric values")
        act_clear_vars.triggered.connect(self._on_clear_variables)
        menu_vars.addAction(act_clear_vars)

        # Tools Menu
        menu_tools = menubar.addMenu("&Tools")

        act_unit_conv = QtGui.QAction("⚡ Quantum &Unit Converter (Energy, Length, Dipole, Polarizability)...", self)
        act_unit_conv.setShortcut("Ctrl+U")
        act_unit_conv.setToolTip("Instant physical constant conversions: Hartree ↔ eV ↔ kcal/mol ↔ kJ/mol ↔ cm⁻¹ ↔ K; Bohr ↔ Å; Debye ↔ a.u.; Polarizability; Field & Pressure (Ctrl+U)")
        act_unit_conv.triggered.connect(self._open_unit_converter)
        menu_tools.addAction(act_unit_conv)

        act_summary_card = QtGui.QAction("📊 &Calculation Summary Card...", self)
        act_summary_card.setShortcut("F4")
        act_summary_card.setToolTip("Open publication-ready 1-page summary card of current QM calculation results (F4)")
        act_summary_card.triggered.connect(self._on_open_summary_card)
        menu_tools.addAction(act_summary_card)

        menu_tools.addSeparator()

        act_quick_orca = QtGui.QAction("🚀 Quick Run &ORCA Calculation...", self)
        act_quick_orca.setShortcut("Ctrl+R")
        act_quick_orca.setToolTip("Quick 1-Click setup and execution of ORCA calculations for coursework")
        act_quick_orca.triggered.connect(self._on_quick_run_orca)
        menu_tools.addAction(act_quick_orca)

        act_generate_qm_input = QtGui.QAction("⚡ Generate &QM Calculation Input...", self)
        act_generate_qm_input.setShortcut("Ctrl+G")
        act_generate_qm_input.setToolTip("Set up and generate input files for Gaussian, ORCA, PySCF, openLOWDIN, or xTB")
        act_generate_qm_input.triggered.connect(self._on_open_qm_dialog)
        menu_tools.addAction(act_generate_qm_input)

        act_generate_brorsen_input = QtGui.QAction("✨ new&BRORSEN Input Builder (Positrons, NEO)...", self)
        act_generate_brorsen_input.setShortcut("Ctrl+B")
        act_generate_brorsen_input.setToolTip("Set up multi-stage input files for newBRORSEN (Positrons, NEO, and CCSD(T))")
        act_generate_brorsen_input.triggered.connect(self._on_open_brorsen_dialog)
        menu_tools.addAction(act_generate_brorsen_input)

        act_job_monitor = QtGui.QAction("🖥️ Calculation &Job Monitor...", self)
        act_job_monitor.setShortcut("Ctrl+J")
        act_job_monitor.setToolTip("View live calculation progress, streaming output logs, and real-time 3D updates")
        act_job_monitor.triggered.connect(self._on_open_job_monitor)
        menu_tools.addAction(act_job_monitor)

        act_workflow_dag = QtGui.QAction("🔀 Visual &Workflow DAG Pipeline Editor (Multi-Stage QM)...", self)
        act_workflow_dag.setShortcut("Ctrl+Shift+D")
        act_workflow_dag.setToolTip("Design and execute chained multi-stage QM workflows (Opt → Freq → TD-DFT / TS → IRC / SP) on an interactive visual DAG canvas (Ctrl+Shift+D)")
        act_workflow_dag.triggered.connect(self._on_open_workflow_dag)
        menu_tools.addAction(act_workflow_dag)

        act_remote_transfer = QtGui.QAction("📁 Remote &File Transfer (SSH / Cluster)...", self)
        act_remote_transfer.setToolTip("Browse and transfer input and output files between local machine and remote cluster")
        act_remote_transfer.triggered.connect(self._on_open_remote_file_manager)
        menu_tools.addAction(act_remote_transfer)

        act_qm_engines = QtGui.QAction("🔧 Configure &QM Engines & Routes...", self)
        act_qm_engines.setToolTip("Detect installed QM packages (ORCA, PySCF, Gaussian, openLOWDIN, xTB, Q-Chem) and configure custom executable routes")
        act_qm_engines.triggered.connect(self._on_configure_qm_engines)
        menu_tools.addAction(act_qm_engines)

        menu_tools.addSeparator()

        act_dihed_scan = QtGui.QAction("🔄 1D &Dihedral Rotor Scan / PES Drive...", self)
        act_dihed_scan.setToolTip("Step selected 4-atom dihedral through 360° with real-time forcefield relaxation (conformer search)")
        act_dihed_scan.triggered.connect(self._on_open_dihedral_scan_dialog)
        menu_tools.addAction(act_dihed_scan)

        act_clash = QtGui.QAction("⚠️ Highlight &Steric Clashes...", self)
        act_clash.setToolTip("Scan molecular geometry and highlight severe non-bonded steric overlaps (< 70% Σr_vdw)")
        act_clash.triggered.connect(self._check_and_highlight_steric_clashes)
        menu_tools.addAction(act_clash)

        act_steric_desc = QtGui.QAction("🛡️ Catalysis &Steric && Coordination Descriptors (%V_bur, Sterimol, τ₄/τ₅)...", self)
        act_steric_desc.setToolTip("Compute SambVca / SEQCROW Percent Buried Volume (%V_bur), Verloop Sterimol (L, B1, B5), Tolman Cone Angle, and Coordination Geometry Indices (τ₄, τ₄', τ₅, Σ)")
        act_steric_desc.triggered.connect(self._on_open_steric_descriptors_dialog)
        menu_tools.addAction(act_steric_desc)

        act_ring_homa = QtGui.QAction("💍 &Ring Aromaticity (HOMA Index) && Planarity Inspector...", self)
        act_ring_homa.setToolTip("Compute Kruszewski–Krygowski HOMA aromaticity index, Hückel π-electron count, and ring planarity RMSD for all rings")
        act_ring_homa.triggered.connect(self._on_open_ring_aromaticity_dialog)
        menu_tools.addAction(act_ring_homa)

        act_iso_kie = QtGui.QAction("⚗️ &Isotopologue Vibrational Shift && KIE Calculator...", self)
        act_iso_kie.setToolTip("Substitute isotopes (H→D, ¹²C→¹³C, ¹⁶O→¹⁸O) and compute frequency shifts Δν, ΔZPE, and Kinetic Isotope Effects (k_H/k_D)")
        act_iso_kie.triggered.connect(self._on_open_isotopologue_dialog)
        menu_tools.addAction(act_iso_kie)

        act_pbc = QtGui.QAction("📦 &Periodic Boundary Conditions (PBC) && Supercell Generator...", self)
        act_pbc.setToolTip("View/edit crystallographic unit cell parameters (a, b, c, α, β, γ), fractional coordinates, and replicate Nx×Ny×Nz supercells")
        act_pbc.triggered.connect(self._on_open_pbc_dialog)
        menu_tools.addAction(act_pbc)

        act_quick_pyscf = QtGui.QAction("⚡ Quick In-Memory DFT / HF (PySCF)...", self)
        act_quick_pyscf.setToolTip("Solve B3LYP / def2-SVP directly in RAM within seconds (no external binaries or files required)")
        act_quick_pyscf.triggered.connect(lambda: self._on_quick_in_memory_pyscf_dft())
        menu_tools.addAction(act_quick_pyscf)

        act_overlay = QtGui.QAction("🔬 &Compare & Align Structures (RMSD Overlay)...", self)
        act_overlay.setShortcut("Ctrl+Shift+A")
        act_overlay.setToolTip("Superimpose two structures with Kabsch SVD alignment and calculate all-atom/heavy-atom RMSD")
        act_overlay.triggered.connect(self._on_open_structure_overlay)
        menu_tools.addAction(act_overlay)

        act_fmo_align = QtGui.QAction("⚡ &Frontier Orbital (FMO) Alignment & Reactivity Matcher...", self)
        act_fmo_align.setToolTip("Compare Donor-Acceptor HOMO/LUMO levels side-by-side with chemical hardness (η) and electrophilicity (ω)")
        act_fmo_align.triggered.connect(self._on_open_fmo_alignment)
        menu_tools.addAction(act_fmo_align)

        act_cdft = QtGui.QAction("⚛️ &Conceptual DFT (CDFT) & Fukui Reactivity Workbench...", self)
        act_cdft.setShortcut("Ctrl+Shift+F")
        act_cdft.setToolTip("Evaluate global CDFT indices (μ, η, ω, N_Nu) and atomic Fukui functions (f+, f-, Δf) with 3D color mapping (Ctrl+Shift+F)")
        act_cdft.triggered.connect(self._on_open_cdft_workbench)
        menu_tools.addAction(act_cdft)

        act_benchmark = QtGui.QAction("⚖️ Cross-Engine &Benchmark & Result Comparison...", self)
        act_benchmark.setShortcut("Ctrl+Shift+M")
        act_benchmark.setToolTip("Compare multiple calculation logs side-by-side across Gaussian, ORCA, PySCF, and xTB (energetics, ΔE, frontier gap, and Kabsch SVD RMSD matrix) (Ctrl+Shift+M)")
        act_benchmark.triggered.connect(self._on_open_benchmark_dialog)
        menu_tools.addAction(act_benchmark)

        act_thermo = QtGui.QAction("⚗️ Reaction &Thermochemistry && Eyring Kinetics (ΔG‡, ΔGrxn, Keq)...", self)
        act_thermo.setShortcut("Alt+Shift+R")
        act_thermo.setToolTip("Compute reaction free energies (ΔGrxn), activation barriers (ΔG‡), equilibrium constant (Keq), and Eyring rate constants k(T) from QM logs (Alt+Shift+R)")
        act_thermo.triggered.connect(self._on_open_reaction_thermo_dialog)
        menu_tools.addAction(act_thermo)

        act_solvation = QtGui.QAction("💧 &Continuum Solvation Free Energy (SMD / PCM / CPCM)...", self)
        act_solvation.setToolTip("Estimate solvent effects, dielectric response (ΔG_el), and cavitation/dispersion (ΔG_cds) across 11 standard solvents")
        act_solvation.triggered.connect(self._on_open_solvation_dialog)
        menu_tools.addAction(act_solvation)

        act_sterics = QtGui.QAction("🛡️ &Catalysis Sterics: Buried Volume (%V_bur) & Sterimol (L, B1, B5)...", self)
        act_sterics.setToolTip("Calculate ligand buried volume (%V_bur), SambVca-style 2D topographic steric altitude maps, and Verloop Sterimol descriptors (L, B1, B5)")
        act_sterics.triggered.connect(self._on_open_sterics_dialog)
        menu_tools.addAction(act_sterics)

        act_marcus = QtGui.QAction("⚡ &Marcus Electron Transfer & Photoredox Kinetics...", self)
        act_marcus.setToolTip("Evaluate inner/outer-sphere reorganization energies (λ_in, λ_out), Marcus activation barrier (ΔG‡), non-adiabatic rate k_ET, and diabatic PES parabolas")
        act_marcus.triggered.connect(self._on_open_marcus_dialog)
        menu_tools.addAction(act_marcus)

        act_spin = QtGui.QAction("⚛️ Open-Shell &Spin Diagnostics & Yamaguchi Projection (<S²>, J)...", self)
        act_spin.setToolTip("Diagnose <S²> spin contamination, compute Yamaguchi approximate spin-projected energies, and Heisenberg exchange coupling (J)")
        act_spin.triggered.connect(self._on_open_spin_diagnostics)
        menu_tools.addAction(act_spin)

        act_redox = QtGui.QAction("⚡ Electrochemical &Redox Potentials & Born-Haber Cycle (E° vs SHE/SCE)...", self)
        act_redox.setToolTip("Calculate standard reduction/oxidation potentials (E° vs. SHE, SCE, Ag/AgCl, Fc/Fc⁺) via thermodynamic Born-Haber solvation cycles")
        act_redox.triggered.connect(self._on_open_redox_dialog)
        menu_tools.addAction(act_redox)

        act_mecp = QtGui.QAction("🔀 &Minimum Energy Crossing Point (MECP) & Spin-Inversion Explorer...", self)
        act_mecp.setToolTip("Interpolate crossing seams (ΔE → 0) between two spin states in two-state reactivity and export ORCA SurfCross MECP search decks")
        act_mecp.triggered.connect(self._on_open_mecp_dialog)
        menu_tools.addAction(act_mecp)

        act_eda = QtGui.QAction("🧩 &Energy Decomposition Analysis (EDA / SAPT / ALMO)...", self)
        act_eda.setToolTip("Decompose interaction energies into Electrostatics, Pauli Repulsion, Orbital/Charge-Transfer, and Dispersion terms from Q-Chem ALMO-EDA or Psi4 SAPT")
        act_eda.triggered.connect(self._on_open_eda_dialog)
        menu_tools.addAction(act_eda)

        act_pes_2d = QtGui.QAction("🗺️ &2D Relaxed PES Scan Grid Builder (ORCA / Gaussian)...", self)
        act_pes_2d.setToolTip("Build two-dimensional coordinate scan grids (distances, angles, dihedrals) and generate ORCA and Gaussian ModRedundant decks")
        act_pes_2d.triggered.connect(self._on_open_pes_2d_builder)
        menu_tools.addAction(act_pes_2d)

        act_huckel = QtGui.QAction("🎓 &Hückel Molecular Orbital (HMO) Theory Teaching Workbench...", self)
        act_huckel.setShortcut("Ctrl+Alt+H")
        act_huckel.setToolTip("Explore Hückel secular determinants |H - E·I| = 0, Frost-Musulin circles, Hund's rule Aufbau electron filling, Coulson bond orders, and LCAO orbital lobes (Ctrl+Alt+H)")
        act_huckel.triggered.connect(self._on_open_huckel_teaching_dialog)
        menu_tools.addAction(act_huckel)

        act_nmr = QtGui.QAction("🧲 &NMR Chemical Shift Referencing & 1D Spectrum Plotter...", self)
        act_nmr.setToolTip("Convert raw QM isotropic shieldings (σ_iso) to referenced chemical shifts (δ ppm for 1H, 13C, 19F, 31P) and synthesize 1D spectra")
        act_nmr.triggered.connect(self._on_open_nmr_dialog)
        menu_tools.addAction(act_nmr)

        act_microkinetics = QtGui.QAction("📈 &Microkinetics & Catalytic Cycle Simulator (TOF, [C] vs. t)...", self)
        act_microkinetics.setToolTip("Simulate multi-step catalytic cycles and reaction networks via numerical ODE rate equation integration with TOF and TON")
        act_microkinetics.triggered.connect(self._on_open_microkinetics_dialog)
        menu_tools.addAction(act_microkinetics)

        act_ibo = QtGui.QAction("🧬 &Intrinsic Bond Orbitals (IBO / IAO) & Lewis Localization...", self)
        act_ibo.setToolTip("Localize canonical molecular orbitals into Lewis chemical bonds (σ, π), lone pairs, and core pairs via PySCF IAO/IBO")
        act_ibo.triggered.connect(self._on_open_ibo_dialog)
        menu_tools.addAction(act_ibo)

        act_ts_builder = QtGui.QAction("🔄 &Reaction Path && Transition State (TS) Guess Interpolator (IDPP / QST2 / NEB-TS)...", self)
        act_ts_builder.setShortcut("Ctrl+Shift+T")
        act_ts_builder.setToolTip("Interpolate smooth, collision-free reaction paths between Reactant and Product geometries using IDPP and generate ORCA NEB-TS or Gaussian QST2/3 inputs (Ctrl+Shift+T)")
        act_ts_builder.triggered.connect(self._on_open_reaction_path_dialog)
        menu_tools.addAction(act_ts_builder)

        act_crest_explorer = QtGui.QAction("🧬 &CREST && Conformer Ensemble Explorer (Boltzmann, RMSD, Overlays)...", self)
        act_crest_explorer.setShortcut("Ctrl+Shift+E")
        act_crest_explorer.setToolTip("Explore, rank, and analyze conformer ensembles with Boltzmann populations, Kabsch RMSDs, duplicate pruning, and batch QM generation (Ctrl+Shift+E)")
        act_crest_explorer.triggered.connect(self._on_open_crest_explorer)
        menu_tools.addAction(act_crest_explorer)

        act_ramachandran = QtGui.QAction("🗺️ &2D Dihedral Conformational Landscape (Ramachandran PES)...", self)
        act_ramachandran.setToolTip("Generate 2D conformational energy contour surfaces E(φ, ψ) across two rotatable dihedral angles")
        act_ramachandran.triggered.connect(self._on_open_ramachandran_2d)
        menu_tools.addAction(act_ramachandran)

        act_irc_sweeper = QtGui.QAction("⚛️ Interactive &IRC Reaction Sweeper & Energy Profile (MEP)...", self)
        act_irc_sweeper.setShortcut("Ctrl+Shift+I")
        act_irc_sweeper.setToolTip("Explore, scrub, and analyze Intrinsic Reaction Coordinate (IRC) Minimum Energy Pathways (MEP) with activation barriers, landmark jumps, and live 3D sync (Ctrl+Shift+I)")
        act_irc_sweeper.triggered.connect(self._on_open_irc_sweeper)
        menu_tools.addAction(act_irc_sweeper)

        act_esp_2d = QtGui.QAction("2D Electrostatic &Potential (ESP) Slicer...", self)
        act_esp_2d.setToolTip("Open Molden file and plot 2D ESP slice contour maps on XY, XZ, or YZ planes")
        act_esp_2d.triggered.connect(self._on_open_esp_2d_dialog)
        menu_tools.addAction(act_esp_2d)

        act_nci_plot = QtGui.QAction("📉 Non-Covalent Interactions (NCI / RDG Plot)...", self)
        act_nci_plot.setToolTip("Plot 2D Reduced Density Gradient s(r) vs sign(λ₂)ρ to reveal H-bonds, dispersion, and steric clash")
        act_nci_plot.triggered.connect(self._on_open_nci_plot_dialog)
        menu_tools.addAction(act_nci_plot)

        act_multiwfn = QtGui.QAction("Multiwfn &Wavefunction Suite (Bonds / Fukui / QTAIM / CM5)...", self)
        act_multiwfn.setToolTip("Run Multiwfn Mayer/Wiberg bond orders, Conceptual DFT Fukui indices, QTAIM critical points, and CM5 charges")
        act_multiwfn.triggered.connect(self._on_open_multiwfn_dialog)
        menu_tools.addAction(act_multiwfn)

        act_nbo = QtGui.QAction("🧪 &NBO Donor–Acceptor Interaction Analyzer...", self)
        act_nbo.setShortcut("Ctrl+Shift+N")
        act_nbo.setToolTip("Inspect Natural Bond Orbital (NBO) second-order perturbation donor-acceptor interactions (E(2), n→σ*, hyperconjugation) (Ctrl+Shift+N)")
        act_nbo.triggered.connect(self._on_open_nbo_dialog)
        menu_tools.addAction(act_nbo)

        act_nbo_net = QtGui.QAction("🕸️ &NBO E(2) Interaction Network Graph...", self)
        act_nbo_net.setToolTip("Visualize donor-acceptor interactions as a topological chord or bipartite graph with edge thicknesses proportional to E(2)")
        act_nbo_net.triggered.connect(self._on_open_nbo_network_dialog)
        menu_tools.addAction(act_nbo_net)

        menu_tools.addSeparator()

        act_plugins = QtGui.QAction("🐍 &Python Script & Plugin Runner...", self)
        act_plugins.setShortcut("Ctrl+Shift+Y")
        act_plugins.setToolTip("Run custom Python scripts or load user plugins from ~/.agymolden/plugins/ (Ctrl+Shift+Y)")
        act_plugins.triggered.connect(self._on_open_script_runner)
        menu_tools.addAction(act_plugins)

        act_addons = QtGui.QAction("🧩 &Add-ons & Scientific Backend Manager...", self)
        act_addons.setToolTip("Inspect and 1-click install optional packages (PySCF, RDKit, ASE, TorchANI, Spglib) directly inside the app")
        act_addons.triggered.connect(self._on_open_addons_manager)
        menu_tools.addAction(act_addons)

        # Surfaces & Electrostatic Potential Menu
        menu_surfaces = menubar.addMenu("&Surfaces")
        self.menu_surfaces = menu_surfaces

        act_esp_3d = QtGui.QAction("🌈 3D Electrostatic Potential (ESP) Mapped Density Surface", self)
        act_esp_3d.setShortcut("Ctrl+Alt+P")
        act_esp_3d.setToolTip("Evaluate and display 3D Electrostatic Potential (ESP) mapped onto total electron density isosurface (ρ = 0.002 a.u.) (Ctrl+Alt+P)")
        act_esp_3d.triggered.connect(self._on_quick_view_esp_map)
        menu_surfaces.addAction(act_esp_3d)

        act_esp_2d = QtGui.QAction("📈 2D Electrostatic Potential (ESP) Contour Map Slicer...", self)
        act_esp_2d.setToolTip("Compute and interactively slice 2D ESP contour isoline heatmaps across any molecular plane")
        act_esp_2d.triggered.connect(self._on_open_esp_2d_dialog)
        menu_surfaces.addAction(act_esp_2d)

        menu_surfaces.addSeparator()

        act_vol_studio = QtGui.QAction("🌐 Molecular Orbitals & Isosurfaces Studio (Cube / Molden)...", self)
        act_vol_studio.setShortcut("Ctrl+Alt+O")
        act_vol_studio.setToolTip("Open full GaussView / Avogadro style Volumetric Studio for MOs, ESP, ELF, NCI/RDG, ALIE, and Cube file arithmetic (Ctrl+Alt+O)")
        act_vol_studio.triggered.connect(self._on_open_volumetric_dialog)
        menu_surfaces.addAction(act_vol_studio)

        act_homo = QtGui.QAction("⚡ 1-Click View HOMO Orbital Surface", self)
        act_homo.setToolTip("Compute and display Highest Occupied Molecular Orbital (HOMO) 3D isosurface lobes")
        act_homo.triggered.connect(self._on_quick_view_homo)
        menu_surfaces.addAction(act_homo)

        act_lumo = QtGui.QAction("⚡ 1-Click View LUMO Orbital Surface", self)
        act_lumo.setToolTip("Compute and display Lowest Unoccupied Molecular Orbital (LUMO) 3D isosurface lobes")
        act_lumo.triggered.connect(self._on_quick_view_lumo)
        menu_surfaces.addAction(act_lumo)

        act_dual_fmo = QtGui.QAction("✨ 1-Click Dual Frontier Overlay (HOMO + LUMO Duality)", self)
        act_dual_fmo.setToolTip("Simultaneously render HOMO (Cyan/Amber) and LUMO (Violet/Emerald) lobes to inspect excitation pathways and donor-acceptor overlap")
        act_dual_fmo.triggered.connect(self._on_quick_view_homo_lumo_duality)
        menu_surfaces.addAction(act_dual_fmo)

        act_spin = QtGui.QAction("🌀 1-Click View Open-Shell Spin Density Lobes", self)
        act_spin.setToolTip("Compute and display alpha - beta open-shell spin density 3D isosurface lobes")
        act_spin.triggered.connect(self._on_quick_view_spin_density)
        menu_surfaces.addAction(act_spin)

        act_gallery = QtGui.QAction("📸 All Molecular Orbitals Snapshot Gallery...", self)
        act_gallery.setToolTip("Batch capture all or selected molecular orbitals into a contact sheet collage or folder of images")
        act_gallery.triggered.connect(self._on_open_orbital_snapshot_gallery)
        menu_surfaces.addAction(act_gallery)

        menu_surfaces.addSeparator()

        act_clear_surf = QtGui.QAction("✖ Clear 3D Isosurfaces / Orbitals", self)
        act_clear_surf.setShortcut("Esc")
        act_clear_surf.setToolTip("Remove active 3D isosurfaces, ESP maps, and orbital lobes from viewport (Esc)")
        act_clear_surf.triggered.connect(self._on_clear_orbital_surface)
        menu_surfaces.addAction(act_clear_surf)

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
        act_toggle_measure.setShortcut("Alt+M")
        act_toggle_measure.setToolTip("Toggle 3D floating distance, angle, and dihedral measurements over selected atoms (Alt+M)")
        def _toggle_meas(checked):
            self.viewer.show_measurements = checked
            self.viewer.update()
        act_toggle_measure.triggered.connect(_toggle_meas)
        menu_view.addAction(act_toggle_measure)

        act_pin_measure = QtGui.QAction("📌 &Pin Current Measurement to 3D View", self)
        act_pin_measure.setShortcut("Alt+P")
        act_pin_measure.setToolTip("Pins the currently measured distance/angle/dihedral permanently to the 3D model for publication figures (Alt+P)")
        def _pin_meas():
            if len(self.viewer.multi_selected) >= 2:
                sel = list(self.viewer.multi_selected)
                if not hasattr(self.viewer, "pinned_measurements"):
                    self.viewer.pinned_measurements = []
                if sel not in self.viewer.pinned_measurements:
                    self.viewer.pinned_measurements.append(sel)
                    self.statusBar().showMessage(f"Pinned measurement for atoms {sel} to 3D view.", 3500)
                else:
                    self.viewer.pinned_measurements.remove(sel)
                    self.statusBar().showMessage(f"Unpinned measurement for atoms {sel}.", 3500)
                self.viewer.update()
            else:
                self.statusBar().showMessage("Select 2, 3, or 4 atoms first to pin a measurement.", 3500)
        act_pin_measure.triggered.connect(_pin_meas)
        menu_view.addAction(act_pin_measure)

        act_clear_pinned = QtGui.QAction("🧹 Clear All &Pinned Measurements", self)
        act_clear_pinned.setToolTip("Clears all pinned distance, angle, and dihedral measurements from the 3D canvas.")
        def _clear_pinned():
            self.viewer.pinned_measurements = []
            self.viewer.update()
            self.statusBar().showMessage("Cleared all pinned measurements.", 3000)
        act_clear_pinned.triggered.connect(_clear_pinned)
        menu_view.addAction(act_clear_pinned)

        self.act_all_bond_lengths = QtGui.QAction("📐 Show All &Bond Lengths (Chemcraft Style)", self)
        self.act_all_bond_lengths.setCheckable(True)
        self.act_all_bond_lengths.setChecked(False)
        self.act_all_bond_lengths.setShortcut("Alt+Shift+B")
        self.act_all_bond_lengths.setToolTip("Toggle 3D bond length overlays (Å) on all covalent bonds (Alt+Shift+B)")
        def _toggle_all_bl(checked):
            self.viewer.show_bond_lengths = bool(checked)
            if hasattr(self, "chk_bond_lengths"):
                self.chk_bond_lengths.blockSignals(True)
                self.chk_bond_lengths.setChecked(bool(checked))
                self.chk_bond_lengths.blockSignals(False)
            self.viewer.update()
        self.act_all_bond_lengths.triggered.connect(_toggle_all_bl)
        menu_view.addAction(self.act_all_bond_lengths)

        act_pub_outlines = QtGui.QAction("🎨 Publication Silhouette &Outlines (IboView / QuteMol Style)", self)
        act_pub_outlines.setCheckable(True)
        act_pub_outlines.setChecked(False)
        act_pub_outlines.setToolTip("Render crisp ink silhouette outlines and hemisphere lighting for publication figures")
        def _toggle_pub_out(checked):
            self.viewer.show_publication_outlines = bool(checked)
            if hasattr(self, "chk_pub_outlines"):
                self.chk_pub_outlines.blockSignals(True)
                self.chk_pub_outlines.setChecked(bool(checked))
                self.chk_pub_outlines.blockSignals(False)
            self.viewer.update()
        act_pub_outlines.triggered.connect(_toggle_pub_out)
        menu_view.addAction(act_pub_outlines)

        act_toggle_nci = QtGui.QAction("🌐 Show &Non-Covalent Interactions (π-π Stacking, Halogen Bonds)", self)
        act_toggle_nci.setCheckable(True)
        act_toggle_nci.setChecked(False)
        act_toggle_nci.setShortcut("Alt+Shift+N")
        act_toggle_nci.setToolTip("Detects and renders 3D non-covalent interactions (π-π ring stacking in purple, halogen bonds in orange) (Alt+Shift+N)")
        act_toggle_nci.triggered.connect(self._on_toggle_nci)
        menu_view.addAction(act_toggle_nci)

        menu_view.addSeparator()

        # Journal Figure Presets Submenu
        menu_journal = menu_view.addMenu("🎨 &Journal Figure Presets...")
        act_pjacs = QtGui.QAction("📄 &JACS / ACS Pastel (Clean White, Ink Outlines)", self)
        act_pjacs.setShortcut("Ctrl+Alt+1")
        act_pjacs.setToolTip("Publication preset: clean white canvas, ink silhouette rim, soft diffuse shading (Ctrl+Alt+1)")
        act_pjacs.triggered.connect(lambda: self.apply_journal_preset("jacs"))
        menu_journal.addAction(act_pjacs)

        act_pnature = QtGui.QAction("✨ &Nature / Wiley Glossy (Specular Phong, White BG)", self)
        act_pnature.setShortcut("Ctrl+Alt+2")
        act_pnature.setToolTip("Publication preset: white canvas, crisp specular Phong highlights (Ctrl+Alt+2)")
        act_pnature.triggered.connect(lambda: self.apply_journal_preset("nature"))
        menu_journal.addAction(act_pnature)

        act_pangew = QtGui.QAction("💎 &Angewandte Bold (High Contrast Outlines)", self)
        act_pangew.setShortcut("Ctrl+Alt+3")
        act_pangew.setToolTip("Publication preset: white canvas with bold silhouette outlines and vivid contrast (Ctrl+Alt+3)")
        act_pangew.triggered.connect(lambda: self.apply_journal_preset("angewandte"))
        menu_journal.addAction(act_pangew)

        act_pmono = QtGui.QAction("🖨️ &Monochrome Print (Grayscale Manuscript)", self)
        act_pmono.setShortcut("Ctrl+Alt+4")
        act_pmono.setToolTip("Publication preset: white canvas, ink outlines for black & white print manuscripts (Ctrl+Alt+4)")
        act_pmono.triggered.connect(lambda: self.apply_journal_preset("monochrome"))
        menu_journal.addAction(act_pmono)

        act_pdark = QtGui.QAction("🌙 &Dark Presentation (Modern Deep Slate)", self)
        act_pdark.setShortcut("Ctrl+Alt+5")
        act_pdark.setToolTip("Presentation preset: deep slate background with vibrant emissive atom colors (Ctrl+Alt+5)")
        act_pdark.triggered.connect(lambda: self.apply_journal_preset("dark"))
        menu_journal.addAction(act_pdark)

        act_lighting_studio = QtGui.QAction("💡 &Lighting, Shaders && Figure Studio...", self)
        act_lighting_studio.setToolTip("Open interactive 3D Lighting, Shading models, and Colorblind-safe palettes studio")
        act_lighting_studio.triggered.connect(self._on_open_lighting_studio)
        menu_journal.addSeparator()
        menu_journal.addAction(act_lighting_studio)
        menu_view.addAction(act_lighting_studio)

        # Camera Projections Submenu
        menu_projections = menu_view.addMenu("📷 &Camera Projections && Alignments...")
        act_cam_xy = QtGui.QAction("Snap to &XY Plane (Top View, +Z)", self)
        act_cam_xy.setShortcut("Ctrl+1")
        act_cam_xy.setToolTip("Snap 3D camera to look directly onto the XY Cartesian plane (Ctrl+1)")
        act_cam_xy.triggered.connect(lambda: self.viewer.set_camera_plane_view("XY"))
        menu_projections.addAction(act_cam_xy)

        act_cam_xz = QtGui.QAction("Snap to &XZ Plane (Front View, +Y)", self)
        act_cam_xz.setShortcut("Ctrl+2")
        act_cam_xz.setToolTip("Snap 3D camera to look directly onto the XZ Cartesian plane (Ctrl+2)")
        act_cam_xz.triggered.connect(lambda: self.viewer.set_camera_plane_view("XZ"))
        menu_projections.addAction(act_cam_xz)

        act_cam_yz = QtGui.QAction("Snap to &YZ Plane (Side View, +X)", self)
        act_cam_yz.setShortcut("Ctrl+3")
        act_cam_yz.setToolTip("Snap 3D camera to look directly onto the YZ Cartesian plane (Ctrl+3)")
        act_cam_yz.triggered.connect(lambda: self.viewer.set_camera_plane_view("YZ"))
        menu_projections.addAction(act_cam_yz)

        menu_projections.addSeparator()

        act_cam_axes = QtGui.QAction("Align to &Principal Inertial Axes", self)
        act_cam_axes.setShortcut("Alt+I")
        act_cam_axes.setToolTip("Align camera view to look along the molecule's Principal Inertial Axes (Alt+I)")
        act_cam_axes.triggered.connect(self.viewer.align_to_principal_axes)
        menu_projections.addAction(act_cam_axes)

        act_cam_flip = QtGui.QAction("&Flip Camera 180° (Opposite Face)", self)
        act_cam_flip.setShortcut("Alt+F")
        act_cam_flip.setToolTip("Flip 3D camera 180 degrees horizontally to view opposite side (Alt+F)")
        act_cam_flip.triggered.connect(self.viewer.flip_camera_180)
        menu_projections.addAction(act_cam_flip)

        act_cam_ortho = QtGui.QAction("Toggle &Perspective / Orthographic Projection", self)
        act_cam_ortho.setShortcut("Alt+Shift+O")
        act_cam_ortho.setToolTip("Switch between perspective and orthographic (parallel) projection (Alt+Shift+O)")
        act_cam_ortho.triggered.connect(self.viewer.toggle_orthographic)
        menu_projections.addAction(act_cam_ortho)

        act_reset_cam = QtGui.QAction("&Reset Camera View", self)
        act_reset_cam.setShortcut("Space")
        act_reset_cam.triggered.connect(self.viewer.auto_center)
        menu_projections.addAction(act_reset_cam)
        menu_view.addAction(act_reset_cam)

        menu_view.addSeparator()

        # Dockable Workspace Panels Submenu
        menu_docks = menu_view.addMenu("🗂️ &Dockable Workspace Panels...")

        act_dock_job = QtGui.QAction("⚡ Calculation &Job Monitor Dock", self)
        act_dock_job.setToolTip("Dock or show the real-time QM calculation job monitor panel")
        act_dock_job.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.get_or_create_job_monitor().show())
        menu_docks.addAction(act_dock_job)

        act_dock_log = QtGui.QAction("📄 QM &Output Log Viewer Dock", self)
        act_dock_log.setToolTip("Dock or show the QM output log viewer and QC Doctor panel")
        act_dock_log.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.get_or_create_output_viewer().show())
        menu_docks.addAction(act_dock_log)

        act_dock_console = QtGui.QAction("🐍 Python &Chemistry Console Dock", self)
        act_dock_console.setToolTip("Dock or show the interactive Python scripting console (Alt+Shift+P)")
        act_dock_console.setShortcut("Alt+Shift+P")
        act_dock_console.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.get_or_create_console().show())
        menu_docks.addAction(act_dock_console)

        act_dock_ts = QtGui.QAction("🔄 Reaction Path & &TS Interpolator Dock", self)
        act_dock_ts.setToolTip("Dock or show the reaction path and Transition State (IDPP) guess generator")
        act_dock_ts.triggered.connect(self._on_open_reaction_path_dialog)
        menu_docks.addAction(act_dock_ts)

        act_dock_hpc = QtGui.QAction("🖥️ HPC &Cluster Queue Dock", self)
        act_dock_hpc.setToolTip("Dock or show the remote Slurm / HPC cluster job queue monitor")
        act_dock_hpc.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.get_or_create_hpc_queue().show())
        menu_docks.addAction(act_dock_hpc)

        act_dock_doctor = QtGui.QAction("🩺 QC &Doctor Diagnostics Dock", self)
        act_dock_doctor.setToolTip("Dock or show calculation failure diagnostics and crash auto-repair remedies")
        act_dock_doctor.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.open_qc_doctor())
        menu_docks.addAction(act_dock_doctor)

        menu_docks.addSeparator()

        act_float_all = QtGui.QAction("🖥️ Multi-Monitor: &Float All Panels", self)
        act_float_all.setToolTip("Detach all docks into independent floating windows for multi-monitor scientific workflows")
        act_float_all.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.float_all_docks())
        menu_docks.addAction(act_float_all)

        act_dock_all = QtGui.QAction("📌 &Snap All Panels into Main Window", self)
        act_dock_all.setToolTip("Dock all floating panels back inside the main window")
        act_dock_all.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.dock_all_panels())
        menu_docks.addAction(act_dock_all)

        act_reset_layout = QtGui.QAction("🔄 &Reset Workspace Layout", self)
        act_reset_layout.setToolTip("Reset dock panels to default hidden/docked state")
        act_reset_layout.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.reset_layout())
        menu_docks.addAction(act_reset_layout)

        # Help Menu
        menu_help = menubar.addMenu("&Help")

        self.act_command_palette = QtGui.QAction("🚀 Quick &Command Palette...", self)
        self.act_command_palette.setShortcuts([
            QtGui.QKeySequence("Ctrl+Shift+P"),
            QtGui.QKeySequence("Ctrl+K"),
        ])
        self.act_command_palette.setToolTip("Instant searchable command palette for all 60+ AgyMolden tools and actions (Ctrl+K / Ctrl+Shift+P)")
        self.act_command_palette.triggered.connect(self._on_open_command_palette)
        menu_help.addAction(self.act_command_palette)

        act_examples = QtGui.QAction("🧪 &Example Molecule && Calculation Gallery...", self)
        act_examples.setShortcut("Ctrl+Shift+G")
        act_examples.setToolTip("Explore bundled Molden wavefunctions, QM output logs, and Z-matrix templates (Ctrl+Shift+G)")
        act_examples.triggered.connect(self._on_open_example_gallery)
        menu_help.addAction(act_examples)

        act_tour = QtGui.QAction("🎓 &Quick Tour && Onboarding Guide...", self)
        act_tour.setToolTip("Interactive 4-step guided walkthrough of AgyMolden core features")
        act_tour.triggered.connect(self._on_open_onboarding_tour)
        menu_help.addAction(act_tour)
        menu_help.addSeparator()

        act_about = QtGui.QAction("&About AgyMolden", self)
        act_about.triggered.connect(self._on_about)
        menu_help.addAction(act_about)

        act_shortcuts = QtGui.QAction("⌨️ &Keyboard && Mouse Shortcuts", self)
        act_shortcuts.setShortcut("F1")
        act_shortcuts.setToolTip("Show searchable cheat sheet of all keyboard and mouse shortcuts (F1 / ?)")
        act_shortcuts.triggered.connect(self._on_show_shortcuts)
        menu_help.addAction(act_shortcuts)

        act_view_log = QtGui.QAction("📋 View Diagnostic &Log (agymolden.log)...", self)
        act_view_log.setShortcut("F3")
        act_view_log.setToolTip("Inspect internal session log file for troubleshooting and error reporting (F3)")
        act_view_log.triggered.connect(self._on_view_diagnostic_log)
        menu_help.addAction(act_view_log)

        act_manual = QtGui.QAction("📖 Open Full User &Manual...", self)
        act_manual.setShortcut("Shift+F1")
        act_manual.triggered.connect(self._on_open_manual)
        menu_help.addAction(act_manual)

        menu_help.addSeparator()
        act_citations = QtGui.QAction("📜 &Cite Used Tools && Methods...", self)
        act_citations.setToolTip("Generate BibTeX and computational methods text acknowledging AgyMolden, UFF, PySCF, IDPP, and Spglib")
        act_citations.triggered.connect(self._on_open_citations_dialog)
        menu_help.addAction(act_citations)

        act_desktop_icon = QtGui.QAction("🖥️ Create &Desktop / App Shortcut...", self)
        act_desktop_icon.setToolTip("Create a convenient 1-click desktop icon for launching AgyMolden on your operating system.")
        act_desktop_icon.triggered.connect(self._on_create_desktop_shortcut)
        menu_help.addAction(act_desktop_icon)

    def _create_toolbar(self):
        toolbar = QtWidgets.QToolBar("File & Editing")
        toolbar.setObjectName("MainControlsToolbar")
        toolbar.setIconSize(QtCore.QSize(18, 18))
        self.addToolBar(toolbar)

        btn_new = QtWidgets.QToolButton()
        btn_new.setText("📄 New")
        btn_new.clicked.connect(self._on_new_molecule)
        toolbar.addWidget(btn_new)

        btn_open = QtWidgets.QToolButton()
        btn_open.setText("📂 Open")
        btn_open.setToolTip("Open any supported file sorted newest-first: .out, .log, .xyz, .molden.input, .molden, .gbw, .cube, .zmat (Ctrl+O)")
        btn_open.clicked.connect(self._on_open_universal_file)
        toolbar.addWidget(btn_open)

        btn_save = QtWidgets.QToolButton()
        btn_save.setText("💾 Save ZMAT")
        btn_save.clicked.connect(self._on_save_zmat)
        toolbar.addWidget(btn_save)

        btn_xyz = QtWidgets.QToolButton()
        btn_xyz.setText("📤 XYZ")
        btn_xyz.clicked.connect(self._on_export_xyz)
        toolbar.addWidget(btn_xyz)

        btn_png = QtWidgets.QToolButton()
        btn_png.setText("📷 PNG")
        btn_png.setToolTip("Export high-resolution PNG image of the 3D view (Ctrl+P)")
        btn_png.clicked.connect(self._on_export_image)
        toolbar.addWidget(btn_png)

        btn_html = QtWidgets.QToolButton()
        btn_html.setText("🌐 HTML")
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

        self.btn_tb_tweezers = QtWidgets.QToolButton()
        self.btn_tb_tweezers.setText("🧲 Tweezers")
        self.btn_tb_tweezers.setCheckable(True)
        self.btn_tb_tweezers.setChecked(False)
        self.btn_tb_tweezers.setToolTip("🧲 Interactive Molecular Tweezers: Click to toggle. When active, simply Left-Click and drag any atom in the 3D viewport to dynamically relax connected atoms in real time!")
        self.btn_tb_tweezers.setStyleSheet("""
            QToolButton:checked {
                background-color: #0284c7;
                color: #ffffff;
                font-weight: bold;
                border: 2px solid #38bdf8;
                border-radius: 4px;
                padding: 2px 6px;
            }
            QToolButton:hover {
                border: 1px solid #0284c7;
            }
        """)
        self.btn_tb_tweezers.toggled.connect(self.act_tweezers.setChecked)
        self.act_tweezers.toggled.connect(self.btn_tb_tweezers.setChecked)
        toolbar.addWidget(self.btn_tb_tweezers)

        toolbar_qm = QtWidgets.QToolBar("Calculations & Surfaces")
        toolbar_qm.setObjectName("QMAndSurfacesToolbar")
        toolbar_qm.setIconSize(QtCore.QSize(18, 18))
        self.addToolBar(toolbar_qm)

        btn_quick_orca = QtWidgets.QToolButton()
        btn_quick_orca.setText("🚀 Run ORCA")
        btn_quick_orca.setToolTip("Quick ORCA calculation studio (Methods, Basis sets, ESP, Opt, Freq)")
        btn_quick_orca.setStyleSheet("color: #2563eb; font-weight: bold;")
        btn_quick_orca.clicked.connect(self._on_quick_run_orca)
        toolbar_qm.addWidget(btn_quick_orca)

        # Pre-flight Calculation Health Status Badge
        self.btn_preflight_badge = QtWidgets.QPushButton("🟢 Ready")
        self.btn_preflight_badge.setFixedWidth(78)
        self.btn_preflight_badge.setStyleSheet(
            "background-color: #15803d; color: #ffffff; font-size: 11px; font-weight: bold; border-radius: 3px; padding: 2px 4px;"
        )
        self.btn_preflight_badge.setToolTip("Pre-flight status: Checks for steric clashes and electron parity before launching calculations.")
        self.btn_preflight_badge.clicked.connect(self._on_open_preflight_check_dialog)
        toolbar_qm.addWidget(self.btn_preflight_badge)

        btn_qm = QtWidgets.QToolButton()
        btn_qm.setText("⚛️ QM Setup")
        btn_qm.setToolTip("Generate calculation input for Gaussian, ORCA, PySCF, openLOWDIN, xTB (Ctrl+G)")
        btn_qm.clicked.connect(self._on_open_qm_dialog)
        toolbar_qm.addWidget(btn_qm)
        self.btn_toolbar_qm_setup = btn_qm

        btn_brorsen = QtWidgets.QToolButton()
        btn_brorsen.setText("✨ newBRORSEN")
        btn_brorsen.setToolTip("Generate input files for newBRORSEN (Positrons, NEO, CCSD(T)) (Ctrl+B)")
        btn_brorsen.clicked.connect(self._on_open_brorsen_dialog)
        toolbar_qm.addWidget(btn_brorsen)
        self.btn_toolbar_brorsen = btn_brorsen

        toolbar_qm.addSeparator()

        btn_esp = QtWidgets.QToolButton()
        btn_esp.setText("🌈 ESP Map")
        btn_esp.setToolTip("1-Click Electrostatic Potential (ESP) mapped onto Electron Density (Ctrl+Alt+P)")
        btn_esp.setStyleSheet("color: #db2777; font-weight: bold;")
        btn_esp.clicked.connect(self._on_quick_view_esp_map)
        toolbar_qm.addWidget(btn_esp)
        self.btn_toolbar_esp = btn_esp

        btn_orbitals = QtWidgets.QToolButton()
        btn_orbitals.setText("🌐 Orbitals/Grid")
        btn_orbitals.setToolTip("Open Volumetric Cube or Molden file to view Molecular Orbitals / ESP isosurfaces (Ctrl+Alt+O)")
        btn_orbitals.clicked.connect(self._on_open_volumetric_dialog)
        toolbar_qm.addWidget(btn_orbitals)

        btn_view_log_tb = QtWidgets.QToolButton()
        btn_view_log_tb.setText("📄 View Log")
        btn_view_log_tb.setToolTip("Open Calculation Output Log & Results Viewer (F2)")
        btn_view_log_tb.clicked.connect(self._on_view_qm_output_file)
        toolbar_qm.addWidget(btn_view_log_tb)

        btn_vib = QtWidgets.QToolButton()
        btn_vib.setText("🎵 Vibrations")
        btn_vib.setToolTip("Compute vibrational frequencies and simulate normal modes")
        btn_vib.clicked.connect(self._on_calculate_vibrations)
        toolbar_qm.addWidget(btn_vib)

        self.btn_toolbar_sym = QtWidgets.QToolButton()
        self.btn_toolbar_sym.setText("🔮 Symmetry")
        self.btn_toolbar_sym.setToolTip("Detect Schoenflies point group symmetry")
        self.btn_toolbar_sym.clicked.connect(lambda _checked=False: self._on_detect_symmetry(switch_tab=True))
        toolbar_qm.addWidget(self.btn_toolbar_sym)

        toolbar_qm.addSeparator()

        btn_cmd_palette = QtWidgets.QToolButton()
        btn_cmd_palette.setText("⚡ Command Palette")
        btn_cmd_palette.setToolTip("Search all tools, calculation engines, and dialogs (Ctrl+K)")
        btn_cmd_palette.setStyleSheet("background-color: #1e293b; color: #38bdf8; font-weight: bold; padding: 2px 8px; border: 1px solid #334155; border-radius: 4px;")
        btn_cmd_palette.clicked.connect(self._on_open_command_palette)
        toolbar_qm.addWidget(btn_cmd_palette)

        toolbar_qm.addSeparator()

        # Workspace Mode Switcher (Action 2: Modern Build / Calculate / Analyze workflows)
        lbl_mode = QtWidgets.QLabel("Mode:")
        lbl_mode.setStyleSheet("font-size: 11px; font-weight: bold; color: #94a3b8; margin-left: 4px;")
        toolbar_qm.addWidget(lbl_mode)

        self.combo_workspace_mode = QtWidgets.QComboBox()
        self.combo_workspace_mode.addItem("🎓 Student / Essentials", "student")
        self.combo_workspace_mode.addItem("🔨 Build & Edit (Full)", "build")
        self.combo_workspace_mode.addItem("⚛️ Calculate (QM)", "calculate")
        self.combo_workspace_mode.addItem("📊 Analyze Results", "analyze")
        self.combo_workspace_mode.setCurrentIndex(1)
        self.combo_workspace_mode.setToolTip(
            "Workspace Perspective Switcher:\n"
            "• Student / Essentials: Decluttered workspace with instant Semiempirical QM (GFN0-xTB) for newcomers.\n"
            "• Build & Edit (Full): Full Z-matrix, 3D builder, and geometry editing suite.\n"
            "• Calculate (QM): Focuses on ORCA, Gaussian, multiwfn, PBC, and pre-flight checks.\n"
            "• Analyze Results: Focuses on Vibrations, Spectra (IR/UV/NMR/DOS), and Molecular Orbitals."
        )
        self.combo_workspace_mode.setStyleSheet(
            "QComboBox { background-color: #0f172a; color: #38bdf8; font-weight: bold; border: 1px solid #334155; border-radius: 4px; padding: 2px 6px; }"
            "QComboBox::drop-down { border: none; }"
        )
        self.combo_workspace_mode.currentIndexChanged.connect(self._on_workspace_mode_changed)
        toolbar_qm.addWidget(self.combo_workspace_mode)

        # Permanent Chemical Summary Label on Status Bar
        self.lbl_chem_summary = QtWidgets.QLabel("Empty Molecule")
        self.lbl_chem_summary.setStyleSheet("color: #38bdf8; font-weight: bold; padding: 0 10px;")
        self.statusBar().addPermanentWidget(self.lbl_chem_summary)

        # Permanent Active Engine / Python Environment Badge on Status Bar
        self.lbl_active_env = QtWidgets.QLabel("QM: Ready")
        self.lbl_active_env.setToolTip("Active Quantum Chemistry Engine & Python Environment. Click to configure routes.")
        self.lbl_active_env.setStyleSheet(
            "background-color: #0f172a; color: #10b981; border: 1px solid #334155; "
            "border-radius: 3px; padding: 1px 6px; font-weight: bold; font-size: 11px; margin-right: 6px;"
        )
        self.lbl_active_env.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.lbl_active_env.mousePressEvent = lambda ev: self._on_configure_qm_engines()
        self.statusBar().addPermanentWidget(self.lbl_active_env)

    def _collect_command_palette_actions(self) -> List[Tuple[str, str, QtGui.QAction]]:
        """Collect all menu actions as (clean_label, shortcut_str, QAction) tuples for the Command Palette."""
        items: List[Tuple[str, str, QtGui.QAction]] = []
        seen = set()
        for act in self.findChildren(QtGui.QAction):
            if act.isSeparator() or act.menu() is not None:
                continue
            raw_txt = act.text() or ""
            clean_txt = raw_txt.replace("&&", "\x00").replace("&", "").replace("\x00", "&").strip()
            if not clean_txt or clean_txt.startswith("(No recent") or clean_txt in seen:
                continue
            seen.add(clean_txt)
            sc = act.shortcut().toString() if not act.shortcut().isEmpty() else ""
            items.append((clean_txt, sc, act))
        return items

    def _on_open_command_palette(self):
        """Open a fast, searchable Quick Command Palette (Ctrl+Shift+P) to trigger any AgyMolden tool or action."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("🚀 Quick Command Palette (Ctrl+Shift+P)")
        dlg.resize(560, 400)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        search_edit = QtWidgets.QLineEdit()
        search_edit.setPlaceholderText("🔍 Type a command (e.g. 'orca', 'symmetry', 'smiles', 'esp', 'outline', 'copy')...")
        layout.addWidget(search_edit)

        list_widget = QtWidgets.QListWidget()
        list_widget.setAlternatingRowColors(True)
        layout.addWidget(list_widget, stretch=1)

        all_items = self._collect_command_palette_actions()

        def _populate(query: str = ""):
            q_tokens = query.strip().lower().split()
            list_widget.clear()
            for label, sc, act in all_items:
                tip = (act.toolTip() or "").lower()
                haystack = f"{label.lower()} {sc.lower()} {tip}"
                if all(tok in haystack for tok in q_tokens):
                    disp = f"{label}   [{sc}]" if sc else label
                    item = QtWidgets.QListWidgetItem(disp)
                    if act.toolTip():
                        item.setToolTip(act.toolTip())
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, act)
                    if not act.isEnabled():
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
                    list_widget.addItem(item)
            if list_widget.count() > 0:
                list_widget.setCurrentRow(0)

        def _trigger_selected():
            cur = list_widget.currentItem()
            if cur is None:
                return
            act = cur.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(act, QtGui.QAction) and act.isEnabled():
                dlg.accept()
                QtCore.QTimer.singleShot(0, act.trigger)

        class SearchKeyFilter(QtCore.QObject):
            def eventFilter(self, obj, ev):
                if ev.type() == QtCore.QEvent.Type.KeyPress:
                    if ev.key() == QtCore.Qt.Key.Key_Down:
                        c_row = list_widget.currentRow()
                        if c_row < list_widget.count() - 1:
                            list_widget.setCurrentRow(c_row + 1)
                        return True
                    elif ev.key() == QtCore.Qt.Key.Key_Up:
                        c_row = list_widget.currentRow()
                        if c_row > 0:
                            list_widget.setCurrentRow(c_row - 1)
                        return True
                    elif ev.key() == QtCore.Qt.Key.Key_Escape:
                        dlg.reject()
                        return True
                return super().eventFilter(obj, ev)

        search_key_filter = SearchKeyFilter(search_edit)
        search_edit.installEventFilter(search_key_filter)

        search_edit.textChanged.connect(_populate)
        search_edit.returnPressed.connect(_trigger_selected)
        list_widget.itemActivated.connect(lambda _item: _trigger_selected())

        _populate()
        search_edit.setFocus()
        dlg.exec()

    def _on_workspace_mode_changed(self, index: int):
        """Switches active workspace perspective (Student/Essentials, Build, Calculate, Analyze) for streamlined workflows."""
        mode_data = self.combo_workspace_mode.currentData()
        if not hasattr(self, "tabs"):
            return

        is_student = (mode_data == "student")
        for adv_btn_attr in ("btn_toolbar_brorsen", "btn_toolbar_qm_setup"):
            btn = getattr(self, adv_btn_attr, None)
            if btn is not None:
                btn.setVisible(not is_student)

        if is_student:
            # Automatically select Semiempirical QM (GFN0-xTB) so students get instant orbitals, ESP & frequencies
            if hasattr(self, "combo_opt_method"):
                for i in range(self.combo_opt_method.count()):
                    if self.combo_opt_method.itemData(i) == "semiempirical":
                        self.combo_opt_method.setCurrentIndex(i)
                        break
            builder_idx = getattr(self, "tab_builder", None)
            if builder_idx is not None:
                idx = self.tabs.indexOf(builder_idx)
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)
            else:
                self.tabs.setCurrentIndex(0)
            self.statusBar().showMessage(
                "🎓 Student / Essentials Mode active: Instant Semiempirical QM enabled (1-Click Optimize, HOMO/LUMO, ESP & IR)!",
                5000,
            )

        elif mode_data == "build":
            # Show builder tab or Z-matrix
            builder_idx = getattr(self, "tab_builder", None)
            if builder_idx is not None:
                idx = self.tabs.indexOf(builder_idx)
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)
            else:
                self.tabs.setCurrentIndex(0)
            self.statusBar().showMessage("Workspace switched to: Build & Edit (Z-Matrix, 3D Builder)", 3500)

        elif mode_data == "calculate":
            # Focus on calculation or symmetry
            sym_tab = getattr(self, "tab_symmetry", None)
            if sym_tab is not None:
                idx = self.tabs.indexOf(sym_tab)
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)
            self.statusBar().showMessage("Workspace switched to: Calculate (QM Setup, ORCA, Pre-flight)", 3500)

        elif mode_data == "analyze":
            # Focus on vibrations or MOs
            vib_tab = getattr(self, "tab_vibrations", None)
            mo_tab = getattr(self, "tab_orbitals", None)
            if vib_tab is not None:
                idx = self.tabs.indexOf(vib_tab)
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)
            elif mo_tab is not None:
                idx = self.tabs.indexOf(mo_tab)
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)
            self.statusBar().showMessage("Workspace switched to: Analyze (Vibrations, Spectra, Orbitals)", 3500)


    # Core logic
