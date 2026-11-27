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
from .mw_menus import MenusMixin


class _FlowLayout(QtWidgets.QLayout):
    """Responsive wrapping layout so toolbar buttons wrap onto new rows instead of hiding off-screen."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, margin: int = 0, h_spacing: int = 6, v_spacing: int = 4):
        super().__init__(parent)
        self._items: List[QtWidgets.QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item: QtWidgets.QLayoutItem):
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QtWidgets.QLayoutItem]:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> Optional[QtWidgets.QLayoutItem]:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> QtCore.Qt.Orientation:
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QtCore.QRect(0, 0, max(width, 480), 0), test_only=True)

    def setGeometry(self, rect: QtCore.QRect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        size = QtCore.QSize(0, 0)
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect: QtCore.QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = eff.x()
        y = eff.y()
        line_height = 0
        for item in self._items:
            wid = item.widget()
            if wid is not None and wid.isHidden():
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h_spacing
            if next_x - self._h_spacing > eff.right() and line_height > 0:
                x = eff.x()
                y = y + line_height + self._v_spacing
                next_x = x + hint.width() + self._h_spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + m.bottom()


class _FlowWidget(QtWidgets.QWidget):
    """Container for _FlowLayout whose minimumSizeHint stays 1-row tall so MainWindow shrinks cleanly."""

    def minimumSizeHint(self) -> QtCore.QSize:
        lay = self.layout()
        if lay is not None:
            return lay.minimumSize()
        return super().minimumSizeHint()


class UIMixin(UIInspectorTabsMixin, MenusMixin):
    def _init_ui(self):
        # Central splitter: Left is 3D Viewport, Right is Control Panel & ZMAT Table
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.main_splitter = splitter
        main_layout.addWidget(splitter)

        # Left Widget: Multi-Molecule Tab Bar + Welcome Banner + 3D Viewport + View controls bar
        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(3)

        mol_tab_row = QtWidgets.QHBoxLayout()
        mol_tab_row.setContentsMargins(0, 0, 0, 0)
        mol_tab_row.setSpacing(2)
        self.mol_tab_bar = QtWidgets.QTabBar()
        self.mol_tab_bar.setTabsClosable(True)
        self.mol_tab_bar.setExpanding(False)
        self.mol_tab_bar.setDocumentMode(True)
        self.mol_tab_bar.addTab("Molecule 1")
        self._mol_tabs = [{}]
        self._active_mol_tab_idx = 0
        self.mol_tab_bar.currentChanged.connect(self._on_switch_mol_tab)
        self.mol_tab_bar.tabCloseRequested.connect(self._on_close_mol_tab)
        mol_tab_row.addWidget(self.mol_tab_bar, stretch=1)

        self.btn_new_mol_tab = QtWidgets.QToolButton()
        self.btn_new_mol_tab.setText("+")
        self.btn_new_mol_tab.setToolTip("Open a new molecule workspace tab")
        self.btn_new_mol_tab.setStyleSheet("font-weight: bold; padding: 2px 8px;")
        self.btn_new_mol_tab.clicked.connect(self._on_new_mol_tab)
        mol_tab_row.addWidget(self.btn_new_mol_tab)
        left_layout.addLayout(mol_tab_row)

        self.welcome_banner = QtWidgets.QFrame()
        self.welcome_banner.setStyleSheet(
            "QFrame { background-color: #eff6ff; border: 1px solid #93c5fd; border-radius: 6px; padding: 4px 8px; }"
            "QPushButton { background-color: #2563eb; color: #ffffff; font-weight: 600; padding: 4px 10px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1d4ed8; }"
        )
        wb_layout = QtWidgets.QHBoxLayout(self.welcome_banner)
        wb_layout.setContentsMargins(6, 4, 6, 4)
        wb_layout.setSpacing(6)
        lbl_wb = QtWidgets.QLabel("<b>Get Started:</b>")
        self.lbl_welcome_banner = lbl_wb
        lbl_wb.setStyleSheet("color: #1e293b; border: none;")
        wb_layout.addWidget(lbl_wb)
        for w_lbl, w_slot in (
            ("📂 Open File...", self._on_open_universal_file),
            ("✏️ 2D → 3D Sketch", lambda: self._on_toggle_2d_3d_split(True)),
            ("🧪 Example Gallery", self._on_open_example_gallery),
            ("🎓 Quick Tour", self._on_open_onboarding_tour),
        ):
            wb_btn = QtWidgets.QPushButton(w_lbl)
            wb_btn.setMinimumHeight(28)
            wb_btn.clicked.connect(w_slot)
            wb_layout.addWidget(wb_btn)
        wb_layout.addStretch()
        self.btn_dismiss_welcome = QtWidgets.QToolButton()
        self.btn_dismiss_welcome.setText("×")
        self.btn_dismiss_welcome.setMinimumSize(26, 26)
        self.btn_dismiss_welcome.setToolTip("Dismiss Get Started banner to maximize 3D canvas space")
        self.btn_dismiss_welcome.setStyleSheet("color: #475569; font-weight: bold; font-size: 14px; border: none;")
        self.btn_dismiss_welcome.clicked.connect(self.welcome_banner.hide)
        wb_layout.addWidget(self.btn_dismiss_welcome)
        left_layout.addWidget(self.welcome_banner)

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

        # Viewer quick controls ribbon (responsive wrapping layout so buttons never hide when right panel is widened)
        ribbon_widget = _FlowWidget()
        view_ctrls = _FlowLayout(ribbon_widget, margin=0, h_spacing=6, v_spacing=4)

        self.btn_mode_toggle = QtWidgets.QPushButton("✏️ Build")
        self.btn_mode_toggle.setCheckable(True)
        self.btn_mode_toggle.setChecked(False)
        self.btn_mode_toggle.setMinimumHeight(28)
        self.btn_mode_toggle.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 4px 10px;")
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
        self.btn_ribbon_tweezers.setMinimumHeight(28)
        self.btn_ribbon_tweezers.setStyleSheet("""
            QPushButton {
                background-color: #dbeafe;
                color: #1d4ed8;
                font-weight: bold;
                padding: 4px 10px;
                border: 1px solid #93c5fd;
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

        self.btn_auto_relax = QtWidgets.QPushButton("⚡ Auto-Relax")
        self.btn_auto_relax.setCheckable(True)
        self.btn_auto_relax.setChecked(False)
        self.btn_auto_relax.setMinimumHeight(28)
        self.btn_auto_relax.setStyleSheet("""
            QPushButton {
                background-color: #ede9fe;
                color: #6d28d9;
                font-weight: bold;
                padding: 4px 10px;
                border: 1px solid #c4b5fd;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #7c3aed;
                color: #ffffff;
                border: 1.5px solid #a78bfa;
            }
        """)
        self.btn_auto_relax.setToolTip("⚡ Auto-Relax (UFF): When active, drawing new atoms or bonds automatically relaxes the 3D geometry with UFF.")
        self.btn_auto_relax.toggled.connect(
            lambda checked: self.statusBar().showMessage(
                f"⚡ Auto-Relax (UFF): {'ON — new atoms/bonds relax automatically' if checked else 'OFF'}", 3000
            ) if self.statusBar() else None
        )
        view_ctrls.addWidget(self.btn_auto_relax)

        self.btn_2d_3d_split = QtWidgets.QPushButton("✏️ 2D | 3D")
        self.btn_2d_3d_split.setCheckable(True)
        self.btn_2d_3d_split.setChecked(False)
        self.btn_2d_3d_split.setMinimumHeight(28)
        self.btn_2d_3d_split.setStyleSheet("""
            QPushButton {
                background-color: #e0f2fe;
                color: #0369a1;
                font-weight: bold;
                padding: 4px 10px;
                border: 1px solid #7dd3fc;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #0284c7;
                color: #ffffff;
                border: 1.5px solid #38bdf8;
            }
        """)
        self.btn_2d_3d_split.setToolTip("Toggle MolView-style 2-pane 2D Sketcher + live 3D Viewport")
        self.btn_2d_3d_split.toggled.connect(self._on_toggle_2d_3d_split)
        view_ctrls.addWidget(self.btn_2d_3d_split)

        self.btn_toggle_right_pane = QtWidgets.QPushButton("◧ Z-Mat")
        self.btn_toggle_right_pane.setCheckable(True)
        self.btn_toggle_right_pane.setChecked(True)
        self.btn_toggle_right_pane.setMinimumHeight(28)
        self.btn_toggle_right_pane.setShortcut("F9")
        self.btn_toggle_right_pane.setStyleSheet("""
            QPushButton {
                background-color: #ccfbf1;
                color: #0f766e;
                font-weight: bold;
                padding: 4px 10px;
                border: 1px solid #5eead4;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #0d9488;
                color: #ffffff;
                border: 1.5px solid #2dd4bf;
            }
        """)
        self.btn_toggle_right_pane.setToolTip("Show Z-Matrix / XYZ / Orbitals panel (switches back from 2D Sketcher, or click again to hide/show panel) (F9)")
        self.btn_toggle_right_pane.toggled.connect(self._on_toggle_right_pane)
        view_ctrls.addWidget(self.btn_toggle_right_pane)

        # Element brush selector kept synced for shortcuts/menus; hidden from ribbon because draw_palette_bar exposes 1-click pills
        self.combo_brush_element = QtWidgets.QComboBox()
        self.combo_brush_element.setFixedWidth(72)
        self.combo_brush_element.setToolTip("Select active drawing & mutation element brush")
        for sym in ("C", "H", "O", "N", "P", "S", "F", "Cl"):
            self.combo_brush_element.addItem(f"🧪 {sym}", sym)
        self.combo_brush_element.currentIndexChanged.connect(
            lambda _idx: self._on_select_brush_element(self.combo_brush_element.currentData() or "C")
        )
        self.combo_brush_element.setVisible(False)

        # Quick Stamp Menu (Benzene, Cyclohexane, Methyl, etc.)
        self.btn_quick_stamp = QtWidgets.QToolButton()
        self.btn_quick_stamp.setText("📌 Stamp ▾")
        self.btn_quick_stamp.setMinimumHeight(28)
        self.btn_quick_stamp.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_quick_stamp.setToolTip("GaussView / Avogadro style: Stamp functional groups or rings onto selected atom in 1 click")
        self.btn_quick_stamp.setStyleSheet("background-color: #0d9488; color: white; font-weight: 600; padding: 4px 8px;")
        menu_stamp = QtWidgets.QMenu(self)
        for st_name in ["Phenyl (-C6H5)", "Methyl (-CH3)", "Hydroxyl (-OH)", "Amino (-NH2)", "Carbonyl (=O)", "Nitro (-NO2)", "Carboxyl (-COOH)", "Water (H2O)"]:
            act = menu_stamp.addAction(st_name)
            act.triggered.connect(lambda _c=False, n=st_name: self._on_quick_stamp_fragment(n))
        self.btn_quick_stamp.setMenu(menu_stamp)
        view_ctrls.addWidget(self.btn_quick_stamp)

        btn_h_ribbon = QtWidgets.QToolButton()
        btn_h_ribbon.setText("💧 Add H")
        btn_h_ribbon.setMinimumHeight(28)
        btn_h_ribbon.setToolTip("Complete valence with hydrogens (Ctrl+H)")
        btn_h_ribbon.setStyleSheet("background-color: #0284c7; color: white; font-weight: 600; padding: 4px 8px;")
        btn_h_ribbon.clicked.connect(self._on_add_hydrogens)
        view_ctrls.addWidget(btn_h_ribbon)

        self.btn_clean_selection = QtWidgets.QPushButton("⚡ Clean")
        self.btn_clean_selection.setMinimumHeight(28)
        self.btn_clean_selection.setToolTip(
            "Clean / pre-optimize selected atoms with UFF (freezes unselected atoms), or optimize whole molecule if none selected"
        )
        self.btn_clean_selection.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 4px 10px;")
        self.btn_clean_selection.clicked.connect(self._on_clean_selection)
        view_ctrls.addWidget(self.btn_clean_selection)

        self.chk_auto_h = QtWidgets.QCheckBox("Adjust H")
        self.chk_auto_h.setChecked(True)
        self.chk_auto_h.setStyleSheet("font-weight: 600; padding: 2px 4px;")
        self.chk_auto_h.setToolTip("Activate/deactivate automatic hydrogen adjustment for 3D editing and 2D → 3D conversion")
        self.chk_auto_h.toggled.connect(self._on_auto_h_toggled)
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
        self.btn_reset_view.setMinimumHeight(28)
        self.btn_reset_view.setToolTip("Center and fit molecule in 3D viewport (Shortcut: Space)")
        self.btn_reset_view.clicked.connect(self.viewer.auto_center)
        view_ctrls.addWidget(self.btn_reset_view)

        btn_planes = QtWidgets.QToolButton()
        btn_planes.setText("📷 Plane ▾")
        btn_planes.setMinimumHeight(28)
        btn_planes.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        btn_planes.setToolTip("Snap 3D camera to XY, XZ, or YZ Cartesian plane")
        menu_planes = QtWidgets.QMenu(self)
        for pl_lbl in ("XY", "XZ", "YZ"):
            act_pl = menu_planes.addAction(f"Snap to {pl_lbl} Plane")
            act_pl.triggered.connect(lambda _c=False, p=pl_lbl: self.viewer.set_camera_plane_view(p))
        btn_planes.setMenu(menu_planes)
        view_ctrls.addWidget(btn_planes)

        btn_axes = QtWidgets.QPushButton("Axes")
        btn_axes.setMinimumWidth(48)
        btn_axes.setMinimumHeight(28)
        btn_axes.setToolTip("Align camera view to look along the molecule's Principal Inertial Axes (Chemcraft / VMD)")
        btn_axes.clicked.connect(lambda: self.viewer.align_to_principal_axes())
        view_ctrls.addWidget(btn_axes)

        self.btn_orient_sym = QtWidgets.QPushButton("📐 Orient")
        self.btn_orient_sym.setMinimumHeight(28)
        self.btn_orient_sym.setStyleSheet("font-weight: 600; padding: 4px 8px; background-color: #6366f1; color: #ffffff;")
        self.btn_orient_sym.setToolTip(
            "Universal Symmetry Orientation (Alt+S):\n"
            "Align molecule coordinates to standard Schoenflies/IUPAC convention for ANY point group\n"
            "(e.g. C2v water in XZ plane with C2 along Z, D2h, C3v, etc.) and center COM at (0,0,0)."
        )
        self.btn_orient_sym.clicked.connect(self._on_align_symmetry)
        view_ctrls.addWidget(self.btn_orient_sym)

        self.btn_camera_proj = QtWidgets.QPushButton("Persp")
        self.btn_camera_proj.setMinimumWidth(54)
        self.btn_camera_proj.setMinimumHeight(28)
        self.btn_camera_proj.setToolTip("Toggle between Perspective and Orthographic parallel projection (VMD / GaussView style) (F5)")
        self.btn_camera_proj.clicked.connect(self._on_toggle_camera_projection)
        view_ctrls.addWidget(self.btn_camera_proj)

        # Compact popup menu for display styles, theme, and 3D overlays so the main bar never overflows
        self.btn_display_menu = QtWidgets.QToolButton()
        self.btn_display_menu.setText("👁️ Display ▾")
        self.btn_display_menu.setMinimumHeight(28)
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

        cam_row = QtWidgets.QHBoxLayout()
        btn_copy_cam = QtWidgets.QPushButton("📷 Copy View")
        btn_copy_cam.setToolTip("Copy current 3D camera angle & zoom to clipboard for matching publication figures")
        btn_copy_cam.clicked.connect(self._on_copy_camera_view)
        cam_row.addWidget(btn_copy_cam)
        btn_paste_cam = QtWidgets.QPushButton("📋 Paste View")
        btn_paste_cam.setToolTip("Apply copied 3D camera angle & zoom from clipboard")
        btn_paste_cam.clicked.connect(self._on_paste_camera_view)
        cam_row.addWidget(btn_paste_cam)
        display_layout.addLayout(cam_row)

        act_display_panel = QtWidgets.QWidgetAction(self)
        act_display_panel.setDefaultWidget(display_panel)
        self.menu_display_overlays.addAction(act_display_panel)
        self.btn_display_menu.setMenu(self.menu_display_overlays)
        view_ctrls.addWidget(self.btn_display_menu)

        ribbon_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        left_layout.addWidget(ribbon_widget)

        # Compact 1-click Element & Bond-Order Draw Strip (Avogadro style — draw without Z-Matrix)
        self.draw_palette_bar = _FlowWidget()
        draw_flow = _FlowLayout(self.draw_palette_bar, margin=0, h_spacing=4, v_spacing=3)
        self.btn_elem_pills: dict[str, QtWidgets.QPushButton] = {}
        for sym in ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br"):
            b = QtWidgets.QPushButton(sym)
            b.setCheckable(True)
            b.setChecked(sym == "C")
            b.setMinimumHeight(28)
            b.setToolTip(f"Select {sym} drawing brush (1 click adds {sym} on empty canvas or starts bond)")
            b.setStyleSheet(
                "QPushButton { padding: 3px 8px; min-width: 28px; font-weight: bold; font-size: 11px; border-radius: 3px; }"
                "QPushButton:checked { background-color: #0284c7; color: #ffffff; }"
            )
            b.clicked.connect(lambda _c=False, s=sym: self._on_quick_element_pill(s))
            self.btn_elem_pills[sym] = b
            draw_flow.addWidget(b)

        self.btn_elem_periodic = QtWidgets.QPushButton("…")
        self.btn_elem_periodic.setMinimumHeight(28)
        self.btn_elem_periodic.setToolTip("Open Periodic Table to pick any element")
        self.btn_elem_periodic.setStyleSheet("padding: 3px 8px; min-width: 28px; font-weight: bold; font-size: 11px; border-radius: 3px;")
        self.btn_elem_periodic.clicked.connect(self._on_add_line)
        draw_flow.addWidget(self.btn_elem_periodic)

        self.active_draw_bond_order = 1
        self.btn_bo_1 = QtWidgets.QPushButton("─ 1")
        self.btn_bo_2 = QtWidgets.QPushButton("═ 2")
        self.btn_bo_3 = QtWidgets.QPushButton("≡ 3")
        for order, btn, tip in (
            (1, self.btn_bo_1, "Single bond order (or apply to selected / diatomic bond)"),
            (2, self.btn_bo_2, "Double bond order (or apply to selected / diatomic bond)"),
            (3, self.btn_bo_3, "Triple bond order (or apply to selected / diatomic bond)"),
        ):
            btn.setCheckable(True)
            btn.setChecked(order == 1)
            btn.setMinimumHeight(28)
            btn.setToolTip(tip)
            btn.setStyleSheet(
                "QPushButton { padding: 3px 8px; font-weight: bold; font-size: 11px; border-radius: 3px; }"
                "QPushButton:checked { background-color: #0d9488; color: #ffffff; }"
            )
            btn.clicked.connect(lambda _c=False, o=order: self._on_quick_bond_order(o))
            draw_flow.addWidget(btn)

        self.draw_palette_bar.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        left_layout.addWidget(self.draw_palette_bar)

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

        self.btn_qm_sum_agemacs = QtWidgets.QPushButton("🚀 Agemacs")
        self.btn_qm_sum_agemacs.setStyleSheet("background-color: #0f766e; color: white; font-weight: bold; padding: 2px 8px; font-size: 11px;")
        self.btn_qm_sum_agemacs.setToolTip("Open active QM output log or Molden file in Agemacs")
        self.btn_qm_sum_agemacs.clicked.connect(self._on_send_qm_output_to_agemacs)
        qm_sum_layout.addWidget(self.btn_qm_sum_agemacs)

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
        self.slider_trajectory.setMinimumWidth(90)
        self.slider_trajectory.valueChanged.connect(self._on_trajectory_slider_changed)
        row1_layout.addWidget(self.slider_trajectory, stretch=1)

        self.lbl_trajectory_step = QtWidgets.QLabel("Step 0/0")
        self.lbl_trajectory_step.setStyleSheet("font-weight: bold; color: #38bdf8; min-width: 65px;")
        row1_layout.addWidget(self.lbl_trajectory_step)

        traj_box_layout.addLayout(row1_layout)

        # Interactive Mini Energy Trajectory Profile Curve (Sparkline)
        from .widget_sparkline import SparklinePlot
        self.sparkline_trajectory = SparklinePlot(self)
        self.sparkline_trajectory.step_clicked.connect(lambda s: self.slider_trajectory.setValue(s))
        traj_box_layout.addWidget(self.sparkline_trajectory)

        # Row 2: Step Energy, Playback Options & Analysis Tools (wrapping flow layout so all buttons stay visible)
        row2_container = _FlowWidget()
        row2_layout = _FlowLayout(row2_container, margin=0, h_spacing=6, v_spacing=4)

        self.combo_traj_mode = QtWidgets.QComboBox()
        self.combo_traj_mode.addItem("Loop 🔁", "loop")
        self.combo_traj_mode.addItem("Bounce ⮂", "bounce")
        self.combo_traj_mode.addItem("Once ➔", "once")
        self.combo_traj_mode.setToolTip("Trajectory animation repeat behavior (Bounce is ideal for IRC pathways)")
        self.combo_traj_mode.setFixedWidth(92)
        row2_layout.addWidget(self.combo_traj_mode)

        self.combo_traj_speed = QtWidgets.QComboBox()
        self.combo_traj_speed.addItem("0.5x", 250)
        self.combo_traj_speed.addItem("1.0x", 120)
        self.combo_traj_speed.addItem("2.0x", 60)
        self.combo_traj_speed.addItem("4.0x", 30)
        self.combo_traj_speed.setCurrentIndex(1)
        self.combo_traj_speed.setToolTip("Animation playback speed")
        self.combo_traj_speed.setFixedWidth(64)
        self.combo_traj_speed.currentIndexChanged.connect(self._on_traj_speed_changed)
        row2_layout.addWidget(self.combo_traj_speed)

        self.lbl_trajectory_energy = QtWidgets.QLabel("")
        self.lbl_trajectory_energy.setStyleSheet("color: #a7f3d0; font-family: monospace; font-size: 11px; white-space: nowrap;")
        row2_layout.addWidget(self.lbl_trajectory_energy)

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
        self.btn_view_loaded_log.setStyleSheet("background-color: #0891b2; color: #ffffff; padding: 3px 10px; font-weight: bold; font-size: 11px; white-space: nowrap;")
        self.btn_view_loaded_log.setToolTip("Open full calculation output log with search and section indexing (F2)")
        self.btn_view_loaded_log.clicked.connect(self._on_view_qm_output_file)
        row2_layout.addWidget(self.btn_view_loaded_log)

        traj_box_layout.addWidget(row2_container)

        self.grp_trajectory.setVisible(False)
        left_layout.addWidget(self.grp_trajectory)

        left_container.setMinimumWidth(300)
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
        self.btn_read_zmat.setStyleSheet("background-color: #2563eb; color: #ffffff; font-size: 11px; font-weight: bold; padding: 4px 6px;")
        self.btn_read_zmat.setToolTip("Open and load a Z-matrix file (*.zmat, *.dat, *.inp, *.txt)")
        self.btn_read_zmat.clicked.connect(self._on_open_zmat)
        self.btn_read_zmat.setVisible(False)  # Accessible via File -> Open File (Ctrl+O) and Raw ZMAT tab

        row_action_layout2.addWidget(self.btn_close_ring)
        row_action_layout2.addWidget(self.btn_reorder_zmat)

        # Collapsible "More Tools" container for secondary Z-Matrix & conformer actions
        self.btn_more_tools = QtWidgets.QPushButton("🧰 More Tools ▾")
        self.btn_more_tools.setCheckable(True)
        self.btn_more_tools.setChecked(False)
        self.btn_more_tools.setStyleSheet("background-color: #0891b2; color: #ffffff; font-size: 11px; font-weight: bold; padding: 4px 8px;")
        self.btn_more_tools.setToolTip("Show/hide secondary tools (Form Bond, Reorder ZMAT, Conformers, Z-Matrix Variables)")
        self.more_tools_widget = QtWidgets.QWidget()
        more_tools_layout = QtWidgets.QVBoxLayout(self.more_tools_widget)
        more_tools_layout.setContentsMargins(0, 0, 0, 0)
        more_tools_layout.setSpacing(5)
        more_tools_layout.addLayout(row_action_layout2)
        self.more_tools_widget.setVisible(False)
        self.btn_more_tools.toggled.connect(
            lambda checked: (
                self.more_tools_widget.setVisible(checked),
                self.btn_more_tools.setText("🧰 More Tools ▴" if checked else "🧰 More Tools ▾"),
            )
        )
        bottom_layout.addWidget(self.btn_more_tools)
        bottom_layout.addWidget(self.more_tools_widget)

        # Row 3: Optimization Method Selector & Optimize Button
        row_action_layout_opt = QtWidgets.QHBoxLayout()
        row_action_layout_opt.setSpacing(5)
        self.combo_opt_method = QtWidgets.QComboBox()
        self.combo_opt_method.addItem("Analytical UFF / MMFF94s", "uff_mmff")
        self.combo_opt_method.addItem("Auto (Hybrid MMFF / UFF)", "auto_hybrid")
        self.combo_opt_method.addItem("GFN0-xTB (Semiempirical QM)", "semiempirical")
        self.combo_opt_method.addItem("🧠 MLIP (ANI-2x / MACE / Hybrid QM)", "mlip")
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
        self.btn_indiv_vars.setStyleSheet("background-color: #6366f1; color: #ffffff; font-size: 11px; font-weight: bold;")
        self.btn_indiv_vars.setToolTip("Assign an individual variable (R1..Rn, A1..An, D1..Dn) to every coordinate")
        self.btn_indiv_vars.clicked.connect(lambda: (menu_zmat_vars.close(), self._on_assign_individual_variables()))

        self.btn_clear_vars = QtWidgets.QPushButton("🔢 Numbers Only (Clear Variables)")
        self.btn_clear_vars.setStyleSheet("background-color: #d97706; color: #ffffff; font-size: 11px; font-weight: bold;")
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
        more_tools_layout.addLayout(row_action_layout_adv)

        # Selected Atom Fine Adjustment Group (Molden Style Slider Adjustment + GaussView SmartSlides)
        self.grp_adjust = QtWidgets.QGroupBox("Interactive Parameter Tuning / SmartSlides (Selected Atom or 2/3/4-Atom Selection)")
        self.grp_adjust.setCheckable(True)
        self.grp_adjust.setChecked(True)
        self.grp_adjust.setToolTip("Toggle to expand or collapse Interactive Parameter Tuning / SmartSlides")
        adjust_layout = QtWidgets.QGridLayout(self.grp_adjust)
        adjust_layout.setHorizontalSpacing(8)
        adjust_layout.setVerticalSpacing(5)
        adjust_layout.setColumnStretch(2, 1)

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
        self.combo_smart_mode.setToolTip(
            "GaussView SmartSlides mode when 2, 3, or 4 atoms are selected in the 3D viewport:\n"
            "• Move Substituent Group: Translates/rotates the connected substituent subgraph\n"
            "• Move Both Groups Symmetrically: Splits displacement 50/50 across both sides\n"
            "• Move Single Atom Only: Moves only the terminal selected atom"
        )
        row_smart_mode.addWidget(self.combo_smart_mode, stretch=1)

        self.btn_invert_center = QtWidgets.QPushButton("🔄 Invert R/S")
        self.btn_invert_center.setMaximumWidth(105)
        self.btn_invert_center.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold; font-size: 11px; padding: 2px 6px;")
        self.btn_invert_center.setToolTip(
            "Invert local tetrahedral/pyramidal stereocenter (select chiral center atom or center + 2 substituents)\n"
            "without flipping the rest of the molecule!"
        )
        self.btn_invert_center.clicked.connect(self._on_invert_stereocenter)
        row_smart_mode.addWidget(self.btn_invert_center)
        adjust_layout.addLayout(row_smart_mode, 0, 0, 1, 3)

        # Bond length slider
        lbl_bond = QtWidgets.QLabel("Bond (Å):")
        lbl_bond.setStyleSheet("font-size: 11px; font-weight: 600;")
        adjust_layout.addWidget(lbl_bond, 2, 0)
        self.spin_bond = QtWidgets.QDoubleSpinBox()
        self.spin_bond.setRange(0.1, 10.0)
        self.spin_bond.setSingleStep(0.01)
        self.spin_bond.setMinimumWidth(88)
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
        self.spin_angle.setMinimumWidth(88)
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
        self.spin_dihedral.setMinimumWidth(88)
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
        raw_header_layout.addWidget(QtWidgets.QLabel("Z-Matrix Text:"))
        self.chk_auto_apply_raw = QtWidgets.QCheckBox("Auto-apply")
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

        raw_vars_layout = QtWidgets.QGridLayout()
        raw_vars_layout.setSpacing(5)
        btn_raw_sym_bonds = QtWidgets.QPushButton("🔗 Symm. Bonds")
        btn_raw_sym_bonds.setStyleSheet("background-color: #0369a1; color: #ffffff; font-weight: bold;")
        btn_raw_sym_bonds.setToolTip("Assign symmetric variables ONLY to bond distances (keep angles & dihedrals fixed)")
        btn_raw_sym_bonds.clicked.connect(self._on_assign_symmetric_bond_variables)
        btn_raw_sym_all = QtWidgets.QPushButton("🔗 Symm. All")
        btn_raw_sym_all.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold;")
        btn_raw_sym_all.clicked.connect(self._on_assign_symmetric_variables)
        btn_raw_indiv = QtWidgets.QPushButton("🔤 All Variables")
        btn_raw_indiv.setStyleSheet("background-color: #6366f1; color: #ffffff; font-weight: bold;")
        btn_raw_indiv.clicked.connect(self._on_assign_individual_variables)
        btn_raw_clear = QtWidgets.QPushButton("🔢 Numbers Only")
        btn_raw_clear.setStyleSheet("background-color: #d97706; color: #ffffff; font-weight: bold;")
        btn_raw_clear.clicked.connect(self._on_clear_variables)
        raw_vars_layout.addWidget(btn_raw_sym_bonds, 0, 0)
        raw_vars_layout.addWidget(btn_raw_sym_all, 0, 1)
        raw_vars_layout.addWidget(btn_raw_indiv, 1, 0)
        raw_vars_layout.addWidget(btn_raw_clear, 1, 1)
        tab_raw_layout.addLayout(raw_vars_layout)

        raw_btn_layout = QtWidgets.QHBoxLayout()
        btn_open_raw = QtWidgets.QPushButton("📂 Open ZMAT")
        btn_open_raw.setStyleSheet("background-color: #0d9488; color: #ffffff; font-weight: bold;")
        btn_open_raw.setToolTip("Open Z-Matrix file...")
        btn_open_raw.clicked.connect(self._on_open_zmat)
        btn_save_raw = QtWidgets.QPushButton("💾 Save ZMAT")
        btn_save_raw.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold;")
        btn_save_raw.setToolTip("Save Z-Matrix file...")
        btn_save_raw.clicked.connect(self._on_save_zmat)
        btn_apply_raw = QtWidgets.QPushButton("⚡ Apply (Ctrl+Enter)")
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
        btn_copy_xyz_tab.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold;")
        btn_copy_xyz_tab.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.text_raw_xyz.toPlainText()))
        btn_clear_xyz = QtWidgets.QPushButton("🧹 Clear")
        btn_clear_xyz.setStyleSheet("background-color: #d97706; color: #ffffff; font-weight: bold;")
        btn_clear_xyz.clicked.connect(lambda: self.text_raw_xyz.clear())
        xyz_quick_actions.addWidget(btn_paste_clip)
        xyz_quick_actions.addWidget(btn_copy_xyz_tab)
        xyz_quick_actions.addWidget(btn_clear_xyz)
        tab_xyz_layout.addLayout(xyz_quick_actions)

        xyz_btn_layout = QtWidgets.QHBoxLayout()
        btn_import_xyz_file = QtWidgets.QPushButton("📂 Open XYZ")
        btn_import_xyz_file.setStyleSheet("background-color: #0d9488; color: #ffffff; font-weight: bold;")
        btn_import_xyz_file.setToolTip("Open Cartesian XYZ file...")
        btn_import_xyz_file.clicked.connect(self._on_import_xyz)
        btn_export_xyz_file = QtWidgets.QPushButton("💾 Save XYZ")
        btn_export_xyz_file.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold;")
        btn_export_xyz_file.setToolTip("Save Cartesian XYZ file...")
        btn_export_xyz_file.clicked.connect(self._on_export_xyz)
        btn_apply_xyz = QtWidgets.QPushButton("⚡ Apply XYZ (Ctrl+Enter)")
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
        btn_save_custom_frag.setVisible(False)
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
        self.btn_freeze_sel.setVisible(False)
        row_freeze.addWidget(self.btn_freeze_sel)

        self.btn_unfreeze_all = QtWidgets.QPushButton("🔥 Unfreeze All (0 frozen)")
        self.btn_unfreeze_all.setStyleSheet("background-color: #d97706; color: #ffffff; font-weight: bold;")
        self.btn_unfreeze_all.setVisible(False)
        self.btn_unfreeze_all.setToolTip("Clear all frozen atom coordinate constraints")
        self.btn_unfreeze_all.clicked.connect(self._on_unfreeze_all_atoms)
        row_freeze.addWidget(self.btn_unfreeze_all)
        tools_layout.addLayout(row_freeze)

        row_invert_scan = QtWidgets.QHBoxLayout()
        row_invert_scan.setSpacing(6)
        btn_invert_local = QtWidgets.QPushButton("🔄 Invert R/S")
        btn_invert_local.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold; padding: 3px 8px;")
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
        self.right_container = right_container
        self.right_stack = QtWidgets.QStackedWidget()
        self.right_stack.setMinimumWidth(260)
        self.right_stack.addWidget(right_container)
        splitter.setHandleWidth(6)
        splitter.addWidget(self.right_stack)

        # Splitter dynamic sizing: responsive in full window view
        # 3D Viewport gets ~65% width by default, inspector panel ~35%
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([820, 460])

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

        self.btn_agemacs_menu = QtWidgets.QToolButton()
        self.btn_agemacs_menu.setText("🚀 Agemacs ▾")
        self.btn_agemacs_menu.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_agemacs_menu.setToolTip("Open or export inputs, outputs, XYZ, Z-Matrix, or any file directly into Agemacs (Alt+Shift+A)")
        self.btn_agemacs_menu.setStyleSheet("background-color: #0d9488; color: #ffffff; font-weight: bold; padding: 2px 8px; border-radius: 4px;")
        menu_agemacs = QtWidgets.QMenu(self)
        for label, handler in (
            ("📄 Open Active File / Geometry in Agemacs", self._on_open_in_agemacs),
            ("📐 Export Cartesian XYZ to Agemacs", self._on_send_xyz_to_agemacs),
            ("📋 Export Z-Matrix to Agemacs", self._on_send_zmat_to_agemacs),
            ("📊 Open Active QM Log / Molden in Agemacs", self._on_send_qm_output_to_agemacs),
        ):
            menu_agemacs.addAction(label, lambda _checked=False, fn=handler: QtCore.QTimer.singleShot(0, fn))
        menu_agemacs.addSeparator()
        menu_agemacs.addAction(
            "📂 Browse & Open Any File in Agemacs...",
            lambda _checked=False, fn=self._on_send_any_file_to_agemacs: QtCore.QTimer.singleShot(0, fn),
        )
        self.btn_agemacs_menu.setMenu(menu_agemacs)
        toolbar.addWidget(self.btn_agemacs_menu)

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

        toolbar_qm = QtWidgets.QToolBar("Calculations & Surfaces")
        self.toolbar_qm = toolbar_qm
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
        btn_view_log_tb.setText("📄 Output Viewer")
        btn_view_log_tb.setToolTip("Open QM Calculation Output Log & Results Viewer (F2)")
        btn_view_log_tb.clicked.connect(self._on_view_qm_output_file)
        toolbar_qm.addWidget(btn_view_log_tb)

        self.btn_unit_conv_tb = QtWidgets.QToolButton()
        self.btn_unit_conv_tb.setText("⚡ Units")
        self.btn_unit_conv_tb.setToolTip("Quantum Chemistry Unit Converter (Hartree, eV, kcal/mol, kJ/mol, cm⁻¹, Bohr, Å, Debye) (Ctrl+U)")
        self.btn_unit_conv_tb.clicked.connect(self._open_unit_converter)
        toolbar_qm.addWidget(self.btn_unit_conv_tb)

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
        btn_cmd_palette.setStyleSheet("background-color: #4f46e5; color: #ffffff; font-weight: bold; padding: 2px 8px; border: 1px solid #4338ca; border-radius: 4px;")
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

        self.btn_status_theme = QtWidgets.QPushButton("🌙 Dark")
        self.btn_status_theme.setToolTip("Toggle between White (light) and Dark color themes")
        self.btn_status_theme.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_status_theme.clicked.connect(
            lambda: self.set_color_theme("dark" if getattr(self.viewer, "bg_color_mode", "white") == "white" else "white")
        )
        self.statusBar().addPermanentWidget(self.btn_status_theme)

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

    def _on_show_shortcuts_cheat_sheet(self):
        """Show a concise keyboard & mouse shortcuts cheat sheet."""
        html = (
            "<h3>⌨️ AgyMolden Quick Shortcuts</h3>"
            "<table>"
            "<tr><td><b>Tab</b></td><td>&nbsp;Toggle Build / Inspect Mode</td></tr>"
            "<tr><td><b>Space</b></td><td>&nbsp;Reset & Center 3D View</td></tr>"
            "<tr><td><b>Ctrl+H</b></td><td>&nbsp;Add Hydrogens (Complete Valence)</td></tr>"
            "<tr><td><b>Ctrl+M</b></td><td>&nbsp;Optimize Geometry (UFF / xTB)</td></tr>"
            "<tr><td><b>Ctrl+R</b></td><td>&nbsp;Run ORCA (1-Click)</td></tr>"
            "<tr><td><b>Ctrl+K</b></td><td>&nbsp;Quick Command Palette</td></tr>"
            "<tr><td><b>Shift+Click</b></td><td>&nbsp;Measure Distance (2), Angle (3), Dihedral (4)</td></tr>"
            "<tr><td><b>Del</b></td><td>&nbsp;Delete Selected Atom(s)</td></tr>"
            "</table>"
        )
        QtWidgets.QMessageBox.information(self, "Keyboard Shortcuts", html)

    def _on_test_qm_setup(self):
        """Check ORCA and xTB executable availability for students."""
        from .qm_detect import detect_qm_executable
        orca = detect_qm_executable("orca")
        xtb = detect_qm_executable("xtb")
        orca_line = f"🟢 <b>ORCA:</b> Found ({orca.resolved_path})" if orca.found else "🔴 <b>ORCA:</b> Not found (use File → Configure QM Engines)"
        xtb_line = f"🟢 <b>xTB:</b> Found ({xtb.resolved_path})" if xtb.found else "🟡 <b>xTB:</b> Not found (optional; UFF/MMFF94 still works)"
        QtWidgets.QMessageBox.information(self, "QM Setup Check", f"{orca_line}<br><br>{xtb_line}")

    def _on_copy_methods_paragraph(self):
        """Copy a ready-to-paste computational methods paragraph to clipboard."""
        text = (
            "All molecular geometries were built and visualized using AgyMolden. "
            "Quantum chemical calculations were performed with ORCA at the B3LYP/def2-SVP level of theory, "
            "and harmonic vibrational frequency analyses were carried out to confirm true minima."
        )
        QtWidgets.QApplication.clipboard().setText(text)
        self.statusBar().showMessage("📋 Methods paragraph copied to clipboard!", 4000)

    def _on_quick_element_pill(self, sym: str):
        """Sync element pill buttons + combo brush and select *sym* for 3D drawing."""
        for s_key, btn in getattr(self, "btn_elem_pills", {}).items():
            btn.blockSignals(True)
            btn.setChecked(s_key == sym)
            btn.blockSignals(False)
        if hasattr(self, "combo_brush_element"):
            idx = self.combo_brush_element.findData(sym)
            if idx >= 0:
                self.combo_brush_element.blockSignals(True)
                self.combo_brush_element.setCurrentIndex(idx)
                self.combo_brush_element.blockSignals(False)
        self._on_select_brush_element(sym)

    def _on_quick_bond_order(self, order: int):
        """Set active bond order and apply immediately if 2 atoms are selected or molecule is diatomic."""
        self.active_draw_bond_order = order
        for o_val, btn in (
            (1, getattr(self, "btn_bo_1", None)),
            (2, getattr(self, "btn_bo_2", None)),
            (3, getattr(self, "btn_bo_3", None)),
        ):
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(o_val == order)
                btn.blockSignals(False)
        sel = list(getattr(self.viewer, "selected_atoms", []) or [])
        if len(sel) == 2 and hasattr(self, "set_bond_order"):
            self.set_bond_order(sel[0], sel[1], target_bo=order)
        elif len(getattr(self, "zatoms", []) or []) == 2 and hasattr(self, "set_bond_order"):
            self.set_bond_order(0, 1, target_bo=order)

    def _on_open_lab_exercise(self):
        """Open the self-contained Interactive Lab Exercise (.agylab) dialog."""
        from .dialog_lab_mode import LabExerciseDialog
        dlg = LabExerciseDialog(self)
        dlg.show()

    def _on_open_in_agemacs(self):
        """Open the current file or exported XYZ geometry in agemacs."""
        import os
        from .companion import send_to_agemacs, send_text_to_agemacs

        for attr in ("current_file_path", "_last_loaded_qm_log_path", "current_file"):
            cur = getattr(self, attr, None)
            if cur:
                cur_s = str(cur)
                if os.path.isfile(cur_s):
                    if not send_to_agemacs([cur_s], self):
                        QtWidgets.QMessageBox.information(
                            self, "Open in agemacs",
                            f"Could not find 'agemacs' on PATH or in standard project folders.\n\nFile ready to open:\n{cur_s}",
                        )
                    return

        if getattr(self, "zatoms", None) and getattr(self, "coords", None) is not None and len(self.coords) > 0:
            lines = [f"{len(self.zatoms)}", "Exported from AgyMolden"]
            for atom, xyz in zip(self.zatoms, self.coords):
                sym = getattr(atom, "symbol", getattr(atom, "element", "X"))
                lines.append(f"{sym:<3s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}")
            if not send_text_to_agemacs("agymolden_export.xyz", "\n".join(lines) + "\n", self):
                QtWidgets.QMessageBox.information(
                    self, "Open in agemacs",
                    "Could not find 'agemacs' on PATH or in standard project folders.",
                )
            return

        QtWidgets.QMessageBox.information(
            self, "Open in agemacs", "No molecule or file is currently loaded to send to agemacs."
        )

    def _on_send_xyz_to_agemacs(self):
        """Export current Cartesian XYZ coordinates directly to agemacs."""
        from .companion import send_text_to_agemacs

        txt = self.text_raw_xyz.toPlainText().strip() if hasattr(self, "text_raw_xyz") else ""
        if not txt and getattr(self, "zatoms", None) and getattr(self, "coords", None) is not None and len(self.coords) > 0:
            lines = [f"{len(self.zatoms)}", "Exported from AgyMolden"]
            for atom, xyz in zip(self.zatoms, self.coords):
                sym = getattr(atom, "symbol", getattr(atom, "element", "X"))
                lines.append(f"{sym:<3s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}")
            txt = "\n".join(lines) + "\n"
        if not txt:
            QtWidgets.QMessageBox.information(self, "Export XYZ to Agemacs", "No molecule loaded.")
            return
        if not send_text_to_agemacs("agymolden_structure.xyz", txt + "\n", self):
            QtWidgets.QMessageBox.information(self, "Open in agemacs", "Could not launch 'agemacs'.")

    def _on_send_zmat_to_agemacs(self):
        """Export current Z-Matrix text directly to agemacs."""
        from .companion import send_text_to_agemacs
        from .zmatrix import format_zmatrix_text

        txt = self.text_raw_zmat.toPlainText().strip() if hasattr(self, "text_raw_zmat") else ""
        if not txt and getattr(self, "zatoms", None):
            txt = format_zmatrix_text(self.zatoms).strip()
        if not txt:
            QtWidgets.QMessageBox.information(self, "Export Z-Matrix to Agemacs", "No Z-Matrix available.")
            return
        if not send_text_to_agemacs("agymolden_structure.zmat", txt + "\n", self):
            QtWidgets.QMessageBox.information(self, "Open in agemacs", "Could not launch 'agemacs'.")

    def _on_send_qm_output_to_agemacs(self):
        """Open active QM output log or Molden file in agemacs."""
        import os
        from .companion import send_to_agemacs

        candidates: List[str] = []
        for attr in (
            "_last_loaded_qm_log_path",
            "_active_molden_path",
            "_loaded_molden_path",
            "current_qm_output_file",
            "current_molden_file",
            "current_file_path",
            "current_file",
        ):
            p = getattr(self, attr, None)
            if p:
                p_s = str(p)
                if os.path.isfile(p_s):
                    candidates.append(p_s)

        monitors = []
        if hasattr(self, "dock_manager") and "job_monitor" in getattr(self.dock_manager, "docks", {}):
            monitors.append(self.dock_manager.docks["job_monitor"].monitor)
        if getattr(self, "_job_monitor_dlg", None) is not None:
            monitors.append(self._job_monitor_dlg)
        for job_mon in monitors:
            runner = getattr(job_mon, "runner", None)
            if runner and getattr(runner, "log_file", None) and os.path.isfile(runner.log_file):
                candidates.append(runner.log_file)

        for p in candidates:
            if not send_to_agemacs([p], self):
                QtWidgets.QMessageBox.information(self, "Open in agemacs", f"Could not launch 'agemacs'.\n\nFile: {p}")
            return
        QtWidgets.QMessageBox.information(self, "Open QM Log in Agemacs", "No QM output, log, or Molden file is currently loaded.")

    def _on_send_any_file_to_agemacs(self):
        """Browse for any input, output, or coordinate file and open it in agemacs."""
        from .companion import send_to_agemacs

        start_dir = self._last_dir() if hasattr(self, "_last_dir") else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open File in Agemacs",
            start_dir,
            "Chemistry Files (*.inp *.gjf *.com *.out *.log *.xyz *.zmat *.molden *.molden.input);;All Files (*)",
        )
        if path and not send_to_agemacs([path], self):
            QtWidgets.QMessageBox.information(self, "Open in agemacs", f"Could not launch 'agemacs'.\n\nFile: {path}")


    # Core logic
