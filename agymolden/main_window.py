"""Main application window for AgyMolden: Modern Molden-inspired Z-Matrix Generator & Editor."""

import os
import math
import copy
import logging
from typing import List, Optional, Dict, Tuple
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets

from .workers import run_in_background

log = logging.getLogger(__name__)

from .elements import ELEMENT_DATA, normalize_symbol, estimate_bond_length, get_atomic_number
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
    find_available_symmetries,
    PointGroupResult,
    align_molecule_to_symmetry,
    enforce_point_group_symmetry,
    calculate_symmetry_representations,
)
from .alignment import (
    RigidFrame,
    capture_rigid_frame,
    apply_rigid_frame,
)
from .molecule_model import MoleculeModel



from .mw_ui import DARK_THEME_QSS, LIGHT_THEME_QSS, UIMixin
from .mw_calculations import CalculationsMixin
from .mw_file_io import FileIOMixin
from .mw_reorder import ReorderMixin
from .mw_editing import EditingMixin


class MainWindow(UIMixin, EditingMixin, CalculationsMixin, FileIOMixin, ReorderMixin, QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AgyMolden - Z-Matrix Molecular Builder")
        # Fixed, screen-bounded window size: never grows with content
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1280, 850)
        self.setMaximumSize(avail.width(), avail.height())
        self.resize(min(1280, avail.width()), min(850, avail.height()))
        self.setAcceptDrops(True)
        self.setStyleSheet(DARK_THEME_QSS)

        # Core molecule state (delegated to self.model as single source of truth)
        self.model = MoleculeModel(self)
        self.zatoms: List[ZAtom] = []
        self.coords: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.variables: Dict[str, float] = {}

        # Undo / Redo stacks
        self.undo_stack: List[Tuple[List[ZAtom], np.ndarray, Dict[str, float]]] = []
        self.redo_stack: List[Tuple[List[ZAtom], np.ndarray, Dict[str, float]]] = []

        # Vibrational modes analysis state
        self.vibrational_analysis: Optional[VibrationalAnalysis] = None

        # Cross-bonds / ring closures outside the Z-matrix spanning tree
        self.cross_bonds: set = set()
        # When True, the bond graph is exactly tree bonds + cross_bonds (no distance guessing)
        self.strict_topology: bool = False
        # Optional rigid-body coordinate frame (preserves XYZ or user-aligned orientation)
        self._rigid_frame: Optional[RigidFrame] = None
        # Frozen atom indices (pinned in 3D space during geometry optimization)
        self.frozen_atoms: set = set()
        # Last selected coordination / hybridization template from Periodic Table
        self.last_coordination_template: str = "auto"

        # State for atom and fragment addition
        self.pending_new_element: Optional[str] = None
        self.pending_picks_needed: int = 0
        # Memory state for clipboard / last selected item to paste on double-click
        self.last_added_element: Optional[str] = None
        self.stored_fragment: Optional[Dict] = None
        self.last_added_fragment: Optional[Dict] = None
        self.last_item_type: Optional[str] = None # None, "element", or "fragment"
        self.pending_add_atom_mode: bool = False
        self._current_sym_res: Optional[PointGroupResult] = None
        self.symmetry_imposed: bool = False
        self.imposed_sym_res: Optional[PointGroupResult] = None
        self.reorder_mode_active: bool = False
        self.reorder_picks: List[int] = []
        self.partial_reorder_mode: Optional[str] = None
        self.partial_reorder_picks: List[int] = []

        self._init_ui()
        self._create_menus()
        self._create_toolbar()
        self._update_viewer_drag_label()

        # Install event filter on this window so Ctrl+Z / Ctrl+Y always fire regardless of focused child widget
        self.installEventFilter(self)

        # Restore window geometry & state from previous session
        self._restore_window_state()

        # Start with no molecule (empty workspace, like Molden)
        self.set_molecule_from_zatoms([])

        # Auto-save timer (every 60 seconds)
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setInterval(60000)
        self._autosave_timer.timeout.connect(self._auto_save_session)
        self._autosave_timer.start()

        # Check for previous crashed / unsaved session
        QtCore.QTimer.singleShot(600, self._check_and_prompt_crash_recovery)

        # Check for first-launch onboarding tour
        QtCore.QTimer.singleShot(1200, self._check_first_run_tour)

    @property
    def zatoms(self) -> List[ZAtom]:
        return self.model.zatoms

    @zatoms.setter
    def zatoms(self, val: List[ZAtom]):
        self.model.zatoms = val

    @property
    def coords(self) -> np.ndarray:
        return self.model.coords

    @coords.setter
    def coords(self, val: np.ndarray):
        self.model.coords = np.asarray(val, dtype=np.float64)

    @property
    def variables(self) -> Dict[str, float]:
        return self.model.variables

    @variables.setter
    def variables(self, val: Dict[str, float]):
        self.model.variables = val

    @property
    def cross_bonds(self) -> set:
        return self.model.cross_bonds

    @cross_bonds.setter
    def cross_bonds(self, val: set):
        self.model.cross_bonds = set(val)

    @property
    def strict_topology(self) -> bool:
        return self.model.strict_topology

    @strict_topology.setter
    def strict_topology(self, val: bool):
        self.model.strict_topology = bool(val)

    @property
    def event_bus(self):
        try:
            from .events import get_event_bus
            return get_event_bus()
        except Exception:
            return None

    def _auto_save_session(self):
        """Silently auto-save current molecule state and calculation history to a hidden project recovery file."""
        if not getattr(self, "zatoms", None):
            return
        try:
            cache_dir = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppDataLocation)
            os.makedirs(cache_dir, exist_ok=True)
            recovery_path = os.path.join(cache_dir, "autosave_recovery.agyproj")

            from .project_bundle import save_project_bundle
            calc_hist = getattr(self, "calculation_history", [])
            viewer_st = {
                "theme": getattr(self, "current_theme", "white"),
                "style": getattr(self, "current_style", "Ball & Stick"),
                "show_labels": getattr(self.viewer, "show_atom_labels", False),
            }
            save_project_bundle(
                filepath=recovery_path,
                zatoms=self.zatoms,
                coords=self.coords,
                variables=self.variables,
                cross_bonds=list(self.cross_bonds),
                calculation_history=calc_hist,
                viewer_state=viewer_st,
                metadata={"title": "Autosave Recovery", "num_atoms": len(self.zatoms)},
            )
        except Exception as e:
            log.debug("Autosave recovery write failed: %s", e)

    def _check_and_prompt_crash_recovery(self):
        """Check if an autosaved session exists from an abnormal exit and prompt the user to recover."""
        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        try:
            cache_dir = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppDataLocation)
            recovery_path = os.path.join(cache_dir, "autosave_recovery.agyproj")
            legacy_recovery = os.path.join(cache_dir, "autosave_recovery.zmat")

            target_recovery = None
            if os.path.isfile(recovery_path) and os.path.getsize(recovery_path) > 10:
                target_recovery = recovery_path
            elif os.path.isfile(legacy_recovery) and os.path.getsize(legacy_recovery) > 10:
                target_recovery = legacy_recovery

            if target_recovery:
                settings = self._settings()
                if not settings.value("clean_shutdown", True, type=bool):
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "🔄 Recover Previous Session?",
                        "AgyMolden detected an unsaved session from your previous run.\n\n"
                        "Would you like to restore your full molecular model, Z-matrix, and calculation history?",
                        QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                        QtWidgets.QMessageBox.StandardButton.Yes,
                    )
                    if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                        self.open_path(target_recovery)
                        self.statusBar().showMessage("Recovered unsaved session from previous run.", 4000)
            settings = self._settings()
            settings.setValue("clean_shutdown", False)
        except Exception as e:
            log.debug("Crash recovery check failed: %s", e)

    def _restore_window_state(self):
        settings = self._settings()
        geom = settings.value("main_window_geom")
        if geom:
            self.restoreGeometry(geom)
        state = settings.value("main_window_state")
        if state:
            self.restoreState(state)
        saved_theme = settings.value("color_theme", "white")
        if saved_theme in ("dark", "white"):
            self.set_color_theme(str(saved_theme))

    def closeEvent(self, event: QtGui.QCloseEvent):
        settings = self._settings()
        settings.setValue("main_window_geom", self.saveGeometry())
        settings.setValue("main_window_state", self.saveState())
        settings.setValue("clean_shutdown", True)
        super().closeEvent(event)

    def _push_undo_state(self):
        """Push current molecule state onto the undo stack and clear redo stack."""
        state = (
            [copy.deepcopy(a) for a in self.zatoms],
            np.copy(self.coords),
            dict(self.variables),
            self._topology_snapshot()
        )
        self.undo_stack.append(state)
        # Limit stack size to 50
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._update_undo_redo_actions()

    def undo(self):
        """Undo last molecular modification."""
        if not self.undo_stack:
            self.statusBar().showMessage("Nothing to undo.", 2500)
            return
        # Current state goes onto redo stack
        current_state = (
            [copy.deepcopy(a) for a in self.zatoms],
            np.copy(self.coords),
            dict(self.variables),
            self._topology_snapshot()
        )
        self.redo_stack.append(current_state)
        if len(self.redo_stack) > 50:
            self.redo_stack.pop(0)

        prev_zatoms, prev_coords, prev_vars, *topo = self.undo_stack.pop()
        self._restore_state(prev_zatoms, prev_coords, prev_vars, topo[0] if topo else None)
        self._update_undo_redo_actions()
        self.statusBar().showMessage("Undo: Restored previous state.", 3000)

    def redo(self):
        """Redo last undone molecular modification."""
        if not self.redo_stack:
            self.statusBar().showMessage("Nothing to redo.", 2500)
            return
        current_state = (
            [copy.deepcopy(a) for a in self.zatoms],
            np.copy(self.coords),
            dict(self.variables),
            self._topology_snapshot()
        )
        self.undo_stack.append(current_state)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

        next_zatoms, next_coords, next_vars, *topo = self.redo_stack.pop()
        self._restore_state(next_zatoms, next_coords, next_vars, topo[0] if topo else None)
        self._update_undo_redo_actions()
        self.statusBar().showMessage("Redo: Restored next state.", 3000)

    def _topology_snapshot(self):
        rf = self._rigid_frame.copy() if getattr(self, "_rigid_frame", None) is not None else None
        return (
            set(self.cross_bonds),
            bool(getattr(self, "strict_topology", False)),
            dict(getattr(self.viewer, "bond_orders", {})),
            rf,
            set(getattr(self, "frozen_atoms", set())),
        )

    def _restore_state(self, zatoms: List[ZAtom], coords: np.ndarray, variables: Dict[str, float], topo=None):
        if topo is not None:
            self.cross_bonds = set(topo[0])
            self.strict_topology = topo[1]
            self.viewer.bond_orders = dict(topo[2])
            self._rigid_frame = topo[3].copy() if len(topo) > 3 and topo[3] is not None else None
            self.frozen_atoms = set(topo[4]) if len(topo) > 4 and topo[4] is not None else set()
        else:
            self.cross_bonds, self.strict_topology, self._rigid_frame = set(), False, None
            self.frozen_atoms = set()
        if hasattr(self, "_update_frozen_buttons_ui"):
            self._update_frozen_buttons_ui()
        self.zatoms = zatoms
        self.coords = np.copy(coords)
        self.variables = variables
        symbols = [a.symbol for a in zatoms]
        self.viewer.selected_atom_idx = None
        self.viewer.multi_selected = []
        self.viewer.highlighted_feature_atoms = {}
        n_at = len(zatoms)
        ub = [(a.ref1, i) for i, a in enumerate(zatoms) if a.ref1 is not None and a.ref1 < i]
        ub += [b for b in self.cross_bonds if b[0] < n_at and b[1] < n_at]
        self.viewer.strict_bonds = self.strict_topology
        self.viewer.set_molecule(symbols, self.coords, user_bonds=ub)
        self.table_widget.load_zatoms(zatoms)
        if hasattr(self, "model") and self.model is not None:
            self.model.zatoms = list(zatoms)
            self.model.coords = np.copy(self.coords)
            self.model.variables = dict(getattr(self, "variables", {}))
            self.model.cross_bonds = set(getattr(self, "cross_bonds", set()))
            self.model.strict_topology = bool(getattr(self, "strict_topology", False))
            self.model.structure_changed.emit()
            self.model.topology_changed.emit()
        self._sync_raw_text()
        self._update_adjust_panel()
        self._update_chemical_summary()
        if len(zatoms) > 0:
            try:
                tol = self.spin_sym_tol.value() if hasattr(self, "spin_sym_tol") else 0.15
                avail = find_available_symmetries(symbols, self.coords, base_tolerance=tol)
                prev_sym = None
                if getattr(self, "symmetry_imposed", False) and getattr(self, "imposed_sym_res", None) is not None:
                    prev_sym = self.imposed_sym_res.symbol
                elif getattr(self, "_current_sym_res", None) is not None:
                    prev_sym = self._current_sym_res.symbol
                chosen_entry = next((item for item in avail if item[0] == prev_sym), None) if prev_sym else None
                if chosen_entry is None:
                    chosen_entry = next((item for item in avail if item[2] == "detected"), avail[0])
                res, role_tag = chosen_entry[1], chosen_entry[2]
                self._current_sym_res = res
                self._populate_available_symmetries(avail, selected_symbol=chosen_entry[0])
                self._refresh_symmetry_ui(res, role_tag=role_tag)
                if getattr(self, "symmetry_imposed", False):
                    self.imposed_sym_res = res
            except Exception as e:
                log.warning(f"Failed to auto-detect/refresh symmetry on restore: {e}", exc_info=True)
        elif len(zatoms) == 0 and getattr(self, "symmetry_imposed", False):
            self.set_impose_symmetry(False)

        try:
            bus = getattr(self, "event_bus", None)
            if bus:
                if len(zatoms) > 0:
                    bus.molecule_loaded.emit(
                        symbols, self.coords, getattr(self, "current_file_path", "") or "Molecule"
                    )
                else:
                    bus.molecule_cleared.emit()
        except (RuntimeError, AttributeError):
            pass

    def _update_undo_redo_actions(self):
        can_undo = len(self.undo_stack) > 0
        can_redo = len(self.redo_stack) > 0
        if hasattr(self, 'act_undo'):
            self.act_undo.setEnabled(can_undo)
        if hasattr(self, 'act_redo'):
            self.act_redo.setEnabled(can_redo)
        if hasattr(self, 'btn_undo'):
            self.btn_undo.setEnabled(can_undo)
        if hasattr(self, 'btn_redo'):
            self.btn_redo.setEnabled(can_redo)

    @property
    def symbols(self) -> List[str]:
        """Convenience property returning element symbols for all atoms in the active molecule."""
        return [a.symbol for a in self.zatoms] if hasattr(self, "zatoms") and self.zatoms else []

    def set_molecule_from_zatoms(
        self,
        zatoms: List[ZAtom],
        exact_coords: Optional[np.ndarray] = None,
        preserve_frame: bool = False,
    ):
        self.zatoms = zatoms
        if len(zatoms) == 0:
            self._rigid_frame = None
            self.coords = np.empty((0, 3), dtype=np.float64)
        elif exact_coords is not None and len(exact_coords) == len(zatoms):
            self.coords = np.array(exact_coords, dtype=np.float64)
            self._rigid_frame = capture_rigid_frame(zatoms, self.coords)
        else:
            local_coords = zmatrix_to_cartesian(zatoms)
            if preserve_frame and getattr(self, "_rigid_frame", None) is not None:
                self.coords = apply_rigid_frame(local_coords, self._rigid_frame)
            else:
                self._rigid_frame = None
                self.coords = local_coords
        symbols = [a.symbol for a in zatoms]

        # Extract user bonds from Z-matrix (sanitized: no H-H bonds, no non-bonded distance references)
        user_bonds = []
        is_pure_h2 = (len(zatoms) == 2 and symbols[0] == "H" and symbols[1] == "H")
        for idx, a in enumerate(zatoms):
            if a.ref1 is not None and a.ref1 < idx:
                s_ref = symbols[a.ref1]
                s_cur = symbols[idx]
                if s_ref == "X" or s_cur == "X":
                    continue
                if s_ref == "H" and s_cur == "H" and not is_pure_h2:
                    continue
                if len(self.coords) == len(symbols):
                    d = float(np.linalg.norm(self.coords[a.ref1] - self.coords[idx]))
                    r_cov_max = (get_cov_radius(s_ref) + get_cov_radius(s_cur)) * 1.38 + 0.25
                    if d > r_cov_max:
                        continue
                user_bonds.append((a.ref1, idx))

        # Include persistent cross_bonds (filtered to valid atom indices)
        n_atoms = len(zatoms)
        if hasattr(self, 'frozen_atoms'):
            self.frozen_atoms = {i for i in self.frozen_atoms if 0 <= i < n_atoms}
        if hasattr(self, "_update_frozen_buttons_ui"):
            self._update_frozen_buttons_ui()
        if hasattr(self, 'cross_bonds'):
            valid_cross = set()
            for u, v in self.cross_bonds:
                if 0 <= u < n_atoms and 0 <= v < n_atoms and u != v:
                    valid_cross.add((min(u, v), max(u, v)))
                    b_pair = (min(u, v), max(u, v))
                    if b_pair not in user_bonds:
                        user_bonds.append(b_pair)
            self.cross_bonds = valid_cross

        self.vibrational_analysis = None
        if hasattr(self, 'table_vibrations'):
            self.table_vibrations.setRowCount(0)
            self.btn_play_vibration.setEnabled(False)
            self.btn_stop_vibration.setEnabled(False)
            self.lbl_vib_info.setText("Click 'Calculate Frequencies' to analyze normal modes.")

        self.viewer.highlighted_feature_atoms = {}
        self.viewer.strict_bonds = bool(getattr(self, 'strict_topology', False))
        self.viewer.set_molecule(symbols, self.coords, user_bonds=user_bonds)

        self.table_widget.load_zatoms(zatoms)
        if hasattr(self, "model") and self.model is not None:
            self.model.zatoms = list(zatoms)
            self.model.coords = np.copy(self.coords)
            self.model.variables = dict(getattr(self, "variables", {}))
            self.model.cross_bonds = set(getattr(self, "cross_bonds", set()))
            self.model.strict_topology = bool(getattr(self, "strict_topology", False))
            self.model.structure_changed.emit()
            self.model.topology_changed.emit()
        self._sync_raw_text()
        self._update_adjust_panel()
        self._update_chemical_summary()
        if len(zatoms) > 0:
            try:
                tol = self.spin_sym_tol.value() if hasattr(self, "spin_sym_tol") else 0.15
                avail = find_available_symmetries(symbols, self.coords, base_tolerance=tol)
                prev_sym = None
                if getattr(self, "symmetry_imposed", False) and getattr(self, "imposed_sym_res", None) is not None:
                    prev_sym = self.imposed_sym_res.symbol
                elif getattr(self, "_current_sym_res", None) is not None:
                    prev_sym = self._current_sym_res.symbol
                chosen_entry = next((item for item in avail if item[0] == prev_sym), None) if prev_sym else None
                if chosen_entry is None:
                    chosen_entry = next((item for item in avail if item[2] == "detected"), avail[0])
                res, role_tag = chosen_entry[1], chosen_entry[2]
                self._current_sym_res = res
                self._populate_available_symmetries(avail, selected_symbol=chosen_entry[0])
                self._refresh_symmetry_ui(res, role_tag=role_tag)
                if getattr(self, "symmetry_imposed", False):
                    self.imposed_sym_res = res
            except Exception as e:
                log.warning(f"Failed to auto-detect/refresh symmetry on restore: {e}", exc_info=True)
        elif len(zatoms) == 0 and getattr(self, "symmetry_imposed", False):
            self.set_impose_symmetry(False)

        try:
            bus = getattr(self, "event_bus", None)
            if bus:
                if len(zatoms) > 0:
                    bus.molecule_loaded.emit(
                        symbols, self.coords, getattr(self, "current_file_path", "") or "Molecule"
                    )
                else:
                    bus.molecule_cleared.emit()
        except (RuntimeError, AttributeError):
            pass

    def update_geometry_live(self):
        """Recompute 3D coordinates from zatoms and refresh viewport without camera reset."""
        local_coords = zmatrix_to_cartesian(self.zatoms)
        if getattr(self, "_rigid_frame", None) is not None and len(local_coords) > 0:
            self.coords = apply_rigid_frame(local_coords, self._rigid_frame)
        else:
            self.coords = local_coords
        self.viewer.symbols = [a.symbol for a in self.zatoms]
        user_bonds = []
        is_pure_h2 = (len(self.zatoms) == 2 and self.viewer.symbols[0] == "H" and self.viewer.symbols[1] == "H")
        for idx, a in enumerate(self.zatoms):
            if a.ref1 is not None and a.ref1 < idx:
                s_ref = self.viewer.symbols[a.ref1]
                s_cur = self.viewer.symbols[idx]
                if s_ref == "X" or s_cur == "X":
                    continue
                if s_ref == "H" and s_cur == "H" and not is_pure_h2:
                    continue
                if len(self.coords) == len(self.viewer.symbols):
                    d = float(np.linalg.norm(self.coords[a.ref1] - self.coords[idx]))
                    r_cov_max = (get_cov_radius(s_ref) + get_cov_radius(s_cur)) * 1.38 + 0.25
                    if d > r_cov_max:
                        continue
                user_bonds.append((a.ref1, idx))
        if hasattr(self, 'cross_bonds'):
            n_at = len(self.zatoms)
            for u, v in self.cross_bonds:
                if 0 <= u < n_at and 0 <= v < n_at:
                    b_pair = (min(u, v), max(u, v))
                    if b_pair not in user_bonds:
                        user_bonds.append(b_pair)
        self.viewer.update_coords(self.coords, user_bonds=user_bonds)
        self._sync_raw_text()
        self._update_chemical_summary()
        try:
            bus = getattr(self, "event_bus", None)
            if bus:
                bus.geometry_changed.emit(self.coords)
        except (RuntimeError, AttributeError):
            pass

    def _update_chemical_summary(self):
        """Update the permanent molecular formula and weight status in real time."""
        if not hasattr(self, "lbl_chem_summary"):
            return
        symbols = [a.symbol for a in self.zatoms if a.symbol != "X"]
        if not symbols:
            self.lbl_chem_summary.setText("Empty Molecule")
            return

        from collections import Counter
        counts = Counter(symbols)
        # Hill system: C first, then H, then alphabetical
        hill_order = []
        if "C" in counts:
            hill_order.append("C")
        if "H" in counts:
            hill_order.append("H")
        for sym in sorted(counts.keys()):
            if sym not in ("C", "H"):
                hill_order.append(sym)

        subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        formula_parts = []
        for sym in hill_order:
            cnt = counts[sym]
            cnt_str = str(cnt).translate(subscript_map) if cnt > 1 else ""
            formula_parts.append(f"{sym}{cnt_str}")
        formula = "".join(formula_parts)

        # Molecular Weight & Total Electrons
        mw = 0.0
        n_electrons = 0
        for sym, cnt in counts.items():
            if sym == "X":
                continue
            info = ELEMENT_DATA.get(sym, {})
            m = info.get("mass", info.get("atomic_mass", 0.0))
            mw += cnt * float(m)
            n_electrons += cnt * get_atomic_number(sym)

        # Aromaticity detection
        aro_str = ""
        try:
            from .aromatic import find_aromatic_rings
            bonds_list = list(getattr(self.viewer, "bonds", []))
            if not bonds_list:
                n_b = len(self.zatoms)
                for idx, a in enumerate(self.zatoms):
                    if a.ref1 is not None and a.ref1 < idx:
                        bonds_list.append((a.ref1, idx))
                if hasattr(self, "cross_bonds"):
                    for u, v in self.cross_bonds:
                        if u < n_b and v < n_b:
                            bonds_list.append((min(u, v), max(u, v)))
                for i in range(min(n_b, len(self.coords))):
                    for j in range(i + 1, min(n_b, len(self.coords))):
                        if (i, j) not in bonds_list and (j, i) not in bonds_list:
                            d = float(np.linalg.norm(self.coords[i] - self.coords[j]))
                            sum_r = get_cov_radius(self.zatoms[i].symbol) + get_cov_radius(self.zatoms[j].symbol)
                            if d <= sum_r * 1.25:
                                bonds_list.append((i, j))
            aro_rings = find_aromatic_rings([a.symbol for a in self.zatoms], self.coords, bonds_list)
            if aro_rings:
                r_cnt = len(aro_rings)
                aro_str = f"  |  <span style='color: #10b981; font-weight: bold;'>🌿 Aromatic ({r_cnt} ring{'s' if r_cnt > 1 else ''})</span>"
        except Exception:
            aro_str = ""

        # Fast steric clash detection (atoms closer than 58% of covalent radii sum or < 0.70 A)
        from .spatial_index import find_steric_clashes
        symbols_all = [a.symbol for a in self.zatoms]
        active_bonds = set(self._current_explicit_bonds()) if hasattr(self, "_current_explicit_bonds") else None
        raw_clashes = find_steric_clashes(symbols_all, self.coords, bonds=active_bonds, threshold_ratio=0.58, min_absolute_dist=0.70)
        clashes = [(c[0], c[1], c[2], c[3], c[4]) for c in raw_clashes]

        clash_str = ""
        if clashes:
            c_lead = clashes[0]
            clash_str = f"  <span style='color: #ef4444; font-weight: bold;'>⚠️ {len(clashes)} Clash! ({c_lead[2]}{c_lead[0]+1}-{c_lead[3]}{c_lead[1]+1}: {c_lead[4]:.2f}Å)</span>"
            self.lbl_chem_summary.setToolTip(
                "Steric clashes detected! Atoms are unphysically close, which may cause QM or force field calculations to fail.\n"
                + "\n".join(f"  • {c[2]}{c[0]+1} and {c[3]}{c[1]+1}: distance {c[4]:.3f} Å" for c in clashes[:5])
            )
            if hasattr(self, "btn_preflight_badge"):
                self.btn_preflight_badge.setText("⚠️ Clash")
                self.btn_preflight_badge.setStyleSheet("background-color: #dc2626; color: white; font-size: 11px; font-weight: bold; border-radius: 3px; padding: 2px 4px;")
                self.btn_preflight_badge.setToolTip(f"Pre-flight Warning: {len(clashes)} severe atomic clash(es) detected (< 0.70 Å). Click to inspect.")
        else:
            self.lbl_chem_summary.setToolTip(f"{formula} ({len(symbols)} atoms, {n_electrons} electrons, molecular weight {mw:.3f} g/mol)")
            if hasattr(self, "btn_preflight_badge"):
                if len(symbols) == 0:
                    self.btn_preflight_badge.setText("⚪ Empty")
                    self.btn_preflight_badge.setStyleSheet("background-color: #475569; color: white; font-size: 11px; font-weight: bold; border-radius: 3px; padding: 2px 4px;")
                    self.btn_preflight_badge.setToolTip("Pre-flight: No atoms in molecule.")
                else:
                    self.btn_preflight_badge.setText("🟢 Ready")
                    self.btn_preflight_badge.setStyleSheet("background-color: #15803d; color: white; font-size: 11px; font-weight: bold; border-radius: 3px; padding: 2px 4px;")
                    self.btn_preflight_badge.setToolTip(f"Pre-flight OK: {formula} ready for calculations (No clashes detected).")

        self.lbl_chem_summary.setText(f"{formula}  |  MW: {mw:.2f} g/mol  |  {n_electrons} e⁻  |  {len(symbols)} atoms{aro_str}{clash_str}")
        self._update_env_badge()

    def _update_env_badge(self):
        """Update the status bar QM engine / Python environment badge."""
        if not hasattr(self, "lbl_active_env"):
            return
        from .qm_detect import detect_qm_executable
        orca_info = detect_qm_executable("orca")
        if orca_info.found:
            self.lbl_active_env.setText("⚡ ORCA: Ready")
            self.lbl_active_env.setStyleSheet(
                "background-color: #064e3b; color: #34d399; border: 1px solid #059669; "
                "border-radius: 3px; padding: 1px 6px; font-weight: bold; font-size: 11px; margin-right: 6px;"
            )
            self.lbl_active_env.setToolTip(f"Active QM: ORCA found at {orca_info.resolved_path}. Click to configure engines.")
        else:
            self.lbl_active_env.setText("⚠️ QM: No Local ORCA")
            self.lbl_active_env.setStyleSheet(
                "background-color: #451a03; color: #fbbf24; border: 1px solid #d97706; "
                "border-radius: 3px; padding: 1px 6px; font-weight: bold; font-size: 11px; margin-right: 6px;"
            )
            self.lbl_active_env.setToolTip("ORCA not found in PATH. Click to select path or configure remote SSH execution.")

    def load_preset_molecule(self, name: str):
        if name in SAMPLE_MOLECULES:
            self._push_undo_state()
            self.cross_bonds = set()
            self.strict_topology = False
            text = SAMPLE_MOLECULES[name]
            zatoms, vars_dict = parse_zmatrix_text(text)
            self.variables = vars_dict
            self.set_molecule_from_zatoms(zatoms)
            self.statusBar().showMessage(f"Loaded preset: {name}", 4000)

    def _sync_raw_text(self):
        if self.variables:
            self.variables = sync_variables_from_atoms(self.zatoms, self.variables)
        txt = format_zmatrix_text(self.zatoms, self.variables)
        if hasattr(self, "text_raw_zmat"):
            self.text_raw_zmat.blockSignals(True)
            self.text_raw_zmat.setPlainText(txt)
            self.text_raw_zmat.blockSignals(False)

        if hasattr(self, "text_raw_xyz") and len(self.zatoms) > 0 and len(self.coords) == len(self.zatoms):
            from .chem_identifiers import format_xyz_string
            symbols = [a.symbol for a in self.zatoms]
            xyz_txt = format_xyz_string(symbols, self.coords, comment="AgyMolden structure")
            self.text_raw_xyz.blockSignals(True)
            self.text_raw_xyz.setPlainText(xyz_txt)
            self.text_raw_xyz.blockSignals(False)
        elif hasattr(self, "text_raw_xyz") and len(self.zatoms) == 0:
            self.text_raw_xyz.blockSignals(True)
            self.text_raw_xyz.clear()
            self.text_raw_xyz.blockSignals(False)

    def _on_table_data_changed(self):
        # A value or variable name edited on a coordinate updates or creates the variable (and all linked coords)
        edit = getattr(self.table_widget, "last_edit", None)
        self.table_widget.last_edit = None
        edited_row = edit[0] if (edit and 0 <= edit[0] < len(self.zatoms)) else None
        edited_attr = edit[1] if edit else None
        if edited_row is not None and edited_attr in ("dist", "angle", "dihedral"):
            atom = self.zatoms[edited_row]
            base, sgn = _split_var(getattr(atom, edited_attr + "_var"))
            if base:
                if base in self.variables:
                    # If user edited a numeric value on an existing variable, or assigned an existing variable name
                    self.variables[base] = sgn * getattr(atom, edited_attr)
                else:
                    # Newly typed variable name in the table
                    self.variables[base] = sgn * getattr(atom, edited_attr)
                apply_variables(self.zatoms, self.variables)
                self.table_widget.load_zatoms(self.zatoms)
                self.update_geometry_live()
            else:
                if self.variables:
                    self.variables = sync_variables_from_atoms(self.zatoms, self.variables)
                mode = self._get_smart_mode()
                sym_res = self._get_active_imposed_symmetry()
                if (mode in ("group", "symmetric") or sym_res is not None) and len(self.coords) == len(self.zatoms):
                    from .smartslides import modify_angle, modify_dihedral
                    symbols = [a.symbol for a in self.zatoms]
                    bonds = self._current_explicit_bonds()
                    new_c = None
                    if edited_attr == "dist" and edited_row >= 1 and atom.ref1 is not None:
                        new_c = self._modify_distance_with_imposed_symmetry(
                            atom.ref1, edited_row, atom.dist, mode=mode
                        )
                    elif edited_attr == "angle" and edited_row >= 2 and atom.ref1 is not None and atom.ref2 is not None:
                        new_c = modify_angle(
                            symbols, self.coords, bonds, atom.ref2, atom.ref1, edited_row, atom.angle, mode=mode
                        )
                    elif (
                        edited_attr == "dihedral"
                        and edited_row >= 3
                        and atom.ref1 is not None
                        and atom.ref2 is not None
                        and atom.ref3 is not None
                    ):
                        new_c = modify_dihedral(
                            symbols,
                            self.coords,
                            bonds,
                            atom.ref3,
                            atom.ref2,
                            atom.ref1,
                            edited_row,
                            atom.dihedral,
                            mode=mode,
                        )
                    if new_c is not None:
                        self._apply_smartslide_coords(new_c, list(self.viewer.multi_selected))
                    else:
                        self.table_widget.load_zatoms(self.zatoms)
                        self.update_geometry_live()
                else:
                    self.table_widget.load_zatoms(self.zatoms)
                    self.update_geometry_live()
        else:
            raw_sym = getattr(self.table_widget, "last_raw_sym_text", "") or ""
            old_sym = getattr(self.table_widget, "last_old_sym", None)
            self.table_widget.last_raw_sym_text = ""
            self.table_widget.last_old_sym = None
            if edited_row is not None and edited_attr == "symbol":
                frag_shortcuts = {
                    "-CO": "Carbonyl Ligand (-CO)",
                    "CO": "Carbonyl Ligand (-CO)" if raw_sym == "CO" else None,
                    "-CN": "Cyano (-CN)",
                    "CN": "Cyano (-CN)",
                    "-OH": "Hydroxyl (-OH)",
                    "OH": "Hydroxyl (-OH)",
                    "-CH3": "Methyl (-CH3)",
                    "CH3": "Methyl (-CH3)",
                    "-NH2": "Amino (-NH2)",
                    "NH2": "Amino (-NH2)",
                    "-NO2": "Nitro (-NO2)",
                    "NO2": "Nitro (-NO2)",
                }
                shortcut_frag_name = frag_shortcuts.get(raw_sym.strip()) or frag_shortcuts.get(raw_sym.strip().upper() if raw_sym.strip() != "Co" else "")
                if shortcut_frag_name and shortcut_frag_name in FRAGMENTS and old_sym is not None:
                    # Restore original symbol on edited_row so _attach_fragment_to_atom_or_orbit sees the pre-edit state
                    self.zatoms[edited_row].symbol = old_sym
                    frag = FRAGMENTS[shortcut_frag_name]
                    n_sites = self._attach_fragment_to_atom_or_orbit(
                        edited_row, frag, conn_idx=int(frag.get("connect_idx", 0)), relax_iters=120
                    )
                    self._update_adjust_panel()
                    sym_note = f" across {n_sites} symmetry-equivalent sites" if n_sites > 1 else ""
                    self.statusBar().showMessage(
                        f"Replaced {old_sym} with {shortcut_frag_name}{sym_note}!", 4500
                    )
                    return

                sym_res = self._get_active_imposed_symmetry()
                if sym_res is not None and old_sym is not None and len(self.coords) == len(self.zatoms):
                    new_sym = self.zatoms[edited_row].symbol
                    self.zatoms[edited_row].symbol = old_sym
                    atom_frag = {"name": new_sym, "symbols": [new_sym], "coords": [(0.0, 0.0, 0.0)], "connect_idx": 0}
                    n_sites = self._attach_fragment_to_atom_or_orbit(
                        edited_row, atom_frag, conn_idx=0, relax_iters=80
                    )
                    self._update_adjust_panel()
                    self.statusBar().showMessage(
                        f"Replaced {n_sites} symmetry-equivalent {old_sym} atoms with {new_sym}!", 4500
                    )
                    return

            self.update_geometry_live()
        # Only run automatic hydrogen adjustment if a symbol was changed to a real heavy element (not 'X' or 'H')
        if (
            edited_row is not None
            and edited_attr == "symbol"
            and self.zatoms[edited_row].symbol not in ("X", "H")
            and getattr(self, "chk_auto_h", None)
            and self.chk_auto_h.isChecked()
        ):
            curr_symbols = [a.symbol for a in self.zatoms]
            curr_coords = np.copy(self.coords)
            adj_symbols, adj_coords = adjust_hydrogens_to_valence(
                curr_symbols,
                curr_coords,
                target_atom_indices=[edited_row],
                user_bonds=list(self.cross_bonds),
            )
            if len(adj_symbols) != len(curr_symbols):
                new_zatoms = cartesian_to_zmatrix(
                    adj_symbols, adj_coords, existing_zatoms=self.zatoms, user_bonds=list(self.cross_bonds)
                )
                self.set_molecule_from_zatoms(new_zatoms)
                return
        self._update_adjust_panel()

    def _on_table_atom_selected(self, row: int):
        self.viewer.smartslide_selection_active = False
        self.viewer.selected_atom_idx = row
        self.viewer.multi_selected = [row]
        self.viewer.update()
        self._update_adjust_panel()

    def _on_table_cell_selected(self, row: int, col: int):
        """Highlight atoms involved in bond, angle, or dihedral when clicking corresponding box."""
        if row < 0 or row >= len(self.zatoms):
            return

        self.viewer.smartslide_selection_active = False

        atom = self.zatoms[row]
        feature_map = {row: "target"}
        msg = f"Atom {atom.symbol}{row+1} selected."

        # Col 2 or 3: Bond (Ref 1 or Length)
        if col in (ZMatrixTableWidget.COL_REF1, ZMatrixTableWidget.COL_DIST):
            if atom.ref1 is not None and atom.ref1 < len(self.zatoms):
                r1 = atom.ref1
                feature_map[r1] = "bond"
                d_val = f"{atom.dist:.3f} Å"
                r1_sym = f"{self.zatoms[r1].symbol}{r1+1}"
                t_sym = f"{atom.symbol}{row+1}"
                msg = f"Bond: {t_sym} - {r1_sym} (Length: {d_val})"

        # Col 4 or 5: Angle (Ref 2 or Angle deg)
        elif col in (ZMatrixTableWidget.COL_REF2, ZMatrixTableWidget.COL_ANG):
            if atom.ref1 is not None and atom.ref1 < len(self.zatoms):
                feature_map[atom.ref1] = "bond"
            if atom.ref2 is not None and atom.ref2 < len(self.zatoms):
                feature_map[atom.ref2] = "angle"
                r1 = atom.ref1
                r2 = atom.ref2
                r1_sym = f"{self.zatoms[r1].symbol}{r1+1}" if r1 is not None else ""
                r2_sym = f"{self.zatoms[r2].symbol}{r2+1}"
                t_sym = f"{atom.symbol}{row+1}"
                msg = f"Angle: {t_sym} - {r1_sym} - {r2_sym} ({atom.angle:.2f}°)"

        # Col 6 or 7: Dihedral (Ref 3 or Dihedral deg)
        elif col in (ZMatrixTableWidget.COL_REF3, ZMatrixTableWidget.COL_DIH):
            if atom.ref1 is not None and atom.ref1 < len(self.zatoms):
                feature_map[atom.ref1] = "bond"
            if atom.ref2 is not None and atom.ref2 < len(self.zatoms):
                feature_map[atom.ref2] = "angle"
            if atom.ref3 is not None and atom.ref3 < len(self.zatoms):
                feature_map[atom.ref3] = "dihedral"
                r1 = atom.ref1
                r2 = atom.ref2
                r3 = atom.ref3
                r1_sym = f"{self.zatoms[r1].symbol}{r1+1}" if r1 is not None else ""
                r2_sym = f"{self.zatoms[r2].symbol}{r2+1}" if r2 is not None else ""
                r3_sym = f"{self.zatoms[r3].symbol}{r3+1}"
                t_sym = f"{atom.symbol}{row+1}"
                msg = f"Dihedral: {t_sym} - {r1_sym} - {r2_sym} - {r3_sym} ({atom.dihedral:.2f}°)"

        self.viewer.highlighted_feature_atoms = feature_map
        self.viewer.selected_atom_idx = row
        self.viewer.multi_selected = list(feature_map.keys())
        self.viewer.update()
        self._update_adjust_panel()
        self.statusBar().showMessage(msg, 6000)

    def _on_viewport_atom_clicked(self, atom_idx: int, button: QtCore.Qt.MouseButton):
        self.table_widget.highlight_row(atom_idx)
        self._update_adjust_panel()

    def _on_measurement_updated(self, msg: str):
        self.lbl_measurement.setText(msg)
        self.statusBar().showMessage(msg, 5000)
        self._update_adjust_panel()
        self._update_live_geom_spinbox()

    def _on_toggle_interaction_mode(self, checked: bool):
        """Toggle between Build mode (False -> 'build') and Inspect mode (True -> 'inspect')."""
        self.set_interaction_mode("inspect" if checked else "build")

    def set_interaction_mode(self, mode: str):
        """Set viewport interaction mode to 'build' or 'inspect' and sync the ribbon button."""
        mode = "inspect" if mode == "inspect" else "build"
        self.viewer.interaction_mode = mode
        if hasattr(self, "btn_mode_toggle"):
            self.btn_mode_toggle.blockSignals(True)
            if mode == "inspect":
                self.btn_mode_toggle.setChecked(True)
                self.btn_mode_toggle.setText("🔍 Inspect")
                self.btn_mode_toggle.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 2px 8px;")
            else:
                self.btn_mode_toggle.setChecked(False)
                self.btn_mode_toggle.setText("✏️ Build")
                self.btn_mode_toggle.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 2px 8px;")
            self.btn_mode_toggle.blockSignals(False)
        self.statusBar().showMessage(
            "🔍 Inspect / Measure Mode: Clicking atoms selects/measures; dragging orbits the camera."
            if mode == "inspect"
            else "✏️ Build / Draw Mode: Dragging from an atom sprouts a new bond/atom; double-clicking mutates/attaches.",
            3500,
        )

    def _update_live_geom_spinbox(self):
        """Show/sync the inline live Distance / Angle / Dihedral spinbox next to the measurement bar when 2, 3, or 4 atoms are selected."""
        if not hasattr(self, "spin_live_geom") or not hasattr(self, "lbl_live_geom"):
            return
        n = len(self.zatoms)
        sel = [i for i in getattr(self.viewer, "multi_selected", []) if 0 <= i < n]
        k = len(sel)
        self.spin_live_geom.blockSignals(True)
        if k == 2:
            d_val = calculate_bond_distance(self.coords[sel[0]], self.coords[sel[1]])
            self.lbl_live_geom.setText("Set d:")
            self.spin_live_geom.setRange(0.10, 15.00)
            self.spin_live_geom.setSingleStep(0.02)
            self.spin_live_geom.setDecimals(3)
            self.spin_live_geom.setSuffix(" Å")
            self.spin_live_geom.setValue(d_val)
            self.lbl_live_geom.setVisible(True)
            self.spin_live_geom.setVisible(True)
        elif k == 3:
            a_val = calculate_angle(self.coords[sel[0]], self.coords[sel[1]], self.coords[sel[2]])
            self.lbl_live_geom.setText("Set ∠:")
            self.spin_live_geom.setRange(0.0, 180.0)
            self.spin_live_geom.setSingleStep(1.0)
            self.spin_live_geom.setDecimals(2)
            self.spin_live_geom.setSuffix("°")
            self.spin_live_geom.setValue(a_val)
            self.lbl_live_geom.setVisible(True)
            self.spin_live_geom.setVisible(True)
        elif k == 4:
            dih_val = calculate_dihedral(self.coords[sel[0]], self.coords[sel[1]], self.coords[sel[2]], self.coords[sel[3]])
            self.lbl_live_geom.setText("Set φ:")
            self.spin_live_geom.setRange(-180.0, 180.0)
            self.spin_live_geom.setSingleStep(2.0)
            self.spin_live_geom.setDecimals(2)
            self.spin_live_geom.setSuffix("°")
            self.spin_live_geom.setValue(dih_val)
            self.lbl_live_geom.setVisible(True)
            self.spin_live_geom.setVisible(True)
        else:
            self.lbl_live_geom.setVisible(False)
            self.spin_live_geom.setVisible(False)
        if hasattr(self, "btn_scan_selected_coord"):
            self.btn_scan_selected_coord.setVisible(k in (2, 3, 4))
        self.spin_live_geom.blockSignals(False)

    def _on_live_geom_spin_changed(self, val: float):
        """Apply direct Distance (2 atoms), Angle (3 atoms), or Dihedral (4 atoms) edit from the inline spinbox."""
        n = len(self.zatoms)
        sel = [i for i in getattr(self.viewer, "multi_selected", []) if 0 <= i < n]
        self.viewer.smartslide_selection_active = len(sel) >= 2
        if len(sel) == 2:
            self._on_spin_bond_changed(val)
        elif len(sel) == 3:
            self._on_spin_angle_changed(val)
        elif len(sel) == 4:
            self._on_spin_dihedral_changed(val)

    def _update_qm_summary_banner(self, res):
        """Populate and display the GaussView-style QM Calculation Summary banner."""
        if not hasattr(self, "grp_qm_summary") or not hasattr(self, "lbl_qm_summary"):
            return
        parts_html = [f"<b>{res.program_name} Summary:</b>"]
        parts_tsv = [f"Program: {res.program_name}"]

        if res.trajectory_energies:
            e_ev = res.trajectory_energies[-1]
            e_ha = e_ev / 27.211386245988
            parts_html.append(f"<b>E:</b> {e_ha:.6f} Ha ({e_ev:.3f} eV)")
            parts_tsv.append(f"E_Ha={e_ha:.8f}\tE_eV={e_ev:.4f}")

        homo = getattr(res, "homo_ev", None)
        lumo = getattr(res, "lumo_ev", None)
        if homo is not None and lumo is not None:
            gap = lumo - homo
            parts_html.append(f"<b>Gap:</b> {gap:.2f} eV (HOMO {homo:.2f}, LUMO {lumo:.2f})")
            parts_tsv.append(f"HOMO_eV={homo:.4f}\tLUMO_eV={lumo:.4f}\tGap_eV={gap:.4f}")

        dip = getattr(res, "dipole_debye", None)
        if dip is not None:
            parts_html.append(f"<b>|μ|:</b> {dip:.3f} D")
            parts_tsv.append(f"Dipole_D={dip:.4f}")

        if res.vibrational_analysis and res.vibrational_analysis.frequencies:
            freqs = res.vibrational_analysis.frequencies
            imag = [f for f in freqs if f < 0.0]
            if imag:
                parts_html.append(f"<span style='color:#f87171;'><b>⚠️ {len(imag)} Imaginary ({imag[0]:.1f} cm⁻¹)</b></span>")
                parts_tsv.append(f"Imaginary_Modes={len(imag)} ({imag[0]:.2f} cm-1)")
            else:
                parts_html.append(f"<span style='color:#4ade80;'><b>✅ 0 Imaginary ({len(freqs)} modes)</b></span>")
                parts_tsv.append(f"Imaginary_Modes=0")

        thermo_bits = []
        for label, attr in (("ZPE", "zpe_ha"), ("H", "enthalpy_ha"), ("G", "gibbs_ha")):
            v = getattr(res, attr, None)
            if v is not None:
                thermo_bits.append(f"{label}={v:.6f} Ha")
                parts_tsv.append(f"{label}_Ha={v:.8f}")
        if thermo_bits:
            parts_html.append("<b>Thermo:</b> " + ", ".join(thermo_bits))

        self._last_qm_summary_tsv = "\t".join(parts_tsv)
        self.lbl_qm_summary.setText(" &nbsp;|&nbsp; ".join(parts_html))
        self.grp_qm_summary.setVisible(True)

    def _on_copy_qm_summary(self):
        txt = getattr(self, "_last_qm_summary_tsv", "") or self.lbl_qm_summary.text()
        QtWidgets.QApplication.clipboard().setText(txt)
        self.statusBar().showMessage("Copied QM Calculation Summary to clipboard!", 3500)

    def _get_smart_mode(self) -> str:
        if hasattr(self, "combo_smart_mode") and self.combo_smart_mode.currentIndex() >= 0:
            return str(self.combo_smart_mode.currentData() or "group")
        return "group"

    def _get_table_feature_selection(self) -> Tuple[Optional[str], List[int]]:
        """
        If the user selected a Bond, Angle, or Dihedral cell in the Z-Matrix table,
        return (feature_type, [i1, i2, ...]) where the atom indices are ordered so that
        the target atom (row) is the moving atom (last index in the tuple).
        """
        fmap = getattr(self.viewer, "highlighted_feature_atoms", None) or {}
        if len(fmap) < 2:
            return None, []
        n = len(self.zatoms)
        target = next((k for k, v in fmap.items() if v == "target" and 0 <= k < n), None)
        bond = next((k for k, v in fmap.items() if v == "bond" and 0 <= k < n), None)
        angle = next((k for k, v in fmap.items() if v == "angle" and 0 <= k < n), None)
        dihedral = next((k for k, v in fmap.items() if v == "dihedral" and 0 <= k < n), None)
        if target is not None and bond is not None and angle is not None and dihedral is not None:
            return "dihedral", [dihedral, angle, bond, target]
        if target is not None and bond is not None and angle is not None:
            return "angle", [angle, bond, target]
        if target is not None and bond is not None:
            return "bond", [bond, target]
        return None, []

    def _get_smartslide_selection(self) -> List[int]:
        """
        Return the active 2/3/4-atom SmartSlides selection ONLY if `multi_selected` is an explicit
        viewport multi-selection (and not a table feature highlight).
        """
        if getattr(self.viewer, "highlighted_feature_atoms", {}):
            return []
        n = len(self.zatoms)
        sel = [i for i in getattr(self.viewer, "multi_selected", []) if 0 <= i < n]
        idx = self.viewer.selected_atom_idx
        if len(sel) >= 2 and (idx is None or idx == sel[0] or getattr(self.viewer, "smartslide_selection_active", False)):
            return sel
        return []

    def _update_adjust_panel(self):
        # Update Z-Matrix Symbolic Variable Sliders (ZOPT Diagnostic)
        self._update_variable_sliders()

        n = len(self.zatoms)
        sel = self._get_smartslide_selection()
        idx = self.viewer.selected_atom_idx

        if n == 0 or (idx is None and not sel):
            self.grp_adjust.setEnabled(False)
            return

        self.grp_adjust.setEnabled(True)

        # Block signals during programmatic update
        self.spin_bond.blockSignals(True)
        self.slider_bond.blockSignals(True)
        self.spin_angle.blockSignals(True)
        self.slider_angle.blockSignals(True)
        self.spin_dihedral.blockSignals(True)
        self.slider_dihedral.blockSignals(True)

        # If 2, 3, or 4 atoms are explicitly multi-selected in the 3D viewport (without table feature highlight),
        # populate SmartSlides with their exact 3D distance, angle, and dihedral!
        if len(sel) >= 2:
            d_val = calculate_bond_distance(self.coords[sel[0]], self.coords[sel[1]])
            self.spin_bond.setEnabled(True)
            self.slider_bond.setEnabled(True)
            self.spin_bond.setValue(d_val)
            self.slider_bond.setValue(int(np.clip(d_val * 100, 50, 400)))

            if len(sel) >= 3:
                a_val = calculate_angle(self.coords[sel[0]], self.coords[sel[1]], self.coords[sel[2]])
                self.spin_angle.setEnabled(True)
                self.slider_angle.setEnabled(True)
                self.spin_angle.setValue(a_val)
                self.slider_angle.setValue(int(np.clip(a_val * 10, 0, 1800)))
            else:
                self.spin_angle.setEnabled(False)
                self.slider_angle.setEnabled(False)
                self.spin_angle.setValue(0.0)

            if len(sel) >= 4:
                dih_val = calculate_dihedral(
                    self.coords[sel[0]], self.coords[sel[1]], self.coords[sel[2]], self.coords[sel[3]]
                )
                self.spin_dihedral.setEnabled(True)
                self.slider_dihedral.setEnabled(True)
                self.spin_dihedral.setValue(dih_val)
                self.slider_dihedral.setValue(int(np.clip(dih_val * 10, -1800, 1800)))
            else:
                self.spin_dihedral.setEnabled(False)
                self.slider_dihedral.setEnabled(False)
                self.spin_dihedral.setValue(0.0)

            self.spin_bond.blockSignals(False)
            self.slider_bond.blockSignals(False)
            self.spin_angle.blockSignals(False)
            self.slider_angle.blockSignals(False)
            self.spin_dihedral.blockSignals(False)
            self.slider_dihedral.blockSignals(False)
            return

        if idx is None or idx < 0 or idx >= n:
            idx = sel[0]
        atom = self.zatoms[idx]

        if idx >= 1:
            self.spin_bond.setEnabled(True)
            self.slider_bond.setEnabled(True)
            self.spin_bond.setValue(atom.dist)
            self.slider_bond.setValue(int(atom.dist * 100))
        else:
            self.spin_bond.setEnabled(False)
            self.slider_bond.setEnabled(False)
            self.spin_bond.setValue(0.0)

        if idx >= 2:
            self.spin_angle.setEnabled(True)
            self.slider_angle.setEnabled(True)
            self.spin_angle.setValue(atom.angle)
            self.slider_angle.setValue(int(atom.angle * 10))
        else:
            self.spin_angle.setEnabled(False)
            self.slider_angle.setEnabled(False)
            self.spin_angle.setValue(0.0)

        if idx >= 3:
            self.spin_dihedral.setEnabled(True)
            self.slider_dihedral.setEnabled(True)
            self.spin_dihedral.setValue(atom.dihedral)
            self.slider_dihedral.setValue(int(atom.dihedral * 10))
        else:
            self.spin_dihedral.setEnabled(False)
            self.slider_dihedral.setEnabled(False)
            self.spin_dihedral.setValue(0.0)

        self.spin_bond.blockSignals(False)
        self.slider_bond.blockSignals(False)
        self.spin_angle.blockSignals(False)
        self.slider_angle.blockSignals(False)
        self.spin_dihedral.blockSignals(False)
        self.slider_dihedral.blockSignals(False)

    def _update_variable_sliders(self):
        """Populate interactive parameter sliders for active symbolic Z-matrix variables (ZOPT style)."""
        if not hasattr(self, "grp_var_sliders") or not hasattr(self, "var_sliders_layout"):
            return

        # Clear existing slider widgets
        while self.var_sliders_layout.count():
            item = self.var_sliders_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not self.variables:
            self.grp_var_sliders.setVisible(False)
            return

        self.grp_var_sliders.setVisible(True)
        # Classify variables by coordinate type: R (bonds), A (angles), D (dihedrals)
        for row, (vname, val) in enumerate(sorted(self.variables.items())):
            lbl = QtWidgets.QLabel(f"<b>{vname}</b>:")
            lbl.setFixedWidth(45)
            self.var_sliders_layout.addWidget(lbl, row, 0)

            spin = QtWidgets.QDoubleSpinBox()
            spin.setFixedWidth(75)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)

            is_angle = vname.upper().startswith("A") or "ANG" in vname.upper()
            is_dihedral = vname.upper().startswith("D") or "DIH" in vname.upper()

            if is_angle:
                spin.setRange(0.0, 180.0)
                spin.setSingleStep(0.5)
                spin.setValue(float(val))
                spin.setSuffix("°")
                slider.setRange(0, 1800)
                slider.setValue(int(np.clip(val * 10, 0, 1800)))
            elif is_dihedral:
                spin.setRange(-180.0, 180.0)
                spin.setSingleStep(1.0)
                spin.setValue(float(val))
                spin.setSuffix("°")
                slider.setRange(-1800, 1800)
                slider.setValue(int(np.clip(val * 10, -1800, 1800)))
            else: # Distance / bond variable
                spin.setRange(0.2, 8.0)
                spin.setSingleStep(0.02)
                spin.setValue(float(val))
                spin.setSuffix(" Å")
                slider.setRange(20, 800)
                slider.setValue(int(np.clip(val * 100, 20, 800)))

            def make_callbacks(var_name, is_ang, is_dih, s_box, s_slider):
                def on_spin_changed(new_v):
                    self._push_undo_state()
                    self.variables[var_name] = new_v
                    from .zmatrix import apply_variables
                    apply_variables(self.zatoms, self.variables)
                    s_slider.blockSignals(True)
                    mult = 10 if (is_ang or is_dih) else 100
                    s_slider.setValue(int(new_v * mult))
                    s_slider.blockSignals(False)
                    self.table_widget.load_zatoms(self.zatoms)
                    self.update_geometry_live()

                def on_slider_changed(slider_val):
                    mult = 10.0 if (is_ang or is_dih) else 100.0
                    new_v = slider_val / mult
                    self.variables[var_name] = new_v
                    from .zmatrix import apply_variables
                    apply_variables(self.zatoms, self.variables)
                    s_box.blockSignals(True)
                    s_box.setValue(new_v)
                    s_box.blockSignals(False)
                    self.table_widget.load_zatoms(self.zatoms)
                    self.update_geometry_live()

                return on_spin_changed, on_slider_changed

            cb_spin, cb_slider = make_callbacks(vname, is_angle, is_dihedral, spin, slider)
            spin.valueChanged.connect(cb_spin)
            slider.sliderPressed.connect(self._push_undo_state)
            slider.valueChanged.connect(cb_slider)

            self.var_sliders_layout.addWidget(spin, row, 1)
            self.var_sliders_layout.addWidget(slider, row, 2)

    def _modify_distance_with_imposed_symmetry(
        self, i1: int, i2: int, target_dist: float, mode: str
    ) -> np.ndarray:
        """
        Modify distance `(i1, i2)` to `target_dist`, and if `symmetry_imposed` is enabled,
        propagate the same bond-distance modification across all symmetry-equivalent bonds
        in the orbit of `i2`.
        """
        from .smartslides import modify_distance
        from .symmetry import get_atom_orbit_and_operations, map_atom_by_operation
        symbols = [a.symbol for a in self.zatoms]
        bonds = self._current_explicit_bonds()
        sym_res = self._get_active_imposed_symmetry()
        if sym_res is None:
            return modify_distance(symbols, self.coords, bonds, i1, i2, target_dist, mode=mode)

        orig_coords = np.copy(self.coords)
        com, orbit_ops = get_atom_orbit_and_operations(
            symbols, orig_coords, i2, result=sym_res, tolerance=max(0.25, sym_res.tolerance * 1.5)
        )
        new_c = np.copy(orig_coords)
        seen_pairs = set()
        for eq_i2, R_op in orbit_ops:
            eq_i1 = map_atom_by_operation(symbols, orig_coords, i1, R_op, com=com)
            pair = (eq_i1, eq_i2)
            if pair in seen_pairs or eq_i1 == eq_i2:
                continue
            seen_pairs.add(pair)
            new_c = modify_distance(symbols, new_c, bonds, eq_i1, eq_i2, target_dist, mode="group")
        return new_c

    def _apply_smartslide_coords(self, new_coords: np.ndarray, keep_selection: List[int]):
        """Update Z-matrix and 3D coordinates from a SmartSlides 2/3/4-atom modification."""
        prev_idx = self.viewer.selected_atom_idx
        symbols = [a.symbol for a in self.zatoms]
        user_bonds = self._current_explicit_bonds()
        old_bo = dict(getattr(self.viewer, "bond_orders", {}))
        new_zatoms = cartesian_to_zmatrix(
            symbols, new_coords, existing_zatoms=self.zatoms, user_bonds=user_bonds
        )
        self.zatoms = new_zatoms
        self.coords = np.asarray(new_coords, dtype=np.float64)
        self._rigid_frame = capture_rigid_frame(self.zatoms, self.coords)
        self.viewer.update_coords(self.coords, user_bonds=user_bonds)
        if old_bo:
            self.viewer.bond_orders = old_bo
        self.viewer.multi_selected = list(keep_selection)
        if prev_idx is not None and prev_idx in keep_selection:
            self.viewer.selected_atom_idx = prev_idx
        else:
            self.viewer.selected_atom_idx = keep_selection[-1] if keep_selection else None
        self.table_widget.load_zatoms(self.zatoms)
        self._sync_raw_text()
        self.viewer.update()

    def _propagate_var(self, idx: int, val_attr: str):
        """If coordinate val_attr of atom idx uses a variable, set the variable and all linked coords."""
        if not self.variables:
            return
        atom = self.zatoms[idx]
        base, sgn = _split_var(getattr(atom, val_attr + "_var"))
        if base in self.variables:
            self.variables[base] = sgn * getattr(atom, val_attr)
            apply_variables(self.zatoms, self.variables)

    def _on_spin_bond_changed(self, val: float):
        sel = self._get_smartslide_selection()
        if len(sel) >= 2:
            self._push_undo_state()
            new_c = self._modify_distance_with_imposed_symmetry(sel[0], sel[1], val, mode=self._get_smart_mode())
            self.slider_bond.blockSignals(True)
            self.slider_bond.setValue(int(np.clip(val * 100, 50, 400)))
            self.slider_bond.blockSignals(False)
            self._apply_smartslide_coords(new_c, sel)
            return
        idx = self.viewer.selected_atom_idx
        if idx is not None and idx >= 1:
            self._push_undo_state()
            atom = self.zatoms[idx]
            mode = self._get_smart_mode()
            sym_res = self._get_active_imposed_symmetry()
            base, _ = _split_var(atom.dist_var)
            if not (base and base in self.variables) and (mode in ("group", "symmetric") or sym_res is not None) and atom.ref1 is not None:
                new_c = self._modify_distance_with_imposed_symmetry(atom.ref1, idx, val, mode=mode)
                self.slider_bond.blockSignals(True)
                self.slider_bond.setValue(int(np.clip(val * 100, 50, 400)))
                self.slider_bond.blockSignals(False)
                self._apply_smartslide_coords(new_c, list(self.viewer.multi_selected) or [idx])
                return
            atom.dist = val
            self._propagate_var(idx, "dist")
            self.slider_bond.blockSignals(True)
            self.slider_bond.setValue(int(val * 100))
            self.slider_bond.blockSignals(False)
            self.table_widget.load_zatoms(self.zatoms)
            self.update_geometry_live()

    def _on_slider_bond_changed(self, val: int):
        f_val = val / 100.0
        sel = self._get_smartslide_selection()
        if len(sel) >= 2:
            new_c = self._modify_distance_with_imposed_symmetry(sel[0], sel[1], f_val, mode=self._get_smart_mode())
            self.spin_bond.blockSignals(True)
            self.spin_bond.setValue(f_val)
            self.spin_bond.blockSignals(False)
            self._apply_smartslide_coords(new_c, sel)
            return
        idx = self.viewer.selected_atom_idx
        if idx is not None and idx >= 1:
            atom = self.zatoms[idx]
            mode = self._get_smart_mode()
            sym_res = self._get_active_imposed_symmetry()
            base, _ = _split_var(atom.dist_var)
            if not (base and base in self.variables) and (mode in ("group", "symmetric") or sym_res is not None) and atom.ref1 is not None:
                new_c = self._modify_distance_with_imposed_symmetry(atom.ref1, idx, f_val, mode=mode)
                self.spin_bond.blockSignals(True)
                self.spin_bond.setValue(f_val)
                self.spin_bond.blockSignals(False)
                self._apply_smartslide_coords(new_c, list(self.viewer.multi_selected) or [idx])
                return
            atom.dist = f_val
            self._propagate_var(idx, "dist")
            self.spin_bond.blockSignals(True)
            self.spin_bond.setValue(f_val)
            self.spin_bond.blockSignals(False)
            self.table_widget.load_zatoms(self.zatoms)
            self.update_geometry_live()

    def _on_spin_angle_changed(self, val: float):
        sel = self._get_smartslide_selection()
        if len(sel) >= 3:
            from .smartslides import modify_angle
            self._push_undo_state()
            symbols = [a.symbol for a in self.zatoms]
            bonds = self._current_explicit_bonds()
            new_c = modify_angle(symbols, self.coords, bonds, sel[0], sel[1], sel[2], val, mode=self._get_smart_mode())
            self.slider_angle.blockSignals(True)
            self.slider_angle.setValue(int(np.clip(val * 10, 0, 1800)))
            self.slider_angle.blockSignals(False)
            self._apply_smartslide_coords(new_c, sel)
            return
        idx = self.viewer.selected_atom_idx
        if idx is not None and idx >= 2:
            self._push_undo_state()
            atom = self.zatoms[idx]
            mode = self._get_smart_mode()
            base, _ = _split_var(atom.angle_var)
            if (
                not (base and base in self.variables)
                and mode in ("group", "symmetric")
                and atom.ref1 is not None
                and atom.ref2 is not None
            ):
                from .smartslides import modify_angle
                symbols = [a.symbol for a in self.zatoms]
                bonds = self._current_explicit_bonds()
                new_c = modify_angle(symbols, self.coords, bonds, atom.ref2, atom.ref1, idx, val, mode=mode)
                self.slider_angle.blockSignals(True)
                self.slider_angle.setValue(int(np.clip(val * 10, 0, 1800)))
                self.slider_angle.blockSignals(False)
                self._apply_smartslide_coords(new_c, list(self.viewer.multi_selected) or [idx])
                return
            atom.angle = val
            self._propagate_var(idx, "angle")
            self.slider_angle.blockSignals(True)
            self.slider_angle.setValue(int(val * 10))
            self.slider_angle.blockSignals(False)
            self.table_widget.load_zatoms(self.zatoms)
            self.update_geometry_live()

    def _on_slider_angle_changed(self, val: int):
        f_val = val / 10.0
        sel = self._get_smartslide_selection()
        if len(sel) >= 3:
            from .smartslides import modify_angle
            symbols = [a.symbol for a in self.zatoms]
            bonds = self._current_explicit_bonds()
            new_c = modify_angle(symbols, self.coords, bonds, sel[0], sel[1], sel[2], f_val, mode=self._get_smart_mode())
            self.spin_angle.blockSignals(True)
            self.spin_angle.setValue(f_val)
            self.spin_angle.blockSignals(False)
            self._apply_smartslide_coords(new_c, sel)
            return
        idx = self.viewer.selected_atom_idx
        if idx is not None and idx >= 2:
            atom = self.zatoms[idx]
            mode = self._get_smart_mode()
            base, _ = _split_var(atom.angle_var)
            if (
                not (base and base in self.variables)
                and mode in ("group", "symmetric")
                and atom.ref1 is not None
                and atom.ref2 is not None
            ):
                from .smartslides import modify_angle
                symbols = [a.symbol for a in self.zatoms]
                bonds = self._current_explicit_bonds()
                new_c = modify_angle(symbols, self.coords, bonds, atom.ref2, atom.ref1, idx, f_val, mode=mode)
                self.spin_angle.blockSignals(True)
                self.spin_angle.setValue(f_val)
                self.spin_angle.blockSignals(False)
                self._apply_smartslide_coords(new_c, list(self.viewer.multi_selected) or [idx])
                return
            atom.angle = f_val
            self._propagate_var(idx, "angle")
            self.spin_angle.blockSignals(True)
            self.spin_angle.setValue(f_val)
            self.spin_angle.blockSignals(False)
            self.table_widget.load_zatoms(self.zatoms)
            self.update_geometry_live()

    def _on_spin_dihedral_changed(self, val: float):
        sel = self._get_smartslide_selection()
        if len(sel) >= 4:
            from .smartslides import modify_dihedral
            self._push_undo_state()
            symbols = [a.symbol for a in self.zatoms]
            bonds = self._current_explicit_bonds()
            new_c = modify_dihedral(
                symbols, self.coords, bonds, sel[0], sel[1], sel[2], sel[3], val, mode=self._get_smart_mode()
            )
            self.slider_dihedral.blockSignals(True)
            self.slider_dihedral.setValue(int(np.clip(val * 10, -1800, 1800)))
            self.slider_dihedral.blockSignals(False)
            self._apply_smartslide_coords(new_c, sel)
            return
        idx = self.viewer.selected_atom_idx
        if idx is not None and idx >= 3:
            self._push_undo_state()
            atom = self.zatoms[idx]
            mode = self._get_smart_mode()
            base, _ = _split_var(atom.dihedral_var)
            if (
                not (base and base in self.variables)
                and mode in ("group", "symmetric")
                and atom.ref1 is not None
                and atom.ref2 is not None
                and atom.ref3 is not None
            ):
                from .smartslides import modify_dihedral
                symbols = [a.symbol for a in self.zatoms]
                bonds = self._current_explicit_bonds()
                new_c = modify_dihedral(
                    symbols, self.coords, bonds, atom.ref3, atom.ref2, atom.ref1, idx, val, mode=mode
                )
                self.slider_dihedral.blockSignals(True)
                self.slider_dihedral.setValue(int(np.clip(val * 10, -1800, 1800)))
                self.slider_dihedral.blockSignals(False)
                self._apply_smartslide_coords(new_c, list(self.viewer.multi_selected) or [idx])
                return
            atom.dihedral = val
            self._propagate_var(idx, "dihedral")
            self.slider_dihedral.blockSignals(True)
            self.slider_dihedral.setValue(int(val * 10))
            self.slider_dihedral.blockSignals(False)
            self.table_widget.load_zatoms(self.zatoms)
            self.update_geometry_live()

    def _on_slider_dihedral_changed(self, val: int):
        f_val = val / 10.0
        sel = self._get_smartslide_selection()
        if len(sel) >= 4:
            from .smartslides import modify_dihedral
            symbols = [a.symbol for a in self.zatoms]
            bonds = self._current_explicit_bonds()
            new_c = modify_dihedral(
                symbols, self.coords, bonds, sel[0], sel[1], sel[2], sel[3], f_val, mode=self._get_smart_mode()
            )
            self.spin_dihedral.blockSignals(True)
            self.spin_dihedral.setValue(f_val)
            self.spin_dihedral.blockSignals(False)
            self._apply_smartslide_coords(new_c, sel)
            return
        idx = self.viewer.selected_atom_idx
        if idx is not None and idx >= 3:
            atom = self.zatoms[idx]
            mode = self._get_smart_mode()
            base, _ = _split_var(atom.dihedral_var)
            if (
                not (base and base in self.variables)
                and mode in ("group", "symmetric")
                and atom.ref1 is not None
                and atom.ref2 is not None
                and atom.ref3 is not None
            ):
                from .smartslides import modify_dihedral
                symbols = [a.symbol for a in self.zatoms]
                bonds = self._current_explicit_bonds()
                new_c = modify_dihedral(
                    symbols, self.coords, bonds, atom.ref3, atom.ref2, atom.ref1, idx, f_val, mode=mode
                )
                self.spin_dihedral.blockSignals(True)
                self.spin_dihedral.setValue(f_val)
                self.spin_dihedral.blockSignals(False)
                self._apply_smartslide_coords(new_c, list(self.viewer.multi_selected) or [idx])
                return
            atom.dihedral = f_val
            self._propagate_var(idx, "dihedral")
            self.spin_dihedral.blockSignals(True)
            self.spin_dihedral.setValue(f_val)
            self.spin_dihedral.blockSignals(False)
            self.table_widget.load_zatoms(self.zatoms)
            self.update_geometry_live()

    def _on_raw_zmat_text_changed(self):
        """Called whenever user types in the Raw ZMAT text editor."""
        if hasattr(self, "chk_auto_apply_raw") and self.chk_auto_apply_raw.isChecked():
            if hasattr(self, "_raw_zmat_timer"):
                self._raw_zmat_timer.start()

    def _on_auto_apply_text_zmat(self):
        """Silently apply parsed raw Z-matrix text to 3D model without popups if valid."""
        text = self.text_raw_zmat.toPlainText().strip()
        if not text:
            return
        try:
            zatoms, vars_dict = parse_zmatrix_text(text)
            if not zatoms:
                return
            # Check if atom count and symbols match current zatoms to preserve smooth live updates
            self.variables = vars_dict
            self._push_undo_state()
            self.set_molecule_from_zatoms(zatoms)
            self.statusBar().showMessage("Auto-applied Z-matrix text updates", 1500)
        except Exception as e:
            # While typing, syntax may be momentarily incomplete - do not disturb the user with error boxes
            log.debug("Debounced live Z-matrix parse incomplete: %s", e)

    def _on_apply_text_zmat(self):
        text = self.text_raw_zmat.toPlainText()
        try:
            zatoms, vars_dict = parse_zmatrix_text(text)
            if not zatoms:
                QtWidgets.QMessageBox.warning(self, "Invalid Z-Matrix", "No valid atom definitions parsed.")
                return
            self._push_undo_state()
            self.variables = vars_dict
            self.set_molecule_from_zatoms(zatoms)
            self.statusBar().showMessage("Applied text Z-matrix successfully", 3000)
        except Exception as e:
            log.exception("Unhandled error in GUI action")
            QtWidgets.QMessageBox.critical(self, "Parsing Error", f"Failed to parse Z-matrix:\n{str(e)}")

    def _on_raw_xyz_text_changed(self):
        """Called whenever user types or pastes into the Cartesian XYZ text editor."""
        if hasattr(self, "chk_auto_apply_xyz") and self.chk_auto_apply_xyz.isChecked():
            if hasattr(self, "_raw_xyz_timer"):
                self._raw_xyz_timer.start()

    def _on_auto_apply_text_xyz(self):
        """Silently auto-apply parsed XYZ text to 3D model without disruptive popups while typing."""
        if not hasattr(self, "text_raw_xyz"):
            return
        text = self.text_raw_xyz.toPlainText().strip()
        if not text:
            return
        try:
            from .zmatrix import parse_xyz
            syms, coords = parse_xyz(text)
            if syms and len(coords) == len(syms):
                from .controllers.import_controller import ImportController
                ImportController(self).load_xyz_data(syms, coords, source_label="Cartesian XYZ")
                self.statusBar().showMessage(f"Auto-applied {len(syms)} Cartesian atoms", 1500)
        except Exception as e:
            log.debug("Debounced live XYZ parse incomplete: %s", e)

    def _on_apply_text_xyz(self):
        """Explicitly apply Cartesian XYZ text with full validation and error reporting (Ctrl+Enter)."""
        if not hasattr(self, "text_raw_xyz"):
            return
        text = self.text_raw_xyz.toPlainText().strip()
        if not text:
            QtWidgets.QMessageBox.information(self, "Empty Input", "Please enter or paste Cartesian XYZ coordinates.")
            return
        try:
            from .zmatrix import parse_xyz
            syms, coords = parse_xyz(text)
            if not syms:
                QtWidgets.QMessageBox.warning(self, "Invalid XYZ", "Could not parse valid Cartesian XYZ atoms from text.")
                return
            from .controllers.import_controller import ImportController
            ImportController(self).load_xyz_data(syms, coords, source_label="Cartesian XYZ")
            self.statusBar().showMessage(f"Applied {len(syms)} Cartesian atoms successfully", 3000)
        except Exception as e:
            log.exception("Failed to parse XYZ text")
            QtWidgets.QMessageBox.critical(self, "Parsing Error", f"Failed to parse Cartesian XYZ:\n{str(e)}")

    def _on_xyz_tab_paste_clipboard(self):
        """Paste system clipboard text into the Cartesian XYZ editor tab."""
        clip_txt = QtWidgets.QApplication.clipboard().text()
        if clip_txt and hasattr(self, "text_raw_xyz"):
            self.text_raw_xyz.setPlainText(clip_txt)
            self._on_apply_text_xyz()

    def _on_style_changed(self, style_name: str):
        self.viewer.style_mode = style_name
        self.viewer.update()

    def _on_theme_changed(self, index: int):
        theme_data = self.combo_theme.itemData(index) or "dark"
        self.set_color_theme(theme_data)

    def set_color_theme(self, theme: str):
        """Switch between dark and white themes for the entire UI, dialogs, and 3D viewport."""
        theme = theme.lower()
        app = QtWidgets.QApplication.instance()
        pal = QtGui.QPalette()
        if theme == "white":
            # Switch icon theme to a light-compatible theme so standard icons (file dialogs, toolbuttons) are visible
            for light_icon_theme in ("breeze", "Breeze_Light", "Adwaita", "Tango", "hicolor"):
                if QtGui.QIcon.hasThemeIcon("go-previous") or QtGui.QIcon.themeName() != light_icon_theme:
                    QtGui.QIcon.setThemeName(light_icon_theme)
                    if not QtGui.QIcon.fromTheme("go-previous").isNull():
                        break

            qss = LIGHT_THEME_QSS
            self.viewer.bg_color_mode = "white"
            pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#f8fafc"))
            pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#0f172a"))
            pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#ffffff"))
            pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#f1f5f9"))
            pal.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#ffffff"))
            pal.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#0f172a"))
            pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#0f172a"))
            pal.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#ffffff"))
            pal.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#0f172a"))
            pal.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ffffff"))
            pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#2563eb"))
            pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
            pal.setColor(QtGui.QPalette.ColorRole.Light, QtGui.QColor("#ffffff"))
            pal.setColor(QtGui.QPalette.ColorRole.Midlight, QtGui.QColor("#f1f5f9"))
            pal.setColor(QtGui.QPalette.ColorRole.Dark, QtGui.QColor("#cbd5e1"))
            pal.setColor(QtGui.QPalette.ColorRole.Mid, QtGui.QColor("#94a3b8"))
            pal.setColor(QtGui.QPalette.ColorRole.Shadow, QtGui.QColor("#64748b"))
            if hasattr(self, "combo_theme"):
                idx = self.combo_theme.findData("white")
                if idx >= 0 and self.combo_theme.currentIndex() != idx:
                    self.combo_theme.blockSignals(True)
                    self.combo_theme.setCurrentIndex(idx)
                    self.combo_theme.blockSignals(False)
            if hasattr(self, "act_theme_white"):
                self.act_theme_white.setChecked(True)
        else:
            # Switch back to dark-compatible icon theme if available
            for dark_icon_theme in ("breeze-dark", "breeze", "Adwaita", "hicolor"):
                if QtGui.QIcon.hasThemeIcon("go-previous") or QtGui.QIcon.themeName() != dark_icon_theme:
                    QtGui.QIcon.setThemeName(dark_icon_theme)
                    if not QtGui.QIcon.fromTheme("go-previous").isNull():
                        break

            qss = DARK_THEME_QSS
            self.viewer.bg_color_mode = "dark"
            pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#1e1e24"))
            pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#fcfcfc"))
            pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#141618"))
            pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#1d1f22"))
            pal.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#292c30"))
            pal.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#fcfcfc"))
            pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#fcfcfc"))
            pal.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#292c30"))
            pal.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#fcfcfc"))
            pal.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#4b4b4b"))
            pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#3daee9"))
            pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
            pal.setColor(QtGui.QPalette.ColorRole.Light, QtGui.QColor("#151618"))
            pal.setColor(QtGui.QPalette.ColorRole.Midlight, QtGui.QColor("#1f2124"))
            pal.setColor(QtGui.QPalette.ColorRole.Dark, QtGui.QColor("#525860"))
            pal.setColor(QtGui.QPalette.ColorRole.Mid, QtGui.QColor("#373b40"))
            pal.setColor(QtGui.QPalette.ColorRole.Shadow, QtGui.QColor("#191919"))
            if hasattr(self, "combo_theme"):
                idx = self.combo_theme.findData("dark")
                if idx >= 0 and self.combo_theme.currentIndex() != idx:
                    self.combo_theme.blockSignals(True)
                    self.combo_theme.setCurrentIndex(idx)
                    self.combo_theme.blockSignals(False)
            if hasattr(self, "act_theme_dark"):
                self.act_theme_dark.setChecked(True)

        self.setPalette(pal)
        self.setStyleSheet(qss)
        if app is not None:
            app.setPalette(pal)
            app.setStyleSheet(qss)

        is_white = (theme == "white")
        # Dynamic theme-sensitive label and frame styles
        if hasattr(self, "lbl_measurement"):
            self.lbl_measurement.setStyleSheet(f"color: {'#0369a1' if is_white else '#38bdf8'}; font-weight: bold;")
        if hasattr(self, "lbl_live_geom"):
            self.lbl_live_geom.setStyleSheet(f"color: {'#0f172a' if is_white else '#e2e8f0'}; font-weight: bold;")
        if hasattr(self, "grp_qm_summary"):
            self.grp_qm_summary.setStyleSheet(
                f"QFrame {{ background-color: {'#e2e8f0' if is_white else '#0f172a'}; "
                f"border: 1px solid {'#cbd5e1' if is_white else '#334155'}; border-radius: 5px; padding: 3px 6px; }}"
            )
        if hasattr(self, "lbl_qm_summary"):
            self.lbl_qm_summary.setStyleSheet(f"color: {'#0f172a' if is_white else '#f8fafc'}; font-size: 11px; border: none;")
        if hasattr(self, "lbl_trajectory_step"):
            self.lbl_trajectory_step.setStyleSheet(f"font-weight: bold; color: {'#0369a1' if is_white else '#38bdf8'}; min-width: 65px;")
        if hasattr(self, "lbl_trajectory_energy"):
            self.lbl_trajectory_energy.setStyleSheet(f"color: {'#047857' if is_white else '#a7f3d0'}; font-family: monospace;")
        if hasattr(self, "lbl_chem_summary"):
            self.lbl_chem_summary.setStyleSheet(f"color: {'#0369a1' if is_white else '#38bdf8'}; font-weight: bold; padding: 0 10px;")
        if hasattr(self, "lbl_choose_sym"):
            self.lbl_choose_sym.setStyleSheet(f"font-weight: bold; color: {'#0f172a' if is_white else '#e2e8f0'};")
        if hasattr(self, "lbl_sym_group"):
            self.lbl_sym_group.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {'#6d28d9' if is_white else '#a78bfa'}; padding: 4px;")
        if hasattr(self, "lbl_sym_details"):
            self.lbl_sym_details.setStyleSheet(f"color: {'#475569' if is_white else '#cbd5e1'}; font-style: italic;")
        if hasattr(self, "lbl_sym_reps"):
            self.lbl_sym_reps.setStyleSheet(
                f"font-family: monospace; font-size: 11px; color: {'#0f172a' if is_white else '#e2e8f0'}; "
                f"background: {'#f1f5f9' if is_white else '#0f172a'}; padding: 6px; border: 1px solid {'#cbd5e1' if is_white else '#334155'}; border-radius: 4px;"
            )
        if hasattr(self, "lbl_vib_info"):
            self.lbl_vib_info.setStyleSheet(f"color: {'#475569' if is_white else '#94a3b8'}; font-style: italic;")
        if hasattr(self, "lbl_mo_status"):
            self.lbl_mo_status.setStyleSheet(f"color: {'#475569' if is_white else '#64748b'}; font-style: italic; font-size: 11px;")
        if hasattr(self, "mo_ladder_widget"):
            self.mo_ladder_widget.update()

        # Update embedded matplotlib figures if present
        fig_face = "#ffffff" if is_white else "#1e1e24"
        ax_face = "#f8fafc" if is_white else "#141418"
        text_color = "#0f172a" if is_white else "#cbd5e1"
        tick_color = "#475569" if is_white else "#94a3b8"
        grid_color = "#cbd5e1" if is_white else "#2e2e38"

        for comp_name in ("spectrum_widget", "ir_widget", "uvvis_widget"):
            w = getattr(self, comp_name, None)
            if w and hasattr(w, "fig") and hasattr(w, "ax"):
                w.fig.patch.set_facecolor(fig_face)
                w.ax.set_facecolor(ax_face)
                w.ax.tick_params(colors=tick_color)
                w.ax.xaxis.label.set_color(text_color)
                w.ax.yaxis.label.set_color(text_color)
                for spine in w.ax.spines.values():
                    spine.set_color(grid_color)
                w.ax.grid(True, color=grid_color, linestyle="--", alpha=0.6)
                if hasattr(w, "canvas"):
                    w.canvas.draw_idle()

        if hasattr(self, "_settings"):
            self._settings().setValue("color_theme", theme)
        self.viewer.update()
        try:
            bus = getattr(self, "event_bus", None)
            if bus:
                bus.theme_changed.emit(theme)
        except (RuntimeError, AttributeError):
            pass

    def apply_journal_preset(self, preset: str):
        """Apply publication-grade visual preset across viewer, theme actions, and status notification."""
        label = self.viewer.apply_journal_preset(preset)
        is_white = (getattr(self.viewer, "bg_color_mode", "white") == "white")
        current_theme = getattr(self, "current_theme", "dark")
        if (is_white and current_theme != "white") or (not is_white and current_theme != "dark"):
            self.set_color_theme("white" if is_white else "dark")
        if hasattr(self, "chk_pub_outlines"):
            self.chk_pub_outlines.blockSignals(True)
            self.chk_pub_outlines.setChecked(self.viewer.show_publication_outlines)
            self.chk_pub_outlines.blockSignals(False)
        self.statusBar().showMessage(f"🎨 Applied Journal Figure Preset: {label}", 4000)

    def _on_toggle_labels(self, enabled: bool):
        self.viewer.show_labels = enabled
        self.viewer.update()

    def _on_toggle_indices(self, enabled: bool):
        self.viewer.show_indices = enabled
        self.viewer.update()

    def _on_toggle_dipole(self, enabled: bool):
        self.viewer.show_dipole = enabled
        self.viewer.update()

    def _on_toggle_charges(self, enabled: bool):
        self.viewer.show_charges = enabled
        self.viewer.update()

    def _on_toggle_hbonds(self, enabled: bool):
        self.viewer.show_hbonds = enabled
        self.viewer.update()

    def _on_toggle_nci(self, enabled: bool):
        self.viewer.show_nci = enabled
        self.viewer.update()
        state_str = "Enabled" if enabled else "Disabled"
        self.statusBar().showMessage(f"Non-Covalent Interactions (π-π, Halogen Bonds): {state_str}", 3000)

    def _on_toggle_forces(self, enabled: bool):
        self.viewer.show_forces = enabled
        self.viewer.update()

    def _on_toggle_camera_projection(self):
        """Toggle between Perspective and Orthographic camera projection."""
        is_ortho = self.viewer.toggle_orthographic()
        if hasattr(self, "btn_camera_proj"):
            self.btn_camera_proj.setText("Ortho" if is_ortho else "Persp")
            self.btn_camera_proj.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold;" if is_ortho else "")
        mode_str = "Orthographic (Parallel)" if is_ortho else "Perspective"
        self.statusBar().showMessage(f"Camera Projection: {mode_str}", 2500)

    def _on_open_cluster_script_generator(self):
        """Open HPC cluster batch submission script generator (SLURM / PBS / LSF / SGE)."""
        from .dialog_cluster_script import ClusterScriptDialog
        dlg = ClusterScriptDialog(self, input_filename="calculation.inp")
        dlg.exec()

    def _on_open_qc_doctor(self):
        """Analyze calculation logs or output files with the AgyMolden QC Doctor."""
        path = getattr(self, "_last_loaded_qm_log_path", None)
        log_text = ""
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    log_text = f.read()
            except Exception:
                pass
        elif hasattr(self, "_job_monitor_dlg") and self._job_monitor_dlg:
            log_text = getattr(self._job_monitor_dlg, "raw_log_text", "")

        if not log_text:
            QtWidgets.QMessageBox.information(
                self,
                "QC Doctor",
                "Please load or run a Quantum Chemistry calculation log (.out / .log) first.",
            )
            return

        from .dialog_qc_doctor import QCDoctorDialog
        dlg = QCDoctorDialog(self, log_text=log_text)
        dlg.exec()

    def _on_open_workflow_pipeline_dialog(self):
        """Open Workflow Pipeline / Multi-Stage Recipe dialog."""
        from .calc_runner.dialog_pipeline import WorkflowPipelineDialog
        symbols = [a.symbol for a in self.zatoms] if self.zatoms else ["O", "H", "H"]
        coords = np.array(self.coords, dtype=np.float64, copy=True) if (self.coords is not None and len(self.coords) > 0) else np.zeros((3, 3))
        dlg = WorkflowPipelineDialog(
            parent=self,
            symbols=symbols,
            coords=coords,
            work_dir=os.path.join(os.getcwd(), "workflow_run"),
        )
        dlg.exec()

    def _on_align_principal_axes(self):
        """Align molecule COM to origin and rotate onto principal inertial axes."""
        from .builtin_plugins.align_principal_axes import PrincipalAxesAlignerPlugin
        from .plugin_api import PluginContext
        plugin = PrincipalAxesAlignerPlugin()
        context = PluginContext(self)
        plugin.run(context)

    def _on_open_command_palette(self):
        """Open modern searchable Command Palette (Ctrl+K)."""
        from .dialog_command_palette import CommandPaletteDialog, CommandItem

        commands = [
            # Calculations & Execution
            CommandItem("AgyMolden QC Doctor (Crash Diagnostician)", "Calculate", "Ctrl+Alt+D", self._on_open_qc_doctor, "doctor diagnostic scf fix repair crash error"),
            CommandItem("Run ORCA Calculation", "Calculate", "Ctrl+R", self._on_quick_run_orca, "orca dft dft-d opt freq sp"),
            CommandItem("Open Universal Job Monitor & Convergence", "Calculate", "Ctrl+J", self._on_open_job_monitor, "monitor scf plot live logs"),
            CommandItem("HPC Batch Script Generator (SLURM / PBS)", "Calculate", "", self._on_open_cluster_script_generator, "cluster slurm pbs sge hpc submit"),
            CommandItem("Generate Quantum Chemistry Input Deck", "Calculate", "Ctrl+G", self._on_open_qm_dialog, "input deck orca gaussian pyscf psi4"),
            CommandItem("Optimize Geometry (Force Field / xTB)", "Calculate", "Ctrl+Shift+O", self._on_optimize_geometry, "minimize uff mmff xtb relax"),
            CommandItem("Workflow Pipeline / Multi-Stage Recipe", "Calculate", "", self._on_open_workflow_pipeline_dialog, "pipeline recipe sequence multistep"),
            CommandItem("Visual Workflow DAG Studio", "Calculate", "", self._on_open_workflow_dag, "dag graph workflow pipeline stages"),
            # Wavefunction & Real-Space Surfaces
            CommandItem("Volumetric Orbitals & Density Studio", "Surfaces", "Ctrl+Alt+O", self._on_open_volumetric_dialog, "orbitals homo lumo density cubes marching"),
            CommandItem("3D Electrostatic Potential (ESP) Map", "Surfaces", "Ctrl+Alt+P", lambda: self._on_open_volumetric_dialog(target_tab="esp"), "esp potential electrostatic charge surface"),
            CommandItem("2D Planar ESP Slice & Contours", "Surfaces", "", self._on_open_esp_2d_dialog, "esp 2d contour slice plane"),
            CommandItem("Multiwfn Real-Space Analysis (NCI / ELF / LOL)", "Surfaces", "", self._on_open_multiwfn_dialog, "multiwfn nci rdg elf lol basin"),
            # Reaction Paths & Conformers
            CommandItem("Interactive IRC Reaction Sweeper", "Reactivity", "", self._on_open_irc_sweeper, "irc sweeper reaction barrier mep path ts"),
            CommandItem("Transition State IDPP Path Interpolator", "Reactivity", "", self._on_open_reaction_path_dialog, "ts idpp transit interpolation saddle"),
            CommandItem("CREST Conformer Ensemble Explorer", "Conformers", "", self._on_open_crest_explorer, "crest ensemble boltzmann cluster conformers"),
            CommandItem("1D/2D Conformer PES Scan", "Conformers", "", self._on_conformer_search, "pes scan dihedral rigid relaxed"),
            # Builder & Structure Tools
            CommandItem("Cycle Welder & Ring Annulation", "Builder", "Ctrl+W", self._on_open_cycle_welder, "cycle ring welder fuse annulate benzene"),
            CommandItem("Add Hydrogens (Valence Fill)", "Builder", "Ctrl+H", self._on_add_hydrogens, "hydrogens add valency protonate"),
            CommandItem("Clean 3D Geometry (UFF / MMFF)", "Builder", "", self._on_optimize_geometry, "clean uff forcefield rough 3d minimize"),
            CommandItem("Periodic Table Element Picker", "Builder", "", lambda: PeriodicTableDialog(self).exec(), "element periodic table c n o"),
            CommandItem("Fragment Library Picker", "Builder", "", self._on_open_fragment_library, "fragments amino acid functional group"),
            CommandItem("Align Principal Axes / Moments of Inertia", "Geometry", "", self._on_align_principal_axes, "align inertia axes center orientation"),
            # Export & Publishing
            CommandItem("Export Scalable Vector Graphics (SVG)", "Export", "", self._on_export_svg, "svg vector export publication drawing image png"),
            CommandItem("Export High-DPI Publication Image", "Export", "Ctrl+Shift+E", self._on_export_image, "dpi png export figure resolution publication"),
            CommandItem("Export Supporting Information (SI) Bundle", "Export", "", self._on_export_reproducibility_bundle, "si bundle zip reproducibility supporting information"),
            CommandItem("Copy Image to Clipboard", "Export", "Ctrl+Shift+C", lambda: self.copy_image_to_clipboard(transparent=True, scale=2, clean_only=True), "copy clipboard figure image"),
            CommandItem("Export Molecular Coordinates (XYZ / PDB / SDF / Mol2 / CIF)", "Export", "", self._on_export_xyz, "export xyz pdb sdf mol2 cif coordinates"),
            CommandItem("Export Tripos Mol2 File", "Export", "", self._on_export_mol2, "export mol2 tripos docking"),
            CommandItem("Export Crystallographic CIF File", "Export", "", self._on_export_cif, "export cif crystal solid state"),
            # App Settings & Help
            CommandItem("Application Preferences & Rendering Options", "Settings", "", self._on_open_preferences, "preferences theme background color rendering"),
            CommandItem("Unit & Energy Converter", "Tools", "", self._open_unit_converter, "convert hartree kcal kj ev cm-1"),
            CommandItem("Interactive Onboarding Tour", "Help", "", self._on_open_onboarding_tour, "tour guide walkthrough welcome help"),
        ]

        dlg = CommandPaletteDialog(self, commands=commands)
        dlg.exec()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        key = event.key()

        # Handle 'b', 'a', 'd' partial reorder shortcuts when not typing in text editor or Z-matrix table
        focus_w = QtWidgets.QApplication.focusWidget()
        in_table = (
            focus_w is self.table_widget
            or (focus_w is not None and self.table_widget.isAncestorOf(focus_w))
        )
        is_typing = isinstance(focus_w, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit))

        # If the user clicked a cell in the Z-Matrix table and starts typing a letter/number,
        # let QTableWidget open its inline editor rather than triggering 3D viewport hotkeys!
        if in_table and not is_typing:
            modifiers = event.modifiers()
            is_ctrl = bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)
            if not is_ctrl and event.text() and event.text().isprintable():
                cur_item = self.table_widget.currentItem()
                if cur_item and (cur_item.flags() & QtCore.Qt.ItemFlag.ItemIsEditable):
                    self.table_widget.editItem(cur_item)
                    editor = QtWidgets.QApplication.focusWidget()
                    if isinstance(editor, QtWidgets.QLineEdit):
                        editor.setText(event.text())
                    return

        if not is_typing:
            modifiers = event.modifiers()
            is_ctrl = bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)
            is_shift = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)
            is_alt = bool(modifiers & QtCore.Qt.KeyboardModifier.AltModifier)

            if is_ctrl:
                if key == QtCore.Qt.Key.Key_K and not is_shift and not is_alt:
                    self._on_open_command_palette()
                    return
                elif key == QtCore.Qt.Key.Key_Z and not is_shift:
                    self.undo()
                    return
                elif key == QtCore.Qt.Key.Key_Y or (key == QtCore.Qt.Key.Key_Z and is_shift):
                    self.redo()
                    return
                elif key == QtCore.Qt.Key.Key_C and not is_shift:
                    self._copy_selected_fragment()
                    return
                elif key == QtCore.Qt.Key.Key_C and is_shift:
                    # Ctrl+Shift+C: Copy publication-quality image to clipboard
                    self.copy_image_to_clipboard(transparent=True, scale=2, clean_only=True)
                    return
                elif key == QtCore.Qt.Key.Key_V and not is_shift:
                    self._on_paste()
                    return
                elif key == QtCore.Qt.Key.Key_V and is_shift:
                    # Ctrl+Shift+V: Direct load XYZ coordinates from clipboard into program
                    self._on_paste_as_new_molecule()
                    return
                elif key == QtCore.Qt.Key.Key_A and not is_shift:
                    # Ctrl+A: Select all atoms in the 3D viewport
                    self.viewer.multi_selected = list(range(len(self.zatoms)))
                    self.viewer.selected_atom_idx = 0 if len(self.zatoms) > 0 else None
                    self.viewer.update()
                    self.statusBar().showMessage(f"Selected all {len(self.zatoms)} atoms", 2500)
                    return
                elif key == QtCore.Qt.Key.Key_D and not is_shift:
                    # Ctrl+D: Deselect all atoms
                    self.viewer.multi_selected = []
                    self.viewer.selected_atom_idx = None
                    self.viewer.update()
                    self.statusBar().showMessage("Deselected all atoms", 2000)
                    return

            # Center / Reset view with Spacebar
            if key == QtCore.Qt.Key.Key_Space and not is_ctrl and not is_shift:
                self.viewer.auto_center()
                self.statusBar().showMessage("Centered 3D view (Space)", 2000)
                return

            # F5: Toggle Perspective / Orthographic projection (VMD / GaussView style)
            if key == QtCore.Qt.Key.Key_F5 and not is_ctrl and not is_shift:
                self._on_toggle_camera_projection()
                return

            # Hotkeys for view planes: X (YZ plane), Y (XZ plane), Z (XY plane)
            if not is_ctrl and not is_alt and not is_shift:
                if key == QtCore.Qt.Key.Key_X:
                    self.viewer.set_camera_plane_view("YZ")
                    self.statusBar().showMessage("Snapped view to YZ plane (along X axis)", 2000)
                    return
                elif key == QtCore.Qt.Key.Key_Y:
                    self.viewer.set_camera_plane_view("XZ")
                    self.statusBar().showMessage("Snapped view to XZ plane (along Y axis)", 2000)
                    return
                elif key == QtCore.Qt.Key.Key_Z:
                    self.viewer.set_camera_plane_view("XY")
                    self.statusBar().showMessage("Snapped view to XY plane (along Z axis)", 2000)
                    return
                elif key in (QtCore.Qt.Key.Key_QuoteLeft, QtCore.Qt.Key.Key_AsciiTilde):
                    # Backquote / Tilde: 180 degree camera flip (Chemcraft style)
                    self.viewer.flip_camera_180()
                    self.statusBar().showMessage("Flipped 3D view 180° horizontally (`/~)", 2000)
                    return

            # Tab key toggles between ✏️ Build and 🔍 Inspect mode
            if key == QtCore.Qt.Key.Key_Tab and not is_ctrl and not is_shift:
                cur_m = getattr(self.viewer, "interaction_mode", "build")
                self.set_interaction_mode("inspect" if cur_m == "build" else "build")
                return

            # Alt+B / Alt+A / Alt+D: Molden Partial Z-Matrix Reorder
            if is_alt:
                if key == QtCore.Qt.Key.Key_B:
                    self._start_partial_reorder("bond")
                    return
                elif key == QtCore.Qt.Key.Key_A:
                    self._start_partial_reorder("angle")
                    return
                elif key == QtCore.Qt.Key.Key_D:
                    self._start_partial_reorder("dihedral")
                    return

            # Bond order change hotkeys: 1, 2, 3 (when a bond is selected or hovered)
            bond_order_map = {
                QtCore.Qt.Key.Key_1: 1,
                QtCore.Qt.Key.Key_2: 2,
                QtCore.Qt.Key.Key_3: 3,
            }
            if key in bond_order_map and not is_ctrl and not is_alt:
                target_bo = bond_order_map[key]
                sel_bond = getattr(self.viewer, "selected_bond", None)
                if sel_bond is not None:
                    u, v = sel_bond
                    self.set_bond_order(u, v, target_bo=target_bo)
                    return
                elif len(self.viewer.multi_selected) == 2:
                    u, v = self.viewer.multi_selected
                    # Verify if this pair is an existing bond
                    curr_bonds = self._current_explicit_bonds()
                    if norm_bond(u, v) in curr_bonds:
                        self.set_bond_order(u, v, target_bo=target_bo)
                        return

            if key in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
                self._on_delete_selected_atom()
                return

            # Quick-Element Hotkeys (Avogadro / GaussView style):
            # Keys: C, H, O, N, S, P, F, B, X, Si (Shift+S), Cl (Shift+C), Br (Shift+B), I
            elem_map = {
                QtCore.Qt.Key.Key_C: "Cl" if is_shift else "C",
                QtCore.Qt.Key.Key_H: "H",
                QtCore.Qt.Key.Key_O: "O",
                QtCore.Qt.Key.Key_N: "N",
                QtCore.Qt.Key.Key_S: "Si" if is_shift else "S",
                QtCore.Qt.Key.Key_P: "P",
                QtCore.Qt.Key.Key_F: "F",
                QtCore.Qt.Key.Key_B: "Br" if is_shift else "B",
                QtCore.Qt.Key.Key_I: "I",
                QtCore.Qt.Key.Key_X: "X",
            }
            if key in elem_map:
                chosen_elem = elem_map[key]
                # If an atom is currently selected in the viewport AND we are in Build mode, mutate its element!
                sel_idx = getattr(self.viewer, "selected_atom_idx", None)
                if (
                    getattr(self.viewer, "interaction_mode", "build") == "build"
                    and sel_idx is not None
                    and 0 <= sel_idx < len(self.zatoms)
                ):
                    old_sym = self.zatoms[sel_idx].symbol
                    if old_sym != chosen_elem:
                        self._push_undo_state()
                        sym_res = self._get_active_imposed_symmetry()
                        if sym_res is not None and len(self.coords) == len(self.zatoms):
                            atom_frag = {"name": chosen_elem, "symbols": [chosen_elem], "coords": [(0.0, 0.0, 0.0)], "connect_idx": 0}
                            n_sites = self._attach_fragment_to_atom_or_orbit(
                                sel_idx, atom_frag, conn_idx=0, relax_iters=80
                            )
                            self.statusBar().showMessage(
                                f"Mutated {n_sites} symmetry-equivalent atoms: {old_sym} ➔ {chosen_elem}", 4000
                            )
                            return
                        # Detect connectivity before mutation to know neighbors
                        curr_bonds = detect_connectivity([a.symbol for a in self.zatoms], self.coords)
                        neighbors = []
                        for u, v in curr_bonds:
                            if u == sel_idx:
                                neighbors.append(v)
                            elif v == sel_idx:
                                neighbors.append(u)

                        self.zatoms[sel_idx].symbol = chosen_elem
                        symbols = [a.symbol for a in self.zatoms]
                        coords = np.copy(self.coords)

                        # Adjust bond length to partner if mutating H to heavy atom (e.g. H -> O), not for X
                        if old_sym == "H" and chosen_elem not in ("H", "X") and neighbors:
                            partner = neighbors[0]
                            p_dir = coords[sel_idx] - coords[partner]
                            p_len = np.linalg.norm(p_dir)
                            if p_len > 1e-4:
                                new_d = estimate_bond_length(chosen_elem, symbols[partner])
                                coords[sel_idx] = coords[partner] + (p_dir / p_len) * new_d

                        if (
                            chosen_elem not in ("X", "H")
                            and getattr(self, "chk_auto_h", None)
                            and self.chk_auto_h.isChecked()
                        ):
                            targets_to_adjust = [sel_idx] + neighbors
                            adj_symbols, adj_coords = adjust_hydrogens_to_valence(symbols, coords, target_atom_indices=targets_to_adjust)
                            new_zatoms = cartesian_to_zmatrix(adj_symbols, adj_coords, existing_zatoms=self.zatoms)
                            self.set_molecule_from_zatoms(new_zatoms)
                        else:
                            self.coords = coords
                            self.table_widget.load_zatoms(self.zatoms)
                            self.update_geometry_live()
                        self.statusBar().showMessage(f"Mutated atom #{sel_idx+1}: {old_sym} ➔ {chosen_elem}", 3500)
                else:
                    # No atom selected: switch drawing brush element
                    self.last_added_element = chosen_elem
                    self.viewer.brush_element = chosen_elem
                    self.last_item_type = "element"
                    self.stored_fragment = None
                    self.last_added_fragment = None
                    self._update_viewer_drag_label()
                    self.viewer.update()
                    self.statusBar().showMessage(f"Active drawing brush switched to: {chosen_elem}", 3000)
                return

        if key == QtCore.Qt.Key.Key_Escape:
            if getattr(self, "pending_add_atom_mode", False):
                self.pending_add_atom_mode = False
                self.statusBar().showMessage("Add atom mode cancelled", 2000)
                return
            if self.viewer.partial_reorder_mode:
                self._cancel_partial_reorder()
                return
            if self.viewer.reordering_active:
                self._cancel_reorder()
                return
            if self.viewer.picking_active:
                self.viewer.picking_active = False
                self.viewer.picking_banner = ""
                self.viewer.picked_references = []
                self.pending_new_element = None
                self.picked_references = []
                self.viewer.update()
                self.statusBar().showMessage("Atom selection aborted!", 3000)
                return
            if getattr(self.viewer, "active_isosurfaces", None):
                self._on_clear_orbital_surface()
                self.statusBar().showMessage("Cleared 3D orbital / isosurface (Esc)", 2500)
                return
        elif key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            if self.viewer.reordering_active:
                self._apply_reorder()
                return

        super().keyPressEvent(event)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress:
            key_event: QtGui.QKeyEvent = event
            key = key_event.key()
            modifiers = key_event.modifiers()
            is_ctrl = bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)
            is_shift = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)

            # Do not intercept if user is typing text in an actual text editor / line edit
            focus_w = QtWidgets.QApplication.focusWidget()
            is_typing = isinstance(focus_w, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit))

            if is_ctrl and not is_typing:
                if key == QtCore.Qt.Key.Key_Z and not is_shift:
                    self.undo()
                    return True
                elif key == QtCore.Qt.Key.Key_Y or (key == QtCore.Qt.Key.Key_Z and is_shift):
                    self.redo()
                    return True
                elif key == QtCore.Qt.Key.Key_C and not is_shift:
                    self._copy_selected_fragment()
                    return
                elif key == QtCore.Qt.Key.Key_C and is_shift:
                    # Ctrl+Shift+C: Copy publication-quality image to clipboard
                    self.copy_image_to_clipboard(transparent=True, scale=2, clean_only=True)
                    return True
                elif key == QtCore.Qt.Key.Key_V and not is_shift:
                    self._on_paste()
                    return
                elif key == QtCore.Qt.Key.Key_V and is_shift:
                    # Ctrl+Shift+V: Direct load XYZ coordinates from clipboard into program
                    self._on_paste_as_new_molecule()
                    return True
                elif key == QtCore.Qt.Key.Key_K and not is_shift:
                    self._on_flush_memory()
                    return True

        return super().eventFilter(watched, event)
