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

log = logging.getLogger(__name__)


class ReorderMixin:
    def _on_reorder_zmat_clicked(self):
        """
        Reorder Z-matrix workflow:
        - If 2 atoms are selected in viewport [A, B]: reorders/adjusts bond for the 2nd atom (B -> ref1 = A).
        - If 3 atoms are selected in viewport [A, B, C]: reorders/adjusts angle for the 3rd atom (C -> ref1 = B, ref2 = A).
        - If 4 atoms are selected in viewport [A, B, C, D]: reorders/adjusts dihedral for the 4th atom (D -> ref1 = C, ref2 = B, ref3 = A).
        - Otherwise (0 or 1 atoms selected): activates interactive sequential clicking mode (Molden ZMEORD).
        """
        if len(self.zatoms) < 2:
            QtWidgets.QMessageBox.information(
                self, "Reorder Z-Matrix", "At least 2 atoms are required to reorder Z-matrix connectivity."
            )
            return

        if self.viewer.reordering_active:
            # Clicking the button while active applies the reorder
            self._apply_reorder()
            return

        # Check if 2, 3, or 4 atoms are currently selected in the viewport (multi_selected)
        n_atoms = len(self.zatoms)
        sel = [idx for idx in self.viewer.multi_selected if 0 <= idx < n_atoms]
        if len(sel) == 2:
            # Bond: [A, B] -> reorders 2nd atom B referencing 1st atom A
            # picks for _apply_partial_reorder: [target, ref1] = [sel[1], sel[0]]
            self._apply_partial_reorder([sel[1], sel[0]], "bond")
            return
        elif len(sel) == 3:
            # Angle: [A, B, C] -> reorders 3rd atom C referencing B (bond) and A (angle)
            # picks for _apply_partial_reorder: [target, ref1, ref2] = [sel[2], sel[1], sel[0]]
            self._apply_partial_reorder([sel[2], sel[1], sel[0]], "angle")
            return
        elif len(sel) == 4:
            # Dihedral: [A, B, C, D] -> reorders 4th atom D referencing C (bond), B (angle), A (dihedral)
            # picks for _apply_partial_reorder: [target, ref1, ref2, ref3] = [sel[3], sel[2], sel[1], sel[0]]
            self._apply_partial_reorder([sel[3], sel[2], sel[1], sel[0]], "dihedral")
            return

        # Cancel any pending 'Add line' picking
        if self.viewer.picking_active:
            self.viewer.picking_active = False
            self.viewer.picking_banner = ""
            self.viewer.picked_references = []
            self.pending_new_element = None

        self.viewer.reordering_active = True
        self.viewer.reorder_list = []
        self.btn_reorder_zmat.setText("✅ Apply Reorder")
        self.btn_reorder_zmat.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: bold; font-size: 11px; padding: 4px 6px;")
        self.viewer.update()
        self.statusBar().showMessage(
            "Reordering Z-Matrix: Click atoms in desired order. Press 'Apply Reorder' or Enter when done, Esc to cancel.", 10000
        )

    def _on_atom_picked_for_reorder(self, atom_idx: int):
        if not self.viewer.reordering_active:
            return

        if atom_idx in self.viewer.reorder_list:
            # If clicked again, remove from reorder list (undo)
            self.viewer.reorder_list.remove(atom_idx)
            self.viewer.update()
            self.statusBar().showMessage(
                f"Removed {self.viewer.symbols[atom_idx]}{atom_idx+1} from reorder sequence ({len(self.viewer.reorder_list)}/{len(self.zatoms)}).", 4000
            )
            return

        self.viewer.reorder_list.append(atom_idx)
        self.viewer.update()
        cur_count = len(self.viewer.reorder_list)
        total_count = len(self.zatoms)
        self.statusBar().showMessage(
            f"Atom #{cur_count}: {self.viewer.symbols[atom_idx]}{atom_idx+1} selected ({cur_count}/{total_count}). Click next or Apply when done.", 5000
        )

        if cur_count == total_count:
            # All atoms picked in sequence! Apply immediately
            self._apply_reorder()

    def _apply_reorder(self):
        if not self.viewer.reordering_active:
            return

        reorder_list = list(self.viewer.reorder_list)
        self.viewer.reordering_active = False
        self.viewer.reorder_list = []
        self.btn_reorder_zmat.setText("🔄 Reorder ZMAT")
        self.btn_reorder_zmat.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold; font-size: 11px; padding: 4px 6px;")
        self.viewer.update()

        n = len(self.zatoms)
        if len(reorder_list) == 0:
            self.statusBar().showMessage("Reordering cancelled (no atoms selected).", 3000)
            return

        self._push_undo_state()

        # Capture existing topology and bond orders before permutation
        orig_bonds = self._current_explicit_bonds()
        orig_bo = self._current_bond_orders()

        # Any unpicked atoms are appended to the end in their original relative order
        remaining = [i for i in range(n) if i not in reorder_list]
        final_order = reorder_list + remaining

        # Mapping from old index to new index
        old_to_new = {old: new for new, old in enumerate(final_order)}
        new_bonds = {norm_bond(old_to_new[u], old_to_new[v]) for u, v in orig_bonds}
        new_bo = {norm_bond(old_to_new[u], old_to_new[v]): order for (u, v), order in orig_bo.items() if order != 1}

        # Permute symbols and Cartesian coordinates
        ordered_symbols = [self.zatoms[i].symbol for i in final_order]
        ordered_coords = self.coords[final_order]

        # Re-derive canonical Z-matrix with the new ordering preserving bonds
        new_zatoms = cartesian_to_zmatrix(ordered_symbols, ordered_coords, user_bonds=list(new_bonds))
        tree = {norm_bond(a.ref1, i) for i, a in enumerate(new_zatoms) if a.ref1 is not None}
        self.cross_bonds = {b for b in new_bonds if b not in tree}
        self.set_molecule_from_zatoms(new_zatoms, exact_coords=ordered_coords)
        self.viewer.bond_orders = new_bo
        self.statusBar().showMessage(
            f"Z-Matrix reordered successfully! New sequence: {', '.join([f'{ordered_symbols[i]}{i+1}' for i in range(min(5, n))])}...", 5000
        )

    def _start_partial_reorder(self, mode: str):
        """
        Start interactive partial Z-matrix adjustment mode:
        'bond' ('b'): click 2 atoms -> sets ref1 for the 1st atom to the 2nd atom.
        'angle' ('a'): click 3 atoms -> sets ref1 (bond) to 2nd atom, ref2 (angle) to 3rd atom.
        'dihedral' ('d'): click 4 atoms -> sets ref1 (bond), ref2 (angle), ref3 (dihedral).
        """
        if len(self.zatoms) < 2:
            self.statusBar().showMessage("At least 2 atoms are required to adjust connectivity.", 3000)
            return

        # Cancel any full reorder or add line picking in progress
        if self.viewer.reordering_active:
            self._cancel_reorder()
        if self.viewer.picking_active:
            self.viewer.picking_active = False
            self.viewer.picking_banner = ""
            self.viewer.picked_references = []
            self.pending_new_element = None

        self.viewer.partial_reorder_mode = mode
        self.viewer.partial_reorder_picks = []

        if mode == "bond":
            self.viewer.partial_reorder_needed = 2
            banner = "Bond Definition ('b'): Click 2 atoms: [1] Target atom, [2] New Bond partner. Esc to cancel."
        elif mode == "angle":
            self.viewer.partial_reorder_needed = 3
            banner = "Angle Definition ('a'): Click 3 atoms: [1] Target atom, [2] Bond partner, [3] Angle partner. Esc to cancel."
        elif mode == "dihedral":
            self.viewer.partial_reorder_needed = 4
            banner = "Dihedral Definition ('d'): Click 4 atoms: [1] Target, [2] Bond, [3] Angle, [4] Dihedral. Esc to cancel."
        else:
            return

        self.viewer.partial_reorder_banner = banner
        self.viewer.update()
        self.statusBar().showMessage(banner, 10000)

    def _cancel_partial_reorder(self):
        if self.viewer.partial_reorder_mode:
            self.viewer.partial_reorder_mode = None
            self.viewer.partial_reorder_picks = []
            self.viewer.partial_reorder_needed = 0
            self.viewer.partial_reorder_banner = ""
            self.viewer.update()
            self.statusBar().showMessage("Partial Z-matrix adjustment cancelled.", 3000)

    def _on_atom_picked_for_partial_reorder(self, atom_idx: int):
        if not self.viewer.partial_reorder_mode:
            return

        picks = self.viewer.partial_reorder_picks
        if atom_idx in picks:
            self.statusBar().showMessage(
                f"Atom {self.viewer.symbols[atom_idx]}{atom_idx+1} already selected! Click a different atom.", 3000
            )
            return

        picks.append(atom_idx)
        self.viewer.update()

        n_picked = len(picks)
        needed = self.viewer.partial_reorder_needed
        mode = self.viewer.partial_reorder_mode

        if n_picked < needed:
            target_idx = picks[0]
            target_sym = f"{self.viewer.symbols[target_idx]}{target_idx+1}"
            if mode == "bond":
                banner = f"Bond Definition for {target_sym}: Now click 2nd atom (Bond partner). Esc to cancel."
            elif mode == "angle":
                if n_picked == 1:
                    banner = f"Angle Definition for {target_sym}: Click 2nd atom (Bond partner). Esc to cancel."
                elif n_picked == 2:
                    bond_sym = f"{self.viewer.symbols[picks[1]]}{picks[1]+1}"
                    banner = f"Angle Definition for {target_sym} bonded to {bond_sym}: Now click 3rd atom (Angle partner). Esc to cancel."
            elif mode == "dihedral":
                if n_picked == 1:
                    banner = f"Dihedral Definition for {target_sym}: Click 2nd atom (Bond partner). Esc to cancel."
                elif n_picked == 2:
                    bond_sym = f"{self.viewer.symbols[picks[1]]}{picks[1]+1}"
                    banner = f"Dihedral Definition for {target_sym}-{bond_sym}: Click 3rd atom (Angle partner). Esc to cancel."
                elif n_picked == 3:
                    ang_sym = f"{self.viewer.symbols[picks[2]]}{picks[2]+1}"
                    banner = f"Dihedral Definition for {target_sym}: Click 4th atom (Dihedral partner). Esc to cancel."
            self.viewer.partial_reorder_banner = banner
            self.viewer.update()
            self.statusBar().showMessage(banner, 8000)
            return

        # All required atoms picked! Apply partial reordering
        self._apply_partial_reorder(picks, mode)

    def _apply_partial_reorder(self, picks: List[int], mode: str):
        self.viewer.partial_reorder_mode = None
        self.viewer.partial_reorder_picks = []
        self.viewer.partial_reorder_needed = 0
        self.viewer.partial_reorder_banner = ""
        self.viewer.update()

        n = len(self.zatoms)
        if mode == "bond":
            # picks: [target, ref1]
            target_orig, ref1_orig = picks[0], picks[1]
            ref2_orig, ref3_orig = None, None
            ref_list = [ref1_orig]
        elif mode == "angle":
            # picks: [target, ref1, ref2]
            target_orig, ref1_orig, ref2_orig = picks[0], picks[1], picks[2]
            ref3_orig = None
            ref_list = [ref1_orig, ref2_orig]
        elif mode == "dihedral":
            # picks: [target, ref1, ref2, ref3]
            target_orig, ref1_orig, ref2_orig, ref3_orig = picks[0], picks[1], picks[2], picks[3]
            ref_list = [ref1_orig, ref2_orig, ref3_orig]
        else:
            return

        self._push_undo_state()

        # Case 1: All reference atoms precede target_orig (max(ref_list) < target_orig).
        # We do NOT change the order of atoms! We only update the Z-matrix references and values for target_orig in place.
        if max(ref_list) < target_orig:
            atom = self.zatoms[target_orig]
            atom.ref1 = ref1_orig
            atom.dist = calculate_bond_distance(self.coords[target_orig], self.coords[ref1_orig])
            atom.dist_var = None

            if mode in ("angle", "dihedral"):
                atom.ref2 = ref2_orig
            elif atom.ref2 == ref1_orig:
                atom.ref2 = next((k for k in range(target_orig) if k != ref1_orig), None)
            if atom.ref2 is not None:
                atom.angle = calculate_angle(self.coords[target_orig], self.coords[atom.ref1], self.coords[atom.ref2])
                atom.angle_var = None

            if mode == "dihedral":
                atom.ref3 = ref3_orig
            elif atom.ref3 in (atom.ref1, atom.ref2):
                atom.ref3 = next((k for k in range(target_orig) if k not in (atom.ref1, atom.ref2)), None)
            if atom.ref2 is not None and atom.ref3 is not None:
                atom.dihedral = calculate_dihedral(self.coords[target_orig], self.coords[atom.ref1], self.coords[atom.ref2], self.coords[atom.ref3])
                atom.dihedral_var = None

            self.table_widget.load_zatoms(self.zatoms)
            self._sync_raw_text()
            self.table_widget.highlight_row(target_orig)
            self.viewer.highlighted_feature_atoms = {}
            self.viewer.selected_atom_idx = target_orig
            self.viewer.multi_selected = [target_orig]
            self._update_adjust_panel()
            self.viewer.update()

            t_name = f"{self.zatoms[target_orig].symbol}{target_orig+1}"
            r1_name = f"{self.zatoms[ref1_orig].symbol}{ref1_orig+1}"
            if mode == "bond":
                msg = f"Adjusted Bond: {t_name} connected to {r1_name} (order preserved)."
            elif mode == "angle":
                r2_name = f"{self.zatoms[ref2_orig].symbol}{ref2_orig+1}"
                msg = f"Adjusted Angle: {t_name} bonded to {r1_name} and angle with {r2_name} (order preserved)."
            else:
                r2_name = f"{self.zatoms[ref2_orig].symbol}{ref2_orig+1}"
                r3_name = f"{self.zatoms[ref3_orig].symbol}{ref3_orig+1}"
                msg = f"Adjusted Dihedral: {t_name} bonded to {r1_name}, angle {r2_name}, dihedral {r3_name} (order preserved)."
            self.statusBar().showMessage(msg, 6000)
            return

        # Capture existing topology and bond orders before permutation
        orig_bonds = self._current_explicit_bonds()
        orig_bo = self._current_bond_orders()

        # Case 2: One or more reference atoms appear after target_orig.
        # In Z-matrix, an atom can only reference preceding atoms.
        # Minimal reorder: move target_orig to just after the maximum reference atom, preserving all other relative atom orders.
        max_ref = max(ref_list)
        order = [i for i in range(n) if i != target_orig]
        insert_pos = order.index(max_ref) + 1
        order.insert(insert_pos, target_orig)

        # Mapping from old index to new index
        old_to_new = {old: new for new, old in enumerate(order)}
        new_bonds = {norm_bond(old_to_new[u], old_to_new[v]) for u, v in orig_bonds}
        new_bo = {norm_bond(old_to_new[u], old_to_new[v]): bo_val for (u, v), bo_val in orig_bo.items() if bo_val != 1}

        # Permute symbols and coordinates
        permuted_symbols = [self.zatoms[i].symbol for i in order]
        permuted_coords = self.coords[order]

        # Build new zatoms preserving bonds
        new_zatoms = cartesian_to_zmatrix(permuted_symbols, permuted_coords, user_bonds=list(new_bonds), strict_bonds=True)

        # Explicitly enforce the requested references on the moved target atom
        new_target_idx = old_to_new[target_orig]
        new_ref1 = old_to_new[ref1_orig]
        target_atom = new_zatoms[new_target_idx]
        target_atom.ref1 = new_ref1
        target_atom.dist = calculate_bond_distance(permuted_coords[new_target_idx], permuted_coords[new_ref1])

        if mode in ("angle", "dihedral"):
            target_atom.ref2 = old_to_new[ref2_orig]
        elif target_atom.ref2 == new_ref1:
            target_atom.ref2 = next((k for k in range(new_target_idx) if k != new_ref1), None)
        if target_atom.ref2 is not None:
            target_atom.angle = calculate_angle(permuted_coords[new_target_idx], permuted_coords[target_atom.ref1], permuted_coords[target_atom.ref2])

        if mode == "dihedral":
            target_atom.ref3 = old_to_new[ref3_orig]
        elif target_atom.ref3 in (target_atom.ref1, target_atom.ref2):
            target_atom.ref3 = next((k for k in range(new_target_idx) if k not in (target_atom.ref1, target_atom.ref2)), None)
        if target_atom.ref2 is not None and target_atom.ref3 is not None:
            target_atom.dihedral = calculate_dihedral(permuted_coords[new_target_idx], permuted_coords[target_atom.ref1], permuted_coords[target_atom.ref2], permuted_coords[target_atom.ref3])

        tree = {norm_bond(a.ref1, i) for i, a in enumerate(new_zatoms) if a.ref1 is not None}
        self.cross_bonds = {b for b in new_bonds if b not in tree}
        self.set_molecule_from_zatoms(new_zatoms, exact_coords=permuted_coords)
        self.viewer.bond_orders = new_bo
        self.table_widget.highlight_row(new_target_idx)
        self.viewer.highlighted_feature_atoms = {}
        self.viewer.selected_atom_idx = new_target_idx
        self.viewer.multi_selected = [new_target_idx]
        self._update_adjust_panel()
        self.viewer.update()

        t_name = f"{permuted_symbols[new_target_idx]}{new_target_idx+1}"
        r1_name = f"{permuted_symbols[new_ref1]}{new_ref1+1}"
        msg = f"Adjusted {mode.capitalize()}: {t_name} moved after its references (now referencing {r1_name})."
        self.statusBar().showMessage(msg, 6000)

    def _cancel_reorder(self):
        if self.viewer.reordering_active:
            self.viewer.reordering_active = False
            self.viewer.reorder_list = []
            self.btn_reorder_zmat.setText("🔄 Reorder ZMAT")
            self.btn_reorder_zmat.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold; font-size: 11px; padding: 4px 6px;")
            self.viewer.update()
            self.statusBar().showMessage("Z-matrix reordering cancelled.", 3000)

