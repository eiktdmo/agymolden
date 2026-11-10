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
    get_symmetry_orbits,
    get_atom_orbit_and_operations,
    map_atom_by_operation,
)

log = logging.getLogger(__name__)



class EditingMixin:
    def _on_add_line(self):
        """
        Molden 'Add Line' workflow:
        1. Opens Periodic Table to choose element.
        2. If 1st atom: placed at origin directly.
        3. If 2nd atom: prompt user to click atom in 3D viewport to define bond.
        4. If 3rd atom: prompt user to click 2 atoms (bond, angle).
        5. If 4th+ atom: prompt user to click 3 atoms (bond, angle, dihedral).
        """
        dlg = PeriodicTableDialog(self, initial_coordination=getattr(self, "last_coordination_template", "auto"))
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        elem = dlg.chosen_element
        if not elem:
            return
        self._on_select_brush_element(elem, coord_tpl=getattr(dlg, "chosen_coordination", "auto") or "auto")

    def _on_select_brush_element(self, elem: str, coord_tpl: str = "auto"):
        """Set active element brush for 3D dragging, canvas double-clicking, and atom mutations."""
        self.last_coordination_template = coord_tpl
        self.last_added_element = elem
        self.viewer.brush_element = elem
        self.last_item_type = "element"
        self.stored_fragment = None
        self.last_added_fragment = None
        self.pending_add_atom_mode = True
        self._update_viewer_drag_label()
        self.statusBar().showMessage(f"Active Drawing Brush: {elem}", 2500)

        n = len(self.zatoms)
        if n == 0:
            self._push_undo_state()
            auto_h = bool(getattr(self, 'chk_auto_h', None) and self.chk_auto_h.isChecked() and elem != "H" and elem != "X")
            if coord_tpl != "auto":
                from .coordination import get_coordination_vectors
                dirs = get_coordination_vectors(elem, coord_tpl)
                syms = [elem]
                pts = [np.array([0.0, 0.0, 0.0])]
                bds = []
                if auto_h and dirs:
                    d_len = estimate_bond_length(elem, "H")
                    for k_i, dv in enumerate(dirs):
                        syms.append("H")
                        pts.append(dv * d_len)
                        bds.append((0, k_i + 1))
                self._commit_explicit_structure(syms, np.array(pts), bds, {}, relax_iters=0, add_h=False, fix_valence=False)
            else:
                self._commit_explicit_structure([elem], np.array([[0.0, 0.0, 0.0]]), [], {}, relax_iters=50, add_h=auto_h)
            self.table_widget.highlight_row(0)
            h_str = " with valence hydrogens" if auto_h else ""
            self.statusBar().showMessage(f"Added initial atom {elem}1{h_str} at origin", 4000)
            self.pending_add_atom_mode = False
            return

        auto_h = bool(getattr(self, 'chk_auto_h', None) and self.chk_auto_h.isChecked())
        auto_h_str = " (with adjusted hydrogens)" if auto_h else ""

        # Activate Molden interactive reference picking (ref1 for bond, ref2 for angle, ref3 for dihedral)
        self.pending_new_element = elem
        self.pending_picks_needed = min(n, 3)
        self.picked_references = []
        self.viewer.picking_active = True
        self.viewer.picked_references = []

        if self.pending_picks_needed == 1:
            banner = f"Adding {elem}: Click atom in 3D viewport to define Bond (or click/drag canvas)."
        elif self.pending_picks_needed == 2:
            banner = f"Adding {elem}: Click 1st atom for Bond, then 2nd for Angle (or click/drag canvas)."
        else:
            banner = f"Adding {elem}: Click 1st atom for Bond, 2nd for Angle, 3rd for Dihedral (or click/drag canvas)."

        self.viewer.picking_banner = banner
        self.statusBar().showMessage(banner, 10000)
        self.viewer.update()

    def _on_quick_stamp_fragment(self, frag_name: str):
        """1-Click stamp functional group or ring onto selected atom or spawn in space."""
        from .fragments import FRAGMENTS
        if frag_name not in FRAGMENTS:
            return

        frag = FRAGMENTS[frag_name]
        idx = self.viewer.selected_atom_idx

        self._push_undo_state()
        if idx is not None and 0 <= idx < len(self.zatoms):
            # Stamp onto selected anchor atom
            conn_idx = int(frag.get("connect_idx", 0))
            self._attach_fragment_to_atom_or_orbit(idx, frag, conn_idx=conn_idx, relax_iters=100)
            self.statusBar().showMessage(f"Stamped {frag_name} onto Atom #{idx + 1}", 3500)
        else:
            # Spawn fragment directly into empty viewport
            syms = list(frag["symbols"])
            pts = np.array(frag["coords"], dtype=np.float64)
            if len(self.zatoms) == 0:
                self.set_molecule_from_cartesian(syms, pts)
                self.statusBar().showMessage(f"Created {frag_name} in empty viewport", 3500)
            else:
                # Offset fragment away from existing center of mass
                com = np.mean(self.coords, axis=0)
                pts_offset = pts + com + np.array([3.5, 0.0, 0.0])
                all_syms = [a.symbol for a in self.zatoms] + syms
                all_coords = np.vstack([self.coords, pts_offset])
                self.set_molecule_from_cartesian(all_syms, all_coords)
                self.statusBar().showMessage(f"Added {frag_name} fragment to viewport", 3500)

    def _on_atom_picked_for_connectivity(self, atom_idx: int):
        if not self.viewer.picking_active or not self.pending_new_element:
            return

        n = len(self.zatoms)
        if atom_idx < 0 or atom_idx >= n:
            self.viewer.picking_active = False
            self.viewer.picking_banner = ""
            self.viewer.picked_references = []
            self.pending_new_element = None
            self.picked_references = []
            return

        if atom_idx in self.picked_references:
            sym_label = self.viewer.symbols[atom_idx] if atom_idx < len(self.viewer.symbols) else ""
            self.statusBar().showMessage(f"Atom {sym_label}{atom_idx+1} already selected! Pick a different atom.", 3000)
            return

        self.picked_references.append(atom_idx)
        self.viewer.picked_references = list(self.picked_references)
        self.viewer.update()

        n_picked = len(self.picked_references)
        elem = self.pending_new_element

        if n_picked < self.pending_picks_needed:
            # Update banner for remaining picks
            if self.pending_picks_needed == 2:
                self.viewer.picking_banner = f"Selected #{self.picked_references[0]+1} for Bond. Now click atom for Angle."
            elif self.pending_picks_needed == 3:
                if n_picked == 1:
                    self.viewer.picking_banner = f"Selected #{self.picked_references[0]+1} for Bond. Click 2nd atom for Angle."
                elif n_picked == 2:
                    self.viewer.picking_banner = f"Selected #{self.picked_references[0]+1}, #{self.picked_references[1]+1}. Click 3rd atom for Dihedral."
            self.statusBar().showMessage(self.viewer.picking_banner, 8000)
            return

        # Done picking all required reference atoms! Construct new ZAtom
        ref1 = self.picked_references[0]
        if ref1 >= n:
            self.viewer.picking_active = False
            self.viewer.picking_banner = ""
            self.viewer.picked_references = []
            self.pending_new_element = None
            self.picked_references = []
            return

        self._push_undo_state()
        sym1 = self.zatoms[ref1].symbol
        def_dist = estimate_bond_length(elem, sym1)

        if n == 1:
            new_atom = ZAtom(symbol=elem, ref1=ref1, dist=def_dist)
        elif n == 2:
            ref2 = self.picked_references[1]
            if ref2 >= n:
                ref2 = 1 if ref1 == 0 else 0
            new_atom = ZAtom(symbol=elem, ref1=ref1, dist=def_dist, ref2=ref2, angle=109.5)
        else:
            ref2 = self.picked_references[1] if self.picked_references[1] < n else (0 if ref1 != 0 else 1)
            ref3 = self.picked_references[2] if self.picked_references[2] < n else (next((k for k in range(n) if k not in (ref1, ref2)), 0))
            new_atom = ZAtom(symbol=elem, ref1=ref1, dist=def_dist, ref2=ref2, angle=109.5, ref3=ref3, dihedral=120.0)

        # Reset picking mode
        self.viewer.picking_active = False
        self.viewer.picking_banner = ""
        self.viewer.picked_references = []
        self.pending_new_element = None
        self.picked_references = []

        self.zatoms.append(new_atom)
        self.set_molecule_from_zatoms(self.zatoms)
        self.table_widget.highlight_row(len(self.zatoms) - 1)
        self.statusBar().showMessage(f"Added atom {new_atom.symbol}{len(self.zatoms)} connected to atom #{ref1+1}", 4000)

    def _update_viewer_drag_label(self):
        if not hasattr(self, "viewer"):
            return
        if getattr(self, "last_item_type", "element") == "fragment":
            frag = self.stored_fragment or self.last_added_fragment
            name = frag.get("name", "Fragment") if frag else "Fragment"
            self.viewer.active_drag_label = f"+ {name}"
        else:
            elem = getattr(self.viewer, "brush_element", None) or getattr(self, "last_added_element", None) or "C"
            auto_h = bool(getattr(self, "chk_auto_h", None) and self.chk_auto_h.isChecked())
            self.viewer.active_drag_label = f"+ {elem}" + (" (with H's)" if auto_h else "")
        self.viewer.update()

    def _on_fragment_combo_changed(self, frag_name: str):
        if frag_name in FRAGMENTS:
            frag = FRAGMENTS[frag_name]
            self.last_added_fragment = dict(frag)
            self.last_added_fragment["name"] = frag_name
            self.last_item_type = "fragment"

            # Populate combo_frag_connect with the fragment's atoms
            if hasattr(self, "combo_frag_connect"):
                self.combo_frag_connect.blockSignals(True)
                self.combo_frag_connect.clear()
                for i, sym in enumerate(frag["symbols"]):
                    self.combo_frag_connect.addItem(f"Atom #{i + 1}: {sym}", i)
                default_idx = int(frag.get("connect_idx", 0))
                if 0 <= default_idx < self.combo_frag_connect.count():
                    self.combo_frag_connect.setCurrentIndex(default_idx)
                self.combo_frag_connect.blockSignals(False)
                self.last_added_fragment["connect_idx"] = default_idx

            self._update_viewer_drag_label()
            self.statusBar().showMessage(
                f"Selected fragment: {frag_name}. Drag from an atom or double-click to attach.", 4500
            )

    def _on_frag_connect_atom_changed(self, idx: int):
        conn_idx = self.combo_frag_connect.currentData() if hasattr(self, "combo_frag_connect") else 0
        if conn_idx is None:
            conn_idx = max(0, idx)
        if self.last_added_fragment is not None:
            self.last_added_fragment["connect_idx"] = conn_idx
            f_syms = self.last_added_fragment.get("symbols", [])
            conn_sym = f_syms[conn_idx] if 0 <= conn_idx < len(f_syms) else ""
            self.statusBar().showMessage(f"Fragment attachment atom set to #{conn_idx + 1} ({conn_sym}).", 3000)

    def _on_canvas_dragged_or_clicked(self, start_pos: QtCore.QPoint, end_pos: QtCore.QPoint, drag_d: int):
        """
        Handle click or drag on empty canvas when Add Atom mode is active.
        Places the atom at the 3D position in the camera view plane,
        completing valence with hydrogens if 'Adjust Hydrogens' is active.
        """
        if not getattr(self, "pending_add_atom_mode", False):
            return
        self.pending_add_atom_mode = False

        elem = getattr(self.viewer, "brush_element", None) or getattr(self, "last_added_element", None) or "C"
        self._push_undo_state()

        # Place atom at canvas end_pos in 3D camera projection plane
        w, h = max(1, self.viewer.width()), max(1, self.viewer.height())
        right_w, up_w = self.viewer._get_camera_axes()
        scale = (self.viewer.zoom * 0.8) / h
        dx = float(end_pos.x() - w * 0.5)
        dy = float(end_pos.y() - h * 0.5)
        centroid = np.mean(self.coords, axis=0) if len(self.coords) > 0 else np.zeros(3)
        placed_pos = centroid + (dx * scale) * right_w - (dy * scale) * up_w

        curr_syms = [a.symbol for a in self.zatoms]
        curr_coords = np.copy(self.coords)
        curr_bonds = self._current_explicit_bonds()
        curr_bo = self._current_bond_orders()
        n_old = len(curr_syms)

        auto_h = bool(getattr(self, "chk_auto_h", None) and self.chk_auto_h.isChecked() and elem != "H" and elem != "X")
        new_syms = [elem]
        new_coords = [placed_pos]
        new_bonds = []
        new_bo = {}

        coord_tpl = getattr(self, "last_coordination_template", "auto") or "auto"
        if auto_h:
            if coord_tpl != "auto":
                from .coordination import get_coordination_vectors
                dirs = get_coordination_vectors(elem, coord_tpl)
                d = estimate_bond_length(elem, "H")
                for k, dvec in enumerate(dirs):
                    new_syms.append("H")
                    new_coords.append(placed_pos + dvec * d)
                    new_bonds.append((0, k + 1))
            elif elem in DEFAULT_VALENCES:
                val = DEFAULT_VALENCES[elem]
                dirs = _calculate_hydrogen_directions([], val)
                d = estimate_bond_length(elem, "H")
                for k, dvec in enumerate(dirs[:val]):
                    new_syms.append("H")
                    new_coords.append(placed_pos + dvec * d)
                    new_bonds.append((0, k + 1))

        combined_syms = curr_syms + new_syms
        combined_coords = np.vstack([curr_coords, np.array(new_coords)]) if n_old > 0 else np.array(new_coords)
        combined_bonds = curr_bonds + [(n_old + u, n_old + v) for u, v in new_bonds]
        combined_bo = dict(curr_bo)
        for (u, v), o in new_bo.items():
            combined_bo[norm_bond(n_old + u, n_old + v)] = o

        self._commit_explicit_structure(combined_syms, combined_coords, combined_bonds, combined_bo, relax_iters=0, add_h=False, fix_valence=False)
        h_str = " with valence hydrogens" if auto_h else ""
        self.statusBar().showMessage(f"Added {elem}{h_str} in free canvas space", 4500)

    def _on_quick_add_atom_dragged(self, anchor_idx: int, dx: float, dy: float):
        """
        Handle Left-Click Drag from an atom.
        Adds the last atom or fragment saved in memory along the exact 3D drag direction.
        - If a fragment is in memory: attaches the complete fragment along the drag direction,
          replacing an H if dragged from H or preserving anchor valence.
        - If an atom is in memory: attaches that atom along the drag direction,
          and if 'Adjust Hydrogens' is checked, adjusts valence and adds geometric hydrogens.
        """
        n = len(self.zatoms)
        if n == 0 or anchor_idx < 0 or anchor_idx >= n:
            return

        self._push_undo_state()

        symbols = [a.symbol for a in self.zatoms]
        coords = np.copy(self.coords)
        bonds = self._current_explicit_bonds()
        bo = self._current_bond_orders()
        adj = build_adj(n, bonds)

        anchor_sym = symbols[anchor_idx]
        anchor_pos = np.copy(coords[anchor_idx])

        # Convert screen drag delta to 3D world unit direction vector
        dir_world = self.viewer.screen_displacement_to_world(dx, dy)
        nd = np.linalg.norm(dir_world)
        dir_u = dir_world / nd if nd > 1e-4 else np.array([1.0, 0.0, 0.0])

        active_frag = (self.stored_fragment or self.last_added_fragment) if getattr(self, "last_item_type", "element") == "fragment" else None
        auto_h = bool(getattr(self, "chk_auto_h", None) and self.chk_auto_h.isChecked())

        if active_frag is not None:
            # === Dragging to add Fragment from memory ===
            f_syms = list(active_frag["symbols"])
            conn_idx = int(active_frag.get("connect_idx", 0))
            if conn_idx < 0 or conn_idx >= len(f_syms):
                conn_idx = 0
            n_sites = self._attach_fragment_to_atom_or_orbit(
                anchor_idx, active_frag, conn_idx=conn_idx, custom_dir_u=dir_u, relax_iters=120
            )
            frag_name = active_frag.get("name", "fragment")
            sym_note = f" across {n_sites} symmetry-equivalent sites" if n_sites > 1 else ""
            self.statusBar().showMessage(
                f"Added {frag_name} ({len(f_syms)} atoms) bound to {anchor_sym}{anchor_idx+1}{sym_note}", 4500
            )
        else:
            # === Dragging to add Atom from memory ===
            elem = getattr(self.viewer, "brush_element", None) or getattr(self, "last_added_element", None) or "C"
            sym_res = self._get_active_imposed_symmetry()
            if sym_res is not None and anchor_sym in ("H", "F", "Cl", "Br", "I") and len(adj[anchor_idx]) == 1:
                # Symmetric substitution of H / halogen orbit by elem
                atom_frag = {"name": elem, "symbols": [elem], "coords": [(0.0, 0.0, 0.0)], "connect_idx": 0}
                n_sites = self._attach_fragment_to_atom_or_orbit(
                    anchor_idx, atom_frag, conn_idx=0, custom_dir_u=None, relax_iters=90
                )
                self.pending_add_atom_mode = False
                self.statusBar().showMessage(
                    f"Replaced {n_sites} symmetry-equivalent {anchor_sym} atoms with {elem}", 4500
                )
                return
            remove = set()
            attach_to: List[int] = []

            if anchor_sym in ("H", "F", "Cl", "Br", "I") and len(adj[anchor_idx]) == 1:
                parent = adj[anchor_idx][0]
                bond_len = estimate_bond_length(elem, symbols[parent])
                p_new = coords[parent] + dir_u * bond_len
                remove.add(anchor_idx)
                attach_to = [parent]
            else:
                bond_len = estimate_bond_length(elem, anchor_sym)
                p_new = anchor_pos + dir_u * bond_len
                attach_to = [anchor_idx]
                h_nbs = [j for j in adj[anchor_idx] if symbols[j] == "H"]
                val = DEFAULT_VALENCES.get(anchor_sym, 4)
                if auto_h and h_nbs:
                    h_nbs.sort(key=lambda h: float(np.dot(coords[h] - anchor_pos, dir_u)), reverse=True)
                    remove.add(h_nbs[0])

            new_idx = n
            all_syms = symbols + [elem]
            all_coords = np.vstack([coords, p_new.reshape((1, 3))])
            all_bonds = list(bonds) + [norm_bond(p, new_idx) for p in attach_to]
            all_bo = dict(bo)
            for p in attach_to:
                all_bo.pop(norm_bond(p, new_idx), None)  # Strictly single bond

            if remove:
                all_syms, all_coords, all_bonds, all_bo, o2n_rem = remove_atoms(all_syms, all_coords, all_bonds, remove, all_bo)
                target_set = {o2n_rem[k] for k in attach_to + [new_idx] if k in o2n_rem}
            else:
                target_set = set(attach_to) | {new_idx}

            self._commit_explicit_structure(
                all_syms, all_coords, all_bonds, all_bo,
                relax_iters=90, add_h=auto_h, fix_valence=auto_h, target_indices=target_set
            )
            self.pending_add_atom_mode = False
            h_str = " (with adjusted hydrogens)" if auto_h else ""
            self.statusBar().showMessage(
                f"Added {elem} bound to {anchor_sym}{anchor_idx+1}{h_str}", 4000
            )

    def _on_cross_bond_drawn(self, atom1_idx: int, atom2_idx: int):
        """
        Handle Avogadro-style drag-and-drop between two existing atoms to form a bond.
        """
        n = len(self.zatoms)
        if atom1_idx < 0 or atom1_idx >= n or atom2_idx < 0 or atom2_idx >= n or atom1_idx == atom2_idx:
            return

        b_key = (min(atom1_idx, atom2_idx), max(atom1_idx, atom2_idx))
        curr_bonds = self._current_explicit_bonds()

        # If already bonded: increment bond order (single -> double -> triple)
        if b_key in curr_bonds:
            self._on_bond_double_clicked(atom1_idx, atom2_idx)
            return

        # Strictly disallow forming bonds between two hydrogen atoms
        if self.zatoms[atom1_idx].symbol == "H" and self.zatoms[atom2_idx].symbol == "H":
            self.statusBar().showMessage("Cannot form a bond between two hydrogen atoms.", 4000)
            return

        # If not bonded: form bond using our clean explicit bond topology mechanism
        self.viewer.multi_selected = [atom1_idx, atom2_idx]
        self._on_close_ring_clicked()

    def _on_bond_torsion_dragged(self, u: int, v: int, delta_deg: float):
        """Rotate substituent branch on atom v around the u -> v bond axis (Alt+Left-Drag)."""
        n = len(self.zatoms)
        if not (0 <= u < n and 0 <= v < n and len(self.coords) == n):
            return

        from .smartslides import _get_rotatable_side_subgraph, rodrigues_rotation
        curr_bonds = self._current_explicit_bonds()

        moving = _get_rotatable_side_subgraph(n, curr_bonds, v, u, None)
        if not moving or len(moving) >= n:
            return

        axis_vec = self.coords[v] - self.coords[u]
        axis_len = float(np.linalg.norm(axis_vec))
        if axis_len < 1e-6:
            return

        axis_u = axis_vec / axis_len
        pivot = self.coords[v].copy()
        R = rodrigues_rotation(axis_u, math.radians(delta_deg))

        new_coords = np.copy(self.coords)
        for idx in moving:
            new_coords[idx] = pivot + R @ (self.coords[idx] - pivot)

        self.coords = new_coords
        # Update Z-matrix values to match new 3D Cartesian coordinates
        new_zatoms = cartesian_to_zmatrix(
            [a.symbol for a in self.zatoms],
            self.coords,
            existing_zatoms=self.zatoms,
            user_bonds=curr_bonds,
            strict_bonds=getattr(self, "strict_topology", False),
        )
        self.zatoms = new_zatoms
        self.table_widget.load_zatoms(self.zatoms)
        self.viewer.update_coords(self.coords, user_bonds=curr_bonds)
        self._sync_raw_text()
        self._update_adjust_panel()
        self._update_chemical_summary()
        self.statusBar().showMessage(f"Torsion: rotated branch on atom #{v+1} around bond {self.zatoms[u].symbol}{u+1}-{self.zatoms[v].symbol}{v+1}", 1500)

    # ------------------------------------------------------------------
    # Explicit-topology helpers (no distance-based bond guessing)
    # ------------------------------------------------------------------
    def _current_explicit_bonds(self) -> List[Tuple[int, int]]:
        """Return the bond list exactly as currently displayed in the viewer (cleaned)."""
        n = len(self.zatoms)
        bonds = None
        vb = getattr(self.viewer, "bonds", None)
        if vb is not None and len(getattr(self.viewer, "symbols", [])) == n:
            bonds = [tuple(b) for b in vb]
        if bonds is None:
            symbols = [a.symbol for a in self.zatoms]
            ub = [(a.ref1, i) for i, a in enumerate(self.zatoms) if a.ref1 is not None and a.ref1 < i]
            ub += list(self.cross_bonds)
            bonds = detect_connectivity(symbols, self.coords, user_bonds=ub,
                                        strict=getattr(self, "strict_topology", False))
        return sorted({norm_bond(u, v) for u, v in bonds if 0 <= u < n and 0 <= v < n and u != v})

    def _current_bond_orders(self) -> Dict[Tuple[int, int], int]:
        n = len(self.zatoms)
        return {
            norm_bond(u, v): val
            for (u, v), val in getattr(self.viewer, "bond_orders", {}).items()
            if 0 <= u < n and 0 <= v < n and u != v and val != 1
        }

    def _get_active_imposed_symmetry(self) -> Optional[PointGroupResult]:
        """Return the active PointGroupResult if Impose Symmetry is enabled and has order > 1."""
        if not getattr(self, "symmetry_imposed", False):
            return None
        res = getattr(self, "imposed_sym_res", None) or getattr(self, "_current_sym_res", None)
        if res is None or res.order <= 1:
            tol = self.spin_sym_tol.value() if hasattr(self, "spin_sym_tol") else 0.15
            symbols = [a.symbol for a in self.zatoms]
            if len(symbols) > 1:
                res = find_symmetry_operations(symbols, self.coords, tolerance=tol)
                self.imposed_sym_res = res
                self._current_sym_res = res
        return res if (res is not None and res.order > 1) else None

    def _attach_fragment_to_atom_or_orbit(
        self,
        anchor_idx: int,
        frag: Dict,
        conn_idx: Optional[int] = None,
        custom_dir_u: Optional[np.ndarray] = None,
        relax_iters: int = 120,
    ) -> int:
        """
        Attach or substitute `frag` at `anchor_idx` (and across its entire symmetry orbit if
        `symmetry_imposed` is active). Returns the number of symmetry-equivalent sites modified.
        """
        symbols = [a.symbol for a in self.zatoms]
        coords = np.copy(self.coords)
        n = len(symbols)
        if n == 0 or anchor_idx < 0 or anchor_idx >= n:
            return 0

        frag_symbols = list(frag["symbols"])
        frag_coords = np.array(frag["coords"], dtype=np.float64)
        if conn_idx is None:
            conn_idx = int(frag.get("connect_idx", 0))
        if conn_idx < 0 or conn_idx >= len(frag_symbols):
            conn_idx = 0

        frag_coords = frag_coords - frag_coords[conn_idx]
        conn_sym = frag_symbols[conn_idx]
        f_bonds = frag.get("bonds")
        if f_bonds is None:
            f_bonds = detect_connectivity(frag_symbols, frag_coords) if len(frag_symbols) > 1 else []

        bonds = self._current_explicit_bonds()
        bo = self._current_bond_orders()
        adj = build_adj(n, bonds)
        auto_h = bool(getattr(self, "chk_auto_h", None) is None or self.chk_auto_h.isChecked())

        anchor_sym = symbols[anchor_idx]
        anchor_pos = np.copy(coords[anchor_idx])

        remove_seed: Optional[int] = None
        parent_seed: int = anchor_idx

        # Monovalent terminal atoms (H, F, Cl, Br, I) bonded to a parent atom are REPLACED
        # when a substituent or element is attached/substituted onto them (e.g. replacing Cl or H by Phenyl).
        if anchor_sym in ("H", "F", "Cl", "Br", "I") and len(adj[anchor_idx]) == 1:
            parent_seed = adj[anchor_idx][0]
            if custom_dir_u is not None:
                dir_u = custom_dir_u
            else:
                v_d = anchor_pos - coords[parent_seed]
                nd = float(np.linalg.norm(v_d))
                dir_u = v_d / nd if nd > 1e-6 else np.array([0.0, 0.0, 1.0])
            bond_len = estimate_bond_length(conn_sym, symbols[parent_seed])
            root_pos = coords[parent_seed] + dir_u * bond_len
            remove_seed = anchor_idx
        else:
            parent_seed = anchor_idx
            heavy_nbs = [j for j in adj[anchor_idx] if symbols[j] != "H"]
            h_nbs = [j for j in adj[anchor_idx] if symbols[j] == "H"]
            mono_nbs = [j for j in adj[anchor_idx] if symbols[j] in ("F", "Cl", "Br", "I") and len(adj[j]) == 1]
            val = DEFAULT_VALENCES.get(anchor_sym, 4)

            # Detect if anchor_idx is an aromatic or sp2 center whose valence is already satisfied at 3 sigma bonds
            is_aromatic_or_sp2_satisfied = False
            if anchor_sym in ("C", "N") and len(adj[anchor_idx]) == 3:
                # Check if any neighbor bond is in aromatic / sp2 range (< 1.44 A) or has explicit bo >= 2
                for nb in adj[anchor_idx]:
                    if bo.get(norm_bond(anchor_idx, nb), 1) >= 2:
                        is_aromatic_or_sp2_satisfied = True
                        break
                    if symbols[nb] in ("C", "N", "O", "S") and float(np.linalg.norm(coords[anchor_idx] - coords[nb])) < 1.44:
                        is_aromatic_or_sp2_satisfied = True
                        break

            tot_b = sum(bo.get(norm_bond(anchor_idx, j), 1) for j in adj[anchor_idx])
            if is_aromatic_or_sp2_satisfied and tot_b < val:
                tot_b = val

            if custom_dir_u is not None:
                dir_u = custom_dir_u
                if (tot_b >= val or auto_h) and h_nbs:
                    h_nbs.sort(key=lambda h: float(np.dot(coords[h] - anchor_pos, dir_u)), reverse=True)
                    remove_seed = h_nbs[0]
                elif (tot_b >= val or auto_h) and mono_nbs:
                    mono_nbs.sort(key=lambda m: float(np.dot(coords[m] - anchor_pos, dir_u)), reverse=True)
                    remove_seed = mono_nbs[0]
            elif h_nbs:
                h_idx = h_nbs[0]
                v_h = coords[h_idx] - anchor_pos
                nv = float(np.linalg.norm(v_h))
                dir_u = v_h / nv if nv > 1e-6 else np.array([0.0, 0.0, 1.0])
                if tot_b >= val or auto_h:
                    remove_seed = h_idx
            elif (tot_b >= val or auto_h) and mono_nbs:
                m_idx = mono_nbs[0]
                v_m = coords[m_idx] - anchor_pos
                nv = float(np.linalg.norm(v_m))
                dir_u = v_m / nv if nv > 1e-6 else np.array([0.0, 0.0, 1.0])
                remove_seed = m_idx
            elif heavy_nbs:
                v_d = anchor_pos - np.mean(coords[heavy_nbs], axis=0)
                nd = float(np.linalg.norm(v_d))
                dir_u = v_d / nd if nd > 1e-6 else np.array([0.0, 0.0, 1.0])
            elif n > 1:
                v_d = anchor_pos - np.mean(coords, axis=0)
                nd = float(np.linalg.norm(v_d))
                dir_u = v_d / nd if nd > 1e-6 else np.array([0.0, 0.0, 1.0])
            else:
                dir_u = np.array([0.0, 0.0, 1.0])

            bond_len = estimate_bond_length(conn_sym, anchor_sym)
            root_pos = anchor_pos + dir_u * bond_len

        f_axis = fragment_outward_axis(frag_coords, root_idx=conn_idx)
        R_align = rotation_between(f_axis, dir_u)
        placed_seed = frag_coords @ R_align.T + root_pos

        # Check if Impose Symmetry is active
        sym_res = self._get_active_imposed_symmetry()
        if sym_res is not None:
            orbit_ref = remove_seed if remove_seed is not None else anchor_idx
            com, orbit_ops = get_atom_orbit_and_operations(
                symbols, coords, orbit_ref, result=sym_res, tolerance=max(0.25, sym_res.tolerance * 1.5)
            )
        else:
            com = np.mean(coords, axis=0)
            orbit_ops = [(remove_seed if remove_seed is not None else anchor_idx, np.eye(3, dtype=np.float64))]

        all_syms = list(symbols)
        all_coords_list = [coords]
        all_bonds = list(bonds)
        all_bo = dict(bo)
        remove_set = set()
        attached_parents = set()
        added_atom_indices = set()

        frag_bo_map = dict(frag.get("bond_orders") or {})
        m_frag = len(frag_symbols)

        for eq_idx, R_op in orbit_ops:
            base_idx = len(all_syms)
            # Transform placed_seed around COM using R_op
            placed_eq = (R_op @ (placed_seed - com).T).T + com
            parent_eq = map_atom_by_operation(symbols, coords, parent_seed, R_op, com=com)
            if remove_seed is not None:
                rem_eq = map_atom_by_operation(symbols, coords, remove_seed, R_op, com=com)
                remove_set.add(rem_eq)

            all_syms.extend(frag_symbols)
            all_coords_list.append(placed_eq)
            for u, v in f_bonds:
                all_bonds.append(norm_bond(base_idx + u, base_idx + v))
            for (u, v), o in frag_bo_map.items():
                all_bo[norm_bond(base_idx + u, base_idx + v)] = o

            root_eq = base_idx + conn_idx
            all_bonds.append(norm_bond(parent_eq, root_eq))
            all_bo.pop(norm_bond(parent_eq, root_eq), None)  # single bond to parent
            attached_parents.add(parent_eq)
            added_atom_indices.update(range(base_idx, base_idx + m_frag))

        all_coords = np.vstack(all_coords_list)
        target_set = set(attached_parents) | set(added_atom_indices)
        if remove_set:
            all_syms, all_coords, all_bonds, all_bo, o2n_rem = remove_atoms(
                all_syms, all_coords, all_bonds, remove_set, all_bo
            )
            target_set = {o2n_rem[k] for k in target_set if k in o2n_rem}

        has_explicit_bo = bool(frag_bo_map)
        # When Impose Symmetry is active (or multiple symmetric sites are substituted simultaneously),
        # keep exact symmetric geometry without force-field distortion so the imposed subgroup holds!
        effective_relax = 0 if sym_res is not None else relax_iters
        self._commit_explicit_structure(
            all_syms,
            all_coords,
            all_bonds,
            all_bo,
            relax_iters=effective_relax,
            add_h=False,
            fix_valence=not (has_explicit_bo or sym_res is not None or len(orbit_ops) > 1),
            target_indices=target_set,
        )
        return len(orbit_ops)

    def _commit_explicit_structure(self, symbols, coords, bonds, bond_orders=None,
                                   relax_iters: int = 120, add_h: Optional[bool] = None,
                                   fix_valence: bool = True,
                                   target_indices: Optional[set] = None):
        """Fix valence on the explicit graph, relax with exactly these bonds, rebuild the Z-matrix
        and push everything to the viewer while keeping every bond (tree + cross bonds).
        With fix_valence=False the given topology is used as-is (no H added/removed)."""
        if add_h is None:
            add_h = bool(getattr(self, "chk_auto_h", None) is None or self.chk_auto_h.isChecked())
        if fix_valence:
            symbols, coords, bonds, bo = fix_valence_explicit(
                symbols, coords, bonds, bond_orders, add_h=add_h, target_indices=target_indices
            )
        else:
            bonds = sorted({norm_bond(u, v) for u, v in bonds})
            bo = {norm_bond(*k): v for k, v in (bond_orders or {}).items() if v != 1}
        if relax_iters > 0 and len(symbols) > 1:
            try:
                # Pass full explicit bond orders (including 1) so optimizer enforces single-bond equilibrium distance
                full_bo = {norm_bond(*k): v for k, v in (bond_orders or {}).items()}
                coords, _ = optimize_geometry_force_field(symbols, coords, user_bonds=bonds,
                                                          max_iter=relax_iters, strict_bonds=True,
                                                          user_bond_orders=full_bo)
            except Exception:
                log.warning("Pre-relaxation failed; continuing without it", exc_info=True)
        symbols, coords, bonds, bo, o2n = bfs_reorder(symbols, coords, bonds, bo)
        if getattr(self, "frozen_atoms", None):
            self.frozen_atoms = {o2n[i] for i in self.frozen_atoms if i in o2n}
            if hasattr(self, "_update_frozen_buttons_ui"):
                self._update_frozen_buttons_ui()
        new_zatoms = cartesian_to_zmatrix(symbols, coords, user_bonds=bonds, strict_bonds=True)
        tree = {norm_bond(a.ref1, i) for i, a in enumerate(new_zatoms) if a.ref1 is not None}
        self.cross_bonds = {b for b in bonds if b not in tree}
        self.strict_topology = True
        self.variables = {}
        self.set_molecule_from_zatoms(new_zatoms, exact_coords=coords)
        self.viewer.bond_orders = {b: o for b, o in bo.items() if o != 1}
        self.viewer.update()
        return o2n

    def _on_close_ring_clicked(self):
        """
        Form Bond action (generalised ring closure):
        Creates a bond between the 2 selected centers. Selecting a hydrogen means "use its parent
        and remove this hydrogen". Existing bonds are kept exactly, no extra bonds are created, the
        excess hydrogens are removed and valence is verified over the whole molecule.
        """
        sel = list(self.viewer.multi_selected)
        if len(sel) < 2 and getattr(self.viewer, "selected_atom_idx", None) is not None:
            sel.append(self.viewer.selected_atom_idx)
        sel = [x for i, x in enumerate(sel) if x not in sel[:i] and 0 <= x < len(self.zatoms)]
        if len(sel) != 2:
            QtWidgets.QMessageBox.information(
                self, "Select 2 Atoms",
                "Please select exactly 2 atoms to form a bond.\n(Tip: Shift+Click two atoms, or Shift+Drag a box.)"
            )
            return

        symbols = [a.symbol for a in self.zatoms]
        coords = np.copy(self.coords)
        bonds = self._current_explicit_bonds()
        bo = self._current_bond_orders()

        try:
            symbols, coords, bonds, bo, a, b, _removed = form_single_bond(
                symbols, coords, bonds, bo, sel[0], sel[1])
        except BondFormationError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot Form Bond", str(exc))
            self.statusBar().showMessage(str(exc), 6000)
            return

        self._push_undo_state()
        s1, s2 = symbols[a], symbols[b]
        target = estimate_bond_length(s1, s2)

        cur_d = float(np.linalg.norm(coords[b] - coords[a]))
        steps = 6 if cur_d > target * 1.5 else 1
        adj_current = build_adj(len(symbols), bonds)
        for step in range(1, steps + 1):
            desired = cur_d - (step / steps) * (cur_d - target)
            mid = 0.5 * (coords[a] + coords[b])
            v = coords[b] - coords[a]
            nv = float(np.linalg.norm(v))
            if nv > 1e-4:
                u = v / nv
                delta_a = (mid - 0.5 * desired * u) - coords[a]
                delta_b = (mid + 0.5 * desired * u) - coords[b]
                coords[a] += delta_a
                coords[b] += delta_b
                for h in adj_current[a]:
                    if symbols[h] == "H":
                        coords[h] += delta_a
                for h in adj_current[b]:
                    if symbols[h] == "H":
                        coords[h] += delta_b
            try:
                coords, _ = optimize_geometry_force_field(symbols, coords, user_bonds=bonds,
                                                          max_iter=60, strict_bonds=True,
                                                          user_bond_orders=bo)
            except Exception:
                log.warning("Pre-relaxation failed; continuing without it", exc_info=True)

        # Topology is final: no hydrogens added, no other bonds formed/changed
        self._commit_explicit_structure(symbols, coords, bonds, bo, relax_iters=300,
                                        add_h=False, fix_valence=False)
        msg = f"Formed single bond {s1}-{s2}"
        if _removed:
            msg += f" (removed {len(_removed)} hydrogen(s) to fulfill valence)"
        self.statusBar().showMessage(msg + ".", 5000)

    def _on_atom_moved_by_drag(self, atom_idx: int, new_x: float, new_y: float, new_z: float):
        """Real-time coordinate update while translating an existing atom with Ctrl + Left-Click Drag.
        If Live Molecular Tweezers relaxation is enabled, relaxes connected atoms concurrently."""
        if hasattr(self, "viewer") and hasattr(self.viewer, "coords") and len(self.viewer.coords) == len(self.coords):
            self.coords = np.array(self.viewer.coords, copy=True)
        if 0 <= atom_idx < len(self.coords):
            self.coords[atom_idx] = np.array([new_x, new_y, new_z], dtype=np.float64)
            if hasattr(self, "viewer") and hasattr(self.viewer, "coords") and 0 <= atom_idx < len(self.viewer.coords):
                self.viewer.coords[atom_idx] = self.coords[atom_idx]

        sym = self.zatoms[atom_idx].symbol if 0 <= atom_idx < len(self.zatoms) else ""

        # Molecular Tweezers (Live UFF relaxation during drag)
        if getattr(self, "live_tweezers_enabled", False) and len(self.zatoms) > 2:
            try:
                symbols = [a.symbol for a in self.zatoms]
                drag_indices = set(getattr(self.viewer, "atom_move_drag_indices", [atom_idx]) or [atom_idx])

                # Pin dragged atom(s) at mouse pointer, allow remaining atoms to relax via UFF analytical gradients
                if len(drag_indices) < len(symbols):
                    user_bonds = list(getattr(self.viewer, "bonds", []))
                    opt_c, e_val = optimize_geometry_force_field(
                        symbols, self.coords, user_bonds=user_bonds, max_iter=8, frozen_indices=drag_indices
                    )
                    self.coords = np.array(opt_c, copy=True)
                    self.viewer.coords = np.array(opt_c, copy=True)
                    self.viewer.compute_bonds()
                    self.viewer.update()
                    self.statusBar().showMessage(
                        f"🧲 Tweezers: Dragging {sym}{atom_idx+1} at ({new_x:.2f}, {new_y:.2f}, {new_z:.2f}) Å | UFF: {e_val:.1f} kcal/mol", 1000
                    )
                else:
                    self.statusBar().showMessage(
                        f"🧲 Tweezers: Translating selected {len(drag_indices)} atom(s)", 1000
                    )
            except Exception:
                self.statusBar().showMessage(
                    f"Moving {sym}{atom_idx+1} to ({new_x:.3f}, {new_y:.3f}, {new_z:.3f}) Å", 1000
                )
        else:
            self.statusBar().showMessage(
                f"Moving {sym}{atom_idx+1} to ({new_x:.3f}, {new_y:.3f}, {new_z:.3f}) Å", 1000
            )

        # Real-time live geometry updates:
        # 1. Update Adjust Panel sliders/spinners for active distance/angle/dihedral
        self._update_adjust_panel()

        # 2. Live update table cell distances for the moving atoms
        drag_indices = set(getattr(self.viewer, "atom_move_drag_indices", [atom_idx]) or [atom_idx])
        if hasattr(self, "table_widget") and len(self.zatoms) == len(self.coords):
            for i, a in enumerate(self.zatoms):
                if a.ref1 is not None and 0 <= a.ref1 < len(self.coords):
                    if i in drag_indices or a.ref1 in drag_indices:
                        d = float(np.linalg.norm(self.coords[i] - self.coords[a.ref1]))
                        item = self.table_widget.item(i, self.table_widget.COL_DIST)
                        if item is not None:
                            item.setText(f"{d:.4f}")

        # 3. Live Cartesian XYZ text streaming (when on Cartesian XYZ tab)
        if hasattr(self, "text_raw_xyz") and hasattr(self, "tabs") and self.tabs.tabText(self.tabs.currentIndex()) == "Cartesian XYZ":
            from .chem_identifiers import format_xyz_string
            symbols = [a.symbol for a in self.zatoms]
            xyz_txt = format_xyz_string(symbols, self.coords, comment="AgyMolden structure")
            self.text_raw_xyz.blockSignals(True)
            self.text_raw_xyz.setPlainText(xyz_txt)
            self.text_raw_xyz.blockSignals(False)

    def _on_toggle_tweezers(self, checked: bool):
        """Toggle interactive live force field relaxation during atom drag."""
        self.live_tweezers_enabled = checked
        if hasattr(self, "viewer"):
            self.viewer.tweezers_active = checked
            # Set crosshair cursor when tweezers are active to show tool readiness
            if checked:
                self.viewer.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            else:
                self.viewer.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        status = "enabled (Left-click & drag any atom to relax live)" if checked else "disabled"
        self.statusBar().showMessage(f"🧲 Molecular Tweezers: {status}", 4000)

    def _on_atom_drag_finished(self, atom_idx: int):
        """Called when mouse release completes atom dragging. Reconstructs canonical Z-matrix."""
        from .alignment import capture_rigid_frame
        if hasattr(self, "viewer") and hasattr(self.viewer, "coords") and len(self.viewer.coords) == len(self.coords):
            self.coords = np.array(self.viewer.coords, copy=True)
        symbols = [a.symbol for a in self.zatoms]
        if self.strict_topology:
            new_zatoms = cartesian_to_zmatrix(symbols, self.coords, existing_zatoms=self.zatoms)
        else:
            new_zatoms = cartesian_to_zmatrix(symbols, self.coords)
        self.zatoms = new_zatoms
        self._rigid_frame = capture_rigid_frame(self.zatoms, self.coords)
        self.table_widget.load_zatoms(self.zatoms)
        self._sync_raw_text()
        self._update_adjust_panel()
        sym = symbols[atom_idx] if 0 <= atom_idx < len(symbols) else ""
        self.statusBar().showMessage(
            f"Moved atom {sym}{atom_idx+1}. Updated Z-matrix geometry.", 4000
        )

    def _on_region_rotation_finished(self):
        """Called when mouse release completes rotating a selected region of atoms."""
        from .alignment import capture_rigid_frame
        self._push_undo_state()
        if hasattr(self, "viewer") and hasattr(self.viewer, "coords") and len(self.viewer.coords) == len(self.coords):
            self.coords = np.array(self.viewer.coords, copy=True)
        symbols = [a.symbol for a in self.zatoms]
        if self.strict_topology:
            new_zatoms = cartesian_to_zmatrix(symbols, self.coords, existing_zatoms=self.zatoms)
        else:
            new_zatoms = cartesian_to_zmatrix(symbols, self.coords)
        self.zatoms = new_zatoms
        self._rigid_frame = capture_rigid_frame(self.zatoms, self.coords)
        self.table_widget.load_zatoms(self.zatoms)
        self._sync_raw_text()
        self._update_adjust_panel()
        n_rot = len(self.viewer.multi_selected)
        self.statusBar().showMessage(
            f"Rotated selected region ({n_rot} atoms). Updated Z-matrix geometry.", 4000
        )

    def _on_atom_drag_started(self, atom_idx: int):
        """Called when user starts dragging an atom. Push undo state once before translation."""
        self._push_undo_state()

    def _on_atom_double_clicked(self, anchor_idx: int):
        """
        Double-clicking an existing atom:
        1. If a fragment is in memory (via Ctrl+C copy or from Preset/Fragment tab):
           Replaces the clicked atom with the stored fragment, placing the fragment's 1st atom
           at the position of the double-clicked atom.
        2. Otherwise:
           Replaces the clicked atom or extends it with the active element brush (Avogadro behavior),
           maintaining proper chemical valence.
        All edits use the explicit bond graph: hydrogens always keep exactly one bond and
        fragments are inserted complete.
        """
        # Cancel any active interactive Molden picking mode so double-click works independently
        self.viewer.picking_active = False
        self.viewer.picking_banner = ""
        self.viewer.picked_references = []
        self.pending_new_element = None
        self.picked_references = []

        if getattr(self.viewer, "interaction_mode", "build") == "inspect":
            self.statusBar().showMessage(
                "🔍 Inspect Mode active: Double-click atom mutation is disabled to protect geometry. Press Tab or click '✏️ Build' to edit.",
                3500,
            )
            return

        if anchor_idx < 0 or anchor_idx >= len(self.zatoms):
            return

        active_frag = (self.stored_fragment or self.last_added_fragment) if getattr(self, "last_item_type", "element") == "fragment" else None
        symbols = [a.symbol for a in self.zatoms]
        coords = np.copy(self.coords)
        n = len(symbols)
        bonds = self._current_explicit_bonds()
        bo = self._current_bond_orders()
        adj = build_adj(n, bonds)
        anchor_sym = symbols[anchor_idx]
        anchor_pos = np.copy(coords[anchor_idx])
        auto_h = bool(getattr(self, 'chk_auto_h', None) and self.chk_auto_h.isChecked())

        self._push_undo_state()

        # Case 1: Replace atom or substitute on atom with fragment in memory
        if active_frag is not None:
            f_syms = list(active_frag["symbols"])
            conn_idx = int(active_frag.get("connect_idx", 0))
            if conn_idx < 0 or conn_idx >= len(f_syms):
                conn_idx = 0
            n_sites = self._attach_fragment_to_atom_or_orbit(
                anchor_idx, active_frag, conn_idx=conn_idx, custom_dir_u=None, relax_iters=150
            )
            frag_name = active_frag.get("name", "fragment")
            sym_note = f" across {n_sites} symmetry-equivalent sites" if n_sites > 1 else ""
            self.statusBar().showMessage(
                f"Attached {frag_name} ({len(f_syms)} atoms) at {anchor_sym}{anchor_idx+1}{sym_note}", 4500
            )
            return

        # Case 2: No fragment in memory - replace or add atom using active element brush
        elem = getattr(self, "last_added_element", None) or getattr(self.viewer, "brush_element", None)
        if not elem:
            heavy_atoms = [a.symbol for a in self.zatoms if a.symbol != "H"]
            elem = heavy_atoms[-1] if heavy_atoms else "C"

        sym_res = self._get_active_imposed_symmetry()
        if sym_res is not None:
            if anchor_sym in ("H", "F", "Cl", "Br", "I"):
                atom_frag = {"name": elem, "symbols": [elem], "coords": [(0.0, 0.0, 0.0)], "connect_idx": 0}
                n_sites = self._attach_fragment_to_atom_or_orbit(
                    anchor_idx, atom_frag, conn_idx=0, custom_dir_u=None, relax_iters=80
                )
                self.statusBar().showMessage(
                    f"Double-click replaced {n_sites} symmetry-equivalent {anchor_sym} atoms with {elem}", 4000
                )
                return
            else:
                # Heavy atom mutation under imposed symmetry
                _, orbit_ops = get_atom_orbit_and_operations(
                    symbols, coords, anchor_idx, result=sym_res, tolerance=max(0.25, sym_res.tolerance * 1.5)
                )
                mutate_indices = sorted({eq_i for eq_i, _ in orbit_ops})
                for idx in mutate_indices:
                    self.zatoms[idx].symbol = elem
                self.table_widget.load_zatoms(self.zatoms)
                self.update_geometry_live()
                self.statusBar().showMessage(
                    f"Double-click replaced {len(mutate_indices)} symmetry-equivalent {anchor_sym} atoms with {elem}", 4000
                )
                return

        new_syms = list(symbols)
        new_coords = np.copy(coords)
        new_bonds = list(bonds)
        if anchor_sym in ("H", "F", "Cl", "Br", "I"):
            # Monovalent terminal atom (H / halogen) keeps its single bond; just becomes elem at a proper distance
            if adj[anchor_idx]:
                parent = adj[anchor_idx][0]
                v_dir = coords[anchor_idx] - coords[parent]
                nv = np.linalg.norm(v_dir)
                v_dir = v_dir / nv if nv > 1e-6 else np.array([1.0, 0.0, 0.0])
                new_coords[anchor_idx] = coords[parent] + v_dir * estimate_bond_length(elem, symbols[parent])
            new_syms[anchor_idx] = elem
            target_set = {anchor_idx} | set(adj[anchor_idx])
            msg = f"Double-click replaced {anchor_sym} with {elem}"
        else:
            h_nbs = [j for j in adj[anchor_idx] if symbols[j] == "H"]
            if h_nbs:
                h_idx = h_nbs[-1]
                v_dir = coords[h_idx] - coords[anchor_idx]
                nv = np.linalg.norm(v_dir)
                v_dir = v_dir / nv if nv > 1e-6 else np.array([1.0, 0.0, 0.0])
                new_syms[h_idx] = elem
                new_coords[h_idx] = coords[anchor_idx] + v_dir * estimate_bond_length(elem, anchor_sym)
                target_set = {anchor_idx, h_idx}
            else:
                nb_vecs = [coords[j] - coords[anchor_idx] for j in adj[anchor_idx]]
                nb_vecs = [v / np.linalg.norm(v) for v in nb_vecs if np.linalg.norm(v) > 1e-4]
                v_dir = _calculate_hydrogen_directions(nb_vecs, 1)[0] if nb_vecs else self.viewer._get_camera_axes()[0]
                new_syms.append(elem)
                new_coords = np.vstack([new_coords, coords[anchor_idx] + v_dir * estimate_bond_length(elem, anchor_sym)])
                nb_pair = norm_bond(anchor_idx, len(new_syms) - 1)
                new_bonds.append(nb_pair)
                bo.pop(nb_pair, None)  # Strictly single bond
                target_set = {anchor_idx, len(new_syms) - 1}
            msg = f"Double-click added {elem} bound to {anchor_sym}{anchor_idx+1}"

        self._commit_explicit_structure(
            new_syms, new_coords, new_bonds, bo,
            relax_iters=80, add_h=auto_h, fix_valence=auto_h, target_indices=target_set
        )
        self.statusBar().showMessage(msg + (" (valence adjusted)" if auto_h else ""), 4000)

    def _on_bond_double_clicked(self, atom1_idx: int, atom2_idx: int):
        """Double-clicking a bond cycles its bond order (1 -> 2 -> 3 -> 1)."""
        self.set_bond_order(atom1_idx, atom2_idx, target_bo=None)

    def set_bond_order(self, atom1_idx: int, atom2_idx: int, target_bo: Optional[int] = None):
        """
        Sets the bond order of bond (atom1_idx, atom2_idx) to target_bo (1, 2, or 3),
        or cycles it if target_bo is None.
        Adjusts the valence of atoms involved in the bond, pruning excess hydrogens
        or adding needed hydrogens geometrically to complete full valence while
        preserving molecular topology and keeping the bond selected.
        """
        # Cancel any active interactive Molden picking mode so bond modification works independently
        self.viewer.picking_active = False
        self.viewer.picking_banner = ""
        self.viewer.picked_references = []
        self.pending_new_element = None
        self.picked_references = []

        n = len(self.zatoms)
        if atom1_idx < 0 or atom1_idx >= n or atom2_idx < 0 or atom2_idx >= n:
            return

        s1 = self.zatoms[atom1_idx].symbol
        s2 = self.zatoms[atom2_idx].symbol

        # Monovalent elements strictly only support single bonds
        MONOVALENT = {"H", "F", "Cl", "Br", "I"}
        if target_bo is not None and target_bo > 1 and (s1 in MONOVALENT or s2 in MONOVALENT):
            mono_sym = s1 if s1 in MONOVALENT else s2
            self.statusBar().showMessage(
                f"Cannot set bond order > 1: {mono_sym} only supports single bonds.", 3500
            )
            return
        elif target_bo is None and (s1 in MONOVALENT or s2 in MONOVALENT):
            mono_sym = s1 if s1 in MONOVALENT else s2
            self.statusBar().showMessage(
                f"Cannot increase bond order: {mono_sym} only supports single bonds.", 3500
            )
            return

        b_key = (min(atom1_idx, atom2_idx), max(atom1_idx, atom2_idx))
        current_bo = self.viewer.bond_orders.get(b_key, 1)

        # Theoretical upper limit on bond order between this pair
        val1 = DEFAULT_VALENCES.get(s1, 4)
        val2 = DEFAULT_VALENCES.get(s2, 4)
        max_pair_bo = min(val1, val2, 3)

        # Detect current connectivity to inspect heavy neighbors
        symbols = [a.symbol for a in self.zatoms]
        coords = np.copy(self.coords)
        curr_bonds = detect_connectivity(symbols, coords)
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for u, v in curr_bonds:
            adj[u].append(v)
            adj[v].append(u)

        # Bond order sum of other heavy neighbors (excluding the current bond)
        def get_other_heavy_bo_sum(atom_i: int, other_i: int) -> int:
            total = 0
            for nb in adj[atom_i]:
                if nb == other_i:
                    continue
                if symbols[nb] != "H":
                    pair_k = (min(atom_i, nb), max(atom_i, nb))
                    total += self.viewer.bond_orders.get(pair_k, 1)
            return total

        other_bo_1 = get_other_heavy_bo_sum(atom1_idx, atom2_idx)
        other_bo_2 = get_other_heavy_bo_sum(atom2_idx, atom1_idx)

        # Maximum bond order allowed without exceeding atom valence
        allowed_max_bo = min(max_pair_bo, val1 - other_bo_1, val2 - other_bo_2)
        if allowed_max_bo < 1:
            allowed_max_bo = 1

        if target_bo is not None:
            if target_bo == current_bo:
                order_names = {1: "Single", 2: "Double", 3: "Triple"}
                bo_name = order_names.get(current_bo, f"{current_bo}-fold")
                self.statusBar().showMessage(
                    f"Bond {s1}{atom1_idx+1} - {s2}{atom2_idx+1} is already a {bo_name} bond.", 2500
                )
                return
            if target_bo > allowed_max_bo:
                self.statusBar().showMessage(
                    f"Cannot set bond order to {target_bo}: valence limit ({allowed_max_bo}) reached for {s1} or {s2}.", 3500
                )
                return
            new_bo = max(1, min(3, target_bo))
        else:
            # Cycle bond order: if at allowed maximum or >= 3, cycle back to 1.
            # Otherwise, increase by 1.
            if current_bo >= allowed_max_bo or current_bo >= 3:
                new_bo = 1
            else:
                if allowed_max_bo <= 1 and current_bo == 1:
                    self.statusBar().showMessage(
                        f"Cannot increase bond order: valence limit reached for {s1} or {s2}.", 3500
                    )
                    return
                new_bo = current_bo + 1

        self._push_undo_state()
        bonds = self._current_explicit_bonds()
        bo = self._current_bond_orders()
        bo[b_key] = new_bo

        # Adjust valence locally on the two atoms involved in the bond:
        # If changing bond order created excess valence, remove excess attached hydrogens
        remove = set()
        adj_map = build_adj(len(symbols), bonds)
        for center, partner in [(atom1_idx, atom2_idx), (atom2_idx, atom1_idx)]:
            val = DEFAULT_VALENCES.get(symbols[center], 4)
            used = sum(bo.get(norm_bond(center, k), 1) for k in adj_map[center] if k not in remove)
            excess = used - val
            if excess > 0:
                hs = [k for k in adj_map[center] if symbols[k] == "H" and k not in remove]
                hs.sort(key=lambda h: np.linalg.norm(coords[h] - coords[partner]))
                remove.update(hs[:excess])

        rem_map = {}
        if remove:
            symbols, coords, bonds, bo, rem_map = remove_atoms(symbols, coords, bonds, remove, bo)

        # Update internuclear distance along the bond axis towards the equilibrium length for new_bo
        curr_u = rem_map.get(atom1_idx, atom1_idx) if remove else atom1_idx
        curr_v = rem_map.get(atom2_idx, atom2_idx) if remove else atom2_idx
        if curr_u != -1 and curr_v != -1 and curr_u < len(symbols) and curr_v < len(symbols):
            su, sv = symbols[curr_u], symbols[curr_v]
            target_d = estimate_bond_length(su, sv, bond_order=new_bo)
            disp_vec = coords[curr_v] - coords[curr_u]
            cur_dist = float(np.linalg.norm(disp_vec))
            if cur_dist > 1e-4 and abs(cur_dist - target_d) > 0.005:
                u_norm = disp_vec / cur_dist
                shift_mag = (target_d - cur_dist) * 0.5
                coords[curr_u] -= u_norm * shift_mag
                coords[curr_v] += u_norm * shift_mag
                # Shift attached hydrogens along with their parent heavy atom
                post_adj = build_adj(len(symbols), bonds)
                for h in post_adj[curr_u]:
                    if symbols[h] == "H" and h != curr_v:
                        coords[h] -= u_norm * shift_mag
                for h in post_adj[curr_v]:
                    if symbols[h] == "H" and h != curr_u:
                        coords[h] += u_norm * shift_mag

        # Pass complete bond orders dictionary including order 1 for the target bond so optimizer respects order 1
        commit_bo = dict(bo)
        if curr_u != -1 and curr_v != -1:
            commit_bo[norm_bond(curr_u, curr_v)] = new_bo

        o2n = self._commit_explicit_structure(
            symbols, coords, bonds, commit_bo,
            relax_iters=120, add_h=False, fix_valence=False
        ) or {}

        # Keep the bond selected in the viewer through removal and BFS reordering
        pre_u = rem_map.get(atom1_idx, atom1_idx) if remove else atom1_idx
        pre_v = rem_map.get(atom2_idx, atom2_idx) if remove else atom2_idx
        new_u = o2n.get(pre_u, pre_u)
        new_v = o2n.get(pre_v, pre_v)
        if new_u != -1 and new_v != -1 and 0 <= new_u < len(self.zatoms) and 0 <= new_v < len(self.zatoms):
            self.viewer.selected_bond = (min(new_u, new_v), max(new_u, new_v))
            self.viewer.selected_atom_idx = new_u
            self.viewer.multi_selected = [new_u, new_v]
            self.viewer.update()

        order_names = {1: "Single", 2: "Double", 3: "Triple"}
        bo_name = order_names.get(new_bo, f"{new_bo}-fold")
        disp_u = new_u if remove and new_u != -1 else atom1_idx
        disp_v = new_v if remove and new_v != -1 else atom2_idx
        self.statusBar().showMessage(
            f"Set bond {symbols[disp_u]}{disp_u+1} - {symbols[disp_v]}{disp_v+1} to {bo_name} bond (order {new_bo}) & maintained molecular topology", 4500
        )

    def _on_canvas_double_clicked(self, pos: QtCore.QPoint):
        """
        Double-clicking empty canvas:
        - If a fragment/group is in memory: pastes it at clicked position in free space.
        - If an atom is in memory: pastes it at clicked position in free space.
        - If memory is empty: displays a menu allowing user to open the Periodic Table
          to pick an element or open/select a chemical fragment/molecule.
        """
        has_frag = bool(self.stored_fragment or self.last_added_fragment)
        has_elem = bool(self.last_added_element)

        # 1. Fragment or group in memory
        if self.last_item_type == "fragment" and has_frag:
            frag = self.stored_fragment or self.last_added_fragment
            self._paste_fragment_to_free_space(frag, screen_pos=pos)
            return

        # 2. Element in memory
        if self.last_item_type == "element" and has_elem:
            elem = self.last_added_element
            self._paste_element_to_free_space(elem, screen_pos=pos)
            return

        # 3. Fragment in memory if last_item_type wasn't element
        if has_frag:
            frag = self.stored_fragment or self.last_added_fragment
            self._paste_fragment_to_free_space(frag, screen_pos=pos)
            return

        # 4. Element in memory if last_item_type wasn't fragment
        if has_elem:
            elem = self.last_added_element
            self._paste_element_to_free_space(elem, screen_pos=pos)
            return

        # 4. Nothing in memory: prompt user to pick element from Periodic Table or add fragment/molecule
        menu = QtWidgets.QMenu(self)
        act_add_atom = menu.addAction("⚛️ Periodic Table (Add Atom)...")
        frag_name = self.combo_fragment.currentText() if hasattr(self, "combo_fragment") else "Water (H2O)"
        act_add_mol = menu.addAction(f"🧪 Chemical Fragment / Molecule ({frag_name})...")
        chosen = menu.exec(self.viewer.mapToGlobal(pos))
        if chosen == act_add_atom:
            self._on_add_line()
        elif chosen == act_add_mol:
            self._on_add_molecule_free_space(screen_pos=pos)

    def _paste_element_to_free_space(self, elem: str, screen_pos: QtCore.QPoint):
        """Place element at screen_pos in 3D camera projection plane."""
        self._push_undo_state()
        w, h = max(1, self.viewer.width()), max(1, self.viewer.height())
        right_w, up_w = self.viewer._get_camera_axes()
        scale = (self.viewer.zoom * 0.8) / h
        dx = float(screen_pos.x() - w * 0.5)
        dy = float(screen_pos.y() - h * 0.5)
        centroid = np.mean(self.coords, axis=0) if len(self.coords) > 0 else np.zeros(3)
        placed_pos = centroid + (dx * scale) * right_w - (dy * scale) * up_w

        curr_syms = [a.symbol for a in self.zatoms]
        curr_coords = np.copy(self.coords)
        curr_bonds = self._current_explicit_bonds()
        curr_bo = self._current_bond_orders()
        n_old = len(curr_syms)

        auto_h = bool(getattr(self, "chk_auto_h", None) and self.chk_auto_h.isChecked() and elem != "H" and elem != "X")
        new_syms = [elem]
        new_coords = [placed_pos]
        new_bonds = []
        new_bo = {}

        coord_tpl = getattr(self, "last_coordination_template", "auto") or "auto"
        if auto_h:
            if coord_tpl != "auto":
                from .coordination import get_coordination_vectors
                dirs = get_coordination_vectors(elem, coord_tpl)
                d = estimate_bond_length(elem, "H")
                for k, dvec in enumerate(dirs):
                    new_syms.append("H")
                    new_coords.append(placed_pos + dvec * d)
                    new_bonds.append((0, k + 1))
            elif elem in DEFAULT_VALENCES:
                val = DEFAULT_VALENCES[elem]
                dirs = _calculate_hydrogen_directions([], val)
                d = estimate_bond_length(elem, "H")
                for k, dvec in enumerate(dirs[:val]):
                    new_syms.append("H")
                    new_coords.append(placed_pos + dvec * d)
                    new_bonds.append((0, k + 1))

        combined_syms = curr_syms + new_syms
        combined_coords = np.vstack([curr_coords, np.array(new_coords)]) if n_old > 0 else np.array(new_coords)
        combined_bonds = curr_bonds + [(n_old + u, n_old + v) for u, v in new_bonds]
        combined_bo = dict(curr_bo)
        for (u, v), o in new_bo.items():
            combined_bo[norm_bond(n_old + u, n_old + v)] = o

        self._commit_explicit_structure(combined_syms, combined_coords, combined_bonds, combined_bo, relax_iters=0, add_h=False, fix_valence=False)

        new_idx = max(0, len(self.zatoms) - 1) if len(self.zatoms) > 0 else None
        self.viewer.selected_atom_idx = new_idx
        self.viewer.multi_selected = [new_idx] if new_idx is not None else []
        self.viewer._update_measurement_info()
        self.viewer.update()

        h_str = " (with valence hydrogens)" if auto_h else ""
        self.statusBar().showMessage(f"Pasted atom {elem}{h_str} at canvas coordinates", 4500)

    def _paste_fragment_to_free_space(self, frag: Dict, screen_pos: QtCore.QPoint):
        """Place a fragment/group in memory at screen_pos in 3D camera projection plane."""
        self._push_undo_state()
        f_syms = list(frag["symbols"])
        f_coords = np.array(frag["coords"], dtype=np.float64)
        f_coords = f_coords - np.mean(f_coords, axis=0)
        f_bonds = frag.get("bonds")
        if f_bonds is None:
            f_bonds = detect_connectivity(f_syms, f_coords, strict=False)

        w, h = max(1, self.viewer.width()), max(1, self.viewer.height())
        right_w, up_w = self.viewer._get_camera_axes()
        scale = (self.viewer.zoom * 0.8) / h
        dx = float(screen_pos.x() - w * 0.5)
        dy = float(screen_pos.y() - h * 0.5)
        centroid = np.mean(self.coords, axis=0) if len(self.coords) > 0 else np.zeros(3)
        placed_coords = f_coords + centroid + (dx * scale) * right_w - (dy * scale) * up_w

        curr_syms = [a.symbol for a in self.zatoms]
        curr_coords = np.copy(self.coords)
        n_old = len(curr_syms)
        existing_bonds = self._current_explicit_bonds()
        existing_bo = self._current_bond_orders()

        combined_symbols = curr_syms + f_syms
        combined_coords = np.vstack([curr_coords, placed_coords]) if n_old > 0 else placed_coords
        combined_bonds = existing_bonds + [(n_old + u, n_old + v) for u, v in f_bonds]
        combined_bo = dict(existing_bo)
        for (u, v), o in (frag.get("bond_orders") or {}).items():
            combined_bo[norm_bond(n_old + u, n_old + v)] = o

        self._commit_explicit_structure(combined_symbols, combined_coords, combined_bonds, combined_bo, relax_iters=0, add_h=False)

        n_new = len(self.zatoms)
        new_indices = [i for i in range(n_old, n_new)] if n_new > n_old else ([n_new - 1] if n_new > 0 else [])
        self.viewer.selected_atom_idx = new_indices[0] if new_indices else None
        self.viewer.multi_selected = new_indices
        self.viewer._update_measurement_info()
        self.viewer.update()

        frag_name = frag.get("name", "fragment")
        self.statusBar().showMessage(f"Pasted {frag_name} ({len(f_syms)} atoms) from memory", 4500)

    def _on_add_molecule_free_space(self, screen_pos: Optional[QtCore.QPoint] = None):
        """
        Add a complete molecule (e.g. Water, Solvents, functional groups) into free space on the canvas.
        Preserves existing molecules without forming spurious covalent bonds between them.
        Ideal for building water clusters, solvent cages, and multi-component systems.
        """
        frag_name = self.combo_fragment.currentText()
        if frag_name not in FRAGMENTS:
            return

        self._push_undo_state()
        frag = FRAGMENTS[frag_name]
        f_syms = list(frag["symbols"])
        f_coords = np.array(frag["coords"], dtype=np.float64)
        # Center fragment at its centroid
        f_coords = f_coords - np.mean(f_coords, axis=0)

        # Internal bonds of the fragment
        f_bonds = detect_connectivity(f_syms, f_coords, strict=False)

        # Determine 3D placement
        if len(self.coords) == 0:
            offset = np.array([0.0, 0.0, 0.0])
        elif screen_pos is not None:
            w, h = max(1, self.viewer.width()), max(1, self.viewer.height())
            right_w, up_w = self.viewer._get_camera_axes()
            scale = (self.viewer.zoom * 0.8) / h
            dx = float(screen_pos.x() - w * 0.5)
            dy = float(screen_pos.y() - h * 0.5)
            centroid = np.mean(self.coords, axis=0)
            offset = centroid + (dx * scale) * right_w - (dy * scale) * up_w
        else:
            # Place adjacent to existing molecules (e.g. centroid + camera_right * (bounding_radius + 3.2 A))
            right_w, up_w = self.viewer._get_camera_axes()
            centroid = np.mean(self.coords, axis=0)
            radii = np.linalg.norm(self.coords - centroid, axis=1)
            max_r = float(np.max(radii)) if len(radii) > 0 else 2.0
            offset = centroid + right_w * (max_r + 3.2)

        placed_coords = f_coords + offset

        curr_symbols = [a.symbol for a in self.zatoms]
        curr_coords = np.copy(self.coords)
        n_old = len(curr_symbols)

        existing_bonds = self._current_explicit_bonds()
        existing_bo = self._current_bond_orders()

        # Combine symbols, coords, and ONLY internal bonds
        combined_symbols = curr_symbols + f_syms
        combined_coords = np.vstack([curr_coords, placed_coords]) if n_old > 0 else placed_coords
        combined_bonds = existing_bonds + [(n_old + u, n_old + v) for u, v in f_bonds]
        combined_bo = dict(existing_bo)

        # Commit explicit structure without auto adding H (molecule is already chemically complete)
        self._commit_explicit_structure(combined_symbols, combined_coords, combined_bonds, combined_bo, relax_iters=0, add_h=False)

        # Store as last added fragment for double-click and drag replacement
        self.last_added_fragment = dict(frag)
        self.last_added_fragment["name"] = frag_name
        self.last_item_type = "fragment"
        self._update_viewer_drag_label()

        # Auto-select the newly added molecule so user can immediately move it with Ctrl+Left Click Drag!
        new_indices = list(range(n_old, n_old + len(f_syms)))
        self.viewer.selected_atom_idx = new_indices[0] if new_indices else None
        self.viewer.multi_selected = new_indices
        self.viewer._update_measurement_info()
        self.viewer.update()

        self.statusBar().showMessage(
            f"Added {frag_name} ({len(f_syms)} atoms) in free canvas space! Tip: Hold Ctrl and drag to position it.", 5000
        )

    def _delete_atom_or_orbit(self, atom_idx: int, source_label: str = "Deleted"):
        """Delete `atom_idx` (and all symmetry-equivalent atoms in its orbit if `symmetry_imposed` is enabled)."""
        if atom_idx < 0 or atom_idx >= len(self.zatoms):
            return
        self._push_undo_state()
        del_sym = self.zatoms[atom_idx].symbol
        del_label = f"{del_sym}{atom_idx+1}"
        is_h = (del_sym == "H")

        symbols = [a.symbol for a in self.zatoms]
        coords = np.copy(self.coords)
        sym_res = self._get_active_imposed_symmetry()
        if sym_res is not None and len(coords) == len(symbols):
            _, orbit_ops = get_atom_orbit_and_operations(
                symbols, coords, atom_idx, result=sym_res, tolerance=max(0.25, sym_res.tolerance * 1.5)
            )
            rem_indices = sorted({eq_i for eq_i, _ in orbit_ops})
        else:
            rem_indices = [atom_idx]

        keep_mask = [i for i in range(len(symbols)) if i not in set(rem_indices)]
        if getattr(self, "frozen_atoms", None):
            o2n_keep = {old_i: new_i for new_i, old_i in enumerate(keep_mask)}
            self.frozen_atoms = {o2n_keep[i] for i in self.frozen_atoms if i in o2n_keep}
            if hasattr(self, "_update_frozen_buttons_ui"):
                self._update_frozen_buttons_ui()
        if not keep_mask:
            self.set_molecule_from_zatoms([])
            self.statusBar().showMessage(f"{source_label} atom {del_label}. Molecule is now empty.", 3000)
            return

        new_symbols = [symbols[i] for i in keep_mask]
        new_coords = coords[keep_mask]

        # If adjust hydrogens is active, adjust valence of remaining atoms EXCEPT when deleting H
        if getattr(self, "chk_auto_h", None) and self.chk_auto_h.isChecked() and not is_h:
            new_symbols, new_coords = adjust_hydrogens_to_valence(new_symbols, new_coords)

        new_zatoms = cartesian_to_zmatrix(new_symbols, new_coords)
        self.viewer.selected_atom_idx = None
        self.set_molecule_from_zatoms(new_zatoms, exact_coords=new_coords)
        if len(rem_indices) > 1:
            self.statusBar().showMessage(
                f"{source_label} {len(rem_indices)} symmetry-equivalent {del_sym} atoms and updated Z-matrix", 4000
            )
        else:
            self.statusBar().showMessage(f"{source_label} atom {del_label} and updated Z-matrix", 3500)

    def _on_viewport_atom_context_menu(self, atom_idx: int, global_pos: QtCore.QPoint):
        """Display an intelligent context menu when right-clicking an atom."""
        if atom_idx < 0 or atom_idx >= len(self.zatoms):
            return

        atom = self.zatoms[atom_idx]
        cur_sym = atom.symbol

        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("QMenu { font-size: 12px; } QMenu::item { padding: 4px 20px 4px 20px; }")

        # Header Title
        title_act = QtGui.QAction(f"Atom #{atom_idx+1}: {cur_sym}", self)
        title_font = title_act.font()
        title_font.setBold(True)
        title_act.setFont(title_font)
        title_act.setEnabled(False)
        menu.addAction(title_act)
        menu.addSeparator()

        # Change Element Submenu
        menu_mutate = menu.addMenu("🔄 Change Element")
        common_elements = ["C", "H", "O", "N", "F", "Cl", "Br", "I", "S", "P", "Si", "B"]
        for el in common_elements:
            act_el = menu_mutate.addAction(f"{el}")
            if el == cur_sym:
                act_el.setText(f"✔ {el}")
            act_el.triggered.connect(lambda _c=False, e=el: self._mutate_atom_element(atom_idx, e))

        act_periodic = menu_mutate.addAction("More Elements (Periodic Table)...")
        act_periodic.triggered.connect(lambda: self._prompt_mutate_periodic_table(atom_idx))

        # Attach Substituent Submenu
        menu_sub = menu.addMenu("🧪 Attach Functional Group")
        popular_frags = [
            "Methyl (-CH3)",
            "Ethyl (-C2H5)",
            "Isopropyl (-CH(CH3)2)",
            "tert-Butyl (-C(CH3)3)",
            "Hydroxyl (-OH)",
            "Methoxy (-OCH3)",
            "Amino (-NH2)",
            "Phenyl (-C6H5)",
            "Pyridyl (-C5H4N)",
            "Nitro (-NO2)",
            "Carboxyl (-COOH)",
            "Cyano (-CN)",
            "Trifluoromethyl (-CF3)",
            "Chlorine (-Cl)",
            "Bromine (-Br)",
            "Fluorine (-F)",
        ]
        for f_name in popular_frags:
            if f_name in FRAGMENTS:
                act_f = menu_sub.addAction(f_name)
                act_f.triggered.connect(lambda _c=False, fn=f_name: self._attach_specific_fragment(atom_idx, fn))

        menu_sub.addSeparator()
        act_all_frags = menu_sub.addAction("Browse All Fragments...")
        act_all_frags.triggered.connect(self._on_open_fragment_library)

        menu.addSeparator()

        # Invert Chirality
        act_invert = menu.addAction("🪞 Invert Chirality at Atom")
        act_invert.triggered.connect(lambda: self._on_invert_chiral_center_at(atom_idx))

        # Adjust connectivity
        act_reorder_b = menu.addAction("Adjust Bond Partner (Alt+B)")
        act_reorder_b.triggered.connect(lambda: self._start_partial_reorder("bond"))

        # Freeze / Unfreeze atom for QM & FF Optimization
        is_frozen = atom_idx in getattr(self, "frozen_atoms", set())
        freeze_lbl = "🔓 Unfreeze Atom Position" if is_frozen else "🔒 Freeze Atom for QM Optimization"
        act_freeze = menu.addAction(freeze_lbl)
        act_freeze.triggered.connect(lambda: self._toggle_freeze_atom(atom_idx))

        menu.addSeparator()

        # Copy Actions
        sel_count = len(getattr(self.viewer, "multi_selected", [])) or 1
        act_copy = menu.addAction(f"📋 Copy Atom{'s' if sel_count > 1 else ''} / Molecule (Ctrl+C)")
        act_copy.triggered.connect(self._copy_selected_fragment)

        act_copy_img = menu.addAction("📷 Copy 3D Image to Clipboard (Ctrl+Shift+C)")
        act_copy_img.triggered.connect(lambda: self.copy_image_to_clipboard(transparent=True, scale=2, clean_only=True))

        act_copy_block = menu.addAction("📄 Copy Coordinates Block (ORCA/Gaussian, Ctrl+Alt+X)")
        act_copy_block.triggered.connect(self._on_copy_coords_block)

        if bool(self.stored_fragment or self.last_added_fragment):
            frag = self.stored_fragment or self.last_added_fragment
            act_paste = menu.addAction(f"📋 Paste Fragment ({frag.get('name', 'Fragment')}) (Ctrl+V)")
            act_paste.triggered.connect(self._on_paste)

        menu.addSeparator()

        # Delete Action
        act_del = menu.addAction("🗑️ Delete Atom (Del)")
        act_del.triggered.connect(lambda: self._delete_atom_or_orbit(atom_idx, source_label="Deleted"))

        menu.exec(global_pos)

    def _on_open_fragment_library(self):
        """Switch to the Presets & Fragments inspector tab."""
        if hasattr(self, "tabs"):
            idx = self.tabs.indexOf(getattr(self, "tab_builder", None))
            if idx >= 0:
                self.tabs.setCurrentIndex(idx)
            else:
                self.tabs.setCurrentIndex(2)

    def _on_viewport_bond_context_menu(self, u: int, v: int, global_pos: QtCore.QPoint):
        """Display context menu when right-clicking a bond."""
        curr_bonds = self._current_explicit_bonds()
        if norm_bond(u, v) not in curr_bonds:
            return

        bo = self._current_bond_orders().get(norm_bond(u, v), 1)
        sym_u, sym_v = self.zatoms[u].symbol, self.zatoms[v].symbol

        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("QMenu { font-size: 12px; } QMenu::item { padding: 4px 20px 4px 20px; }")

        title_act = QtGui.QAction(f"Bond: {sym_u}{u+1} - {sym_v}{v+1}", self)
        title_font = title_act.font()
        title_font.setBold(True)
        title_act.setFont(title_font)
        title_act.setEnabled(False)
        menu.addAction(title_act)
        menu.addSeparator()

        # Bond Orders
        act_s = menu.addAction("Single Bond (Key: 1)")
        if bo == 1: act_s.setText("✔ Single Bond (Key: 1)")
        act_s.triggered.connect(lambda: self.set_bond_order(u, v, target_bo=1))

        act_d = menu.addAction("Double Bond (Key: 2)")
        if bo == 2: act_d.setText("✔ Double Bond (Key: 2)")
        act_d.triggered.connect(lambda: self.set_bond_order(u, v, target_bo=2))

        act_t = menu.addAction("Triple Bond (Key: 3)")
        if bo == 3: act_t.setText("✔ Triple Bond (Key: 3)")
        act_t.triggered.connect(lambda: self.set_bond_order(u, v, target_bo=3))

        # Stretch / Tweak Bond Length (GaussView Style)
        act_stretch = menu.addAction("📏 Stretch Bond Length (Interactive Slider)...")
        act_stretch.triggered.connect(lambda: self._activate_bond_tweak_bar(u, v))

        # Weld / Fuse Molecular Ring onto this Bond
        act_weld = menu.addAction("💍 Weld / Fuse Ring onto this Bond (Ctrl+Shift+W)...")
        act_weld.triggered.connect(lambda: self._on_open_cycle_welder_for_bond(u, v))

        menu.addSeparator()

        # Break / Delete Bond
        act_break = menu.addAction("✂️ Delete / Break Bond")
        act_break.triggered.connect(lambda: self._break_bond(u, v))

        menu.exec(global_pos)

    def _on_viewport_canvas_context_menu(self, global_pos: QtCore.QPoint):
        """Display context menu when right-clicking empty 3D canvas."""
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("QMenu { font-size: 12px; } QMenu::item { padding: 4px 20px 4px 20px; }")

        act_center = menu.addAction("🎯 Center & Reset View (Space)")
        act_center.triggered.connect(self.viewer.auto_center)

        act_add_line = menu.addAction("➕ Add Atom Line...")
        act_add_line.triggered.connect(self._on_add_line)

        act_add_h = menu.addAction("💧 Complete Valence with Hydrogens (Ctrl+H)")
        act_add_h.triggered.connect(self._on_add_hydrogens)

        act_opt = menu.addAction("⚡ Optimize Geometry (Ctrl+M)")
        act_opt.triggered.connect(self._on_optimize_geometry)

        act_build_name = menu.addAction("🧪 Build from Name / SMILES / PubChem (Ctrl+Shift+B)...")
        act_build_name.triggered.connect(self._on_prompt_smiles_or_inchi)

        if len(self.zatoms) > 0:
            act_copy_block = menu.addAction("📄 Copy Coordinates Block (ORCA/Gaussian, Ctrl+Alt+X)")
            act_copy_block.triggered.connect(self._on_copy_coords_block)

        # Paste Action in Free Space
        has_pasteable = bool(self.stored_fragment or self.last_added_fragment)
        clip_txt = QtWidgets.QApplication.clipboard().text().strip()
        if not has_pasteable and clip_txt:
            has_pasteable = True
        if has_pasteable:
            frag_lbl = (self.stored_fragment or self.last_added_fragment or {}).get("name", "Fragment / Molecule")
            act_paste = menu.addAction(f"📋 Paste {frag_lbl} (Ctrl+V)")
            local_pos = self.viewer.mapFromGlobal(global_pos)
            act_paste.triggered.connect(lambda: self._paste_fragment_to_free_space(
                self.stored_fragment or self.last_added_fragment, screen_pos=local_pos
            ) if (self.stored_fragment or self.last_added_fragment) else self._on_paste())

        menu.addSeparator()
        act_esp = menu.addAction("🌈 3D Electrostatic Potential (ESP) Map (Ctrl+Alt+P)")
        act_esp.triggered.connect(self._on_quick_view_esp_map)

        act_vol = menu.addAction("🌐 Volumetric Studio & MOs (Ctrl+Alt+O)...")
        act_vol.triggered.connect(self._on_open_volumetric_dialog)

        act_light = menu.addAction("💡 3D Lighting & Figure Studio...")
        act_light.triggered.connect(self._on_open_lighting_studio)

        if getattr(self.viewer, "active_isosurfaces", None):
            act_clear_surf = menu.addAction("✖ Clear 3D Orbital / Isosurface (Esc)")
            act_clear_surf.triggered.connect(self._on_clear_orbital_surface)

        menu.addSeparator()

        act_sel_all = menu.addAction("Select All Atoms (Ctrl+A)")
        act_sel_all.triggered.connect(lambda: setattr(self.viewer, 'multi_selected', list(range(len(self.zatoms)))) or self.viewer.update())

        act_desel = menu.addAction("Deselect All (Ctrl+D)")
        act_desel.triggered.connect(lambda: setattr(self.viewer, 'multi_selected', []) or setattr(self.viewer, 'selected_atom_idx', None) or self.viewer.update())

        menu.exec(global_pos)

    def _on_open_lighting_studio(self):
        """Open 3D Lighting, Shaders & Publication Studio."""
        from .dialog_lighting import LightingStudioDialog
        dlg = LightingStudioDialog(self.viewer, parent=self)
        dlg.exec()

    def _mutate_atom_element(self, atom_idx: int, new_element: str):
        """Mutate an atom's element symbol cleanly (and across its symmetry orbit if Impose Symmetry is enabled)."""
        if atom_idx < 0 or atom_idx >= len(self.zatoms):
            return
        old_sym = self.zatoms[atom_idx].symbol
        if old_sym == new_element:
            return
        self._push_undo_state()

        symbols = [a.symbol for a in self.zatoms]
        coords = np.copy(self.coords)
        sym_res = self._get_active_imposed_symmetry()
        if sym_res is not None and len(coords) == len(symbols):
            _, orbit_ops = get_atom_orbit_and_operations(
                symbols, coords, atom_idx, result=sym_res, tolerance=max(0.25, sym_res.tolerance * 1.5)
            )
            mutate_indices = sorted({eq_i for eq_i, _ in orbit_ops})
        else:
            mutate_indices = [atom_idx]

        for idx in mutate_indices:
            self.zatoms[idx].symbol = new_element

        # Update bond distances for newly mutated terminal atoms (or atoms with ref1) so C-Cl -> C-H / C-C etc.
        for idx in mutate_indices:
            ref1 = self.zatoms[idx].ref1
            if ref1 is not None and 0 <= ref1 < len(self.zatoms):
                parent_sym = self.zatoms[ref1].symbol
                self.zatoms[idx].dist = estimate_bond_length(new_element, parent_sym)

        auto_h = bool(getattr(self, "chk_auto_h", None) and self.chk_auto_h.isChecked())
        if auto_h and new_element not in ("H", "X") and sym_res is None:
            from .alignment import apply_rigid_frame
            new_syms = [a.symbol for a in self.zatoms]
            local_coords = zmatrix_to_cartesian(self.zatoms)
            if getattr(self, "_rigid_frame", None) is not None and len(local_coords) > 0:
                new_coords = apply_rigid_frame(local_coords, self._rigid_frame)
            else:
                new_coords = local_coords
            curr_bonds = self._current_explicit_bonds()
            curr_bo = self._current_bond_orders()
            self._commit_explicit_structure(
                new_syms, new_coords, curr_bonds, curr_bo,
                relax_iters=80, add_h=True, fix_valence=True, target_indices=set(mutate_indices)
            )
        else:
            self.table_widget.load_zatoms(self.zatoms)
            self.update_geometry_live()
        if len(mutate_indices) > 1:
            self.statusBar().showMessage(
                f"Mutated {len(mutate_indices)} symmetry-equivalent {old_sym} atoms ➔ {new_element}", 4000
            )
        else:
            self.statusBar().showMessage(f"Mutated atom #{atom_idx+1}: {old_sym} ➔ {new_element}", 3000)

    def _prompt_mutate_periodic_table(self, atom_idx: int):
        """Open periodic table dialog to choose a mutation element."""
        dlg = PeriodicTableDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted and dlg.chosen_element:
            self._mutate_atom_element(atom_idx, dlg.chosen_element)

    def _attach_specific_fragment(self, atom_idx: int, frag_name: str):
        """Attach a specific fragment from FRAGMENTS to the selected atom."""
        if frag_name not in FRAGMENTS or atom_idx < 0 or atom_idx >= len(self.coords):
            return
        self._push_undo_state()
        frag = FRAGMENTS[frag_name]
        conn_idx = int(frag.get("connect_idx", 0))
        n_sites = self._attach_fragment_to_atom_or_orbit(atom_idx, frag, conn_idx=conn_idx, relax_iters=100)
        self.statusBar().showMessage(f"Attached {frag_name} fragment at atom #{atom_idx+1}!", 3500)

    def _break_bond(self, u: int, v: int):
        """Delete / break a specific covalent bond."""
        curr_bonds = self._current_explicit_bonds()
        pair = norm_bond(u, v)
        if pair in curr_bonds:
            self._push_undo_state()
            new_bonds = [b for b in curr_bonds if b != pair]
            self.cross_bonds = {b for b in self.cross_bonds if b != pair}
            self.strict_topology = True
            new_zatoms = cartesian_to_zmatrix(
                [a.symbol for a in self.zatoms],
                self.coords,
                user_bonds=new_bonds,
                strict_bonds=True,
                existing_zatoms=self.zatoms,
            )
            self.set_molecule_from_zatoms(new_zatoms)
            self.statusBar().showMessage(f"Broken bond between atom #{u+1} and atom #{v+1}", 3000)

    def _on_viewport_atom_right_clicked_delete(self, atom_idx: int):
        """Right-click on an atom deletes it directly (Avogadro behavior)."""
        self._delete_atom_or_orbit(atom_idx, source_label="Right-click removed")

    def _invert_molecule_coordinates(self, plane: str = "inversion"):
        """Invert molecule coordinates: inversion through origin or reflection across XY, XZ, YZ planes."""
        if len(self.coords) == 0:
            return
        self._push_undo_state()
        new_coords = np.copy(self.coords)
        if plane == "inversion":
            new_coords = -new_coords
            label = "Inverted chirality / point of inversion (x, y, z ➔ -x, -y, -z)"
        elif plane == "xy":
            new_coords[:, 2] = -new_coords[:, 2]
            label = "Mirrored across XY plane (z ➔ -z)"
        elif plane == "xz":
            new_coords[:, 1] = -new_coords[:, 1]
            label = "Mirrored across XZ plane (y ➔ -y)"
        elif plane == "yz":
            new_coords[:, 0] = -new_coords[:, 0]
            label = "Mirrored across YZ plane (x ➔ -x)"
        else:
            new_coords = -new_coords
            label = "Inverted molecule coordinates"

        symbols = [a.symbol for a in self.zatoms]
        curr_bonds = self._current_explicit_bonds()
        new_zatoms = cartesian_to_zmatrix(symbols, new_coords, user_bonds=curr_bonds, strict_bonds=True)
        self.set_molecule_from_zatoms(new_zatoms, exact_coords=new_coords)
        self.statusBar().showMessage(f"Chirality / Symmetry: {label}", 4000)

    def _check_and_highlight_steric_clashes(self):
        """Detect and highlight severe non-bonded steric clashes with interatomic distance < 0.70 * (r_vdw1 + r_vdw2)."""
        if len(self.coords) < 2:
            return
        from .spatial_index import find_steric_clashes
        symbols = [a.symbol for a in self.zatoms]
        bonds = set(self._current_explicit_bonds())
        raw_clashes = find_steric_clashes(symbols, self.coords, bonds=bonds, threshold_ratio=0.70)

        if raw_clashes:
            clash_atoms = sorted({idx for c in raw_clashes for idx in (c[0], c[1])})
            self.viewer.multi_selected = clash_atoms
            self.viewer.selected_atom_idx = clash_atoms[0]
            self.viewer.update()
            c_desc = ", ".join(f"{u_sym}{u+1}···{v_sym}{v+1} ({d:.2f} Å)" for u, v, u_sym, v_sym, d, _ in raw_clashes[:4])
            self.statusBar().showMessage(f"⚠️ Detected {len(raw_clashes)} severe steric clash(es): {c_desc} (Selected in 3D)", 5000)
        else:
            self.statusBar().showMessage("✅ Clean geometry: Zero severe steric clashes detected (all non-bonded distances > 70% Σr_vdw).", 3500)

    def _open_unit_converter(self):
        """Open physical constant energy unit converter dialog."""
        from .dialog_unit_converter import UnitConverterDialog
        last_ha = 1.0
        last_qm = getattr(self, "_last_qm_res", None)
        if last_qm and getattr(last_qm, "trajectory_energies", None):
            last_ha = abs(last_qm.trajectory_energies[-1] / 27.211386245988)
        dlg = UnitConverterDialog(self, initial_hartree=last_ha)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _center_molecule_to_origin(self, mode: str = "com"):
        """Translate molecule so center-of-mass or selected anchor atom is precisely at origin (0, 0, 0)."""
        if len(self.coords) == 0:
            return
        self._push_undo_state()
        new_coords = np.copy(self.coords)
        if mode == "anchor" and hasattr(self.viewer, "selected_atom_idx") and self.viewer.selected_atom_idx is not None:
            anchor = self.viewer.selected_atom_idx
            shift = -new_coords[anchor]
            msg = f"Shifted atom #{anchor+1} ({self.zatoms[anchor].symbol}) to Cartesian Origin (0, 0, 0)"
        else:
            # Center of Mass
            masses = []
            for a in self.zatoms:
                info = ELEMENT_DATA.get(a.symbol, {})
                masses.append(info.get("mass", 12.0))
            mass_arr = np.array(masses)
            com = np.sum(new_coords * mass_arr[:, None], axis=0) / np.sum(mass_arr)
            shift = -com
            msg = f"Shifted molecular Center of Mass to Cartesian Origin (0, 0, 0)"

        new_coords += shift
        symbols = [a.symbol for a in self.zatoms]
        curr_bonds = self._current_explicit_bonds()
        new_zatoms = cartesian_to_zmatrix(symbols, new_coords, user_bonds=curr_bonds, strict_bonds=True)
        self.set_molecule_from_zatoms(new_zatoms, exact_coords=new_coords)
        self.statusBar().showMessage(msg, 3500)

    def _on_bond_tweak_slider_changed(self, int_val: int):
        """Live update bond length as user drags the GaussView-style slider."""
        dist = int_val / 100.0
        if hasattr(self, "lbl_tweak_val"):
            self.lbl_tweak_val.setText(f"{dist:.3f} Å")

        if not hasattr(self, "_active_tweak_bond") or not self._active_tweak_bond:
            return
        u, v = self._active_tweak_bond
        if u >= len(self.coords) or v >= len(self.coords):
            return

        # Move the subtree on atom v along the bond axis
        adj = build_adj(len(self.coords), self._current_explicit_bonds())
        # Find all atoms connected to v without crossing edge (u, v)
        sub_atoms = set()
        queue = [v]
        visited = {u, v}
        while queue:
            curr = queue.pop(0)
            sub_atoms.add(curr)
            for nbr in adj.get(curr, []):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)

        p_u = self.coords[u]
        p_v = self.coords[v]
        curr_vec = p_v - p_u
        curr_len = np.linalg.norm(curr_vec)
        if curr_len < 1e-4:
            return
        unit_vec = curr_vec / curr_len
        target_vec = unit_vec * dist
        shift = target_vec - curr_vec

        new_coords = np.copy(self.coords)
        for idx in sub_atoms:
            new_coords[idx] += shift

        self.coords = new_coords
        self.viewer.set_molecule(
            symbols=[a.symbol for a in self.zatoms],
            coords=self.coords,
            user_bonds=self._current_explicit_bonds()
        )
        self.viewer.update()

    def _on_bond_tweak_slider_released(self):
        """Sync Z-matrix table once user finishes dragging the bond tweak slider."""
        if not hasattr(self, "_active_tweak_bond") or not self._active_tweak_bond:
            return
        self._push_undo_state()
        symbols = [a.symbol for a in self.zatoms]
        curr_bonds = self._current_explicit_bonds()
        new_zatoms = cartesian_to_zmatrix(symbols, self.coords, user_bonds=curr_bonds, strict_bonds=True)
        self.set_molecule_from_zatoms(new_zatoms, exact_coords=self.coords)
        self.statusBar().showMessage("Updated geometry from interactive bond slider.", 3000)

    def _activate_bond_tweak_bar(self, u: int, v: int):
        """Show and initialize the GaussView-style interactive bond stretch slider."""
        if u >= len(self.coords) or v >= len(self.coords):
            return
        self._active_tweak_bond = (u, v)
        curr_d = float(np.linalg.norm(self.coords[u] - self.coords[v]))
        if hasattr(self, "grp_bond_tweak"):
            self.grp_bond_tweak.setVisible(True)
            self.lbl_tweak_bond.setText(f"Stretch Bond: {self.zatoms[u].symbol}{u+1} — {self.zatoms[v].symbol}{v+1}")
            self.slider_bond_tweak.blockSignals(True)
            self.slider_bond_tweak.setValue(int(curr_d * 100.0))
            self.slider_bond_tweak.blockSignals(False)
            self.lbl_tweak_val.setText(f"{curr_d:.3f} Å")

    def _copy_viewport_image_to_clipboard(self):
        """Render high-resolution offscreen image of the current viewport and copy directly to the system clipboard."""
        try:
            pixmap = self.viewer.grab()
            if not pixmap.isNull():
                clipboard = QtWidgets.QApplication.clipboard()
                clipboard.setPixmap(pixmap)
                self.statusBar().showMessage("📋 Copied high-res 3D molecular image to clipboard!", 3500)
            else:
                self.statusBar().showMessage("Failed to capture viewport image.", 3000)
        except Exception as e:
            self.statusBar().showMessage(f"Clipboard copy error: {e}", 3000)

    def _toggle_freeze_atom(self, atom_idx: int):
        """Toggle frozen state of an atom for QM input generation and local minimization."""
        if not hasattr(self, "frozen_atoms"):
            self.frozen_atoms = set()
        if atom_idx in self.frozen_atoms:
            self.frozen_atoms.remove(atom_idx)
            self.statusBar().showMessage(f"Unfroze atom #{atom_idx+1} ({self.zatoms[atom_idx].symbol})", 3000)
        else:
            self.frozen_atoms.add(atom_idx)
            self.statusBar().showMessage(f"Locked / Frozen atom #{atom_idx+1} for QM optimization", 3000)
        if hasattr(self, "_update_frozen_buttons_ui"):
            self._update_frozen_buttons_ui()

    def _on_delete_selected_atom(self):
        n = len(self.zatoms)
        multi = sorted({i for i in getattr(self.viewer, "multi_selected", []) if 0 <= i < n})
        if len(multi) > 1:
            self._push_undo_state()
            symbols = [a.symbol for a in self.zatoms]
            coords = np.copy(self.coords)
            bonds = self._current_explicit_bonds()
            bo = self._current_bond_orders()
            rem_set = set(multi)
            new_syms, new_coords, new_bonds, new_bo, o2n_del = remove_atoms(symbols, coords, bonds, rem_set, bo)
            if getattr(self, "frozen_atoms", None):
                self.frozen_atoms = {o2n_del[i] for i in self.frozen_atoms if i in o2n_del}
                if hasattr(self, "_update_frozen_buttons_ui"):
                    self._update_frozen_buttons_ui()
            self.viewer.selected_atom_idx = None
            self.viewer.multi_selected = []
            if not new_syms:
                self.set_molecule_from_zatoms([])
                self.statusBar().showMessage(f"Deleted all {len(multi)} selected atoms. Molecule is now empty.", 3500)
                return
            self._commit_explicit_structure(new_syms, new_coords, new_bonds, new_bo, relax_iters=0, add_h=False, fix_valence=False)
            self.statusBar().showMessage(f"Deleted {len(multi)} selected atoms and updated Z-matrix.", 4000)
            return

        idx = self.viewer.selected_atom_idx
        if idx is None and len(multi) == 1:
            idx = multi[0]
        if idx is None or idx < 0 or idx >= n:
            QtWidgets.QMessageBox.warning(self, "No Atom Selected", "Please click an atom to select it for deletion.")
            return
        self._delete_atom_or_orbit(idx, source_label="Deleted")

    def _on_attach_fragment(self):
        """Attach a functional group (e.g. -CH3, -OH, -Phenyl, -CO) at selected atom (and its symmetry orbit if imposed)."""
        idx = self.viewer.selected_atom_idx
        if idx is None or idx < 0 or idx >= len(self.coords):
            QtWidgets.QMessageBox.warning(self, "Selection Required", "Please select an anchor atom in the 3D viewport or table first.")
            return

        self._push_undo_state()

        frag_name = self.combo_fragment.currentText()
        frag = FRAGMENTS[frag_name]

        # Determine chosen fragment attachment atom index
        conn_idx = 0
        if hasattr(self, "combo_frag_connect") and self.combo_frag_connect.currentIndex() >= 0:
            conn_idx = self.combo_frag_connect.currentData()
            if conn_idx is None:
                conn_idx = self.combo_frag_connect.currentIndex()
        else:
            conn_idx = int(frag.get("connect_idx", 0))

        frag_symbols = list(frag["symbols"])
        if conn_idx < 0 or conn_idx >= len(frag_symbols):
            conn_idx = 0
        conn_sym = frag_symbols[conn_idx]
        anchor_sym = self.zatoms[idx].symbol

        self.last_added_fragment = dict(frag)
        self.last_added_fragment["name"] = frag_name
        self.last_added_fragment["connect_idx"] = conn_idx
        self.last_item_type = "fragment"
        self._update_viewer_drag_label()

        n_sites = self._attach_fragment_to_atom_or_orbit(
            idx, frag, conn_idx=conn_idx, custom_dir_u=None, relax_iters=120
        )
        sym_note = f" across all {n_sites} symmetry-equivalent atoms" if n_sites > 1 else f" at {anchor_sym}{idx+1}"
        self.statusBar().showMessage(
            f"Attached {frag_name} fragment (via #{conn_idx+1}: {conn_sym}){sym_note}!",
            4500
        )

    def _copy_selected_fragment(self):
        """
        Copy selected group of atoms as a fragment into memory (Ctrl+C).
        The first atom in the selection is preserved as the primary anchor (placed at origin [0, 0, 0]).
        When an atom is subsequently double-clicked, it is replaced by this stored fragment.
        """
        sel = list(self.viewer.multi_selected)
        if not sel and getattr(self.viewer, "selected_atom_idx", None) is not None:
            sel = [self.viewer.selected_atom_idx]

        # Deduplicate preserving order
        sel = [x for i, x in enumerate(sel) if x not in sel[:i] and 0 <= x < len(self.zatoms)]
        if not sel and len(self.zatoms) > 0:
            # If nothing is explicitly selected, default to copying the entire molecule
            sel = list(range(len(self.zatoms)))

        if not sel:
            self.statusBar().showMessage("No molecule or atoms to copy. Workspace is empty.", 3500)
            return

        symbols = [self.zatoms[i].symbol for i in sel]
        first_idx = sel[0]
        anchor_coords = self.coords[first_idx]
        rel_coords = (self.coords[sel] - anchor_coords).tolist()

        pos = {old: k for k, old in enumerate(sel)}
        frag_bonds = [(pos[u], pos[v]) for u, v in self._current_explicit_bonds() if u in pos and v in pos]
        frag_bo = {}
        for (u, v), o in self._current_bond_orders().items():
            if u in pos and v in pos:
                frag_bo[norm_bond(pos[u], pos[v])] = o
        self.stored_fragment = {
            "name": f"Selection ({'-'.join(symbols[:3])}{'...' if len(symbols) > 3 else ''})",
            "symbols": symbols,
            "coords": rel_coords,
            "bonds": frag_bonds,
            "bond_orders": frag_bo,
            "connect_idx": 0,
            "default_bond": estimate_bond_length(symbols[0], "C"),
        }
        # Clear last_added_fragment so stored_fragment has precedence
        self.last_added_fragment = None
        self.last_item_type = "fragment"
        self._update_viewer_drag_label()

        # Also copy XYZ format to system clipboard for seamless pasting across apps or within AgyMolden
        try:
            from .chem_identifiers import format_xyz_string
            xyz_txt = format_xyz_string(symbols, self.coords[sel], comment=f"AgyMolden Selection ({len(sel)} atoms)")
            QtWidgets.QApplication.clipboard().setText(xyz_txt)
        except Exception as e:
            log.debug("Failed copying selection XYZ to clipboard: %s", e)

        first_atom_label = f"{symbols[0]}{first_idx + 1}"
        self.statusBar().showMessage(
            f"Copied {len(sel)} atom(s) into clipboard / memory (Anchor: {first_atom_label}). Press Ctrl+V, double-click canvas, or double-click an atom to attach/paste.",
            5000
        )

    def _on_paste(self):
        """
        Paste copied molecule/fragment into the 3D viewport (Ctrl+V).
        - If active molecule is empty, and clipboard contains XYZ/SMILES/InChI, loads it directly as the new molecule.
        - If an atom is selected, attaches or merges the fragment near the selected atom.
        - Otherwise, places the fragment in free space with a slight offset so it is distinct.
        """
        # 1. Check if there is an in-memory stored fragment or last added fragment
        frag = self.stored_fragment or self.last_added_fragment
        clip_parsed_syms = None
        clip_parsed_coords = None

        # 2. If no in-memory fragment, check system clipboard for XYZ text
        if not frag:
            clip_text = QtWidgets.QApplication.clipboard().text().strip()
            if clip_text:
                try:
                    from .zmatrix import parse_xyz
                    parsed_syms, parsed_coords = parse_xyz(clip_text)
                    if parsed_syms and len(parsed_coords) == len(parsed_syms):
                        clip_parsed_syms = parsed_syms
                        clip_parsed_coords = np.array(parsed_coords, dtype=np.float64)
                        # If active molecule is currently empty, load XYZ directly as full molecule
                        if len(self.zatoms) == 0:
                            from .controllers.import_controller import ImportController
                            ImportController(self).load_xyz_data(
                                clip_parsed_syms, clip_parsed_coords, source_label="Pasted XYZ"
                            )
                            return
                        f_coords = clip_parsed_coords
                        anchor_c = f_coords[0] if len(f_coords) > 0 else np.zeros(3)
                        rel_c = (f_coords - anchor_c).tolist()
                        frag = {
                            "name": f"Clipboard ({len(parsed_syms)} atoms)",
                            "symbols": parsed_syms,
                            "coords": rel_c,
                            "bonds": detect_connectivity(parsed_syms, f_coords, strict=False),
                            "bond_orders": {},
                            "connect_idx": 0,
                            "default_bond": estimate_bond_length(parsed_syms[0], "C"),
                        }
                except Exception:
                    frag = None

                # If not standard XYZ, check if clipboard contains a SMILES, InChI, or Chemical Name
                if not frag and len(clip_text) < 500 and not clip_text.startswith("{") and not clip_text.startswith("<"):
                    try:
                        from .chem_identifiers import parse_smiles_or_inchi_to_3d
                        s_syms, s_coords, s_bonds, s_bo = parse_smiles_or_inchi_to_3d(clip_text)
                        if s_syms and len(s_coords) == len(s_syms):
                            if len(self.zatoms) == 0:
                                from .controllers.import_controller import ImportController
                                ImportController(self).load_xyz_data(
                                    s_syms, np.array(s_coords, dtype=np.float64), source_label=f"Pasted '{clip_text[:20]}'"
                                )
                                return
                            f_coords = np.array(s_coords, dtype=np.float64)
                            anchor_c = f_coords[0] if len(f_coords) > 0 else np.zeros(3)
                            rel_c = (f_coords - anchor_c).tolist()
                            frag = {
                                "name": f"Pasted ({clip_text[:20]})",
                                "symbols": s_syms,
                                "coords": rel_c,
                                "bonds": s_bonds,
                                "bond_orders": s_bo,
                                "connect_idx": 0,
                                "default_bond": estimate_bond_length(s_syms[0], "C"),
                            }
                            self.statusBar().showMessage(f"Converted '{clip_text}' from clipboard to 3D fragment ({len(s_syms)} atoms)!", 4000)
                    except Exception:
                        pass

        if not frag:
            self.statusBar().showMessage("Clipboard / fragment memory is empty. Select atoms and press Ctrl+C first, or copy Cartesian XYZ coordinates.", 3500)
            return

        w, h = max(1, self.viewer.width()), max(1, self.viewer.height())
        center_screen = QtCore.QPoint(int(w * 0.55), int(h * 0.55))
        self._paste_fragment_to_free_space(frag, screen_pos=center_screen)

    def _on_paste_as_new_molecule(self):
        """
        Directly parse Cartesian XYZ coordinates or SMILES from clipboard and set as the active molecule (Ctrl+Shift+V).
        Supports:
          - Standard XYZ format (with atom count header & comment)
          - Headerless XYZ coordinate blocks (Element X Y Z)
          - Comma-delimited coordinates
          - SMILES, InChI, or chemical name
        """
        clip_text = QtWidgets.QApplication.clipboard().text().strip()
        if not clip_text:
            QtWidgets.QMessageBox.information(
                self,
                "Clipboard Empty",
                "System clipboard is empty.\n\nCopy Cartesian XYZ coordinates (or SMILES / chemical name) to your clipboard and try again.",
            )
            return

        from .zmatrix import parse_xyz
        try:
            syms, coords = parse_xyz(clip_text)
            if syms and len(coords) == len(syms):
                from .controllers.import_controller import ImportController
                ImportController(self).load_xyz_data(syms, coords, source_label="Clipboard XYZ")
                return
        except Exception:
            pass

        # If not standard XYZ, try SMILES/InChI
        if len(clip_text) < 500 and not clip_text.startswith("{") and not clip_text.startswith("<"):
            try:
                from .chem_identifiers import parse_smiles_or_inchi_to_3d
                s_syms, s_coords, s_bonds, s_bo = parse_smiles_or_inchi_to_3d(clip_text)
                if s_syms and len(s_coords) == len(s_syms):
                    from .controllers.import_controller import ImportController
                    ImportController(self).load_xyz_data(
                        s_syms, np.array(s_coords, dtype=np.float64), source_label=f"Clipboard '{clip_text[:20]}'"
                    )
                    return
            except Exception:
                pass

        QtWidgets.QMessageBox.warning(
            self,
            "Could Not Parse Coordinates",
            "Could not parse valid Cartesian XYZ coordinates from clipboard.\n\n"
            "Format expected:\n"
            "  O  0.000  0.000  0.117\n"
            "  H  0.000  0.758 -0.469\n"
            "  H  0.000 -0.758 -0.469",
        )

    def _on_copy_as_xyz(self):
        """Copy current structure in Cartesian XYZ format to clipboard."""
        if len(self.zatoms) == 0:
            self.statusBar().showMessage("Molecule is empty. Nothing to copy.", 3000)
            return
        from .chem_identifiers import format_xyz_string
        symbols = [a.symbol for a in self.zatoms]
        xyz_txt = format_xyz_string(symbols, self.coords, comment="AgyMolden structure")
        QtWidgets.QApplication.clipboard().setText(xyz_txt)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Copied Cartesian XYZ to clipboard!", self)
        self.statusBar().showMessage(f"Copied XYZ ({len(symbols)} atoms) to clipboard.", 3500)

    def _on_copy_coords_block(self):
        """Copy headerless 'Element X Y Z' block to clipboard (ready for ORCA / Gaussian / Q-Chem input files)."""
        if len(self.zatoms) == 0:
            self.statusBar().showMessage("Molecule is empty. Nothing to copy.", 3000)
            return
        from .chem_identifiers import format_coords_block_string
        symbols = [a.symbol for a in self.zatoms]
        block_txt = format_coords_block_string(symbols, self.coords)
        QtWidgets.QApplication.clipboard().setText(block_txt)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Copied ORCA/Gaussian coordinates block to clipboard!", self)
        self.statusBar().showMessage(
            f"Copied coordinates block ({len(symbols)} atoms, no header) to clipboard — ready to paste into ORCA/Gaussian!",
            4000,
        )

    def _on_copy_as_smiles(self):
        """Generate and copy SMILES string to clipboard."""
        if len(self.zatoms) == 0:
            self.statusBar().showMessage("Molecule is empty. Nothing to copy.", 3000)
            return
        try:
            from .chem_identifiers import get_smiles
            symbols = [a.symbol for a in self.zatoms]
            bonds = self._current_explicit_bonds()
            bo = self._current_bond_orders()
            smi = get_smiles(symbols, self.coords, bonds, bo)
            QtWidgets.QApplication.clipboard().setText(smi)
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"Copied SMILES: {smi}", self)
            self.statusBar().showMessage(f"Copied SMILES: {smi}", 4500)
        except Exception as e:
            log.exception("Failed to generate SMILES")
            QtWidgets.QMessageBox.warning(self, "SMILES Generation Error", f"Could not generate SMILES:\n{str(e)}")

    def _on_copy_as_inchi(self):
        """Generate and copy InChI string to clipboard."""
        if len(self.zatoms) == 0:
            self.statusBar().showMessage("Molecule is empty. Nothing to copy.", 3000)
            return
        try:
            from .chem_identifiers import get_inchi_and_key
            symbols = [a.symbol for a in self.zatoms]
            bonds = self._current_explicit_bonds()
            bo = self._current_bond_orders()
            inchi, _ = get_inchi_and_key(symbols, self.coords, bonds, bo)
            QtWidgets.QApplication.clipboard().setText(inchi)
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Copied InChI to clipboard!", self)
            self.statusBar().showMessage(f"Copied InChI: {inchi}", 5000)
        except Exception as e:
            log.exception("Failed to generate InChI")
            QtWidgets.QMessageBox.warning(self, "InChI Generation Error", f"Could not generate InChI:\n{str(e)}")

    def _on_copy_as_inchikey(self):
        """Generate and copy InChIKey to clipboard."""
        if len(self.zatoms) == 0:
            self.statusBar().showMessage("Molecule is empty. Nothing to copy.", 3000)
            return
        try:
            from .chem_identifiers import get_inchi_and_key
            symbols = [a.symbol for a in self.zatoms]
            bonds = self._current_explicit_bonds()
            bo = self._current_bond_orders()
            _, inchikey = get_inchi_and_key(symbols, self.coords, bonds, bo)
            QtWidgets.QApplication.clipboard().setText(inchikey)
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"Copied InChIKey: {inchikey}", self)
            self.statusBar().showMessage(f"Copied InChIKey: {inchikey}", 4500)
        except Exception as e:
            log.exception("Failed to generate InChIKey")
            QtWidgets.QMessageBox.warning(self, "InChIKey Generation Error", f"Could not generate InChIKey:\n{str(e)}")

    def _on_copy_supporting_info(self, style: str = "acs"):
        """Format and copy a publication-ready Supporting Information (SI) entry (ACS, Markdown, or LaTeX) to clipboard."""
        if len(self.zatoms) == 0:
            self.statusBar().showMessage("Molecule is empty. Nothing to copy.", 3000)
            return
        from .chem_identifiers import format_supporting_information_entry
        symbols = [a.symbol for a in self.zatoms]
        pt_grp = getattr(self.viewer, "current_point_group_label", "") or "C1"
        title = getattr(self, "title_override", "") or "Optimized Structure"
        qm_res = getattr(self, "_last_qm_res", None)
        e_ha = None
        zpe_ha = None
        h_ha = None
        g_ha = None
        dip = None
        homo = None
        lumo = None
        n_imag = None
        low_freq = None
        if qm_res is not None:
            if getattr(qm_res, "trajectory_energies", None):
                e_ha = float(qm_res.trajectory_energies[-1]) / 27.211386245988
            zpe_ha = getattr(qm_res, "zpe_ha", None)
            h_ha = getattr(qm_res, "enthalpy_ha", None)
            g_ha = getattr(qm_res, "gibbs_ha", None)
            dip = getattr(qm_res, "dipole_debye", None)
            homo = getattr(qm_res, "homo_ev", None)
            lumo = getattr(qm_res, "lumo_ev", None)
            vib = getattr(qm_res, "vibrational_analysis", None)
            if vib and getattr(vib, "frequencies", None):
                freqs = list(vib.frequencies)
                n_imag = sum(1 for f in freqs if f < 0.0)
                low_freq = float(min(freqs)) if freqs else None

        si_text = format_supporting_information_entry(
            symbols=symbols,
            coords=self.coords,
            title=title,
            point_group=pt_grp,
            energy_ha=e_ha,
            zpe_ha=zpe_ha,
            enthalpy_ha=h_ha,
            gibbs_ha=g_ha,
            dipole_debye=dip,
            homo_ev=homo,
            lumo_ev=lumo,
            n_imag=n_imag,
            lowest_freq_cm1=low_freq,
            style=style,
        )
        QtWidgets.QApplication.clipboard().setText(si_text)
        style_label = {"acs": "ACS/RSC Plain Text", "markdown": "Markdown", "latex": "LaTeX Table"}.get(style, style)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"Copied Supporting Information ({style_label}) to clipboard!", self)
        self.statusBar().showMessage(
            f"📋 Copied Supporting Information ({style_label}, {len(symbols)} atoms) to clipboard!", 4500
        )

    def _on_open_steric_descriptors_dialog(self):
        """Open the Catalysis & Organometallic Steric Descriptors Inspector (%V_bur, Sterimol, Tolman Cone Angle)."""
        if len(self.zatoms) == 0:
            QtWidgets.QMessageBox.information(self, "Steric Descriptors", "Please load or build a molecule first.")
            return

        from .dialog_steric_descriptors import StericDescriptorsDialog
        symbols = [a.symbol for a in self.zatoms]
        coords = np.array(self.coords, dtype=np.float64, copy=True)
        sel = self._get_ordered_selection()
        center_idx = sel[0] if len(sel) >= 1 else 0
        attached_idx = sel[1] if len(sel) >= 2 else (1 if len(symbols) > 1 and center_idx == 0 else 0)

        dlg = StericDescriptorsDialog(
            parent=self,
            symbols=symbols,
            coords=coords,
            center_idx=center_idx,
            attached_idx=attached_idx,
        )
        dlg.exec()

    def _on_open_ring_aromaticity_dialog(self):
        """Open the Ring Aromaticity (HOMA Index) & Planarity Inspector Dialog (Multiwfn / Krygowski style)."""
        from .controllers import RingController
        RingController(self).open_ring_aromaticity_dialog()

    def _on_open_pbc_dialog(self):
        """Open the Periodic Boundary Conditions (PBC) & Supercell Generator dialog."""
        from .dialog_pbc import PBCDialog
        from .forcefield import detect_connectivity, detect_bond_orders
        from .zmatrix import cartesian_to_zmatrix, norm_bond

        symbols = [a.symbol for a in self.zatoms]
        coords = np.array(self.coords, dtype=np.float64, copy=True)
        unit_cell = getattr(self, "unit_cell", None) or getattr(self.viewer, "unit_cell", None)

        dlg = PBCDialog(
            parent=self,
            symbols=symbols,
            coords=coords,
            unit_cell=unit_cell,
            viewer=self.viewer,
        )

        def _on_cell_changed(new_cell):
            self.unit_cell = new_cell
            if self.viewer:
                self.viewer.unit_cell = new_cell
                self.viewer.update()

        def _on_supercell_applied(new_symbols, new_coords, new_unit_cell):
            self._push_undo_state()
            self.unit_cell = new_unit_cell
            detected_bonds = detect_connectivity(new_symbols, new_coords, strict=False)
            detected_bo = detect_bond_orders(new_symbols, new_coords, detected_bonds)
            zatoms = cartesian_to_zmatrix(new_symbols, new_coords, user_bonds=detected_bonds, strict_bonds=True)
            tree = {norm_bond(a.ref1, i) for i, a in enumerate(zatoms) if a.ref1 is not None}
            self.cross_bonds = {norm_bond(u, v) for u, v in detected_bonds if norm_bond(u, v) not in tree}
            self.strict_topology = True
            self.set_molecule_from_zatoms(zatoms, exact_coords=new_coords)
            if self.viewer:
                self.viewer.unit_cell = new_unit_cell
                self.viewer.show_unit_cell = True
                self.viewer.bond_orders = {norm_bond(u, v): o for (u, v), o in detected_bo.items() if o != 1}
                self.viewer.auto_center()
                self.viewer.update()
            self.statusBar().showMessage(
                f"Generated {len(new_symbols)}-atom supercell (a={new_unit_cell.a:.2f}, b={new_unit_cell.b:.2f}, c={new_unit_cell.c:.2f} Å).",
                5000,
            )

        dlg.cellChanged.connect(_on_cell_changed)
        dlg.supercellApplied.connect(_on_supercell_applied)
        dlg.exec()

    def _on_flush_memory(self):
        """
        Flush clipboard / memory state (Ctrl+K).
        Clears stored fragments, last added elements and fragments.
        Subsequent double-clicks on empty canvas will open option 1:
        Periodic Table or Add Fragment/Molecule picker.
        """
        self.last_added_element = None
        self.stored_fragment = None
        self.last_added_fragment = None
        self.last_item_type = None
        self._update_viewer_drag_label()
        self.statusBar().showMessage("Flushed memory clipboard (Ctrl+K). Double-click canvas to choose new element/fragment.", 4000)

    def _get_ordered_selection(self) -> List[int]:
        """Return currently selected atom indices in selection order."""
        n = len(self.zatoms)
        sel = [i for i in getattr(self.viewer, "multi_selected", []) if 0 <= i < n]
        if not sel:
            prim = getattr(self.viewer, "selected_atom_idx", None)
            if prim is not None and 0 <= prim < n:
                sel = [prim]
        return sel

    def _apply_rigid_coords_transform(self, new_coords: np.ndarray, status_msg: str):
        """
        Apply a rigid 3D coordinate transformation (translation/rotation) to self.coords,
        update _rigid_frame so subsequent Z-matrix edits preserve this exact orientation,
        refresh the 3D viewport and measurements, and automatically refresh symmetry operations
        if symmetry was already detected or if the Symmetry tab is active.
        """
        from .alignment import capture_rigid_frame
        self.coords = np.asarray(new_coords, dtype=np.float64)
        self._rigid_frame = capture_rigid_frame(self.zatoms, self.coords)
        user_bonds = self._current_explicit_bonds()
        self.viewer.update_coords(self.coords, user_bonds=user_bonds)
        self.viewer.fit_camera()
        self.viewer._update_measurement_info()
        self.viewer.update()

        # If symmetry analysis is active or Symmetry tab is visible, refresh symmetry matrices in the new frame!
        if getattr(self, "_current_sym_res", None) is not None or (
            hasattr(self, "tabs") and self.tabs.currentIndex() == 4
        ):
            try:
                prev_sym = self._current_sym_res.symbol if getattr(self, "_current_sym_res", None) else None
                self._on_detect_symmetry(switch_tab=False, preferred_symbol=prev_sym)
            except Exception:
                log.warning("Could not refresh symmetry after alignment", exc_info=True)

        self.statusBar().showMessage(status_msg, 5000)

    def _on_recenter(self):
        """
        Center molecule to origin (0, 0, 0):
        - If 1 atom is selected: places that atom at (0, 0, 0).
        - If >1 atoms are selected: places the centroid of the selected atoms at (0, 0, 0).
        - If 0 atoms are selected: places the molecular Center of Mass (COM) at (0, 0, 0).
        """
        if len(self.zatoms) == 0:
            self.viewer.auto_center()
            return
        sel = self._get_ordered_selection()
        if len(sel) == 1:
            self._on_center_molecule_mode("atom")
        elif len(sel) > 1:
            self._push_undo_state()
            sub_centroid = np.mean(self.coords[sel], axis=0)
            new_coords = self.coords - sub_centroid
            self._apply_rigid_coords_transform(
                new_coords,
                f"Centered centroid of {len(sel)} selected atoms at origin (0, 0, 0).",
            )
        else:
            self._on_center_molecule_mode("com")

    def _on_center_molecule_mode(self, mode: str = "com"):
        """
        Center molecule at origin (0, 0, 0) using:
          - mode='com': Center of Mass (COM)
          - mode='centroid': Geometric Centroid
          - mode='atom': Currently selected atom
        """
        from .alignment import center_at_com, center_at_centroid, center_at_atom
        if len(self.zatoms) == 0:
            return
        symbols = [a.symbol for a in self.zatoms]
        if mode == "atom":
            sel = self._get_ordered_selection()
            if not sel:
                QtWidgets.QMessageBox.information(
                    self,
                    "Center Atom at Origin",
                    "Please click an atom in the 3D viewport or Z-matrix table first to place it at (0, 0, 0).",
                )
                return
            idx = sel[0]
            self._push_undo_state()
            new_coords, _ = center_at_atom(self.coords, idx)
            self._apply_rigid_coords_transform(
                new_coords,
                f"Centered atom {symbols[idx]}{idx+1} at origin (0.000, 0.000, 0.000).",
            )
        elif mode == "centroid":
            self._push_undo_state()
            new_coords, _ = center_at_centroid(self.coords)
            self._apply_rigid_coords_transform(
                new_coords,
                "Centered molecular geometric centroid at origin (0.000, 0.000, 0.000).",
            )
        else:
            self._push_undo_state()
            new_coords, _ = center_at_com(symbols, self.coords)
            self._apply_rigid_coords_transform(
                new_coords,
                "Centered molecular Center of Mass (COM) at origin (0.000, 0.000, 0.000).",
            )

    def _on_align_to_axis(self, target_axis: str = "+Z"):
        """Align selected 2-atom bond/vector, 1-atom position vector, or principal axis along target_axis."""
        from .alignment import align_atoms_to_axis
        if len(self.zatoms) == 0:
            return
        symbols = [a.symbol for a in self.zatoms]
        sel = self._get_ordered_selection()
        self._push_undo_state()
        rot_coords, _, desc = align_atoms_to_axis(symbols, self.coords, sel, target_axis=target_axis)
        self._apply_rigid_coords_transform(rot_coords, desc)

    def _on_align_to_plane(self, target_plane: str = "XZ"):
        """Align plane of 3 selected atoms (or whole molecule best-fit plane) into XY, XZ, or YZ."""
        from .alignment import align_plane_to_cartesian
        if len(self.zatoms) < 2:
            return
        symbols = [a.symbol for a in self.zatoms]
        sel = self._get_ordered_selection()
        self._push_undo_state()
        rot_coords, _, desc = align_plane_to_cartesian(
            symbols, self.coords, sel, target_plane=target_plane
        )
        self._apply_rigid_coords_transform(rot_coords, desc)

    def _on_open_structure_overlay(self):
        """Open Multi-Structure 3D Superposition & RMSD Alignment Dialog."""
        from .dialog_structure_overlay import StructureOverlayDialog
        if len(self.zatoms) == 0:
            QtWidgets.QMessageBox.information(self, "No Molecule", "Please open or build a molecule first.")
            return

        symbols = [a.symbol for a in self.zatoms]
        traj = getattr(self, "trajectory_coords", None)
        energies = getattr(self, "trajectory_energies", None)

        dlg = StructureOverlayDialog(
            parent=self,
            symbols=symbols,
            coords=self.coords,
            trajectory=traj,
            trajectory_energies=energies,
            viewer=self.viewer,
        )
        dlg.exec()

    def _on_open_reaction_path_dialog(self):
        """Open Reaction Path & Transition State (TS) Guess Interpolator Dialog (IDPP / QST2 / NEB-TS)."""
        from .dialog_reaction_path import ReactionPathDialog
        from .zmatrix import cartesian_to_zmatrix

        symbols = [a.symbol for a in self.zatoms] if self.zatoms else self.symbols
        coords = self.coords if len(self.coords) > 0 else None

        def _on_adopted(adopted_syms, adopted_coords):
            self._push_undo_state()
            curr_bonds = self._current_explicit_bonds() if hasattr(self, "_current_explicit_bonds") else None
            new_zatoms = cartesian_to_zmatrix(adopted_syms, adopted_coords, user_bonds=curr_bonds, strict_bonds=True)
            self.set_molecule_from_zatoms(new_zatoms, exact_coords=adopted_coords)
            self.statusBar().showMessage("🎯 Adopted IDPP Transition State (TS) Guess into 3D Editor", 4000)

        def _on_preview_traj(p_syms, p_frames):
            if hasattr(self, "trajectory_coords"):
                self.trajectory_coords = p_frames
                self.trajectory_energies = [0.0] * len(p_frames)
                if hasattr(self, "slider_trajectory"):
                    self.slider_trajectory.setEnabled(True)
                    self.slider_trajectory.setRange(0, len(p_frames) - 1)
                    self.slider_trajectory.setValue(0)
                if hasattr(self, "lbl_trajectory_step"):
                    self.lbl_trajectory_step.setText(f"1/{len(p_frames)}")
                if len(p_frames) > 0:
                    self.viewer.set_molecule(p_syms, p_frames[0])
            self.statusBar().showMessage(f"Loaded {len(p_frames)} reaction path frames into player", 4000)

        if hasattr(self, "dock_manager"):
            dock = self.dock_manager.open_reaction_path(
                initial_symbols=symbols,
                initial_coords=coords,
                on_adopted=_on_adopted,
                on_preview_traj=_on_preview_traj,
            )
        else:
            dlg = ReactionPathDialog(
                parent=self,
                initial_symbols=symbols,
                initial_coords=coords,
            )
            dlg.path_adopted.connect(_on_adopted)
            dlg.trajectory_preview.connect(_on_preview_traj)
            dlg.exec()

    def _on_open_crest_explorer(self, filepath: Optional[str] = None):
        """Open CREST & Conformer Ensemble Explorer Dialog (Ctrl+Shift+E)."""
        from .dialog_conformer_ensemble import ConformerEnsembleDialog

        symbols = [a.symbol for a in self.zatoms] if self.zatoms else self.symbols
        traj_coords = getattr(self, "trajectory_coords", None)
        traj_energies = getattr(self, "trajectory_energies", None)
        orig_coords = np.array(self.coords, copy=True) if len(self.coords) > 0 else None

        dlg = ConformerEnsembleDialog(
            parent=self,
            filepath=filepath,
            symbols=symbols if traj_coords else None,
            trajectory_coords=traj_coords,
            trajectory_energies=traj_energies,
        )

        def _on_preview(p_syms, p_coords):
            if hasattr(self, "viewer") and self.viewer and len(p_coords) > 0:
                self.viewer.update_coords(p_coords)

        dlg.conformer_preview.connect(_on_preview)
        res = dlg.exec()
        # If user closed or rejected dialog without adopting, restore original active coordinates
        if res != QtWidgets.QDialog.DialogCode.Accepted and orig_coords is not None and hasattr(self, "viewer") and self.viewer:
            self.viewer.update_coords(orig_coords)

    def _on_open_irc_sweeper(self, pathway: Optional[Any] = None):
        """Open Interactive IRC Reaction Sweeper & Energy Profile Dialog (Ctrl+Shift+I)."""
        from .dialog_irc_sweeper import IRCReactionSweeperDialog
        from .irc_sweeper import assemble_irc_pathway

        symbols = [a.symbol for a in self.zatoms] if self.zatoms else getattr(self, "symbols", [])
        traj_coords = getattr(self, "trajectory_coords", None)
        traj_energies = getattr(self, "trajectory_energies", None)
        qm_res = getattr(self, "_last_qm_res", None)

        if pathway is None:
            # Check if cached on last QM result
            pathway = getattr(qm_res, "irc_pathway", None)

        if pathway is None and traj_coords and len(traj_coords) > 1:
            # Build pathway from currently loaded trajectory
            e_ha = [e / 27.211386245988 for e in traj_energies] if traj_energies else [float(i) for i in range(len(traj_coords))]
            irc_pts = getattr(qm_res, "irc_path_points", None)
            pathway = assemble_irc_pathway(
                symbols=symbols,
                coords_list=traj_coords,
                energies_ha=e_ha,
                rxn_coords=irc_pts,
            )

        dlg = IRCReactionSweeperDialog(parent=self, viewer=self.viewer, pathway=pathway)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        self._irc_sweeper_dlg = dlg

    def _on_invert_chirality(self):
        if not self.zatoms:
            return
        self._push_undo_state()
        for a in self.zatoms:
            a.dihedral = -a.dihedral
        self.table_widget.load_zatoms(self.zatoms)
        self.update_geometry_live()
        self.statusBar().showMessage("Inverted all dihedrals (enantiomer/chirality flipped)", 3000)

    def _on_invert_stereocenter(self):
        """Invert the chiral stereocenter configuration at selected atom or active selection."""
        from .controllers import FragmentController
        FragmentController(self).invert_stereocenter()

    def _on_fragment_category_changed(self, category: str):
        """Filter the fragment combo box by the chosen category."""
        from .controllers import FragmentController
        FragmentController(self).on_fragment_category_changed(category)

    def _on_save_custom_fragment(self):
        """Save the selected subgraph (or entire molecule) to the persistent Custom User Library."""
        from .controllers import FragmentController
        FragmentController(self).save_custom_fragment()

    def _update_frozen_buttons_ui(self):
        froz = set(getattr(self, "frozen_atoms", set()) or set())
        n_f = len(froz)
        if hasattr(self, "btn_unfreeze_all"):
            self.btn_unfreeze_all.setText(f"🔥 Unfreeze All ({n_f} frozen)")
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.frozen_atoms = froz
            self.viewer.update()

    def _on_freeze_selected_atoms(self):
        """Pin/freeze currently selected atoms during geometry optimization."""
        sel = self._get_ordered_selection()
        if not sel:
            QtWidgets.QMessageBox.information(
                self,
                "Freeze Selected Atoms",
                "Select one or more atoms in the 3D viewport (Click or Shift+Click) to freeze their coordinates during optimization.",
            )
            return
        if not hasattr(self, "frozen_atoms"):
            self.frozen_atoms = set()
        self.frozen_atoms.update(sel)
        self._update_frozen_buttons_ui()
        labels = ", ".join(f"{self.zatoms[i].symbol}{i+1}" for i in sel)
        self.statusBar().showMessage(
            f"Frozen {len(self.frozen_atoms)} atom(s) ({labels}) for constrained geometry optimization.",
            4500,
        )

    def _on_unfreeze_all_atoms(self):
        """Clear all frozen atom constraints."""
        self.frozen_atoms = set()
        self._update_frozen_buttons_ui()
        self.statusBar().showMessage("Cleared all frozen atom constraints.", 3500)

    def _on_open_dihedral_scan_dialog(self):
        """Open interactive 1D Dihedral Rotor Conformer Scan dialog with embedded PES plotter."""
        from .controllers import FragmentController
        FragmentController(self).open_dihedral_scan_dialog()

    def _on_add_hydrogens(self):
        """Complete valence of selected atoms (or all atoms if none selected) by adding hydrogens geometrically."""
        if len(self.zatoms) == 0:
            QtWidgets.QMessageBox.information(
                self, "Add Hydrogens", "Please add at least one atom before completing valence."
            )
            return

        current_symbols = [a.symbol for a in self.zatoms]
        user_bonds = self._current_explicit_bonds()

        # Check if user has specific atoms selected (via 3D viewer or table widget)
        selected_indices = set()
        if getattr(self.viewer, "multi_selected", None):
            selected_indices.update(self.viewer.multi_selected)
        if getattr(self.viewer, "selected_atom_idx", None) is not None:
            selected_indices.add(self.viewer.selected_atom_idx)

        # Filter to valid indices
        target_indices = [i for i in sorted(list(selected_indices)) if 0 <= i < len(self.zatoms)]
        if len(target_indices) == 0:
            target_indices = None  # None indicates whole molecule

        old_bo = dict(getattr(self.viewer, "bond_orders", {}))
        new_symbols, new_coords, count_added = add_hydrogens_to_valence(
            current_symbols, self.coords, user_bonds=user_bonds, target_atom_indices=target_indices,
            user_bond_orders=old_bo
        )
        if count_added == 0:
            if target_indices:
                selected_labels = ", ".join(f"{current_symbols[i]}{i+1}" for i in target_indices)
                self.statusBar().showMessage(f"Selected atom(s) ({selected_labels}) already have their valence fully satisfied!", 3500)
            else:
                self.statusBar().showMessage("All atoms in the molecule already have their valence fully satisfied!", 3500)
            return

        self._push_undo_state()
        new_zatoms = cartesian_to_zmatrix(new_symbols, new_coords, user_bonds=user_bonds)
        self.set_molecule_from_zatoms(new_zatoms, exact_coords=new_coords)
        if old_bo:
            self.viewer.bond_orders = old_bo
            self.viewer.update()
        if target_indices:
            selected_labels = ", ".join(f"{current_symbols[i]}{i+1}" for i in target_indices)
            self.statusBar().showMessage(f"Added {count_added} hydrogen(s) to selected atom(s) ({selected_labels})!", 4000)
        else:
            self.statusBar().showMessage(f"Completed valence: added {count_added} hydrogen atoms to molecule!", 4000)

    def _on_open_cycle_welder(self):
        """Open Molecular Cycle Welder & Ring Fusion Studio (Ctrl+Shift+W)."""
        from .controllers import RingController
        RingController(self).open_cycle_welder()

    def _on_open_cycle_welder_for_bond(self, u: int, v: int):
        """Select bond (u, v) and open Cycle Welder Dialog."""
        from .controllers import RingController
        RingController(self).open_cycle_welder_for_bond(u, v)

