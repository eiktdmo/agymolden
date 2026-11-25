"""
Interactive 2D Molecular Sketcher Canvas & 3D Builder Dialog for AgyMolden.
Draws 2D bond-line structures and converts them directly to 3D Cartesian/Z-Matrix geometries.
Features interactive drawing, atom moving/re-positioning, intuitive functional group stamping,
live canonical SMILES preview, undo/redo (Ctrl+Z), and robust 3D ETKDG/UFF coordinate embedding.
"""

from __future__ import annotations

import copy
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

log = logging.getLogger(__name__)


@dataclass
class SketchAtom2D:
    x: float
    y: float
    symbol: str = "C"


@dataclass
class SketchBond2D:
    atom1: int
    atom2: int
    order: int = 1


# Predefined functional group templates (smiles, connection atom index)
FUNCTIONAL_GROUPS: Dict[str, Tuple[str, int]] = {
    "Ph (Phenyl)": ("c1ccccc1", 0),
    "Me (Methyl)": ("C", 0),
    "Et (Ethyl)": ("CC", 1),
    "tBu (tert-Butyl)": ("CC(C)C", 1),
    "NO2 (Nitro)": ("[N+](=O)[O-]", 0),
    "CF3 (Trifluoromethyl)": ("C(F)(F)F", 0),
    "COOH (Carboxylic Acid)": ("C(=O)O", 0),
    "CHO (Formyl / Aldehyde)": ("C=O", 0),
    "OMe (Methoxy)": ("OC", 0),
    "NH2 (Amino)": ("N", 0),
    "SO3H (Sulfonic Acid)": ("S(=O)(=O)O", 0),
}


class SketchCanvas2D(QtWidgets.QWidget):
    """Interactive 2D bond-line sketching canvas with undo/redo support, atom repositioning, and group stamping."""

    molecule_changed = QtCore.pyqtSignal()
    status_hint = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(540, 420)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.atoms: List[SketchAtom2D] = []
        self.bonds: List[SketchBond2D] = []
        self.undo_stack: List[Tuple[List[SketchAtom2D], List[SketchBond2D]]] = []
        self.redo_stack: List[Tuple[List[SketchAtom2D], List[SketchBond2D]]] = []
        self.current_tool = "draw"  # "draw", "move", "stamp", "weld_ring"
        self.current_element = "C"
        self.current_bond_order = 1
        self.active_stamp_group: Optional[str] = None
        self.weld_ring_sides: int = 6
        self.weld_ring_aromatic: bool = True
        self.moving_atom_idx: Optional[int] = None
        self.selected_atom_idx: Optional[int] = None
        self.drag_start_idx: Optional[int] = None
        self.current_mouse_pos: Optional[QtCore.QPointF] = None
        self.hovered_atom_idx: Optional[int] = None
        self.hovered_bond_idx: Optional[int] = None

    def push_undo_state(self):
        """Saves current state for undo."""
        st = (
            [SketchAtom2D(x=a.x, y=a.y, symbol=a.symbol) for a in self.atoms],
            [SketchBond2D(atom1=b.atom1, atom2=b.atom2, order=b.order) for b in self.bonds],
        )
        self.undo_stack.append(st)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        """Revert to previous canvas state (Ctrl+Z)."""
        if not self.undo_stack:
            return
        curr = (
            [SketchAtom2D(x=a.x, y=a.y, symbol=a.symbol) for a in self.atoms],
            [SketchBond2D(atom1=b.atom1, atom2=b.atom2, order=b.order) for b in self.bonds],
        )
        self.redo_stack.append(curr)
        prev_atoms, prev_bonds = self.undo_stack.pop()
        self.atoms = prev_atoms
        self.bonds = prev_bonds
        self.drag_start_idx = None
        self.moving_atom_idx = None
        self.selected_atom_idx = None
        self.hovered_atom_idx = None
        self.update()
        self.molecule_changed.emit()

    def redo(self):
        """Redo previously undone action (Ctrl+Y / Ctrl+Shift+Z)."""
        if not self.redo_stack:
            return
        curr = (
            [SketchAtom2D(x=a.x, y=a.y, symbol=a.symbol) for a in self.atoms],
            [SketchBond2D(atom1=b.atom1, atom2=b.atom2, order=b.order) for b in self.bonds],
        )
        self.undo_stack.append(curr)
        nxt_atoms, nxt_bonds = self.redo_stack.pop()
        self.atoms = nxt_atoms
        self.bonds = nxt_bonds
        self.drag_start_idx = None
        self.moving_atom_idx = None
        self.selected_atom_idx = None
        self.hovered_atom_idx = None
        self.update()
        self.molecule_changed.emit()

    def load_from_3d(
        self,
        symbols: List[str],
        coords: np.ndarray,
        bonds: List[Tuple[int, int]],
        bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
        strip_h: bool = True,
    ) -> bool:
        """Populate the 2D canvas from a 3D molecule without emitting molecule_changed."""
        if not symbols or len(coords) == 0:
            return False

        heavy_indices = [i for i, s in enumerate(symbols) if s != "H"] if strip_h else list(range(len(symbols)))
        if not heavy_indices:
            heavy_indices = list(range(len(symbols)))
        idx_map = {old_i: new_i for new_i, old_i in enumerate(heavy_indices)}
        sub_syms = [symbols[i] for i in heavy_indices]
        sub_coords = np.asarray(coords, dtype=np.float64)[heavy_indices]
        sub_bonds: List[Tuple[int, int, int]] = []
        bo_dict = bond_orders or {}
        seen_pairs = set()
        for u, v in bonds:
            if u in idx_map and v in idx_map:
                nu, nv = idx_map[u], idx_map[v]
                if nu != nv:
                    pair = (min(nu, nv), max(nu, nv))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        order = bo_dict.get((min(u, v), max(u, v)), 1)
                        sub_bonds.append((pair[0], pair[1], int(order)))

        coords_2d: Optional[np.ndarray] = None
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            rwmol = Chem.RWMol()
            for s in sub_syms:
                rwmol.AddAtom(Chem.Atom(s))
            bo_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
            for u, v, order in sub_bonds:
                rwmol.AddBond(u, v, bo_map.get(order, Chem.BondType.SINGLE))
            mol = rwmol.GetMol()
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            coords_2d = np.array([[conf.GetAtomPosition(i).x, -conf.GetAtomPosition(i).y] for i in range(len(sub_syms))])
        except Exception:
            coords_2d = None

        if coords_2d is None:
            centered = sub_coords - np.mean(sub_coords, axis=0)
            if len(sub_syms) >= 2:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                coords_2d = centered @ vt[:2].T
            else:
                coords_2d = np.zeros((len(sub_syms), 2), dtype=np.float64)
            coords_2d[:, 1] = -coords_2d[:, 1]

        cx = self.width() / 2.0 if self.width() > 40 else 220.0
        cy = self.height() / 2.0 if self.height() > 40 else 190.0
        if len(coords_2d) > 0:
            coords_2d = coords_2d - np.mean(coords_2d, axis=0)
            span = max(1e-6, float(np.max(np.abs(coords_2d))))
            scale = min(38.0, 140.0 / span) if len(sub_syms) > 1 else 38.0
        else:
            scale = 38.0

        self.push_undo_state()
        self.atoms = [
            SketchAtom2D(x=float(cx + coords_2d[i, 0] * scale), y=float(cy + coords_2d[i, 1] * scale), symbol=sub_syms[i])
            for i in range(len(sub_syms))
        ]
        self.bonds = [SketchBond2D(atom1=u, atom2=v, order=order) for u, v, order in sub_bonds]
        self.drag_start_idx = None
        self.moving_atom_idx = None
        self.selected_atom_idx = None
        self.update()
        return True

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        key = event.key()
        mod = event.modifiers()
        if (mod & QtCore.Qt.KeyboardModifier.ControlModifier):
            if key == QtCore.Qt.Key.Key_Z:
                if mod & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    self.redo()
                else:
                    self.undo()
                event.accept()
                return
            elif key == QtCore.Qt.Key.Key_Y:
                self.redo()
                event.accept()
                return
        if key in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            if self.hovered_atom_idx is not None:
                self.delete_atom(self.hovered_atom_idx)
                event.accept()
                return
        if key == QtCore.Qt.Key.Key_Escape and self.current_tool in ("weld_ring", "stamp"):
            self.current_tool = "draw"
            self.active_stamp_group = None
            self.hovered_bond_idx = None
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            self.status_hint.emit("Back to Draw mode.")
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def delete_atom(self, atom_idx: int):
        if 0 <= atom_idx < len(self.atoms):
            self.push_undo_state()
            self.atoms.pop(atom_idx)
            self.bonds = [
                SketchBond2D(
                    atom1=(b.atom1 if b.atom1 < atom_idx else b.atom1 - 1),
                    atom2=(b.atom2 if b.atom2 < atom_idx else b.atom2 - 1),
                    order=b.order,
                )
                for b in self.bonds
                if b.atom1 != atom_idx and b.atom2 != atom_idx
            ]
            self.hovered_atom_idx = None
            self.moving_atom_idx = None
            self.update()
            self.molecule_changed.emit()

    def clear_canvas(self):
        if self.atoms or self.bonds:
            self.push_undo_state()
        self.atoms.clear()
        self.bonds.clear()
        self.selected_atom_idx = None
        self.moving_atom_idx = None
        self.update()
        self.molecule_changed.emit()

    def add_fragment_ring(self, n_sides: int = 6, aromatic: bool = False):
        """Adds a regular n-sided ring centered on canvas or clicked atom."""
        self.push_undo_state()
        cx = self.width() / 2.0 if self.width() > 0 else 240.0
        cy = self.height() / 2.0 if self.height() > 0 else 190.0
        r = 50.0

        start_idx = len(self.atoms)
        for i in range(n_sides):
            angle = (2.0 * math.pi * i) / n_sides - math.pi / 2.0
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            self.atoms.append(SketchAtom2D(x=x, y=y, symbol="C"))

        for i in range(n_sides):
            a1 = start_idx + i
            a2 = start_idx + ((i + 1) % n_sides)
            b_order = 2 if (aromatic and n_sides == 6 and i % 2 == 0) else 1
            self.bonds.append(SketchBond2D(atom1=a1, atom2=a2, order=b_order))

        self.update()
        self.molecule_changed.emit()

    def add_functional_group(self, group_key: str, target_atom_idx: Optional[int] = None, screen_pos: Optional[QtCore.QPointF] = None):
        """Attaches a predefined functional group to target_atom_idx or stamps at screen_pos."""
        if group_key not in FUNCTIONAL_GROUPS:
            return
        smi, attach_pt = FUNCTIONAL_GROUPS[group_key]

        frag_syms: List[str] = []
        frag_pos: List[Tuple[float, float]] = []
        frag_bonds: List[Tuple[int, int, int]] = []

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except Exception:
                    logging.getLogger(__name__).debug("Ignored exception", exc_info=True)
                AllChem.Compute2DCoords(mol)
                conf = mol.GetConformer()
                frag_syms = [a.GetSymbol() for a in mol.GetAtoms()]
                frag_pos = [(conf.GetAtomPosition(i).x * 42.0, -conf.GetAtomPosition(i).y * 42.0) for i in range(mol.GetNumAtoms())]
                bo_map = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 1}
                for b in mol.GetBonds():
                    frag_bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bo_map.get(b.GetBondType(), 1)))
        except Exception:
            logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

        if not frag_syms:
            if group_key.startswith("Ph"):
                frag_syms = ["C"] * 6
                frag_pos = [(42.0 * math.cos(2 * math.pi * i / 6), 42.0 * math.sin(2 * math.pi * i / 6)) for i in range(6)]
                frag_bonds = [(i, (i + 1) % 6, 2 if i % 2 == 0 else 1) for i in range(6)]
            elif group_key.startswith("Et"):
                frag_syms = ["C", "C"]
                frag_pos = [(42.0, 0.0), (0.0, 0.0)]
                frag_bonds = [(0, 1, 1)]
            elif group_key.startswith("tBu"):
                frag_syms = ["C", "C", "C", "C"]
                frag_pos = [(42.0, 0.0), (0.0, 0.0), (0.0, 42.0), (0.0, -42.0)]
                frag_bonds = [(1, 0, 1), (1, 2, 1), (1, 3, 1)]
            elif group_key.startswith("NO2"):
                frag_syms = ["N", "O", "O"]
                frag_pos = [(0.0, 0.0), (36.0, 24.0), (36.0, -24.0)]
                frag_bonds = [(0, 1, 2), (0, 2, 1)]
            elif group_key.startswith("CF3"):
                frag_syms = ["C", "F", "F", "F"]
                frag_pos = [(0.0, 0.0), (42.0, 0.0), (0.0, 42.0), (0.0, -42.0)]
                frag_bonds = [(0, 1, 1), (0, 2, 1), (0, 3, 1)]
            elif group_key.startswith("COOH"):
                frag_syms = ["C", "O", "O"]
                frag_pos = [(0.0, 0.0), (36.0, 24.0), (36.0, -24.0)]
                frag_bonds = [(0, 1, 2), (0, 2, 1)]
            elif group_key.startswith("CHO"):
                frag_syms = ["C", "O"]
                frag_pos = [(0.0, 0.0), (40.0, 0.0)]
                frag_bonds = [(0, 1, 2)]
            elif group_key.startswith("OMe"):
                frag_syms = ["O", "C"]
                frag_pos = [(0.0, 0.0), (40.0, 0.0)]
                frag_bonds = [(0, 1, 1)]
            elif group_key.startswith("NH2"):
                frag_syms = ["N"]
                frag_pos = [(0.0, 0.0)]
            elif group_key.startswith("SO3H"):
                frag_syms = ["S", "O", "O", "O"]
                frag_pos = [(0.0, 0.0), (40.0, 0.0), (0.0, 40.0), (0.0, -40.0)]
                frag_bonds = [(0, 1, 2), (0, 2, 2), (0, 3, 1)]
            else:
                frag_syms = ["C"]
                frag_pos = [(0.0, 0.0)]

        self.push_undo_state()
        start_idx = len(self.atoms)

        if target_atom_idx is not None and 0 <= target_atom_idx < len(self.atoms):
            anchor = self.atoms[target_atom_idx]
            connected = [b.atom2 if b.atom1 == target_atom_idx else b.atom1 for b in self.bonds if b.atom1 == target_atom_idx or b.atom2 == target_atom_idx]
            if connected:
                avg_nx = sum(self.atoms[c].x for c in connected) / len(connected)
                avg_ny = sum(self.atoms[c].y for c in connected) / len(connected)
                dx = anchor.x - avg_nx
                dy = anchor.y - avg_ny
                d_len = math.hypot(dx, dy) or 1.0
                offset_x = anchor.x + (dx / d_len) * 55.0
                offset_y = anchor.y + (dy / d_len) * 55.0
            else:
                offset_x = anchor.x + 55.0
                offset_y = anchor.y
        elif screen_pos is not None:
            offset_x = screen_pos.x()
            offset_y = screen_pos.y()
        else:
            offset_x = self.width() / 2.0 if self.width() > 0 else 240.0
            offset_y = self.height() / 2.0 if self.height() > 0 else 190.0

        att_x, att_y = frag_pos[attach_pt] if attach_pt < len(frag_pos) else (0.0, 0.0)

        for sym, (px, py) in zip(frag_syms, frag_pos):
            self.atoms.append(SketchAtom2D(x=offset_x + (px - att_x), y=offset_y + (py - att_y), symbol=sym))

        for u, v, o in frag_bonds:
            self.bonds.append(SketchBond2D(atom1=start_idx + u, atom2=start_idx + v, order=o))

        if target_atom_idx is not None and 0 <= target_atom_idx < len(self.atoms):
            self.bonds.append(SketchBond2D(atom1=target_atom_idx, atom2=start_idx + attach_pt, order=1))

        self.update()
        self.molecule_changed.emit()

    def get_canonical_smiles(self) -> str:
        """Returns the canonical SMILES string for the currently drawn canvas topology."""
        if not self.atoms:
            return ""
        try:
            from rdkit import Chem
            rwmol = Chem.RWMol()
            for a in self.atoms:
                rwmol.AddAtom(Chem.Atom(a.symbol))
            bo_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
            for b in self.bonds:
                if b.atom1 < len(self.atoms) and b.atom2 < len(self.atoms):
                    rwmol.AddBond(b.atom1, b.atom2, bo_map.get(b.order, Chem.BondType.SINGLE))
            mol = rwmol.GetMol()
            try:
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol)
            except Exception:
                return Chem.MolToSmiles(mol, kekuleSmiles=True)
        except Exception:
            counts = {}
            for a in self.atoms:
                counts[a.symbol] = counts.get(a.symbol, 0) + 1
            parts = []
            for el in ["C", "H"]:
                if el in counts:
                    parts.append(f"{el}{counts.pop(el)}")
            for el in sorted(counts.keys()):
                parts.append(f"{el}{counts[el]}")
            return "".join(parts)

    def _find_atom_near(self, pos: QtCore.QPointF, tol: float = 18.0) -> Optional[int]:
        for i, at in enumerate(self.atoms):
            dist = math.hypot(at.x - pos.x(), at.y - pos.y())
            if dist <= tol:
                return i
        return None

    def _find_bond_near(self, pos: QtCore.QPointF, tol: float = 12.0) -> Optional[int]:
        best_idx: Optional[int] = None
        best_dist = tol
        px, py = pos.x(), pos.y()
        for i, b in enumerate(self.bonds):
            if b.atom1 >= len(self.atoms) or b.atom2 >= len(self.atoms):
                continue
            a1, a2 = self.atoms[b.atom1], self.atoms[b.atom2]
            dx, dy = a2.x - a1.x, a2.y - a1.y
            len_sq = dx * dx + dy * dy
            if len_sq < 1e-6:
                continue
            t = ((px - a1.x) * dx + (py - a1.y) * dy) / len_sq
            if t < 0.16 or t > 0.84:
                continue
            proj_x = a1.x + t * dx
            proj_y = a1.y + t * dy
            dist = math.hypot(px - proj_x, py - proj_y)
            if dist <= best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _atom_bond_order_sum(self, atom_idx: int) -> int:
        return sum(b.order for b in self.bonds if b.atom1 == atom_idx or b.atom2 == atom_idx)

    @staticmethod
    def _atom_max_valence(symbol: str) -> int:
        return {
            "C": 4, "Si": 4, "Ge": 4,
            "N": 3, "P": 3, "As": 3, "B": 3,
            "O": 2, "S": 2, "Se": 2,
            "H": 1, "F": 1, "Cl": 1, "Br": 1, "I": 1,
        }.get(symbol, 4)

    def _has_bond_between(self, u_i: int, v_i: int) -> bool:
        return any((b.atom1 == u_i and b.atom2 == v_i) or (b.atom1 == v_i and b.atom2 == u_i) for b in self.bonds)

    def _validate_weld_topology(
        self,
        existing: List[Optional[int]],
        attach_atom_idx: Optional[int],
        require_new_topology: bool = False,
    ) -> bool:
        """Check valence on existing atoms using actual new bonds added after vertex merging."""
        added_deg: Dict[int, int] = {}
        n = len(existing)
        new_atoms = sum(1 for x in existing if x is None)
        new_bonds = 0
        for i in range(n):
            a = existing[i]
            b = existing[(i + 1) % n]
            if a is not None and b is not None:
                if a == b or self._has_bond_between(a, b):
                    continue
            new_bonds += 1
            if a is not None:
                added_deg[a] = added_deg.get(a, 0) + 1
            if b is not None:
                added_deg[b] = added_deg.get(b, 0) + 1
        if attach_atom_idx is not None:
            new_bonds += 1
            added_deg[attach_atom_idx] = added_deg.get(attach_atom_idx, 0) + 1
            if existing[0] is not None:
                added_deg[existing[0]] = added_deg.get(existing[0], 0) + 1
        if require_new_topology and new_atoms == 0 and new_bonds == 0:
            return False
        for ex_idx, deg in added_deg.items():
            if self._atom_bond_order_sum(ex_idx) + deg > self._atom_max_valence(self.atoms[ex_idx].symbol):
                return False
        return True

    def _pick_weld_target(self, pos: QtCore.QPointF) -> Tuple[Optional[int], Optional[int]]:
        """Apothem-matched magnetic snap: snap to valid bonds when floating ring edge approaches, or directly over an atom."""
        hit_atom = self._find_atom_near(pos, tol=13.0)
        if hit_atom is not None:
            return None, hit_atom

        r_est = max(32.0, min(56.0, 50.0 * self.weld_ring_sides / 6.0))
        bond_snap_tol = max(46.0, r_est + 12.0)

        px, py = pos.x(), pos.y()
        candidates: List[Tuple[float, int]] = []
        for i, b in enumerate(self.bonds):
            if b.atom1 >= len(self.atoms) or b.atom2 >= len(self.atoms):
                continue
            a1, a2 = self.atoms[b.atom1], self.atoms[b.atom2]
            dx, dy = a2.x - a1.x, a2.y - a1.y
            len_sq = dx * dx + dy * dy
            if len_sq < 1e-6:
                continue
            t = max(0.0, min(1.0, ((px - a1.x) * dx + (py - a1.y) * dy) / len_sq))
            proj_x = a1.x + t * dx
            proj_y = a1.y + t * dy
            dist = math.hypot(px - proj_x, py - proj_y)
            if dist <= bond_snap_tol:
                candidates.append((dist, i))

        if not candidates:
            return None, None

        candidates.sort(key=lambda item: item[0])
        for dist, bi in candidates:
            _, _, _, is_valid = self._compute_weld_ring_vertices(
                self.weld_ring_sides,
                target_bond_idx=bi,
                screen_pos=pos,
                aromatic=self.weld_ring_aromatic,
            )
            if is_valid:
                return bi, None

        # Only snap to an invalid (full-valence) bond when cursor is right on top of it
        if candidates[0][0] <= 12.0:
            return candidates[0][1], None
        return None, None

    def _compute_weld_ring_vertices(
        self,
        n_sides: int,
        target_bond_idx: Optional[int] = None,
        target_atom_idx: Optional[int] = None,
        screen_pos: Optional[QtCore.QPointF] = None,
        aromatic: bool = False,
    ) -> Tuple[List[Tuple[float, float]], List[Optional[int]], Optional[int], bool]:
        """Compute 2D polygon vertices, atom merges, exocyclic anchor, and valence validity."""
        n_sides = max(3, int(n_sides))

        if target_bond_idx is not None and 0 <= target_bond_idx < len(self.bonds):
            b = self.bonds[target_bond_idx]
            u, v = b.atom1, b.atom2
            p1 = (self.atoms[u].x, self.atoms[u].y)
            p2 = (self.atoms[v].x, self.atoms[v].y)
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            L = math.hypot(dx, dy)
            if L < 1e-4:
                return [], [], None, False
            mx, my = 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])
            nx, ny = -dy / L, dx / L

            nbrs = [
                (bnd.atom2 if bnd.atom1 == idx else bnd.atom1)
                for idx in (u, v)
                for bnd in self.bonds
                if (bnd.atom1 == idx or bnd.atom2 == idx) and {bnd.atom1, bnd.atom2} != {u, v}
            ]
            dot_nbr = 0.0
            if nbrs:
                cx_nbr = sum(self.atoms[k].x for k in nbrs) / len(nbrs)
                cy_nbr = sum(self.atoms[k].y for k in nbrs) / len(nbrs)
                dot_nbr = (cx_nbr - mx) * nx + (cy_nbr - my) * ny
            if abs(dot_nbr) >= 4.0:
                if dot_nbr > 0:
                    nx, ny = -nx, -ny
            elif screen_pos is not None:
                if (screen_pos.x() - mx) * nx + (screen_pos.y() - my) * ny < 0:
                    nx, ny = -nx, -ny

            apothem = L / (2.0 * math.tan(math.pi / n_sides))
            R = L / (2.0 * math.sin(math.pi / n_sides))
            cx, cy = mx + nx * apothem, my + ny * apothem

            ang_u = math.atan2(p1[1] - cy, p1[0] - cx)
            ang_v = math.atan2(p2[1] - cy, p2[0] - cx)
            delta = (ang_v - ang_u) % (2.0 * math.pi)
            step = (-2.0 * math.pi / n_sides) if delta > math.pi else (2.0 * math.pi / n_sides)

            pts: List[Tuple[float, float]] = [p1, p2]
            existing: List[Optional[int]] = [u, v]
            for k in range(2, n_sides):
                ang = ang_u + k * step
                px, py = cx + R * math.cos(ang), cy + R * math.sin(ang)
                merged: Optional[int] = None
                for idx_ex, at in enumerate(self.atoms):
                    if idx_ex not in (u, v) and math.hypot(at.x - px, at.y - py) <= 16.0:
                        merged = idx_ex
                        px, py = at.x, at.y
                        break
                pts.append((px, py))
                existing.append(merged)
            is_valid = self._validate_weld_topology(existing, None, require_new_topology=True)
            return pts, existing, None, is_valid

        R = max(32.0, min(56.0, 50.0 * n_sides / 6.0))
        if target_atom_idx is not None and 0 <= target_atom_idx < len(self.atoms):
            val = self._atom_bond_order_sum(target_atom_idx)
            max_val = self._atom_max_valence(self.atoms[target_atom_idx].symbol)
            if val >= max_val:
                return [], [], None, False

            ax, ay = self.atoms[target_atom_idx].x, self.atoms[target_atom_idx].y
            nbrs = [
                (bnd.atom2 if bnd.atom1 == target_atom_idx else bnd.atom1)
                for bnd in self.bonds
                if bnd.atom1 == target_atom_idx or bnd.atom2 == target_atom_idx
            ]
            if nbrs:
                cx_nbr = sum(self.atoms[k].x for k in nbrs) / len(nbrs)
                cy_nbr = sum(self.atoms[k].y for k in nbrs) / len(nbrs)
                ox, oy = ax - cx_nbr, ay - cy_nbr
                olen = math.hypot(ox, oy) or 1.0
                ux, uy = ox / olen, oy / olen
            else:
                ux, uy = 1.0, 0.0

            if val + 2 <= max_val and not aromatic:
                # Spiro fusion on low-valence atom (adds 2 single bonds)
                cx, cy = ax + R * ux, ay + R * uy
                ang0 = math.atan2(ay - cy, ax - cx)
                pts = [(ax, ay)]
                existing: List[Optional[int]] = [target_atom_idx]
                for k in range(1, n_sides):
                    ang = ang0 + k * (2.0 * math.pi / n_sides)
                    px, py = cx + R * math.cos(ang), cy + R * math.sin(ang)
                    merged = None
                    for idx_ex, at in enumerate(self.atoms):
                        if idx_ex != target_atom_idx and math.hypot(at.x - px, at.y - py) <= 16.0:
                            merged = idx_ex
                            px, py = at.x, at.y
                            break
                    pts.append((px, py))
                    existing.append(merged)
                is_valid = self._validate_weld_topology(existing, None, require_new_topology=True)
                return pts, existing, None, is_valid

            # Exocyclic single-bond ring attachment (e.g., Biphenyl / Phenylcyclopentane)
            v0x, v0y = ax + 46.0 * ux, ay + 46.0 * uy
            cx, cy = v0x + R * ux, v0y + R * uy
            ang0 = math.atan2(v0y - cy, v0x - cx)
            pts = [(cx + R * math.cos(ang0 + k * (2.0 * math.pi / n_sides)), cy + R * math.sin(ang0 + k * (2.0 * math.pi / n_sides))) for k in range(n_sides)]
            existing = [None] * n_sides
            is_valid = self._validate_weld_topology(existing, target_atom_idx)
            return pts, existing, target_atom_idx, is_valid

        cx = screen_pos.x() if screen_pos is not None else (self.width() / 2.0 if self.width() > 0 else 240.0)
        cy = screen_pos.y() if screen_pos is not None else (self.height() / 2.0 if self.height() > 0 else 190.0)
        pts = [
            (
                cx + R * math.cos(2.0 * math.pi * k / n_sides - math.pi / 2.0),
                cy + R * math.sin(2.0 * math.pi * k / n_sides - math.pi / 2.0),
            )
            for k in range(n_sides)
        ]
        return pts, [None] * n_sides, None, True

    def weld_ring(
        self,
        n_sides: int = 6,
        aromatic: bool = False,
        target_bond_idx: Optional[int] = None,
        target_atom_idx: Optional[int] = None,
        screen_pos: Optional[QtCore.QPointF] = None,
    ) -> bool:
        """Weld a regular n-sided ring onto a bond (fused), atom (spiro/exocyclic), or free canvas."""
        pts, existing, attach_atom_idx, is_valid = self._compute_weld_ring_vertices(
            n_sides,
            target_bond_idx=target_bond_idx,
            target_atom_idx=target_atom_idx,
            screen_pos=screen_pos,
            aromatic=aromatic,
        )
        if not is_valid or len(pts) < 3:
            self.status_hint.emit("Cannot weld here: target atom already has full valence (4).")
            return False

        self.push_undo_state()
        ring_indices: List[int] = []
        for (px, py), ex_idx in zip(pts, existing):
            if ex_idx is not None:
                ring_indices.append(ex_idx)
            else:
                ring_indices.append(len(self.atoms))
                self.atoms.append(SketchAtom2D(x=px, y=py, symbol="C"))

        def _find_bond(u_i: int, v_i: int) -> Optional[SketchBond2D]:
            for bnd in self.bonds:
                if (bnd.atom1 == u_i and bnd.atom2 == v_i) or (bnd.atom1 == v_i and bnd.atom2 == u_i):
                    return bnd
            return None

        def _has_double_bond(idx_a: int) -> bool:
            return any(bnd.order == 2 and (bnd.atom1 == idx_a or bnd.atom2 == idx_a) for bnd in self.bonds)

        start_double_phase = 0
        if aromatic and n_sides == 6 and target_bond_idx is not None and 0 <= target_bond_idx < len(self.bonds):
            shared_b = self.bonds[target_bond_idx]
            if shared_b.order == 2 or _has_double_bond(ring_indices[0]) or _has_double_bond(ring_indices[1]):
                start_double_phase = 0
            else:
                start_double_phase = 1

        for i in range(len(ring_indices)):
            u_i = ring_indices[i]
            v_i = ring_indices[(i + 1) % len(ring_indices)]
            if u_i == v_i or _find_bond(u_i, v_i) is not None:
                continue
            b_order = 1
            if aromatic and n_sides == 6 and (i % 2 == start_double_phase):
                if not _has_double_bond(u_i) and not _has_double_bond(v_i):
                    b_order = 2
            self.bonds.append(SketchBond2D(atom1=min(u_i, v_i), atom2=max(u_i, v_i), order=b_order))

        if attach_atom_idx is not None and 0 <= attach_atom_idx < len(self.atoms):
            u_a, v_a = attach_atom_idx, ring_indices[0]
            if u_a != v_a and _find_bond(u_a, v_a) is None:
                self.bonds.append(SketchBond2D(atom1=min(u_a, v_a), atom2=max(u_a, v_a), order=1))

        self.update()
        self.molecule_changed.emit()
        return True

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.setFocus()
        pos = event.position()

        # 0. Ring Welder Mode (one-shot by default; hold Shift to chain multiple welds)
        if self.current_tool == "weld_ring":
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                hit_bond, hit_atom = self._pick_weld_target(pos)
                ok = self.weld_ring(
                    self.weld_ring_sides,
                    self.weld_ring_aromatic,
                    target_bond_idx=hit_bond,
                    target_atom_idx=hit_atom,
                    screen_pos=pos,
                )
                if ok and not (event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                    self.current_tool = "draw"
                    self.hovered_bond_idx = None
                    self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                    self.status_hint.emit("Welded ring! Back to Draw mode.")
                    self.update()
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                self.current_tool = "draw"
                self.hovered_bond_idx = None
                self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                self.status_hint.emit("Exited Ring Welder mode.")
                self.update()
            return

        hit_atom = self._find_atom_near(pos)

        # 1. Stamping Mode
        if self.current_tool == "stamp" and self.active_stamp_group:
            self.add_functional_group(self.active_stamp_group, target_atom_idx=hit_atom, screen_pos=pos)
            self.current_tool = "draw"
            self.active_stamp_group = None
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            self.status_hint.emit("Stamped group successfully! Back to Draw mode.")
            return

        # 2. Move / Reposition Atom Mode
        if self.current_tool == "move" or (event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
            if hit_atom is not None:
                self.push_undo_state()
                self.moving_atom_idx = hit_atom
                self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
            return

        # 3. Draw Mode (Default)
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if hit_atom is not None:
                self.drag_start_idx = hit_atom
            else:
                # Add new atom
                self.push_undo_state()
                new_idx = len(self.atoms)
                self.atoms.append(SketchAtom2D(x=pos.x(), y=pos.y(), symbol=self.current_element))
                self.drag_start_idx = new_idx
                self.molecule_changed.emit()
            self.update()

        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            if hit_atom is not None:
                self.delete_atom(hit_atom)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        self.current_mouse_pos = event.position()
        if self.current_tool == "weld_ring":
            self.hovered_bond_idx, self.hovered_atom_idx = self._pick_weld_target(event.position())
        else:
            self.hovered_atom_idx = self._find_atom_near(event.position())
            self.hovered_bond_idx = None

        # If currently dragging/repositioning an atom:
        if self.moving_atom_idx is not None and 0 <= self.moving_atom_idx < len(self.atoms):
            self.atoms[self.moving_atom_idx].x = event.position().x()
            self.atoms[self.moving_atom_idx].y = event.position().y()
            self.update()
            self.molecule_changed.emit()
            return

        # Cursor feedback
        if self.current_tool in ("stamp", "weld_ring"):
            self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        elif self.current_tool == "move" or (event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
            if self.hovered_atom_idx is not None:
                self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        # Finish moving atom
        if self.moving_atom_idx is not None:
            self.moving_atom_idx = None
            if self.current_tool == "move":
                self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            self.update()
            self.molecule_changed.emit()
            return

        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.drag_start_idx is not None:
            pos = event.position()
            hit_atom = self._find_atom_near(pos)
            start_i = self.drag_start_idx

            if hit_atom is not None and hit_atom != start_i:
                # Connected to another atom: Add or cycle bond
                self.push_undo_state()
                existing = next(
                    (b for b in self.bonds if (b.atom1 == start_i and b.atom2 == hit_atom) or (b.atom1 == hit_atom and b.atom2 == start_i)),
                    None
                )
                if existing:
                    existing.order = (existing.order % 3) + 1
                else:
                    self.bonds.append(SketchBond2D(atom1=min(start_i, hit_atom), atom2=max(start_i, hit_atom), order=self.current_bond_order))
                self.molecule_changed.emit()
            elif hit_atom is None:
                # Dragged to empty space: spawn atom connected to start
                dist = math.hypot(pos.x() - self.atoms[start_i].x, pos.y() - self.atoms[start_i].y)
                if dist > 14.0:
                    self.push_undo_state()
                    new_idx = len(self.atoms)
                    self.atoms.append(SketchAtom2D(x=pos.x(), y=pos.y(), symbol=self.current_element))
                    self.bonds.append(SketchBond2D(atom1=min(start_i, new_idx), atom2=max(start_i, new_idx), order=self.current_bond_order))
                    self.molecule_changed.emit()

            self.drag_start_idx = None
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QtGui.QColor("#1e293b"))

        # Draw Bonds
        pen_single = QtGui.QPen(QtGui.QColor("#e2e8f0"), 2.4)
        pen_double = QtGui.QPen(QtGui.QColor("#38bdf8"), 2.4)
        pen_triple = QtGui.QPen(QtGui.QColor("#f43f5e"), 2.4)

        for b in self.bonds:
            if b.atom1 >= len(self.atoms) or b.atom2 >= len(self.atoms):
                continue
            a1 = self.atoms[b.atom1]
            a2 = self.atoms[b.atom2]
            p1 = QtCore.QPointF(a1.x, a1.y)
            p2 = QtCore.QPointF(a2.x, a2.y)

            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            length = math.hypot(dx, dy)
            if length < 1e-4:
                continue

            ux = -dy / length
            uy = dx / length

            if b.order == 1:
                painter.setPen(pen_single)
                painter.drawLine(p1, p2)
            elif b.order == 2:
                painter.setPen(pen_double)
                offset = 3.2
                p1a = QtCore.QPointF(p1.x() + ux * offset, p1.y() + uy * offset)
                p2a = QtCore.QPointF(p2.x() + ux * offset, p2.y() + uy * offset)
                p1b = QtCore.QPointF(p1.x() - ux * offset, p1.y() - uy * offset)
                p2b = QtCore.QPointF(p2.x() - ux * offset, p2.y() - uy * offset)
                painter.drawLine(p1a, p2a)
                painter.drawLine(p1b, p2b)
            elif b.order == 3:
                painter.setPen(pen_triple)
                offset = 4.2
                painter.drawLine(p1, p2)
                p1a = QtCore.QPointF(p1.x() + ux * offset, p1.y() + uy * offset)
                p2a = QtCore.QPointF(p2.x() + ux * offset, p2.y() + uy * offset)
                p1b = QtCore.QPointF(p1.x() - ux * offset, p1.y() - uy * offset)
                p2b = QtCore.QPointF(p2.x() - ux * offset, p2.y() - uy * offset)
                painter.drawLine(p1a, p2a)
                painter.drawLine(p1b, p2b)

        # Highlight hovered target bond/atom in Ring Welder mode & draw clear fuse/attach indicator
        if self.current_tool == "weld_ring" and self.current_mouse_pos is not None:
            is_bond_snap = self.hovered_bond_idx is not None and 0 <= self.hovered_bond_idx < len(self.bonds)
            is_atom_snap = (not is_bond_snap) and (self.hovered_atom_idx is not None and 0 <= self.hovered_atom_idx < len(self.atoms))
            preview_pts, preview_existing, attach_idx, is_valid = self._compute_weld_ring_vertices(
                self.weld_ring_sides,
                target_bond_idx=self.hovered_bond_idx if is_bond_snap else None,
                target_atom_idx=self.hovered_atom_idx if is_atom_snap else None,
                screen_pos=self.current_mouse_pos,
                aromatic=self.weld_ring_aromatic,
            )

            if is_bond_snap:
                hb = self.bonds[self.hovered_bond_idx]
                if hb.atom1 < len(self.atoms) and hb.atom2 < len(self.atoms):
                    p_u = QtCore.QPointF(self.atoms[hb.atom1].x, self.atoms[hb.atom1].y)
                    p_v = QtCore.QPointF(self.atoms[hb.atom2].x, self.atoms[hb.atom2].y)
                    bond_col = QtGui.QColor("#f59e0b" if is_valid else "#ef4444")
                    painter.setPen(QtGui.QPen(bond_col, 6.5, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
                    painter.drawLine(p_u, p_v)
                    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    painter.setPen(QtGui.QPen(bond_col, 2.6))
                    painter.drawEllipse(p_u, 17.0, 17.0)
                    painter.drawEllipse(p_v, 17.0, 17.0)
                    for ex_m in preview_existing[2:]:
                        if ex_m is not None and 0 <= ex_m < len(self.atoms):
                            painter.setPen(QtGui.QPen(QtGui.QColor("#10b981"), 2.5))
                            painter.drawEllipse(QtCore.QPointF(self.atoms[ex_m].x, self.atoms[ex_m].y), 17.0, 17.0)
            elif is_atom_snap:
                ha = self.atoms[self.hovered_atom_idx]
                atom_col = QtGui.QColor("#06b6d4" if is_valid else "#ef4444")
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.setPen(QtGui.QPen(atom_col, 2.8))
                painter.drawEllipse(QtCore.QPointF(ha.x, ha.y), 17.0, 17.0)

            if len(preview_pts) >= 3:
                if not is_valid:
                    stroke_col = QtGui.QColor("#ef4444")
                    fill_col = QtGui.QColor(239, 68, 68, 45)
                    badge_txt = "✖ FULL"
                elif is_bond_snap:
                    stroke_col = QtGui.QColor("#22c55e")
                    fill_col = QtGui.QColor(34, 197, 94, 55)
                    has_peri = any(x is not None for x in preview_existing[2:])
                    badge_txt = "🔗 PERI-FUSE" if has_peri else "🔗 FUSE"
                elif is_atom_snap:
                    stroke_col = QtGui.QColor("#06b6d4")
                    fill_col = QtGui.QColor(6, 182, 212, 50)
                    badge_txt = "— ATTACH" if attach_idx is not None else "⚛ SPIRO"
                else:
                    stroke_col = QtGui.QColor("#38bdf8")
                    fill_col = QtGui.QColor(56, 189, 248, 35)
                    badge_txt = "PLACE"

                poly = QtGui.QPolygonF([QtCore.QPointF(px, py) for px, py in preview_pts])
                painter.setPen(QtGui.QPen(stroke_col, 2.5, QtCore.Qt.PenStyle.SolidLine if (is_bond_snap or is_atom_snap) else QtCore.Qt.PenStyle.DashLine))
                painter.setBrush(QtGui.QBrush(fill_col))
                painter.drawPolygon(poly)

                if attach_idx is not None and 0 <= attach_idx < len(self.atoms):
                    painter.drawLine(
                        QtCore.QPointF(self.atoms[attach_idx].x, self.atoms[attach_idx].y),
                        QtCore.QPointF(preview_pts[0][0], preview_pts[0][1]),
                    )

                for (px, py), ex_idx in zip(preview_pts, preview_existing):
                    if ex_idx is None:
                        painter.setBrush(QtGui.QBrush(stroke_col))
                        painter.setPen(QtCore.Qt.PenStyle.NoPen)
                        painter.drawEllipse(QtCore.QPointF(px, py), 4.0, 4.0)

                cx_p = sum(px for px, _ in preview_pts) / len(preview_pts)
                cy_p = sum(py for _, py in preview_pts) / len(preview_pts)
                badge_rect = QtCore.QRectF(cx_p - 42.0, cy_p - 10.0, 84.0, 20.0)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(15, 23, 42, 215)))
                painter.setPen(QtGui.QPen(stroke_col, 1.2))
                painter.drawRoundedRect(badge_rect, 4.0, 4.0)
                painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Weight.Bold))
                painter.setPen(QtGui.QColor("#ffffff"))
                painter.drawText(badge_rect, QtCore.Qt.AlignmentFlag.AlignCenter, badge_txt)

        # Draw live drag line
        if self.drag_start_idx is not None and self.current_mouse_pos is not None:
            if 0 <= self.drag_start_idx < len(self.atoms):
                a_start = self.atoms[self.drag_start_idx]
                pen_guide = QtGui.QPen(QtGui.QColor("#38bdf8"), 1.8, QtCore.Qt.PenStyle.DashLine)
                painter.setPen(pen_guide)
                painter.drawLine(QtCore.QPointF(a_start.x, a_start.y), self.current_mouse_pos)

        # Stamping ghost preview
        if self.current_tool == "stamp" and self.active_stamp_group and self.current_mouse_pos is not None:
            painter.setPen(QtGui.QPen(QtGui.QColor("#10b981"), 1.8, QtCore.Qt.PenStyle.DotLine))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(16, 185, 129, 40)))
            r_stamp = 20.0
            painter.drawEllipse(self.current_mouse_pos, r_stamp, r_stamp)
            painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
            painter.setPen(QtGui.QColor("#10b981"))
            painter.drawText(
                QtCore.QRectF(self.current_mouse_pos.x() + 15, self.current_mouse_pos.y() - 10, 100, 20),
                QtCore.Qt.AlignmentFlag.AlignLeft,
                self.active_stamp_group.split()[0],
            )

        # Draw Atoms
        elem_colors = {
            "C": ("#334155", "#cbd5e1"),
            "H": ("#94a3b8", "#0f172a"),
            "O": ("#ef4444", "#ffffff"),
            "N": ("#3b82f6", "#ffffff"),
            "S": ("#eab308", "#000000"),
            "P": ("#f97316", "#ffffff"),
            "F": ("#10b981", "#ffffff"),
            "Cl": ("#22c55e", "#ffffff"),
            "Br": ("#b45309", "#ffffff"),
            "I": ("#8b5cf6", "#ffffff"),
        }

        font = QtGui.QFont("DejaVu Sans", 10, QtGui.QFont.Weight.Bold)
        painter.setFont(font)

        for i, at in enumerate(self.atoms):
            r = 13.0
            bg_hex, fg_hex = elem_colors.get(at.symbol, ("#64748b", "#ffffff"))
            if i == self.hovered_atom_idx:
                painter.setPen(QtGui.QPen(QtGui.QColor("#38bdf8"), 2.5))
            else:
                painter.setPen(QtGui.QPen(QtGui.QColor("#0f172a"), 1.5))

            painter.setBrush(QtGui.QBrush(QtGui.QColor(bg_hex)))
            painter.drawEllipse(QtCore.QPointF(at.x, at.y), r, r)

            painter.setPen(QtGui.QColor(fg_hex))
            text_rect = QtCore.QRectF(at.x - r, at.y - r, 2 * r, 2 * r)
            painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, at.symbol)

        # On-canvas mode banner so instructions are always visible even in compact 2D | 3D mode
        if self.current_tool in ("weld_ring", "stamp"):
            banner_txt = (
                "💍 Ring Welder: Hover near bond to FUSE or atom to ATTACH · Click to weld (Shift=chain, Esc/Right-click=exit)"
                if self.current_tool == "weld_ring"
                else "📌 Stamp Mode: Click an atom to attach or empty canvas to place · Esc/Right-click=exit"
            )
            banner_col = QtGui.QColor("#0d9488" if self.current_tool == "weld_ring" else "#10b981")
            b_rect = QtCore.QRectF(6.0, 6.0, max(120.0, self.width() - 12.0), 22.0)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(15, 23, 42, 220)))
            painter.setPen(QtGui.QPen(banner_col, 1.2))
            painter.drawRoundedRect(b_rect, 4.0, 4.0)
            painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Weight.Bold))
            painter.setPen(QtGui.QColor("#f8fafc"))
            painter.drawText(b_rect.adjusted(6.0, 0.0, -6.0, 0.0), QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft, banner_txt)


class Sketcher2DPane(QtWidgets.QWidget):
    """
    Widget housing the 2D bond-line sketcher, quick ring & functional group palettes,
    element selector, live canonical SMILES badge, and 3D geometry generator.
    """

    structure_generated = QtCore.pyqtSignal(list, np.ndarray, list, dict)
    sync_from_3d_requested = QtCore.pyqtSignal()
    switch_to_inspector_requested = QtCore.pyqtSignal()
    accepted_requested = QtCore.pyqtSignal()
    rejected_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None, embedded: bool = False):
        super().__init__(parent)
        self._embedded = bool(embedded)
        self._live_timer = QtCore.QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(50)
        self._live_timer.timeout.connect(lambda: self._on_convert_to_3d(close_dialog=False, fast_live=True))
        self._init_ui()

    def hideEvent(self, event: QtGui.QHideEvent):
        if getattr(self, "_live_timer", None) is not None:
            self._live_timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent):
        if getattr(self, "_live_timer", None) is not None:
            self._live_timer.stop()
        super().closeEvent(event)

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        if self._embedded:
            main_layout.setContentsMargins(4, 4, 4, 4)
            main_layout.setSpacing(4)
            self.setMinimumSize(284, 200)
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)

        # Left Palette Bar (scrollable in embedded split mode so pane can resize freely)
        side_container = QtWidgets.QWidget()
        self.side_container = side_container
        side_bar = QtWidgets.QVBoxLayout(side_container)
        side_bar.setSpacing(4 if self._embedded else 6)
        side_bar.setContentsMargins(2, 2, 6, 2) if self._embedded else side_bar.setContentsMargins(0, 0, 0, 0)

        # Mode Selector
        side_bar.addWidget(QtWidgets.QLabel("<b>Active Tool:</b>"))
        self.btn_tool_draw = QtWidgets.QPushButton("✏️ Draw" if self._embedded else "✏️ Draw Bonds")
        self.btn_tool_draw.setCheckable(True)
        self.btn_tool_draw.setChecked(True)
        self.btn_tool_draw.setToolTip("Left-drag to create/connect atoms (hold Shift to move)")
        self.btn_tool_draw.setStyleSheet("QPushButton:checked { background-color: #0284c7; color: white; font-weight: bold; }")
        self.btn_tool_draw.clicked.connect(self._set_draw_mode)
        side_bar.addWidget(self.btn_tool_draw)

        self.btn_tool_move = QtWidgets.QPushButton("✥ Move Atom" if self._embedded else "✥ Move Atom (Drag)")
        self.btn_tool_move.setCheckable(True)
        self.btn_tool_move.setChecked(False)
        self.btn_tool_move.setToolTip("Click and drag any existing atom on the canvas to reposition it (or hold Shift in Draw mode)")
        self.btn_tool_move.setStyleSheet("QPushButton:checked { background-color: #0d9488; color: white; font-weight: bold; }")
        self.btn_tool_move.clicked.connect(self._set_move_mode)
        side_bar.addWidget(self.btn_tool_move)

        side_bar.addSpacing(6)
        lbl_elem = QtWidgets.QLabel("<b>Elements:</b>")
        side_bar.addWidget(lbl_elem)

        self.btn_group_elems = QtWidgets.QButtonGroup(self)
        self._elem_buttons: dict[str, QtWidgets.QPushButton] = {}
        elem_grid = QtWidgets.QGridLayout()
        common_elems = ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"]
        for idx, el in enumerate(common_elems):
            btn = QtWidgets.QPushButton(el)
            btn.setCheckable(True)
            if el == "C":
                btn.setChecked(True)
            btn.setMinimumWidth(36)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _, sym=el: self._set_element(sym))
            self.btn_group_elems.addButton(btn)
            self._elem_buttons[el] = btn
            elem_grid.addWidget(btn, idx // 2, idx % 2)
        side_bar.addLayout(elem_grid)

        self.btn_other_elem = QtWidgets.QPushButton("⚛️ Other..." if self._embedded else "⚛️ Other Element...")
        self.btn_other_elem.setCheckable(True)
        self.btn_other_elem.setToolTip("Open the full Periodic Table to select any element (B, Si, Li, Fe, Pt, etc.)")
        self.btn_other_elem.setStyleSheet("QPushButton:checked { background-color: #0284c7; color: white; font-weight: bold; }")
        self.btn_other_elem.clicked.connect(self._on_pick_other_element)
        side_bar.addWidget(self.btn_other_elem)

        side_bar.addSpacing(6)
        side_bar.addWidget(QtWidgets.QLabel("<b>Bonds:</b>"))
        btn_b1 = QtWidgets.QPushButton("Single (—)")
        btn_b1.clicked.connect(lambda: self._set_bond_order(1))
        side_bar.addWidget(btn_b1)

        btn_b2 = QtWidgets.QPushButton("Double (=)")
        btn_b2.clicked.connect(lambda: self._set_bond_order(2))
        side_bar.addWidget(btn_b2)

        btn_b3 = QtWidgets.QPushButton("Triple (≡)")
        btn_b3.clicked.connect(lambda: self._set_bond_order(3))
        side_bar.addWidget(btn_b3)

        side_bar.addSpacing(6)
        side_bar.addWidget(QtWidgets.QLabel("<b>💍 Ring Welder:</b>" if self._embedded else "<b>💍 Ring Welder & Templates:</b>"))
        self._ring_weld_buttons: List[QtWidgets.QPushButton] = []
        ring_grid = QtWidgets.QGridLayout()
        ring_specs = [
            ("⌬ Benz", 6, True, "Weld/place Aromatic Benzene ring (click bond to fuse, atom for spiro, or canvas to place)"),
            ("⬡ C6", 6, False, "Weld/place Cyclohexane 6-ring (click bond to fuse, atom for spiro, or canvas to place)"),
            ("⏢ C5", 5, False, "Weld/place Cyclopentane 5-ring (click bond to fuse, atom for spiro, or canvas to place)"),
            ("◻ C4", 4, False, "Weld/place Cyclobutane 4-ring (click bond to fuse, atom for spiro, or canvas to place)"),
            ("△ C3", 3, False, "Weld/place Cyclopropane 3-ring (click bond to fuse, atom for spiro, or canvas to place)"),
            ("⬡ C7", 7, False, "Weld/place Cycloheptane 7-ring (click bond to fuse, atom for spiro, or canvas to place)"),
        ]
        for idx, (r_lbl, r_sides, r_arom, r_tip) in enumerate(ring_specs):
            btn_r = QtWidgets.QPushButton(r_lbl)
            btn_r.setCheckable(True)
            btn_r.setToolTip(r_tip)
            btn_r.setStyleSheet("QPushButton { font-size: 11px; padding: 3px; } QPushButton:checked { background-color: #0d9488; color: #ffffff; font-weight: bold; }")
            btn_r.clicked.connect(lambda _chk=False, n=r_sides, ar=r_arom, b=btn_r: self._prime_ring_welder(n, ar, b))
            self._ring_weld_buttons.append(btn_r)
            ring_grid.addWidget(btn_r, idx // 2, idx % 2)
        side_bar.addLayout(ring_grid)

        # Quick Stamp Palette Grid
        fg_grid = QtWidgets.QGridLayout()
        fg_keys = list(FUNCTIONAL_GROUPS.keys())
        for idx, k in enumerate(fg_keys[:6]):
            short_lbl = k.split()[0]
            btn_fg = QtWidgets.QPushButton(f"+{short_lbl}")
            btn_fg.setToolTip(f"Stamp {k}: Click to prime, then click any atom or canvas position to stamp!")
            btn_fg.setStyleSheet("font-size: 11px; padding: 3px;")
            btn_fg.clicked.connect(lambda _, grp=k: self._prime_stamp(grp))
            fg_grid.addWidget(btn_fg, idx // 2, idx % 2)
        side_bar.addLayout(fg_grid)

        self.combo_fg = QtWidgets.QComboBox()
        self.combo_fg.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_fg.setMinimumContentsLength(8)
        self.combo_fg.setMinimumWidth(0)
        self.combo_fg.addItem("— More Groups —")
        for fg_name in fg_keys[6:]:
            self.combo_fg.addItem(fg_name)
        self.combo_fg.activated.connect(self._on_stamp_fg_selected)
        side_bar.addWidget(self.combo_fg)

        side_bar.addSpacing(6)
        side_bar.addWidget(QtWidgets.QLabel("<b>History:</b>"))
        btn_undo = QtWidgets.QPushButton("↶ Undo" if self._embedded else "↶ Undo (Ctrl+Z)")
        btn_undo.setToolTip("Undo last sketch action (Ctrl+Z)")
        btn_undo.clicked.connect(self._on_undo)
        side_bar.addWidget(btn_undo)

        btn_redo = QtWidgets.QPushButton("↷ Redo" if self._embedded else "↷ Redo (Ctrl+Y)")
        btn_redo.setToolTip("Redo last undone action (Ctrl+Y / Ctrl+Shift+Z)")
        btn_redo.clicked.connect(self._on_redo)
        side_bar.addWidget(btn_redo)

        btn_clear = QtWidgets.QPushButton("🧹 Clear" if self._embedded else "🧹 Clear All")
        btn_clear.clicked.connect(self._on_clear)
        side_bar.addWidget(btn_clear)

        side_bar.addStretch()
        if self._embedded:
            side_scroll = QtWidgets.QScrollArea()
            side_scroll.setWidgetResizable(True)
            side_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            side_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            side_scroll.setWidget(side_container)
            side_container.adjustSize()
            sb_extent = side_scroll.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent) or 16
            side_w = max(168, side_container.sizeHint().width() + sb_extent + 8)
            side_scroll.setFixedWidth(side_w)
            self.side_scroll = side_scroll
            self.setMinimumSize(side_w + 152, 200)
            main_layout.addWidget(side_scroll)
        else:
            main_layout.addWidget(side_container)

        # Center: Canvas & Bottom Action Bar (wrapped in QWidget so it never overlaps side_scroll)
        center_widget = QtWidgets.QWidget()
        center_widget.setMinimumWidth(140)
        center_layout = QtWidgets.QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)

        # Top Bar: Live SMILES badge & quick chemical name builder
        top_bar = QtWidgets.QHBoxLayout()
        if not self._embedded:
            top_bar.addWidget(QtWidgets.QLabel("<b>Live SMILES:</b>"))
        self.lbl_smiles_badge = QtWidgets.QLineEdit()
        self.lbl_smiles_badge.setReadOnly(True)
        self.lbl_smiles_badge.setMinimumWidth(40)
        self.lbl_smiles_badge.setPlaceholderText("(Draw structure to view canonical SMILES)")
        self.lbl_smiles_badge.setStyleSheet("background-color: #0f172a; color: #38bdf8; font-family: monospace; font-weight: bold; padding: 4px;")
        top_bar.addWidget(self.lbl_smiles_badge, stretch=1)

        btn_load_smiles = QtWidgets.QPushButton("📥 SMILES" if self._embedded else "📥 Import SMILES...")
        btn_load_smiles.setToolTip("Load an existing SMILES string directly into the 2D canvas")
        btn_load_smiles.clicked.connect(self._on_import_smiles)
        top_bar.addWidget(btn_load_smiles)

        if self._embedded:
            btn_from_3d = QtWidgets.QPushButton("⬅ From 3D")
            btn_from_3d.setToolTip("Project the current 3D molecule into the 2D sketcher canvas")
            btn_from_3d.clicked.connect(self.sync_from_3d_requested.emit)
            top_bar.addWidget(btn_from_3d)

            btn_to_zmat = QtWidgets.QPushButton("📋 Z-Matrix")
            btn_to_zmat.setStyleSheet("background-color: #0d9488; color: #ffffff; font-weight: bold;")
            btn_to_zmat.setToolTip("Switch right pane back to Z-Matrix / XYZ / Orbitals inspector")
            btn_to_zmat.clicked.connect(self.switch_to_inspector_requested.emit)
            top_bar.addWidget(btn_to_zmat)

        center_layout.addLayout(top_bar)

        self.canvas = SketchCanvas2D(self)
        if self._embedded:
            self.canvas.setMinimumSize(100, 120)
        self.canvas.molecule_changed.connect(self._update_live_smiles)
        self.canvas.molecule_changed.connect(self._schedule_live_3d)
        self.canvas.status_hint.connect(self._on_status_hint)
        center_layout.addWidget(self.canvas, stretch=1)

        bottom_bar = QtWidgets.QHBoxLayout()
        self.lbl_info = QtWidgets.QLabel("Draw: Drag to connect atoms. Move: Shift+Drag (or ✥ Move Atom). Delete: Right-click.")
        self.lbl_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.lbl_info.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.lbl_info.setMinimumWidth(0)
        if self._embedded:
            self.lbl_info.setVisible(False)
        bottom_bar.addWidget(self.lbl_info, stretch=1)

        self.chk_add_hs = QtWidgets.QCheckBox("Add H" if self._embedded else "Add Implicit Hydrogens")
        self.chk_add_hs.setChecked(True)
        self.chk_add_hs.setToolTip("Automatically add valence hydrogens when converting to 3D")
        self.chk_add_hs.toggled.connect(self._schedule_live_3d)
        bottom_bar.addWidget(self.chk_add_hs)

        self.chk_live_3d = QtWidgets.QCheckBox("⚡ Live" if self._embedded else "⚡ Live 2D → 3D")
        self.chk_live_3d.setChecked(self._embedded)
        self.chk_live_3d.setToolTip("Automatically update the 3D viewport whenever you edit the 2D structure")
        self.chk_live_3d.toggled.connect(lambda checked: self._schedule_live_3d() if checked else None)
        bottom_bar.addWidget(self.chk_live_3d)

        btn_build_3d = QtWidgets.QPushButton("🚀 3D" if self._embedded else "🚀 Convert to 3D Geometry ▶")
        btn_build_3d.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 14px;")
        btn_build_3d.clicked.connect(lambda: self._on_convert_to_3d(close_dialog=not self._embedded))
        bottom_bar.addWidget(btn_build_3d)

        if not self._embedded:
            btn_cancel = QtWidgets.QPushButton("Cancel")
            btn_cancel.clicked.connect(self.rejected_requested.emit)
            bottom_bar.addWidget(btn_cancel)

        center_layout.addLayout(bottom_bar)
        main_layout.addWidget(center_widget, stretch=1)

    def _set_draw_mode(self):
        self.btn_tool_draw.setChecked(True)
        self.btn_tool_move.setChecked(False)
        for b in getattr(self, "_ring_weld_buttons", []):
            b.setChecked(False)
        self.canvas.current_tool = "draw"
        self.canvas.active_stamp_group = None
        self.canvas.hovered_bond_idx = None
        self.canvas.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.lbl_info.setText("Draw Mode: Left-drag to create/connect atoms. Shift+Drag to reposition. Right-click to delete.")

    def _set_move_mode(self):
        self.btn_tool_draw.setChecked(False)
        self.btn_tool_move.setChecked(True)
        for b in getattr(self, "_ring_weld_buttons", []):
            b.setChecked(False)
        self.canvas.current_tool = "move"
        self.canvas.active_stamp_group = None
        self.canvas.hovered_bond_idx = None
        self.canvas.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
        self.lbl_info.setText("Move Mode: Left-click and drag any atom on canvas to reposition it cleanly.")

    def _prime_stamp(self, group_key: str):
        self.canvas.current_tool = "stamp"
        self.canvas.active_stamp_group = group_key
        self.canvas.hovered_bond_idx = None
        self.btn_tool_draw.setChecked(False)
        self.btn_tool_move.setChecked(False)
        for b in getattr(self, "_ring_weld_buttons", []):
            b.setChecked(False)
        self.canvas.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.lbl_info.setText(f"Stamp Mode [{group_key}]: Click on any atom to attach, or click anywhere on empty canvas to place!")

    def _prime_ring_welder(self, n_sides: int, aromatic: bool, btn: QtWidgets.QPushButton):
        if len(self.canvas.atoms) == 0:
            self.canvas.add_fragment_ring(n_sides, aromatic=aromatic)
            self._set_draw_mode()
            return
        if (
            self.canvas.current_tool == "weld_ring"
            and self.canvas.weld_ring_sides == n_sides
            and self.canvas.weld_ring_aromatic == aromatic
        ):
            self._set_draw_mode()
            return
        self.canvas.current_tool = "weld_ring"
        self.canvas.weld_ring_sides = n_sides
        self.canvas.weld_ring_aromatic = aromatic
        self.canvas.active_stamp_group = None
        self.btn_tool_draw.setChecked(False)
        self.btn_tool_move.setChecked(False)
        for b in getattr(self, "_ring_weld_buttons", []):
            b.setChecked(b is btn)
        self.canvas.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.lbl_info.setText(
            f"Ring Welder [{n_sides}-ring]: Click a bond to fuse or atom to attach (Hold Shift to weld multiple, Right-click/Esc to cancel)."
        )

    def _on_status_hint(self, msg: str):
        self.lbl_info.setText(msg)
        self._set_draw_mode()

    def _set_element(self, sym: str):
        self.canvas.current_element = sym
        if sym in self._elem_buttons:
            self._elem_buttons[sym].setChecked(True)
            self.btn_other_elem.setChecked(False)
            self.btn_other_elem.setText("⚛️ Other..." if self._embedded else "⚛️ Other Element...")
        else:
            self.btn_group_elems.setExclusive(False)
            for b in self._elem_buttons.values():
                b.setChecked(False)
            self.btn_group_elems.setExclusive(True)
            self.btn_other_elem.setChecked(True)
            self.btn_other_elem.setText(f"⚛️ {sym} (Other...)")
        self._set_draw_mode()

    def _on_pick_other_element(self):
        from .periodic_table import PeriodicTableDialog

        dlg = PeriodicTableDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted and dlg.chosen_element:
            self._set_element(dlg.chosen_element)
        else:
            self.btn_other_elem.setChecked(self.canvas.current_element not in self._elem_buttons)

    def _set_bond_order(self, bo: int):
        self.canvas.current_bond_order = bo
        self._set_draw_mode()

    def _on_undo(self):
        self.canvas.undo()

    def _on_redo(self):
        self.canvas.redo()

    def _on_clear(self):
        self.canvas.clear_canvas()

    def _on_stamp_fg_selected(self, idx: int):
        if idx <= 0:
            return
        key = self.combo_fg.currentText()
        self.combo_fg.setCurrentIndex(0)
        self._prime_stamp(key)

    def _schedule_live_3d(self):
        if getattr(self, "chk_live_3d", None) is not None and self.chk_live_3d.isChecked():
            self._live_timer.start()

    def _update_live_smiles(self):
        smi = self.canvas.get_canonical_smiles()
        self.lbl_smiles_badge.setText(smi)

    def _on_import_smiles(self):
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Import SMILES", "Enter SMILES string (e.g. c1ccccc1, CC(=O)O, C1=CC=CC=C1):"
        )
        if ok and text.strip():
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem
                mol = Chem.MolFromSmiles(text.strip())
                if mol is not None:
                    AllChem.Compute2DCoords(mol)
                    conf = mol.GetConformer()
                    self.canvas.push_undo_state()
                    self.canvas.atoms.clear()
                    self.canvas.bonds.clear()
                    cx = self.canvas.width() / 2.0 if self.canvas.width() > 0 else 240.0
                    cy = self.canvas.height() / 2.0 if self.canvas.height() > 0 else 190.0
                    for i in range(mol.GetNumAtoms()):
                        pos = conf.GetAtomPosition(i)
                        self.canvas.atoms.append(SketchAtom2D(x=cx + pos.x * 38.0, y=cy - pos.y * 38.0, symbol=mol.GetAtomWithIdx(i).GetSymbol()))
                    bo_map = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 1}
                    for b in mol.GetBonds():
                        self.canvas.bonds.append(SketchBond2D(atom1=b.GetBeginAtomIdx(), atom2=b.GetEndAtomIdx(), order=bo_map.get(b.GetBondType(), 1)))
                    self.canvas.update()
                    self.canvas.molecule_changed.emit()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "SMILES Parse Error", f"Could not parse SMILES: {e}")

    def _on_convert_to_3d(self, close_dialog: bool = True, fast_live: bool = False):
        """Converts 2D sketched connectivity to 3D Cartesian coordinates using ETKDG / UFF."""
        if len(self.canvas.atoms) == 0:
            if close_dialog:
                QtWidgets.QMessageBox.warning(self, "Empty Canvas", "Please sketch at least one atom before converting.")
            return

        symbols = [a.symbol for a in self.canvas.atoms]
        bonds = [(min(b.atom1, b.atom2), max(b.atom1, b.atom2)) for b in self.canvas.bonds]
        bond_orders = {(min(b.atom1, b.atom2), max(b.atom1, b.atom2)): b.order for b in self.canvas.bonds}
        add_hs = self.chk_add_hs.isChecked()

        final_symbols = symbols
        final_coords: Optional[np.ndarray] = None
        final_bonds = bonds
        final_bo = bond_orders

        # Strategy 1: If RDKit is available, build Mol, Sanitize, AddHs, Embed ETKDGv3, and UFF optimize
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            rwmol = Chem.RWMol()
            for s in symbols:
                rwmol.AddAtom(Chem.Atom(s))

            bo_map = {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
            }
            for (u, v), order in bond_orders.items():
                rwmol.AddBond(u, v, bo_map.get(order, Chem.BondType.SINGLE))

            mol = rwmol.GetMol()
            Chem.SanitizeMol(mol)
            if add_hs:
                mol = Chem.AddHs(mol)

            res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if res < 0:
                res = AllChem.EmbedMolecule(mol, useRandomCoords=True)

            if mol.GetNumConformers() > 0:
                try:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=80 if fast_live else 300)
                except Exception:
                    logging.getLogger(__name__).debug("Ignored exception", exc_info=True)
                conf = mol.GetConformer()
                final_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
                final_coords = conf.GetPositions().astype(np.float64)

                r_bonds = []
                r_bo = {}
                rdkit_bo_inv = {
                    Chem.BondType.SINGLE: 1,
                    Chem.BondType.DOUBLE: 2,
                    Chem.BondType.TRIPLE: 3,
                    Chem.BondType.AROMATIC: 1,
                }
                for b in mol.GetBonds():
                    u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                    norm_b = (min(u, v), max(u, v))
                    r_bonds.append(norm_b)
                    r_bo[norm_b] = rdkit_bo_inv.get(b.GetBondType(), 1)
                final_bonds = r_bonds
                final_bo = r_bo
        except Exception as e:
            log.info("RDKit 3D embedding notice: %s. Falling back to pure-Python force field.", e)

        # Strategy 2: Pure-Python Fallback if RDKit not present or embedding failed
        if final_coords is None:
            n = len(symbols)
            init_coords = np.zeros((n, 3), dtype=np.float64)
            for i, a in enumerate(self.canvas.atoms):
                init_coords[i, 0] = a.x * 0.035
                init_coords[i, 1] = -a.y * 0.035
                init_coords[i, 2] = np.random.uniform(-0.1, 0.1)

            init_coords -= np.mean(init_coords, axis=0)

            curr_syms = list(symbols)
            curr_coords = init_coords
            curr_bonds = list(bonds)
            curr_bo = dict(bond_orders)

            if add_hs:
                try:
                    from .forcefield import add_hydrogens_to_valence, detect_connectivity
                    if not curr_bonds and len(curr_syms) > 1:
                        curr_bonds = detect_connectivity(curr_syms, curr_coords)
                    h_syms, h_coords, _n_added = add_hydrogens_to_valence(
                        curr_syms, curr_coords, user_bonds=curr_bonds or None, user_bond_orders=curr_bo or None
                    )
                    orig_bonds = list(curr_bonds)
                    curr_syms = h_syms
                    curr_coords = h_coords
                    det_bonds = detect_connectivity(curr_syms, curr_coords)
                    bond_set = { (min(u, v), max(u, v)) for u, v in orig_bonds } | { (min(u, v), max(u, v)) for u, v in det_bonds }
                    curr_bonds = sorted(bond_set)
                    for b in curr_bonds:
                        if b not in curr_bo:
                            curr_bo[b] = 1
                except Exception as e:
                    log.warning("Pure-Python hydrogen addition notice: %s", e)

            from .forcefield import optimize_geometry_force_field
            try:
                opt_coords, _ = optimize_geometry_force_field(
                    curr_syms,
                    curr_coords,
                    user_bonds=curr_bonds,
                    max_iter=200,
                    user_bond_orders=curr_bo,
                )
                final_coords = opt_coords
            except Exception:
                final_coords = curr_coords

            final_symbols = curr_syms
            final_bonds = curr_bonds
            final_bo = curr_bo

        self.structure_generated.emit(final_symbols, final_coords, final_bonds, final_bo)
        if close_dialog and not self._embedded:
            self.accepted_requested.emit()


class MoleculeSketcher2DDialog(QtWidgets.QDialog):
    """Modal dialog wrapper around Sketcher2DPane for standalone Ctrl+Shift+K usage."""

    structure_generated = QtCore.pyqtSignal(list, np.ndarray, list, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D Molecular Bond-Line Sketcher & 3D Builder")
        self.resize(880, 580)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._pane = Sketcher2DPane(self, embedded=False)
        layout.addWidget(self._pane)
        self.canvas = self._pane.canvas
        self.chk_add_hs = self._pane.chk_add_hs
        self.chk_live_3d = self._pane.chk_live_3d
        self.lbl_smiles_badge = self._pane.lbl_smiles_badge
        self._pane.structure_generated.connect(self.structure_generated.emit)
        self._pane.accepted_requested.connect(self.accept)
        self._pane.rejected_requested.connect(self.reject)

    def _on_convert_to_3d(self, close_dialog: bool = True, fast_live: bool = False):
        self._pane._on_convert_to_3d(close_dialog=close_dialog, fast_live=fast_live)
