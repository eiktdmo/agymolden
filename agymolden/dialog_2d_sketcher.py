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
        self.current_tool = "draw"  # "draw", "move", "stamp"
        self.current_element = "C"
        self.current_bond_order = 1
        self.active_stamp_group: Optional[str] = None
        self.moving_atom_idx: Optional[int] = None
        self.selected_atom_idx: Optional[int] = None
        self.drag_start_idx: Optional[int] = None
        self.current_mouse_pos: Optional[QtCore.QPointF] = None
        self.hovered_atom_idx: Optional[int] = None

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
                    pass
                AllChem.Compute2DCoords(mol)
                conf = mol.GetConformer()
                frag_syms = [a.GetSymbol() for a in mol.GetAtoms()]
                frag_pos = [(conf.GetAtomPosition(i).x * 42.0, -conf.GetAtomPosition(i).y * 42.0) for i in range(mol.GetNumAtoms())]
                bo_map = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 1}
                for b in mol.GetBonds():
                    frag_bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bo_map.get(b.GetBondType(), 1)))
        except Exception:
            pass

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

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.setFocus()
        pos = event.position()
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
        self.hovered_atom_idx = self._find_atom_near(event.position())

        # If currently dragging/repositioning an atom:
        if self.moving_atom_idx is not None and 0 <= self.moving_atom_idx < len(self.atoms):
            self.atoms[self.moving_atom_idx].x = event.position().x()
            self.atoms[self.moving_atom_idx].y = event.position().y()
            self.update()
            self.molecule_changed.emit()
            return

        # Cursor feedback
        if self.current_tool == "stamp":
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


class MoleculeSketcher2DDialog(QtWidgets.QDialog):
    """
    Dialog housing the 2D bond-line sketcher, quick ring & functional group palettes,
    element selector, live canonical SMILES badge, and 3D geometry generator.
    """

    structure_generated = QtCore.pyqtSignal(list, np.ndarray, list, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D Molecular Bond-Line Sketcher & 3D Builder")
        self.resize(880, 580)
        self._init_ui()

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # Left Palette Bar
        side_bar = QtWidgets.QVBoxLayout()
        side_bar.setSpacing(6)

        # Mode Selector
        side_bar.addWidget(QtWidgets.QLabel("<b>Active Tool:</b>"))
        self.btn_tool_draw = QtWidgets.QPushButton("✏️ Draw Bonds")
        self.btn_tool_draw.setCheckable(True)
        self.btn_tool_draw.setChecked(True)
        self.btn_tool_draw.setStyleSheet("QPushButton:checked { background-color: #0284c7; color: white; font-weight: bold; }")
        self.btn_tool_draw.clicked.connect(self._set_draw_mode)
        side_bar.addWidget(self.btn_tool_draw)

        self.btn_tool_move = QtWidgets.QPushButton("✥ Move Atom (Drag)")
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
        elem_grid = QtWidgets.QGridLayout()
        common_elems = ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"]
        for idx, el in enumerate(common_elems):
            btn = QtWidgets.QPushButton(el)
            btn.setCheckable(True)
            if el == "C":
                btn.setChecked(True)
            btn.setFixedWidth(36)
            btn.clicked.connect(lambda _, sym=el: self._set_element(sym))
            self.btn_group_elems.addButton(btn)
            elem_grid.addWidget(btn, idx // 2, idx % 2)
        side_bar.addLayout(elem_grid)

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
        side_bar.addWidget(QtWidgets.QLabel("<b>Templates & Stamp Groups:</b>"))
        btn_ring6 = QtWidgets.QPushButton("⌬ Benzene")
        btn_ring6.clicked.connect(lambda: self.canvas.add_fragment_ring(6, aromatic=True))
        side_bar.addWidget(btn_ring6)

        btn_ring6_sat = QtWidgets.QPushButton("⬡ Cyclohexane")
        btn_ring6_sat.clicked.connect(lambda: self.canvas.add_fragment_ring(6, aromatic=False))
        side_bar.addWidget(btn_ring6_sat)

        btn_ring5 = QtWidgets.QPushButton("⏢ 5-Ring")
        btn_ring5.clicked.connect(lambda: self.canvas.add_fragment_ring(5, aromatic=False))
        side_bar.addWidget(btn_ring5)

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
        self.combo_fg.addItem("— More Groups —")
        for fg_name in fg_keys[6:]:
            self.combo_fg.addItem(fg_name)
        self.combo_fg.activated.connect(self._on_stamp_fg_selected)
        side_bar.addWidget(self.combo_fg)

        side_bar.addSpacing(6)
        side_bar.addWidget(QtWidgets.QLabel("<b>History:</b>"))
        btn_undo = QtWidgets.QPushButton("↶ Undo (Ctrl+Z)")
        btn_undo.setToolTip("Undo last sketch action (Ctrl+Z)")
        btn_undo.clicked.connect(self._on_undo)
        side_bar.addWidget(btn_undo)

        btn_redo = QtWidgets.QPushButton("↷ Redo (Ctrl+Y)")
        btn_redo.setToolTip("Redo last undone action (Ctrl+Y / Ctrl+Shift+Z)")
        btn_redo.clicked.connect(self._on_redo)
        side_bar.addWidget(btn_redo)

        btn_clear = QtWidgets.QPushButton("🧹 Clear All")
        btn_clear.clicked.connect(self._on_clear)
        side_bar.addWidget(btn_clear)

        side_bar.addStretch()
        main_layout.addLayout(side_bar)

        # Center: Canvas & Bottom Action Bar
        center_layout = QtWidgets.QVBoxLayout()

        # Top Bar: Live SMILES badge & quick chemical name builder
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(QtWidgets.QLabel("<b>Live SMILES:</b>"))
        self.lbl_smiles_badge = QtWidgets.QLineEdit()
        self.lbl_smiles_badge.setReadOnly(True)
        self.lbl_smiles_badge.setPlaceholderText("(Draw structure to view canonical SMILES)")
        self.lbl_smiles_badge.setStyleSheet("background-color: #0f172a; color: #38bdf8; font-family: monospace; font-weight: bold; padding: 4px;")
        top_bar.addWidget(self.lbl_smiles_badge, stretch=1)

        btn_load_smiles = QtWidgets.QPushButton("📥 Import SMILES...")
        btn_load_smiles.setToolTip("Load an existing SMILES string directly into the 2D canvas")
        btn_load_smiles.clicked.connect(self._on_import_smiles)
        top_bar.addWidget(btn_load_smiles)

        center_layout.addLayout(top_bar)

        self.canvas = SketchCanvas2D(self)
        self.canvas.molecule_changed.connect(self._update_live_smiles)
        self.canvas.status_hint.connect(self._on_status_hint)
        center_layout.addWidget(self.canvas, stretch=1)

        bottom_bar = QtWidgets.QHBoxLayout()
        self.lbl_info = QtWidgets.QLabel("Draw: Drag to connect atoms. Move: Shift+Drag (or ✥ Move Atom). Delete: Right-click.")
        self.lbl_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        bottom_bar.addWidget(self.lbl_info)
        bottom_bar.addStretch()

        self.chk_add_hs = QtWidgets.QCheckBox("Add Implicit Hydrogens")
        self.chk_add_hs.setChecked(True)
        self.chk_add_hs.setToolTip("Automatically add valence hydrogens when converting to 3D")
        bottom_bar.addWidget(self.chk_add_hs)

        btn_build_3d = QtWidgets.QPushButton("🚀 Convert to 3D Geometry ▶")
        btn_build_3d.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 18px;")
        btn_build_3d.clicked.connect(self._on_convert_to_3d)
        bottom_bar.addWidget(btn_build_3d)

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        bottom_bar.addWidget(btn_cancel)

        center_layout.addLayout(bottom_bar)
        main_layout.addLayout(center_layout, stretch=1)

    def _set_draw_mode(self):
        self.btn_tool_draw.setChecked(True)
        self.btn_tool_move.setChecked(False)
        self.canvas.current_tool = "draw"
        self.canvas.active_stamp_group = None
        self.canvas.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.lbl_info.setText("Draw Mode: Left-drag to create/connect atoms. Shift+Drag to reposition. Right-click to delete.")

    def _set_move_mode(self):
        self.btn_tool_draw.setChecked(False)
        self.btn_tool_move.setChecked(True)
        self.canvas.current_tool = "move"
        self.canvas.active_stamp_group = None
        self.canvas.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
        self.lbl_info.setText("Move Mode: Left-click and drag any atom on canvas to reposition it cleanly.")

    def _prime_stamp(self, group_key: str):
        self.canvas.current_tool = "stamp"
        self.canvas.active_stamp_group = group_key
        self.btn_tool_draw.setChecked(False)
        self.btn_tool_move.setChecked(False)
        self.canvas.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.lbl_info.setText(f"Stamp Mode [{group_key}]: Click on any atom to attach, or click anywhere on empty canvas to place!")

    def _on_status_hint(self, msg: str):
        self.lbl_info.setText(msg)
        self._set_draw_mode()

    def _set_element(self, sym: str):
        self.canvas.current_element = sym
        self._set_draw_mode()

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

    def _on_convert_to_3d(self):
        """Converts 2D sketched connectivity to 3D Cartesian coordinates using ETKDG / UFF."""
        if len(self.canvas.atoms) == 0:
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
                    AllChem.UFFOptimizeMolecule(mol, maxIters=300)
                except Exception:
                    pass
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
        self.accept()
