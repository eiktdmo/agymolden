"""
Unified Molecule Model & State Manager for AgyMolden.

Decouples the core molecular representation from UI mixins and viewers.
Acts as a central, reactive data model providing:
- Cohesive atomic data (symbols, Cartesian coordinates, Z-matrix atoms, variables)
- Topology and bonding representation (explicit bonds, cross-bonds, bond orders)
- Centralized Undo / Redo history management
- Qt signal notification for synchronized updates across inspector tabs and 3D viewers
"""

import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Any
from PyQt6 import QtCore

from .zmatrix import ZAtom, cartesian_to_zmatrix, zmatrix_to_cartesian
from .topology import norm_bond
from .forcefield import detect_connectivity, detect_bond_orders


@dataclass
class MoleculeSnapshot:
    """Immutable snapshot of the molecule state for undo/redo."""
    zatoms: List[ZAtom]
    coords: np.ndarray
    variables: Dict[str, float]
    cross_bonds: Set[Tuple[int, int]]
    strict_topology: bool
    bond_orders: Dict[Tuple[int, int], int] = field(default_factory=dict)


class MoleculeModel(QtCore.QObject):
    """
    Reactive molecular state model emitting signals on structure mutations.

    Signals:
    - structure_changed: Emitted whenever atoms, coordinates, or Z-matrix change.
    - topology_changed: Emitted when bonds or bond orders change.
    - selection_changed: Emitted when atom selection changes.
    """
    structure_changed = QtCore.pyqtSignal()
    topology_changed = QtCore.pyqtSignal()
    selection_changed = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.zatoms: List[ZAtom] = []
        self.coords: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.variables: Dict[str, float] = {}
        self.cross_bonds: Set[Tuple[int, int]] = set()
        self.strict_topology: bool = False
        self.bond_orders: Dict[Tuple[int, int], int] = {}
        self.selected_indices: List[int] = []

        self._undo_stack: List[MoleculeSnapshot] = []
        self._redo_stack: List[MoleculeSnapshot] = []
        self.max_history: int = 50

    @property
    def symbols(self) -> List[str]:
        return [a.symbol for a in self.zatoms]

    @property
    def num_atoms(self) -> int:
        return len(self.zatoms)

    def is_empty(self) -> bool:
        return len(self.zatoms) == 0

    def capture_snapshot(self) -> MoleculeSnapshot:
        """Create a deep-copied snapshot of current state."""
        return MoleculeSnapshot(
            zatoms=[copy.deepcopy(a) for a in self.zatoms],
            coords=np.copy(self.coords),
            variables=dict(self.variables),
            cross_bonds=set(self.cross_bonds),
            strict_topology=self.strict_topology,
            bond_orders=dict(self.bond_orders),
        )

    def push_undo_state(self):
        """Record current state to the undo stack."""
        snap = self.capture_snapshot()
        self._undo_stack.append(snap)
        if len(self._undo_stack) > self.max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> bool:
        """Revert to the prior state."""
        if not self._undo_stack:
            return False
        current = self.capture_snapshot()
        self._redo_stack.append(current)
        prev = self._undo_stack.pop()
        self._restore_from_snapshot(prev)
        return True

    def redo(self) -> bool:
        """Re-apply the undone state."""
        if not self._redo_stack:
            return False
        current = self.capture_snapshot()
        self._undo_stack.append(current)
        nxt = self._redo_stack.pop()
        self._restore_from_snapshot(nxt)
        return True

    def _restore_from_snapshot(self, snap: MoleculeSnapshot):
        self.zatoms = [copy.deepcopy(a) for a in snap.zatoms]
        self.coords = np.copy(snap.coords)
        self.variables = dict(snap.variables)
        self.cross_bonds = set(snap.cross_bonds)
        self.strict_topology = snap.strict_topology
        self.bond_orders = dict(snap.bond_orders)
        self.structure_changed.emit()
        self.topology_changed.emit()

    def set_molecule(
        self,
        symbols: List[str],
        coords: np.ndarray,
        bonds: Optional[List[Tuple[int, int]]] = None,
        bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
        exact_coords: bool = True,
    ):
        """Load or update molecule from symbols and coordinates."""
        coords = np.array(coords, dtype=np.float64)
        if bonds is None:
            bonds = detect_connectivity(symbols, coords)
        if bond_orders is None:
            bond_orders = detect_bond_orders(symbols, coords, bonds)

        zatoms = cartesian_to_zmatrix(symbols, coords, user_bonds=bonds, strict_bonds=True)
        tree = {norm_bond(a.ref1, i) for i, a in enumerate(zatoms) if a.ref1 is not None}
        cross = {norm_bond(u, v) for u, v in bonds if norm_bond(u, v) not in tree}

        self.zatoms = zatoms
        self.coords = coords if exact_coords else zmatrix_to_cartesian(zatoms)
        self.variables = {}
        self.cross_bonds = cross
        self.strict_topology = True
        self.bond_orders = {norm_bond(u, v): o for (u, v), o in bond_orders.items() if o != 1}

        self.structure_changed.emit()
        self.topology_changed.emit()

    def get_explicit_bonds(self) -> List[Tuple[int, int]]:
        """Return full list of explicit covalent bonds (tree bonds + ring closure cross bonds)."""
        bonds = set()
        for i, a in enumerate(self.zatoms):
            if a.ref1 is not None and 0 <= a.ref1 < len(self.zatoms):
                bonds.add(norm_bond(a.ref1, i))
        bonds.update(self.cross_bonds)
        return list(bonds)
