"""Unit and Invariant Tests for MoleculeModel Architecture.

Verifies:
1. Signal emission upon structure and topology changes.
2. Accurate snapshot capture and state restoration.
3. Undo/Redo stack depth limits and reversibility.
4. Correct extraction of explicit covalent connectivity.
"""

import numpy as np
import pytest

from agymolden.molecule_model import MoleculeModel, MoleculeSnapshot
from agymolden.zmatrix import ZAtom


WATER_SYMBOLS = ["O", "H", "H"]
WATER_COORDS = np.array([
    [0.0000, 0.0000, 0.1173],
    [0.0000, 0.7572, -0.4692],
    [0.0000, -0.7572, -0.4692],
], dtype=np.float64)


def test_molecule_model_initialization_and_set():
    """Verify loading a molecule sets coordinates, symbols, and emits signals."""
    model = MoleculeModel()
    assert model.is_empty()

    signals_received = {"structure": 0, "topology": 0}

    def on_struct():
        signals_received["structure"] += 1

    def on_topo():
        signals_received["topology"] += 1

    model.structure_changed.connect(on_struct)
    model.topology_changed.connect(on_topo)

    model.set_molecule(WATER_SYMBOLS, WATER_COORDS)

    assert not model.is_empty()
    assert model.num_atoms == 3
    assert model.symbols == ["O", "H", "H"]
    assert signals_received["structure"] == 1
    assert signals_received["topology"] == 1

    bonds = model.get_explicit_bonds()
    assert len(bonds) == 2


def test_molecule_model_undo_redo_cycle():
    """Verify state snapshotting, undo, and redo within MoleculeModel."""
    model = MoleculeModel()
    model.set_molecule(WATER_SYMBOLS, WATER_COORDS)
    assert model.num_atoms == 3

    # State 1: push undo and mutate by adding carbon
    model.push_undo_state()
    model.zatoms.append(ZAtom("C", ref1=0, dist=1.43))
    model.coords = np.vstack([model.coords, [0.0, 0.0, 1.55]])
    assert model.num_atoms == 4
    assert model.symbols == ["O", "H", "H", "C"]

    # Undo
    assert model.undo() is True
    assert model.num_atoms == 3
    assert model.symbols == ["O", "H", "H"]

    # Redo
    assert model.redo() is True
    assert model.num_atoms == 4
    assert model.symbols == ["O", "H", "H", "C"]


def test_molecule_model_history_limit():
    """Verify undo stack size does not exceed max_history."""
    model = MoleculeModel()
    model.max_history = 5
    model.set_molecule(WATER_SYMBOLS, WATER_COORDS)

    for i in range(10):
        model.push_undo_state()
        model.zatoms.append(ZAtom("H", ref1=0, dist=1.0 + 0.1 * i))

    assert len(model._undo_stack) == 5
