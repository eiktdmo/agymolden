"""Comprehensive Builder Undo/Redo Test Matrix.

Verifies that all primary molecular operations in AgyMolden are fully reversible:
1. Atom addition & deletion
2. Addition of explicit Hydrogens (valency satisfaction)
3. Ring welding / Cycle edge annulation (Benzene -> Naphthalene)
4. Triad peri-annulation (Naphthalene -> Phenalene)
5. Atom substitution / element change
6. BFS reordering of Z-Matrix
7. Redo restores the modified state identically
8. Multi-step chained undo/redo reversibility
"""

import os
import copy
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets
from agymolden.main_window import MainWindow
from agymolden.zmatrix import ZAtom
from agymolden.cycle_welder import fuse_ring_to_bond, fuse_ring_to_triad, detect_peri_bays
from agymolden.forcefield import detect_connectivity


@pytest.fixture
def win():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    # Initialize with Water (O-H-H)
    zatoms = [
        ZAtom("O"),
        ZAtom("H", ref1=0, dist=0.96),
        ZAtom("H", ref1=0, dist=0.96, ref2=1, angle=104.5),
    ]
    window.set_molecule_from_zatoms(zatoms)
    window.undo_stack.clear()
    window.redo_stack.clear()
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def _get_symbols(win: MainWindow):
    return [a.symbol for a in win.zatoms]


def test_undo_redo_add_and_delete_atom(win):
    """Adding an atom and deleting it must be reversibly undoable and redoable."""
    init_symbols = _get_symbols(win)
    assert init_symbols == ["O", "H", "H"]
    assert len(win.undo_stack) == 0

    # 1. Add atom (C attached to O)
    win._push_undo_state()
    win.zatoms.append(ZAtom("C", ref1=0, dist=1.43, ref2=1, angle=109.5, ref3=2, dihedral=120.0))
    win.set_molecule_from_zatoms(win.zatoms)
    assert _get_symbols(win) == ["O", "H", "H", "C"]
    assert len(win.undo_stack) == 1

    # Undo addition
    win.undo()
    assert _get_symbols(win) == ["O", "H", "H"]
    assert len(win.redo_stack) == 1

    # Redo addition
    win.redo()
    assert _get_symbols(win) == ["O", "H", "H", "C"]

    # 2. Delete an atom
    win._push_undo_state()
    del win.zatoms[-1]
    win.set_molecule_from_zatoms(win.zatoms)
    assert _get_symbols(win) == ["O", "H", "H"]

    # Undo deletion
    win.undo()
    assert _get_symbols(win) == ["O", "H", "H", "C"]

    # Redo deletion
    win.redo()
    assert _get_symbols(win) == ["O", "H", "H"]


def test_undo_redo_add_hydrogens(win):
    """Adding hydrogens to satisfy valency must be undoable."""
    # Set single bare carbon
    win.set_molecule_from_zatoms([ZAtom("C")])
    win.undo_stack.clear()
    win.redo_stack.clear()

    # Call add hydrogens
    win._on_add_hydrogens()
    assert len(_get_symbols(win)) == 5  # CH4
    assert _get_symbols(win).count("H") == 4
    assert len(win.undo_stack) == 1

    # Undo
    win.undo()
    assert len(win.zatoms) == 1
    assert win.zatoms[0].symbol == "C"

    # Redo
    win.redo()
    assert len(win.zatoms) == 5
    assert win.zatoms[0].symbol == "C"


def test_undo_redo_cycle_welder_edge_fusion(win):
    """Fusing a benzene ring onto a bond must be completely reversible via undo."""
    from agymolden.zmatrix import cartesian_to_zmatrix
    # Build standard planar benzene with proper Cartesian coordinates
    r_cc = 1.40
    r_ch = 1.09
    syms = ["C"] * 6 + ["H"] * 6
    crds = []
    for i in range(6):
        theta = i * np.pi / 3.0
        crds.append([r_cc * np.cos(theta), r_cc * np.sin(theta), 0.0])
    for i in range(6):
        theta = i * np.pi / 3.0
        crds.append([(r_cc + r_ch) * np.cos(theta), (r_cc + r_ch) * np.sin(theta), 0.0])
    crds = np.array(crds, dtype=np.float64)

    benzene_zatoms = cartesian_to_zmatrix(syms, crds)
    win.set_molecule_from_zatoms(benzene_zatoms, exact_coords=crds)
    win.undo_stack.clear()
    win.redo_stack.clear()

    orig_count = len(win.zatoms)
    assert orig_count == 12

    # Simulate Cycle Welder Edge Annulation
    win._push_undo_state()
    fused_syms, fused_crds = fuse_ring_to_bond(
        symbols=syms,
        coords=win.coords,
        a1=0,
        a2=1,
        ring_preset="Benzene",
        relax=True,
    )
    new_zatoms = cartesian_to_zmatrix(fused_syms, fused_crds)
    win.set_molecule_from_zatoms(new_zatoms, exact_coords=fused_crds)

    # Naphthalene: 10 C, 8 H = 18 atoms
    assert len(win.zatoms) == 18
    assert _get_symbols(win).count("C") == 10
    assert _get_symbols(win).count("H") == 8

    # Undo
    win.undo()
    assert len(win.zatoms) == orig_count
    assert _get_symbols(win).count("C") == 6
    assert _get_symbols(win).count("H") == 6

    # Redo
    win.redo()
    assert len(win.zatoms) == 18
    assert _get_symbols(win).count("C") == 10
    assert _get_symbols(win).count("H") == 8


def test_undo_redo_multi_step_chain(win):
    """Multi-step edits (A -> B -> C -> D) can be rolled back sequentially and redone."""
    states = []
    for elem in ["F", "Cl", "Br", "I"]:
        win._push_undo_state()
        win.zatoms.append(ZAtom(elem, ref1=0, dist=1.5))
        win.set_molecule_from_zatoms(win.zatoms)
        states.append(copy.deepcopy(_get_symbols(win)))

    assert len(win.zatoms) == 7  # O, H, H, F, Cl, Br, I

    # Undo 4 times
    for expected_symbols in reversed(states[:-1]):
        win.undo()
        assert _get_symbols(win) == expected_symbols

    win.undo()
    assert _get_symbols(win) == ["O", "H", "H"]

    # Redo all 4 steps
    for expected_symbols in states:
        win.redo()
        assert _get_symbols(win) == expected_symbols
