"""Tests for Molecular Cycle Welder & Ring Fusion Engine."""

import numpy as np
import pytest

from agymolden.cycle_welder import (
    fuse_ring_to_bond,
    bridge_cycle_atoms,
    RING_PRESETS,
)
from agymolden.dialog_cycle_welder import CycleWelderDialog


def _create_benzene():
    R = 1.40
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    c_coords = np.column_stack([R * np.cos(angles), R * np.sin(angles), np.zeros(6)])
    h_coords = np.column_stack([(R + 1.08) * np.cos(angles), (R + 1.08) * np.sin(angles), np.zeros(6)])
    symbols = ["C"] * 6 + ["H"] * 6
    coords = np.vstack([c_coords, h_coords])
    return symbols, coords


def _create_cyclohexane():
    # Standard cyclohexane chair conformation
    symbols = ["C"] * 6 + ["H"] * 12
    coords = np.array([
        [-1.261,  0.728,  0.225],
        [ 0.000,  1.456, -0.225],
        [ 1.261,  0.728,  0.225],
        [ 1.261, -0.728, -0.225],
        [ 0.000, -1.456,  0.225],
        [-1.261, -0.728, -0.225],
        # Hydrogens
        [-2.158,  1.246, -0.155],
        [-1.261,  0.728,  1.325],
        [ 0.000,  1.456, -1.325],
        [ 0.000,  2.492,  0.155],
        [ 2.158,  1.246, -0.155],
        [ 1.261,  0.728,  1.325],
        [ 2.158, -1.246,  0.155],
        [ 1.261, -0.728, -1.325],
        [ 0.000, -1.456,  1.325],
        [ 0.000, -2.492, -0.155],
        [-2.158, -1.246,  0.155],
        [-1.261, -0.728, -1.325],
    ])
    return symbols, coords


def test_fuse_benzene_to_naphthalene():
    symbols, coords = _create_benzene()
    # Fuse Benzene onto bond 0-1
    new_syms, new_coords = fuse_ring_to_bond(
        symbols=symbols,
        coords=coords,
        a1=0,
        a2=1,
        ring_preset="Benzene",
        relax=True,
    )

    # Naphthalene: C10H8
    assert new_syms.count("C") == 10
    assert new_syms.count("H") == 8
    assert len(new_syms) == 18
    assert len(new_coords) == 18

    # Check that fused bond distance is preserved near ~1.40 A
    L = np.linalg.norm(new_coords[1] - new_coords[0])
    assert 1.30 < L < 1.50


def test_fuse_heterocycles():
    symbols, coords = _create_benzene()

    # 1. Fuse Pyrrole -> Indole (C8H7N)
    ind_syms, ind_coords = fuse_ring_to_bond(
        symbols=symbols, coords=coords, a1=0, a2=1, ring_preset="Pyrrole", relax=True
    )
    assert ind_syms.count("C") == 8
    assert ind_syms.count("H") == 7
    assert ind_syms.count("N") == 1

    # 2. Fuse Furan -> Benzofuran (C8H6O)
    fur_syms, fur_coords = fuse_ring_to_bond(
        symbols=symbols, coords=coords, a1=0, a2=1, ring_preset="Furan", relax=True
    )
    assert fur_syms.count("C") == 8
    assert fur_syms.count("H") == 6
    assert fur_syms.count("O") == 1

    # 3. Fuse Thiophene -> Benzothiophene (C8H6S)
    th_syms, th_coords = fuse_ring_to_bond(
        symbols=symbols, coords=coords, a1=0, a2=1, ring_preset="Thiophene", relax=True
    )
    assert th_syms.count("C") == 8
    assert th_syms.count("H") == 6
    assert th_syms.count("S") == 1

    # 4. Fuse Pyridine -> Quinoline (C9H7N)
    quin_syms, quin_coords = fuse_ring_to_bond(
        symbols=symbols, coords=coords, a1=0, a2=1, ring_preset="Pyridine", relax=True
    )
    assert quin_syms.count("C") == 9
    assert quin_syms.count("H") == 7
    assert quin_syms.count("N") == 1


def test_bicyclic_norbornane_bridge():
    symbols, coords = _create_cyclohexane()
    # Atoms 1 (C0,1,0) and 4 (C0,-1,0) are 1,4-bridgeheads across the cyclohexane ring
    b_syms, b_coords = bridge_cycle_atoms(
        symbols=symbols, coords=coords, a1=1, a2=4, bridge_type="-CH2-", relax=True
    )
    # Cyclohexane (C6H12) - 2H + CH2 (C + 2H) = C7H12 (Norbornane / Bicyclo[2.2.1]heptane)
    assert b_syms.count("C") == 7
    assert b_syms.count("H") == 12


def test_cycle_welder_dialog(qapp):
    class MockWindow:
        def __init__(self):
            syms, crds = _create_benzene()
            from agymolden.zmatrix import ZAtom
            self.zatoms = [ZAtom(symbol=s) for s in syms]
            self.coords = crds
            self.selected_atoms = [0, 1]
            self.viewer = None

    win = MockWindow()
    dlg = CycleWelderDialog(win)
    assert dlg.btn_fuse.isEnabled() is True
    assert "Target Bond" in dlg.lbl_atoms.text()
    dlg.close()
