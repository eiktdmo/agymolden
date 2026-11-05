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


def test_cycle_welder_dropdown_and_execution(qapp):
    class MockViewer:
        selected_bond = None
        multi_selected = []
        selected_atom_idx = None
        bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        def update(self):
            pass
        def auto_center(self):
            pass

    class MockWindow:
        def __init__(self):
            syms, crds = _create_benzene()
            from agymolden.zmatrix import ZAtom
            self.zatoms = [ZAtom(symbol=s) for s in syms]
            self.coords = crds
            self.viewer = MockViewer()
            self.updated_with_zatoms = False

        def set_molecule_from_zatoms(self, zatoms, exact_coords=None):
            self.zatoms = zatoms
            self.coords = exact_coords
            self.updated_with_zatoms = True

    win = MockWindow()
    # Even without pre-selecting any atoms, dialog auto-populates all bonds in the molecule!
    dlg = CycleWelderDialog(win)
    assert dlg.combo_bonds.count() == 6
    assert dlg.btn_fuse.isEnabled() is True
    assert dlg.target_bond == (0, 1)

    # Change to bond 2 (2-3)
    dlg.combo_bonds.setCurrentIndex(2)
    assert dlg.target_bond == (2, 3)

    # Execute fuse: Benzene + Benzene -> Naphthalene
    dlg._on_execute_fuse()
    assert win.updated_with_zatoms is True
    assert len(win.zatoms) == 18  # C10H8
    dlg.close()


def test_naphthalene_peri_annulation_to_phenalene():
    from agymolden.chem_identifiers import parse_smiles_or_inchi_to_3d
    from agymolden.cycle_welder import detect_peri_bays, fuse_ring_to_triad

    symbols, coords, bonds, bo = parse_smiles_or_inchi_to_3d("c1ccc2ccccc2c1")
    # Naphthalene has 18 atoms: 10 C, 8 H
    assert symbols.count("C") == 10
    assert symbols.count("H") == 8

    # 1. Detect peri bays
    bays = detect_peri_bays(symbols, coords, bonds)
    assert len(bays) == 2  # Top and bottom bays

    # Find the top bay (the one with the top bridgehead carbon)
    a_L, a_C, a_R = bays[0]

    # 2. Fuse Benzene ring across the bay triad -> Phenalene
    new_syms, new_coords = fuse_ring_to_triad(
        symbols=symbols,
        coords=coords,
        a_left=a_L,
        a_center=a_C,
        a_right=a_R,
        ring_preset="Benzene",
        apex_saturation="neutral",
        relax=True,
    )
    # 1H-phenalene: C13H10
    assert new_syms.count("C") == 13
    assert new_syms.count("H") == 10

    # Verify bond connectivity after relaxation
    from agymolden.forcefield import detect_connectivity
    det_bonds = detect_connectivity(new_syms, new_coords, strict=False)
    assert len(det_bonds) >= 24  # Well-connected polycyclic system


def test_naphthalene_peri_annulation_to_acenaphthene():
    from agymolden.chem_identifiers import parse_smiles_or_inchi_to_3d
    from agymolden.cycle_welder import detect_peri_bays, fuse_ring_to_triad

    symbols, coords, bonds, bo = parse_smiles_or_inchi_to_3d("c1ccc2ccccc2c1")
    bays = detect_peri_bays(symbols, coords, bonds)
    a_L, a_C, a_R = bays[0]

    # Fuse 5-membered ring across bay -> Acenaphthene
    new_syms, new_coords = fuse_ring_to_triad(
        symbols=symbols,
        coords=coords,
        a_left=a_L,
        a_center=a_C,
        a_right=a_R,
        ring_preset="Cyclopentyl",
        relax=True,
    )
    assert new_syms.count("C") == 12  # 10 + 2 new carbons
    assert new_syms.count("H") == 10


def test_edge_fusion_on_bridgehead_bond_detects_clash():
    from agymolden.chem_identifiers import parse_smiles_or_inchi_to_3d
    from agymolden.cycle_welder import fuse_ring_to_bond

    symbols, coords, bonds, bo = parse_smiles_or_inchi_to_3d("c1ccc2ccccc2c1")
    # Find a bridgehead atom (degree 3 heavy, 0 H)
    from agymolden.cycle_welder import _find_attached_hydrogens
    bh_atoms = [i for i, s in enumerate(symbols) if s == "C" and len(_find_attached_hydrogens(symbols, coords, i)) == 0]
    assert len(bh_atoms) == 2
    bh = bh_atoms[0]
    # Find an adjacent alpha carbon
    from agymolden.cycle_welder import _find_adjacent_heavy_neighbors
    neighs = _find_adjacent_heavy_neighbors(symbols, coords, bh, -1)
    alpha = [n for n in neighs if len(_find_attached_hydrogens(symbols, coords, n)) > 0][0]

    # Edge fusion on this bond should detect severe steric collision and guide to Peri-Annulation
    with pytest.raises(ValueError, match="severe steric collision|Peri-Annulation"):
        fuse_ring_to_bond(symbols, coords, a1=bh, a2=alpha, ring_preset="Benzene", relax=False)


def test_dialog_peri_annulation_flow(qapp):
    from agymolden.chem_identifiers import parse_smiles_or_inchi_to_3d
    from agymolden.zmatrix import ZAtom
    symbols, coords, bonds, bo = parse_smiles_or_inchi_to_3d("c1ccc2ccccc2c1")

    class MockViewer:
        def __init__(self, b_list):
            self.selected_bond = None
            self.multi_selected = [4, 3, 2]  # User selected 3 atoms in the top bay!
            self.selected_atom_idx = 3
            self.bonds = b_list

        def update(self):
            pass

        def auto_center(self):
            pass

    class MockWindow:
        def __init__(self):
            self.zatoms = [ZAtom(symbol=s) for s in symbols]
            self.coords = coords
            self.viewer = MockViewer(bonds)
            self.updated_with_zatoms = False

        def set_molecule_from_zatoms(self, zatoms, exact_coords=None):
            self.zatoms = zatoms
            self.coords = exact_coords
            self.updated_with_zatoms = True

    win = MockWindow()
    dlg = CycleWelderDialog(win)

    # Should automatically switch to Peri-Annulation tab because 3 atoms were selected!
    assert dlg.tabs.currentIndex() == 1
    assert dlg.combo_bays.count() >= 2
    assert dlg.btn_fuse_triad.isEnabled() is True
    assert "Target Bay" in dlg.lbl_bay_status.text()

    # Execute peri-annulation
    dlg._on_execute_fuse_triad()
    assert win.updated_with_zatoms is True
    assert len(win.zatoms) in (22, 23)  # Phenalene C13H9 or C13H10
    dlg.close()
