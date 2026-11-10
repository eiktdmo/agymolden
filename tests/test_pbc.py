import pytest
import numpy as np
from agymolden.pbc import UnitCell, create_box_from_molecule, replicate_supercell


def test_unit_cell_orthorhombic():
    cell = UnitCell(a=10.0, b=12.0, c=15.0, alpha=90.0, beta=90.0, gamma=90.0)
    assert pytest.approx(cell.volume, rel=1e-5) == 1800.0

    M = cell.lattice_vectors
    assert np.allclose(M[0], [10.0, 0.0, 0.0])
    assert np.allclose(M[1], [0.0, 12.0, 0.0])
    assert np.allclose(M[2], [0.0, 0.0, 15.0])

    G = cell.metric_tensor
    assert pytest.approx(G[0, 0]) == 100.0
    assert pytest.approx(G[1, 1]) == 144.0
    assert pytest.approx(G[2, 2]) == 225.0
    assert pytest.approx(G[0, 1]) == 0.0


def test_unit_cell_triclinic_and_reciprocal():
    cell = UnitCell(a=5.0, b=6.0, c=7.0, alpha=70.0, beta=80.0, gamma=100.0)
    assert cell.volume > 0.0

    # Reciprocal vectors satisfy a_i . b_j = 2pi * delta_ij
    M = cell.lattice_vectors
    B = cell.reciprocal_lattice_vectors
    prod = M @ B.T
    expected = 2.0 * np.pi * np.eye(3)
    assert np.allclose(prod, expected, atol=1e-5)


def test_fractional_cartesian_roundtrip():
    cell = UnitCell(a=8.0, b=9.0, c=10.0, alpha=85.0, beta=95.0, gamma=110.0)
    orig_cart = np.array([
        [1.2, 3.4, 5.6],
        [7.0, 1.0, 8.5],
        [0.1, 0.2, 0.3],
    ])
    frac = cell.cartesian_to_fractional(orig_cart)
    recon_cart = cell.fractional_to_cartesian(frac)
    assert np.allclose(orig_cart, recon_cart, atol=1e-6)


def test_coordinate_wrapping():
    cell = UnitCell(a=10.0, b=10.0, c=10.0)
    coords = np.array([
        [15.5, -2.3, 24.1],
    ])
    wrapped = cell.wrap_coordinates(coords, to_range="[0,1)")
    frac = cell.cartesian_to_fractional(wrapped)
    assert np.all(frac >= 0.0) and np.all(frac < 1.0)
    assert pytest.approx(frac[0, 0]) == 0.55
    assert pytest.approx(frac[0, 1]) == 0.77
    assert pytest.approx(frac[0, 2]) == 0.41


def test_box_edges_and_vertices():
    cell = UnitCell(a=4.0, b=5.0, c=6.0)
    verts = cell.get_box_vertices()
    assert len(verts) == 8
    edges = cell.get_box_edges()
    assert len(edges) == 12

    labels = [e[2] for e in edges]
    assert labels.count("a") == 4
    assert labels.count("b") == 4
    assert labels.count("c") == 4


def test_create_box_from_molecule():
    coords = np.array([
        [-1.0, 2.0, 3.0],
        [4.0, 6.0, 1.0],
    ])
    cell = create_box_from_molecule(coords, padding=5.0)
    assert pytest.approx(cell.a) == 5.0 + 10.0  # (4 - (-1)) + 2*5 = 15.0
    assert pytest.approx(cell.b) == 4.0 + 10.0  # (6 - 2) + 10 = 14.0
    assert pytest.approx(cell.c) == 2.0 + 10.0  # (3 - 1) + 10 = 12.0
    assert cell.alpha == 90.0


def test_supercell_replication():
    cell = UnitCell(a=5.0, b=5.0, c=5.0)
    symbols = ["Na", "Cl"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.5, 2.5, 2.5],
    ])
    # 2x2x2 supercell
    rep_syms, rep_coords, new_cell = replicate_supercell(symbols, coords, cell, nx=2, ny=2, nz=2)
    assert len(rep_syms) == 16
    assert rep_coords.shape == (16, 3)
    assert pytest.approx(new_cell.a) == 10.0
    assert pytest.approx(new_cell.b) == 10.0
    assert pytest.approx(new_cell.c) == 10.0
    assert pytest.approx(new_cell.volume) == 1000.0


def test_supercell_duplicate_removal():
    cell = UnitCell(a=4.0, b=4.0, c=4.0)
    # Atom placed exactly at face boundary (4.0, 0, 0) and (0, 0, 0)
    symbols = ["C", "C"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ])
    # When replicating along X by 2, (0 + 4.0, 0, 0) overlaps with (4.0, 0, 0)
    rep_syms, rep_coords, _ = replicate_supercell(
        symbols, coords, cell, nx=2, ny=1, nz=1, remove_duplicates=True, duplicate_tol=0.05
    )
    # Out of 4 total, 1 duplicate removed -> 3 unique atoms
    assert len(rep_syms) == 3


def test_pbc_dialog_and_supercell_flow(qapp):
    from agymolden.dialog_pbc import PBCDialog

    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.75, 0.58],
        [0.0, -0.75, 0.58],
    ])
    dlg = PBCDialog(None, symbols=symbols, coords=coords)
    assert dlg.tabs.count() == 4
    assert dlg.tbl_frac.rowCount() == 3

    # Change parameters
    dlg.spin_a.setValue(12.0)
    dlg.spin_b.setValue(12.0)
    dlg.spin_c.setValue(12.0)
    assert pytest.approx(dlg.unit_cell.volume) == 1728.0

    # Test supercell application
    applied = []
    dlg.supercellApplied.connect(lambda s, c, u: applied.append((s, c, u)))
    dlg.spin_nx.setValue(2)
    dlg.spin_ny.setValue(2)
    dlg.spin_nz.setValue(2)
    dlg._on_apply_supercell()

    assert len(applied) == 1
    new_syms, new_coords, new_cell = applied[0]
    assert len(new_syms) == 24
    assert pytest.approx(new_cell.a) == 24.0
    assert pytest.approx(new_cell.volume) == 1728.0 * 8.0
