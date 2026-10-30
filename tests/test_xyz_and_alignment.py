"""Comprehensive tests for Phase 1 (XYZ import/export & ring/bond/frame preservation)
and Phase 2 (Precision 3D Centering, Axis/Plane Alignment & Symmetry Matrix generation)."""

import os
import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("OpenGL")

from PyQt6 import QtWidgets
from agymolden.zmatrix import parse_xyz, write_xyz, ZAtom, zmatrix_to_cartesian
from agymolden.alignment import (
    center_at_com,
    center_at_centroid,
    center_at_atom,
    align_atoms_to_axis,
    align_plane_to_cartesian,
    capture_rigid_frame,
    apply_rigid_frame,
)
from agymolden.symmetry import find_symmetry_operations
from agymolden.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ==============================================================================
# Phase 1 Tests: XYZ Parsing & MainWindow Import
# ==============================================================================

def test_parse_xyz_with_blank_comment_line():
    """Verify that an XYZ file with a blank comment line on line 2 does NOT drop Atom 1."""
    xyz_text = (
        "3\n"
        "\n"
        "O   0.000000   0.000000   0.117300\n"
        "H   0.000000   0.757200  -0.469200\n"
        "H   0.000000  -0.757200  -0.469200\n"
    )
    symbols, coords = parse_xyz(xyz_text, is_path=False)
    assert symbols == ["O", "H", "H"]
    assert coords.shape == (3, 3)
    np.testing.assert_allclose(coords[0], [0.0, 0.0, 0.1173], atol=1e-6)


def test_parse_xyz_atomic_numbers_labels_and_fortran_exponents():
    """Verify atomic numbers (6 -> C, 1 -> H), atom labels (Cl1 -> Cl), and D exponents."""
    xyz_text = (
        "4\n"
        "Formaldehyde with mixed tokens and D exponents\n"
        "6     0.000000D+00   0.000000D+00   0.000000D+00\n"
        "O1    0.000000D+00   0.000000D+00   1.205000D+00\n"
        "1     0.940000d+00   0.000000d+00  -0.540000d+00\n"
        "CL2  -0.940000E+00   0.000000E+00  -0.540000E+00\n"
    )
    symbols, coords = parse_xyz(xyz_text, is_path=False)
    assert symbols == ["C", "O", "H", "Cl"]
    assert coords.shape == (4, 3)
    assert np.isclose(coords[1, 2], 1.205)


def test_mainwindow_load_xyz_preserves_exact_coordinates_and_ring_bonds(qapp, tmp_path):
    """Verify loading an XYZ file into MainWindow preserves exact 3D coordinates,
    ring closure cross-bonds, and double/aromatic bond orders."""
    # Benzene ring in the XY plane centered at (2.0, 3.0, -1.5)
    r_cc = 1.397
    r_ch = 2.481
    angles = [k * np.pi / 3.0 for k in range(6)]
    center = np.array([2.0, 3.0, -1.5])
    coords_list = []
    symbols = ["C"] * 6 + ["H"] * 6
    for ang in angles:
        coords_list.append(center + np.array([r_cc * np.cos(ang), r_cc * np.sin(ang), 0.0]))
    for ang in angles:
        coords_list.append(center + np.array([r_ch * np.cos(ang), r_ch * np.sin(ang), 0.0]))
    orig_coords = np.array(coords_list, dtype=np.float64)

    xyz_path = str(tmp_path / "benzene_shifted.xyz")
    # Write with a blank comment line to test both together
    with open(xyz_path, "w") as f:
        f.write("12\n\n")
        for s, (x, y, z) in zip(symbols, orig_coords):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")

    win = MainWindow()
    win.open_path(xyz_path)

    # All 12 atoms read
    assert len(win.zatoms) == 12
    # Exact 3D Cartesian coordinates from the XYZ file are preserved (not reset to Atom0=0,0,0)!
    np.testing.assert_allclose(win.coords, orig_coords, atol=1e-5)
    np.testing.assert_allclose(win.viewer.coords, orig_coords, atol=1e-5)

    # All 12 covalent bonds (including the ring-closing C1-C6 bond) are present
    bonds = win._current_explicit_bonds()
    assert len(bonds) == 12
    assert (0, 5) in bonds


# ==============================================================================
# Phase 2 Tests: Centering, Axis/Plane Alignment & Symmetry Matrices
# ==============================================================================

def test_center_com_centroid_and_selected_atom(qapp):
    """Verify centering at COM, centroid, and a specific selected atom."""
    win = MainWindow()
    # Water shifted away from origin
    zatoms = [
        ZAtom(symbol="O"),
        ZAtom(symbol="H", ref1=0, dist=0.96),
        ZAtom(symbol="H", ref1=0, dist=0.96, ref2=1, angle=104.5),
    ]
    shifted_coords = zmatrix_to_cartesian(zatoms) + np.array([5.0, -3.0, 2.0])
    win.set_molecule_from_zatoms(zatoms, exact_coords=shifted_coords)

    # 1. Center Selected Atom (H2, index 1) at origin (0, 0, 0)
    win.viewer.selected_atom_idx = 1
    win.viewer.multi_selected = [1]
    win._on_center_molecule_mode("atom")
    np.testing.assert_allclose(win.coords[1], [0.0, 0.0, 0.0], atol=1e-7)

    # 2. Center at Geometric Centroid
    win._on_center_molecule_mode("centroid")
    np.testing.assert_allclose(np.mean(win.coords, axis=0), [0.0, 0.0, 0.0], atol=1e-7)

    # 3. Center at Center of Mass (COM)
    win._on_center_molecule_mode("com")
    from agymolden.alignment import compute_center_of_mass
    com = compute_center_of_mass(["O", "H", "H"], win.coords)
    np.testing.assert_allclose(com, [0.0, 0.0, 0.0], atol=1e-7)


def test_align_axis_plane_and_canonical_symmetry_matrices(qapp):
    """
    Verify that centering an atom/COM and aligning a molecule to a chosen axis and plane
    preserves that exact orientation and produces textbook-clean 3x3 symmetry matrices.
    """
    win = MainWindow()
    # Start with water rotated arbitrarily in 3D space
    zatoms = [
        ZAtom(symbol="O"),
        ZAtom(symbol="H", ref1=0, dist=0.96),
        ZAtom(symbol="H", ref1=0, dist=0.96, ref2=1, angle=104.5),
    ]
    base_coords = zmatrix_to_cartesian(zatoms)
    # Apply arbitrary rotation and translation
    ang1, ang2 = 0.73, -1.12
    R_arb = np.array([
        [np.cos(ang1), -np.sin(ang1), 0.0],
        [np.sin(ang1),  np.cos(ang1), 0.0],
        [0.0,           0.0,          1.0],
    ]) @ np.array([
        [1.0, 0.0,           0.0],
        [0.0, np.cos(ang2), -np.sin(ang2)],
        [0.0, np.sin(ang2),  np.cos(ang2)],
    ])
    tilted_coords = (R_arb @ base_coords.T).T + np.array([3.2, -1.8, 4.5])
    win.set_molecule_from_zatoms(zatoms, exact_coords=tilted_coords)

    # Step 1: Center Oxygen (atom 0) at the origin (0, 0, 0)
    win.viewer.selected_atom_idx = 0
    win.viewer.multi_selected = [0]
    win._on_center_molecule_mode("atom")
    np.testing.assert_allclose(win.coords[0], [0.0, 0.0, 0.0], atol=1e-7)

    # Step 2: Select all 3 atoms [0, 1, 2] and align the molecular plane into the YZ plane (x = 0)
    win.viewer.multi_selected = [0, 1, 2]
    win._on_align_to_plane("YZ")
    # Every atom in water must now have x == 0!
    np.testing.assert_allclose(win.coords[:, 0], [0.0, 0.0, 0.0], atol=1e-6)

    # Step 3: Clear selection so principal symmetry axis (C2 bisector) aligns along +Z
    win.viewer.multi_selected = []
    win.viewer.selected_atom_idx = None
    win._on_align_to_axis("+Z")
    # Re-align plane into YZ so water lies in YZ with C2 along Z
    win.viewer.multi_selected = [0, 1, 2]
    win._on_align_to_plane("YZ")
    np.testing.assert_allclose(win.coords[:, 0], [0.0, 0.0, 0.0], atol=1e-6)

    # Step 4: Detect symmetry and check the 3x3 matrices!
    win._on_detect_symmetry()
    res = win._current_sym_res
    assert res is not None
    assert res.symbol == "C2v"
    assert res.order == 4

    # In C2v with C2 along Z and molecular plane in YZ (or XZ):
    # Every 3x3 matrix in res.operations must be a clean diagonal matrix with entries in {-1, +1} and 0 off-diagonal!
    expected_diags = {
        (1.0, 1.0, 1.0),    # E
        (-1.0, -1.0, 1.0),  # C2(z)
        (1.0, -1.0, 1.0),   # sigma(xz)
        (-1.0, 1.0, 1.0),   # sigma(yz)
    }
    found_diags = set()
    for op in res.operations:
        M = op.matrix
        # Off-diagonal elements must be zero
        off_diag = M - np.diag(np.diag(M))
        np.testing.assert_allclose(off_diag, np.zeros((3, 3)), atol=1e-6)
        diag_tuple = tuple(np.round(np.diag(M), 4))
        found_diags.add(diag_tuple)

    assert found_diags == expected_diags


def test_aligned_orientation_preserved_during_zmatrix_slider_edit(qapp):
    """Verify that after aligning a bond along +X and plane in XY, tuning a bond length
    in the Z-matrix preserves the aligned XY plane and +X bond axis."""
    win = MainWindow()
    zatoms = [
        ZAtom(symbol="O"),
        ZAtom(symbol="H", ref1=0, dist=0.96),
        ZAtom(symbol="H", ref1=0, dist=0.96, ref2=1, angle=104.5),
    ]
    win.set_molecule_from_zatoms(zatoms)

    # Center O1 at (0,0,0), align O1->H2 bond along +X, and align water plane into XY
    win.viewer.multi_selected = [0]
    win._on_center_molecule_mode("atom")
    win.viewer.multi_selected = [0, 1]
    win._on_align_to_axis("+X")
    win.viewer.multi_selected = [0, 1, 2]
    win._on_align_to_plane("XY")

    # Verify O1 is at origin, H2 is on +X axis, and all atoms have z == 0 (in XY plane)
    np.testing.assert_allclose(win.coords[0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(win.coords[1], [0.96, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(win.coords[:, 2], [0.0, 0.0, 0.0], atol=1e-6)

    # Now select H3 (index 2) and stretch its O-H bond from 0.96 to 1.15 Å via the spinbox
    win.viewer.selected_atom_idx = 2
    win._on_spin_bond_changed(1.15)

    # O1 must still be at (0,0,0), H2 must still be on +X, and the molecule must still lie in the XY plane (z == 0)!
    np.testing.assert_allclose(win.coords[0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(win.coords[1], [0.96, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(win.coords[:, 2], [0.0, 0.0, 0.0], atol=1e-6)
    assert np.isclose(np.linalg.norm(win.coords[2] - win.coords[0]), 1.15, atol=1e-5)
