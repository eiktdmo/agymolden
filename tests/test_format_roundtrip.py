"""Format Round-Trip and Geometric Invariant Tests.

Verifies lossless round-trip serialization and geometric invariance across supported formats:
1. XYZ -> Z-Matrix -> Cartesian (Internal coordinates / Pairwise distances)
2. XYZ -> MDL MOL/SDF -> XYZ
3. XYZ -> Protein Data Bank (PDB) -> XYZ
4. Multi-stage Pipeline: XYZ -> Z-Matrix -> XYZ -> SDF -> PDB -> XYZ
5. Distance matrix preservation: max |D_ij - D'_ij| < 10^-3 Å
6. Covalent connectivity graph preservation across all conversions.
"""

import os
import tempfile
import numpy as np
import pytest
from typing import List, Tuple

from agymolden.zmatrix import (
    ZAtom,
    cartesian_to_zmatrix,
    zmatrix_to_cartesian,
    write_xyz,
    parse_xyz,
    write_mol_sdf,
    parse_mol_sdf,
    write_pdb,
    parse_pdb,
)
from agymolden.forcefield import detect_connectivity


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute (N, N) pairwise Euclidean distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=-1)


# Test Molecule 1: Water (H2O)
WATER_SYMBOLS = ["O", "H", "H"]
WATER_COORDS = np.array([
    [0.0000, 0.0000, 0.1173],
    [0.0000, 0.7572, -0.4692],
    [0.0000, -0.7572, -0.4692],
], dtype=np.float64)

# Test Molecule 2: Benzene (C6H6)
BENZENE_SYMBOLS = ["C"] * 6 + ["H"] * 6
BENZENE_COORDS = np.array([
    [ 1.3970,  0.0000, 0.0000],
    [ 0.6985,  1.2100, 0.0000],
    [-0.6985,  1.2100, 0.0000],
    [-1.3970,  0.0000, 0.0000],
    [-0.6985, -1.2100, 0.0000],
    [ 0.6985, -1.2100, 0.0000],
    [ 2.4810,  0.0000, 0.0000],
    [ 1.2405,  2.1486, 0.0000],
    [-1.2405,  2.1486, 0.0000],
    [-2.4810,  0.0000, 0.0000],
    [-1.2405, -2.1486, 0.0000],
    [ 1.2405, -2.1486, 0.0000],
], dtype=np.float64)

# Test Molecule 3: Ethanol (C2H6O)
ETHANOL_SYMBOLS = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
ETHANOL_COORDS = np.array([
    [-1.233,  0.379,  0.000],
    [ 0.000, -0.485,  0.000],
    [ 1.189,  0.312,  0.000],
    [-2.128, -0.245,  0.000],
    [-1.258,  1.021,  0.887],
    [-1.258,  1.021, -0.887],
    [ 0.039, -1.130,  0.885],
    [ 0.039, -1.130, -0.885],
    [ 1.956, -0.266,  0.000],
], dtype=np.float64)


MOLECULES = [
    ("Water", WATER_SYMBOLS, WATER_COORDS),
    ("Benzene", BENZENE_SYMBOLS, BENZENE_COORDS),
    ("Ethanol", ETHANOL_SYMBOLS, ETHANOL_COORDS),
]


@pytest.mark.parametrize("name,symbols,coords", MOLECULES)
def test_cartesian_to_zmatrix_roundtrip(name, symbols, coords):
    """XYZ -> Z-matrix -> Cartesian must preserve pairwise distance matrix within 10^-3 Å."""
    orig_dist = _pairwise_distances(coords)

    # 1. Convert to Z-Matrix
    zatoms = cartesian_to_zmatrix(symbols, coords)
    assert len(zatoms) == len(symbols)

    # 2. Reconstruct 3D Cartesian coordinates
    rebuilt_coords = zmatrix_to_cartesian(zatoms)
    assert rebuilt_coords.shape == coords.shape

    # 3. Check pairwise distance invariance
    rebuilt_dist = _pairwise_distances(rebuilt_coords)
    max_dist_error = np.max(np.abs(orig_dist - rebuilt_dist))
    assert max_dist_error < 1e-3, f"[{name}] Z-matrix roundtrip altered distances by {max_dist_error:.5f} Å"


@pytest.mark.parametrize("name,symbols,coords", MOLECULES)
def test_xyz_roundtrip(tmp_path, name, symbols, coords):
    """Write XYZ -> Read XYZ must preserve coordinates to float precision."""
    xyz_path = str(tmp_path / f"{name}.xyz")
    write_xyz(xyz_path, symbols, coords, comment=f"Test {name}")

    parsed_syms, parsed_coords = parse_xyz(xyz_path)
    assert parsed_syms == symbols
    assert parsed_coords.shape == coords.shape
    assert np.allclose(parsed_coords, coords, atol=1e-3)


@pytest.mark.parametrize("name,symbols,coords", MOLECULES)
def test_sdf_roundtrip(tmp_path, name, symbols, coords):
    """Write SDF -> Read SDF must preserve coordinates and symbols within 10^-3 Å."""
    sdf_path = str(tmp_path / f"{name}.sdf")
    bonds = detect_connectivity(symbols, coords)
    write_mol_sdf(sdf_path, symbols, coords, bonds=bonds, title=name)

    res = parse_mol_sdf(sdf_path)
    assert res is not None
    parsed_syms, parsed_coords, parsed_bonds, _ = res

    assert parsed_syms == symbols
    assert parsed_coords.shape == coords.shape
    assert np.allclose(parsed_coords, coords, atol=1e-3)


@pytest.mark.parametrize("name,symbols,coords", MOLECULES)
def test_pdb_roundtrip(tmp_path, name, symbols, coords):
    """Write PDB -> Read PDB must preserve coordinates within standard PDB resolution (0.001 Å)."""
    pdb_path = str(tmp_path / f"{name}.pdb")
    bonds = detect_connectivity(symbols, coords)
    write_pdb(pdb_path, symbols, coords, bonds=bonds)

    parsed_syms, parsed_coords, parsed_bonds = parse_pdb(pdb_path)
    assert parsed_syms == symbols
    assert parsed_coords.shape == coords.shape
    assert np.allclose(parsed_coords, coords, atol=1e-3)


@pytest.mark.parametrize("name,symbols,coords", MOLECULES)
def test_full_format_pipeline_chain(tmp_path, name, symbols, coords):
    """Chain: XYZ -> Z-matrix -> Cartesian -> SDF -> PDB -> XYZ.

    Asserts pairwise distance matrix is strictly preserved at every transformation.
    """
    orig_dist = _pairwise_distances(coords)
    orig_bonds = set(tuple(sorted(b)) for b in detect_connectivity(symbols, coords))

    # Stage 1: XYZ -> Z-Matrix -> Cartesian
    zatoms = cartesian_to_zmatrix(symbols, coords)
    c1 = zmatrix_to_cartesian(zatoms)
    d1 = _pairwise_distances(c1)
    assert np.max(np.abs(orig_dist - d1)) < 1e-3

    # Stage 2: Cartesian -> SDF -> Parse
    sdf_path = str(tmp_path / f"{name}_chain.sdf")
    write_mol_sdf(sdf_path, symbols, c1, bonds=detect_connectivity(symbols, c1))
    parsed_syms2, c2, _, _ = parse_mol_sdf(sdf_path)
    assert parsed_syms2 == symbols
    d2 = _pairwise_distances(c2)
    assert np.max(np.abs(orig_dist - d2)) < 1e-3

    # Stage 3: SDF coords -> PDB -> Parse
    pdb_path = str(tmp_path / f"{name}_chain.pdb")
    write_pdb(pdb_path, symbols, c2, bonds=detect_connectivity(symbols, c2))
    parsed_syms3, c3, _ = parse_pdb(pdb_path)
    assert parsed_syms3 == symbols
    d3 = _pairwise_distances(c3)
    assert np.max(np.abs(orig_dist - d3)) < 1e-3

    # Stage 4: PDB coords -> XYZ -> Parse
    xyz_path = str(tmp_path / f"{name}_chain.xyz")
    write_xyz(xyz_path, parsed_syms3, c3)
    final_syms, final_coords = parse_xyz(xyz_path)
    assert final_syms == symbols
    final_dist = _pairwise_distances(final_coords)
    assert np.max(np.abs(orig_dist - final_dist)) < 1e-3

    # Stage 5: Topology graph invariance
    final_bonds = set(tuple(sorted(b)) for b in detect_connectivity(final_syms, final_coords))
    assert final_bonds == orig_bonds, f"[{name}] Bond topology altered in round-trip pipeline"
