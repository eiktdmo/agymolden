"""
Tests for Chemical Identifiers & Copying Features (XYZ, SMILES, InChI, InChIKey).
"""

import numpy as np
import pytest

from agymolden.chem_identifiers import (
    format_xyz_string,
    get_smiles,
    get_inchi_and_key,
    _HAS_RDKIT,
)


def test_format_xyz_string():
    symbols = ["C", "H", "H", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.09],
        [1.03, 0.0, -0.36],
        [-0.51, -0.89, -0.36],
        [-0.51, 0.89, -0.36],
    ])
    txt = format_xyz_string(symbols, coords, comment="Methane test")
    lines = txt.strip().splitlines()
    assert lines[0] == "5"
    assert lines[1] == "Methane test"
    assert lines[2].startswith("C ")
    assert len(lines) == 7


def test_smiles_and_inchi_generation():
    if not _HAS_RDKIT:
        pytest.skip("RDKit not installed")

    # Water H2O
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.12],
        [0.0, 0.76, -0.48],
        [0.0, -0.76, -0.48],
    ])
    bonds = [(0, 1), (0, 2)]
    bo = {(0, 1): 1, (0, 2): 1}

    smi = get_smiles(symbols, coords, bonds, bo)
    assert "O" in smi

    inchi, key = get_inchi_and_key(symbols, coords, bonds, bo)
    assert inchi.startswith("InChI=1S/H2O")
    assert len(key) == 27
    assert key.startswith("XLYOFNOQVPJJNP")


def test_parse_smiles_and_inchi_to_3d():
    if not _HAS_RDKIT:
        pytest.skip("RDKit not installed")

    from agymolden.chem_identifiers import parse_smiles_or_inchi_to_3d

    # Test SMILES -> 3D coordinates & connectivity
    syms, coords, bonds, bo = parse_smiles_or_inchi_to_3d("c1ccccc1", add_hydrogens=True)
    assert len(syms) == 12  # C6H6
    assert coords.shape == (12, 3)
    assert len(bonds) >= 12

    # Test InChI -> 3D coordinates
    syms_eth, coords_eth, bonds_eth, bo_eth = parse_smiles_or_inchi_to_3d(
        "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", add_hydrogens=True
    )
    assert len(syms_eth) == 9  # C2H6O (ethanol)
    assert "O" in syms_eth
    assert coords_eth.shape == (9, 3)

