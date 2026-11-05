"""Unit tests for PDB and SDF/MOL parser, writer, and connectivity roundtrips."""

import numpy as np
import pytest

from agymolden.zmatrix import (
    parse_pdb,
    write_pdb,
    parse_mol_sdf,
    write_mol_sdf,
    cartesian_to_zmatrix,
    zmatrix_to_cartesian,
)


def test_pdb_writer_and_parser_roundtrip(tmp_path):
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ])
    bonds = [(0, 1), (0, 2)]
    pdb_path = str(tmp_path / "water.pdb")

    # Write PDB
    write_pdb(pdb_path, symbols, coords, bonds=bonds, title="Water Test")

    # Parse PDB back
    p_syms, p_coords, p_bonds = parse_pdb(pdb_path, is_path=True)
    assert p_syms == ["O", "H", "H"]
    assert p_coords.shape == (3, 3)
    np.testing.assert_allclose(p_coords, coords, atol=1e-3)
    assert (0, 1) in p_bonds
    assert (0, 2) in p_bonds


def test_sdf_writer_and_parser_roundtrip(tmp_path):
    # Ethylene with a double bond
    symbols = ["C", "C", "H", "H", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.665],
        [0.0, 0.0, -0.665],
        [0.0, 0.923, 1.232],
        [0.0, -0.923, 1.232],
        [0.0, 0.923, -1.232],
        [0.0, -0.923, -1.232],
    ])
    bonds = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
    bo_dict = {(0, 1): 2, (0, 2): 1, (0, 3): 1, (1, 4): 1, (1, 5): 1}
    sdf_path = str(tmp_path / "ethylene.sdf")

    # Write SDF
    write_mol_sdf(sdf_path, symbols, coords, bonds=bonds, bond_orders=bo_dict, title="Ethylene")

    # Parse SDF back
    s_syms, s_coords, s_bonds, s_bo = parse_mol_sdf(sdf_path, is_path=True)
    assert s_syms == symbols
    assert s_coords.shape == (6, 3)
    np.testing.assert_allclose(s_coords, coords, atol=1e-4)
    assert len(s_bonds) == 5
    assert (0, 1) in s_bonds
    assert s_bo.get((0, 1)) == 2  # Double bond preserved


def test_pdb_sdf_to_zmatrix_roundtrip(tmp_path):
    symbols = ["C", "H", "H", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.09],
        [1.03, 0.0, -0.36],
        [-0.51, -0.89, -0.36],
        [-0.51, 0.89, -0.36],
    ])
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
    pdb_path = str(tmp_path / "methane.pdb")
    write_pdb(pdb_path, symbols, coords, bonds=bonds)

    s, c, b = parse_pdb(pdb_path)
    zatoms = cartesian_to_zmatrix(s, c, user_bonds=b, strict_bonds=True)
    assert len(zatoms) == 5
    c_recon = zmatrix_to_cartesian(zatoms)
    # Check that pairwise internal distances match exactly
    for (i, j) in bonds:
        d_orig = np.linalg.norm(coords[i] - coords[j])
        d_recon = np.linalg.norm(c_recon[i] - c_recon[j])
        assert pytest.approx(d_orig, abs=1e-3) == d_recon


def test_spectrum_vector_exports(tmp_path):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(5, 3), dpi=90)
    ax = fig.add_subplot(111)
    ax.plot([400, 1500, 3000], [0.1, 0.8, 0.3], color="crimson")
    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Intensity")

    pdf_out = str(tmp_path / "plot.pdf")
    svg_out = str(tmp_path / "plot.svg")
    png_out = str(tmp_path / "plot.png")

    fig.savefig(pdf_out, dpi=300, bbox_inches="tight")
    fig.savefig(svg_out, bbox_inches="tight")
    fig.savefig(png_out, dpi=300, bbox_inches="tight")

    assert (tmp_path / "plot.pdf").stat().st_size > 500
    assert (tmp_path / "plot.svg").stat().st_size > 500
    assert (tmp_path / "plot.png").stat().st_size > 500


def test_parse_cif_crystal_structure(tmp_path):
    from agymolden.zmatrix import parse_cif
    cif_content = """
    data_rutile
    _cell_length_a                     4.5937(5)
    _cell_length_b                     4.5937(5)
    _cell_length_c                     2.9587(3)
    _cell_angle_alpha                  90.00
    _cell_angle_beta                   90.00
    _cell_angle_gamma                  90.00

    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Ti1 Ti 0.0000 0.0000 0.0000
    Ti2 Ti 0.5000 0.5000 0.5000
    O1  O  0.3053 0.3053 0.0000
    O2  O  0.6947 0.6947 0.0000
    """
    cif_file = tmp_path / "rutile.cif"
    cif_file.write_text(cif_content, encoding="utf-8")

    symbols, coords, cell = parse_cif(str(cif_file))
    assert symbols == ["Ti", "Ti", "O", "O"]
    assert coords.shape == (4, 3)
    assert cell["a"] == pytest.approx(4.5937, rel=1e-4)
    assert cell["c"] == pytest.approx(2.9587, rel=1e-4)

    # Ti1 at (0, 0, 0)
    np.testing.assert_allclose(coords[0], [0.0, 0.0, 0.0], atol=1e-5)
    # Ti2 at (0.5*a, 0.5*b, 0.5*c)
    np.testing.assert_allclose(coords[1], [0.5 * 4.5937, 0.5 * 4.5937, 0.5 * 2.9587], atol=1e-4)


