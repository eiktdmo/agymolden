"""
Tests for Volumetric Grid & Isosurface extraction:
1. Parsing Gaussian .cube files (Electron density, MOs)
2. Parsing Molden .molden format files ([Atoms], [MO])
3. Marching Cubes isosurface mesh generation (positive and negative orbital lobes)
"""

import tempfile
import numpy as np
import pytest

from agymolden.volumetric import (
    parse_cube_file,
    parse_molden_file,
    extract_isosurface,
    generate_orbital_surfaces,
    VolumetricData,
    _HAS_MARCHING_CUBES,
)


def test_cube_file_parsing_and_marching_cubes():
    if not _HAS_MARCHING_CUBES:
        pytest.skip("scikit-image marching_cubes not available")

    # Construct synthetic Gaussian .cube file for H2 molecule
    cube_content = """CP-DFT Calculation
MO #1 Orbital
  2   -2.000000   -2.000000   -2.000000
 10    0.400000    0.000000    0.000000
 10    0.000000    0.400000    0.000000
 10    0.000000    0.000000    0.400000
  1    1.000000    0.000000    0.000000   -0.700000
  1    1.000000    0.000000    0.000000    0.700000
"""
    # 1000 grid points
    grid_vals = []
    for x in np.linspace(-2, 2, 10):
        for y in np.linspace(-2, 2, 10):
            for z in np.linspace(-2, 2, 10):
                # Harmonic p_z-like orbital lobe (+ on z > 0, - on z < 0)
                val = z * np.exp(-(x**2 + y**2 + z**2) / 1.5)
                grid_vals.append(f"{val:12.5E}")

    # Format 6 numbers per line
    for i in range(0, len(grid_vals), 6):
        cube_content += " ".join(grid_vals[i:i+6]) + "\n"

    with tempfile.NamedTemporaryFile("w", suffix=".cube", delete=False) as f:
        f.write(cube_content)
        cube_path = f.name

    vol = parse_cube_file(cube_path)
    assert vol.voxels_x == 10
    assert vol.voxels_y == 10
    assert vol.voxels_z == 10
    assert len(vol.symbols) == 2
    assert vol.symbols == ["H", "H"]
    assert vol.grid_data.shape == (10, 10, 10)

    # Test Marching Cubes extraction
    lobes = generate_orbital_surfaces(vol, isovalue=0.08)
    assert len(lobes) == 2  # positive and negative lobes
    pos_lobe, neg_lobe = lobes[0], lobes[1]
    assert len(pos_lobe.vertices) > 0
    assert len(pos_lobe.faces) > 0
    assert len(neg_lobe.vertices) > 0


def test_molden_file_parsing():
    molden_content = """[Molden Format]
[Title]
Water Test
[Atoms] (AU)
O 1 8 0.000000 0.000000 0.221700
H 2 1 0.000000 1.437500 -0.887000
H 3 1 0.000000 -1.437500 -0.887000
[FREQ]
 1595.0
 3657.0
 3756.0
[MO]
 Sym= 1A1
 Ene= -20.5500
 Spin= Alpha
 Occup= 2.000000
   1   0.99420
   2   0.02580
 Sym= 2A1
 Ene= -1.3500
 Spin= Alpha
 Occup= 2.000000
   1  -0.23300
   2   0.84400
 Sym= 1B2
 Ene= -0.7100
 Spin= Alpha
 Occup= 2.000000
   3   0.61200
"""
    with tempfile.NamedTemporaryFile("w", suffix=".molden", delete=False) as f:
        f.write(molden_content)
        molden_path = f.name

    data = parse_molden_file(molden_path)
    assert len(data.symbols) == 3
    assert data.symbols == ["O", "H", "H"]
    assert len(data.orbitals) == 3
    assert data.orbitals[0].sym == "1A1"
    assert data.orbitals[0].occupancy == 2.0
    assert data.frequencies == [1595.0, 3657.0, 3756.0]


def test_evaluate_molden_orbital_grid_symmetry():
    pytest.importorskip("pyscf")
    import os
    from agymolden.volumetric import evaluate_molden_orbital_grid, extract_isosurface

    sample_molden = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "C6H6_OPT_HF_cc-pvdz_hf_electrons.molden")
    if not os.path.exists(sample_molden):
        pytest.skip(f"{sample_molden} not found")

    vol = evaluate_molden_orbital_grid(sample_molden, orbital_index=20, nx=40, ny=40, nz=40)
    assert vol is not None
    assert vol.is_orbital is True
    assert vol.voxels_x == 40
    assert len(vol.symbols) == 12

    mesh_pos = extract_isosurface(vol, isovalue=0.03)
    mesh_neg = extract_isosurface(vol, isovalue=-0.03)
    assert mesh_pos is not None
    assert mesh_neg is not None
    assert len(mesh_pos.faces) == len(mesh_neg.faces)

    # Check center of mass alignment
    atom_com = np.mean(vol.atom_coords, axis=0)
    mesh_pos_com = np.mean(mesh_pos.vertices, axis=0)
    mesh_neg_com = np.mean(mesh_neg.vertices, axis=0)
    assert np.allclose(mesh_pos_com, atom_com, atol=1e-3)
    assert np.allclose(mesh_neg_com, atom_com, atol=1e-3)


def test_advanced_wavefunction_analyses():
    """Verify ESP with extrema, NCI/RDG, ALIE, and MO decomposition."""
    pytest.importorskip("pyscf")
    import os
    from agymolden.volumetric import (
        compute_molden_esp_mapped_density,
        compute_molden_nci_rdg,
        compute_molden_alie_mapped_density,
        compute_molden_orbital_decomposition,
    )

    sample = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "C2H6O_OPT_HF_cc-pvdz_hf_electrons.molden")
    if not os.path.exists(sample):
        pytest.skip(f"{sample} not found")

    # 1. ESP Mapped Density & Extrema
    esp = compute_molden_esp_mapped_density(sample, nx=20, ny=20, nz=20, colormap_name="coolwarm", find_extrema=True)
    assert esp is not None
    assert len(esp.vertices) > 0
    assert esp.vertex_colors is not None
    assert len(esp.vertex_colors) == len(esp.vertices)
    assert esp.extrema_points is not None and len(esp.extrema_points) > 0

    # 2. NCI / RDG
    nci = compute_molden_nci_rdg(sample, nx=20, ny=20, nz=20, rdg_isovalue=0.5)
    assert nci is not None
    assert len(nci.vertices) > 0
    assert nci.vertex_colors is not None

    # 3. ALIE Mapped Density
    alie = compute_molden_alie_mapped_density(sample, nx=20, ny=20, nz=20)
    assert alie is not None
    assert len(alie.vertices) > 0
    assert alie.extrema_points is not None

    # 4. MO Decomposition
    decomp = compute_molden_orbital_decomposition(sample, orbital_index=12)
    assert len(decomp) > 0
    assert decomp[0]["percentage"] > 0.0


def test_unrestricted_beta_orbital_evaluation_and_decomposition():
    """Verify UHF/UKS Molden files with Alpha and Beta spin channels select mo_coeff[1] for Beta orbitals."""
    pytest.importorskip("pyscf")
    from agymolden.volumetric import (
        evaluate_molden_orbital_grid,
        compute_molden_orbital_decomposition,
    )

    uhf_molden = """[Molden Format]
[Title]
UHF H2 Test
[Atoms] (AU)
H 1 1 0.000000 0.000000 -0.700000
H 2 1 0.000000 0.000000  0.700000
[GTO]
  1 0
 s    1 1.00
      0.5000000000D+00  0.1000000000D+01

  2 0
 s    1 1.00
      0.5000000000D+00  0.1000000000D+01

[MO]
 Sym= 1a
 Ene= -0.5000
 Spin= Alpha
 Occup= 1.000000
   1   1.000000
   2   0.000000
 Sym= 2a
 Ene= 0.2000
 Spin= Alpha
 Occup= 0.000000
   1   0.000000
   2   1.000000
 Sym= 1b
 Ene= -0.4500
 Spin= Beta
 Occup= 1.000000
   1   0.000000
   2   1.000000
 Sym= 2b
 Ene= 0.2500
 Spin= Beta
 Occup= 0.000000
   1   1.000000
   2   0.000000
"""
    with tempfile.NamedTemporaryFile("w", suffix=".molden", delete=False) as f:
        f.write(uhf_molden)
        path = f.name

    # Orbital 0 (Alpha #1) is localized on Atom 1 (z < 0); Orbital 2 (Beta #1) is localized on Atom 2 (z > 0)
    vol_alpha = evaluate_molden_orbital_grid(path, orbital_index=0, nx=12, ny=12, nz=12)
    vol_beta = evaluate_molden_orbital_grid(path, orbital_index=2, nx=12, ny=12, nz=12)
    assert vol_alpha is not None and vol_beta is not None
    assert "[Alpha]" in vol_alpha.title
    assert "[Beta]" in vol_beta.title
    assert not np.allclose(vol_alpha.grid_data, vol_beta.grid_data)

    decomp_alpha = compute_molden_orbital_decomposition(path, orbital_index=0)
    decomp_beta = compute_molden_orbital_decomposition(path, orbital_index=2)
    assert len(decomp_alpha) == 2 and len(decomp_beta) == 2
    assert decomp_alpha[0]["atom_idx"] == 0
    assert decomp_beta[0]["atom_idx"] == 1



