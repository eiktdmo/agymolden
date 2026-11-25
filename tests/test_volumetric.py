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


def test_fallback_molden_6d_10f_sp_and_electron_density(tmp_path):
    """Verify 6D/10F Cartesian normalization, sp shells, MO spacing variants, and total electron density."""
    from agymolden.volumetric import (
        BOHR_TO_ANGSTROM,
        _MO_CTX_CACHE,
        compute_molden_electron_density_grid,
        evaluate_molden_orbital_grid,
        extract_isosurface,
    )

    _MO_CTX_CACHE.clear()
    molden_txt = """[Molden Format]
[Atoms] (AU)
C 1 6 0.0 0.0 0.0
[6D]
[10F]
[GTO]
  1 0
 sp 2 1.00
  3.000000 0.600000 0.400000
  0.800000 0.500000 0.700000
 d 1 1.00
  0.500000 1.000000
 f 1 1.00
  0.400000 1.000000

[MO]
 Sym = 1s
 Ene = -10.0
 Spin = Alpha
 Occup = 2.0
   1 1.0
 Sym = 2px
 Ene = -0.5
 Spin = Alpha
 Occup = 2.0
   2 1.0
 Sym = 3d_xx
 Ene = 0.2
 Spin = Alpha
 Occup = 0.0
   5 1.0
 Sym = 3d_xy
 Ene = 0.2
 Spin = Alpha
 Occup = 0.0
   8 1.0
 Sym = 4f_xxx
 Ene = 0.5
 Spin = Alpha
 Occup = 0.0
  11 1.0
 Sym = 4f_xyz
 Ene = 0.5
 Spin = Alpha
 Occup = 0.0
  20 1.0
"""
    m_path = tmp_path / "cart_sp.molden"
    m_path.write_text(molden_txt, encoding="utf-8")

    # Every orbital (s, p, 6D xx/xy, 10F xxx/xyz) must integrate to 1.0
    for idx in range(6):
        vol = evaluate_molden_orbital_grid(str(m_path), orbital_index=idx, nx=35, ny=35, nz=35, margin_bohr=6.0)
        assert vol is not None
        dx = float(np.linalg.norm(vol.step_x)) / BOHR_TO_ANGSTROM
        dy = float(np.linalg.norm(vol.step_y)) / BOHR_TO_ANGSTROM
        dz = float(np.linalg.norm(vol.step_z)) / BOHR_TO_ANGSTROM
        norm = float(np.sum(vol.grid_data ** 2)) * dx * dy * dz
        assert norm == pytest.approx(1.0, abs=0.02)

    # Total electron density from 2 doubly-occupied MOs must integrate to ~4.0 electrons
    dens = compute_molden_electron_density_grid(str(m_path), nx=35, ny=35, nz=35, margin_bohr=6.0)
    assert dens is not None
    assert dens.is_orbital is False
    assert float(np.min(dens.grid_data)) >= 0.0
    dx = float(np.linalg.norm(dens.step_x)) / BOHR_TO_ANGSTROM
    dy = float(np.linalg.norm(dens.step_y)) / BOHR_TO_ANGSTROM
    dz = float(np.linalg.norm(dens.step_z)) / BOHR_TO_ANGSTROM
    n_elec = float(np.sum(dens.grid_data)) * dx * dy * dz
    assert n_elec == pytest.approx(4.0, abs=0.05)

    if _HAS_MARCHING_CUBES:
        mesh = extract_isosurface(dens, isovalue=0.02)
        assert mesh is not None
        assert len(mesh.vertices) > 0


def test_default_margin_prevents_lumo_clipping():
    """Default margin_bohr (5.5 Bohr) must keep benzene LUMO boundary amplitude well below 0.03."""
    pytest.importorskip("pyscf")
    import os
    from agymolden.volumetric import evaluate_molden_orbital_grid

    sample_molden = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "examples", "C6H6_OPT_HF_cc-pvdz_hf_electrons.molden"
    )
    if not os.path.exists(sample_molden):
        pytest.skip(f"{sample_molden} not found")

    vol = evaluate_molden_orbital_grid(sample_molden, orbital_index=21, nx=32, ny=32, nz=32)
    assert vol is not None
    g = vol.grid_data
    boundary_max = float(max(
        np.max(np.abs(g[0, :, :])), np.max(np.abs(g[-1, :, :])),
        np.max(np.abs(g[:, 0, :])), np.max(np.abs(g[:, -1, :])),
        np.max(np.abs(g[:, :, 0])), np.max(np.abs(g[:, :, -1])),
    ))
    assert boundary_max < 0.01


def test_build_density_colormap_slice_mesh():
    """Verify 3D Density Color Map slice mesh generation for log and linear scales across planes."""
    from agymolden.volumetric import VolumetricData, build_density_colormap_slice_mesh

    nx, ny, nz = 20, 20, 20
    xs = np.linspace(-3.0, 3.0, nx)
    ys = np.linspace(-3.0, 3.0, ny)
    zs = np.linspace(-3.0, 3.0, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.exp(-(X ** 2 + Y ** 2 + Z ** 2))
    step = 6.0 / (nx - 1)
    vol = VolumetricData(
        title="Synthetic Density",
        comment="test",
        symbols=["C", "C"],
        atom_coords=np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]),
        origin=np.array([-3.0, -3.0, -3.0]),
        step_x=np.array([step, 0.0, 0.0]),
        step_y=np.array([0.0, step, 0.0]),
        step_z=np.array([0.0, 0.0, step]),
        voxels_x=nx,
        voxels_y=ny,
        voxels_z=nz,
        grid_data=grid,
        is_orbital=False,
    )

    for plane in ("AUTO", "XY", "XZ", "YZ"):
        mesh_log = build_density_colormap_slice_mesh(vol, plane=plane, colormap_name="viridis", log_scale=True, resolution=24)
        assert len(mesh_log.vertices) == 24 * 24
        assert len(mesh_log.faces) == 2 * 23 * 23
        assert mesh_log.vertex_colors is not None and mesh_log.vertex_colors.shape == (24 * 24, 3)
        assert float(np.min(mesh_log.vertex_colors)) >= 0.0 and float(np.max(mesh_log.vertex_colors)) <= 1.0
        assert mesh_log.esp_range is not None and mesh_log.esp_range[0] < mesh_log.esp_range[1]
        assert "log10" in mesh_log.colorbar_title

    mesh_lin = build_density_colormap_slice_mesh(vol, plane="XY", colormap_name="plasma", log_scale=False, resolution=20)
    assert mesh_lin.esp_range is not None and mesh_lin.esp_range[1] > 0.5
    assert "Density" in mesh_lin.colorbar_title

    if _HAS_MARCHING_CUBES:
        from agymolden.volumetric import build_density_mapped_vdw_mesh
        vdw_mesh = build_density_mapped_vdw_mesh(vol, vdw_scale=0.85, colormap_name="viridis", log_scale=False, grid_n=28)
        assert vdw_mesh is not None
        assert len(vdw_mesh.vertices) > 50 and len(vdw_mesh.faces) > 50
        assert vdw_mesh.vertex_colors is not None
        assert vdw_mesh.esp_range is not None and vdw_mesh.esp_range[0] < vdw_mesh.esp_range[1]



def test_elf_far_field_decay_uhf_and_nci_bgr(tmp_path):
    """Verify ELF decays cleanly to 0 at box edges (RHF & UHF) and NCI masks covalent cores with BGR colors."""
    pytest.importorskip("pyscf")
    import os
    from agymolden.volumetric import compute_molden_elf_grid, compute_molden_nci_rdg

    sample = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "examples", "C2H6O_OPT_HF_cc-pvdz_hf_electrons.molden"
    )
    if not os.path.exists(sample):
        pytest.skip(f"{sample} not found")

    elf = compute_molden_elf_grid(sample, nx=24, ny=24, nz=24)
    assert elf is not None
    assert float(np.min(elf.grid_data)) >= 0.0
    assert float(np.max(elf.grid_data)) <= 1.0
    assert float(elf.grid_data[0, 0, 0]) < 0.05

    nci = compute_molden_nci_rdg(sample, nx=24, ny=24, nz=24, rdg_isovalue=0.5)
    assert nci is not None and len(nci.vertices) > 0
    assert nci.vertex_colors is not None
    # True BGR colormap: green channel should be prominent in vdW regions (not all white/gray)
    assert float(np.max(nci.vertex_colors[:, 1])) > 0.5

    # UHF open-shell ELF check
    uhf_molden = """[Molden Format]
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
   1   0.707107
   2   0.707107
 Sym= 1b
 Ene= -0.4000
 Spin= Beta
 Occup= 0.000000
   1   0.707107
   2  -0.707107
"""
    uhf_p = tmp_path / "uhf_elf.molden"
    uhf_p.write_text(uhf_molden, encoding="utf-8")
    elf_uhf = compute_molden_elf_grid(str(uhf_p), nx=14, ny=14, nz=14)
    assert elf_uhf is not None
    assert float(np.min(elf_uhf.grid_data)) >= 0.0
    assert float(np.max(elf_uhf.grid_data)) <= 1.0


def test_positron_species_title_and_annihilation_density(tmp_path):
    """Verify positron density title auto-detection and shared-grid annihilation density rho_- * rho_+."""
    from agymolden.volumetric import (
        _MO_CTX_CACHE,
        compute_molden_annihilation_density_grid,
        compute_molden_density_grid,
    )

    _MO_CTX_CACHE.clear()
    h_elec = """[Molden Format]
[Atoms] (AU)
H 1 1 0.0 0.0 0.0
[GTO]
  1 0
 s 1 1.00
  1.000000 1.000000
[MO]
 Sym= 1s
 Ene= -0.5
 Spin= Alpha
 Occup= 1.0
   1 1.0
"""
    h_pos = """[Molden Format]
[Atoms] (AU)
H 1 1 0.0 0.0 0.0
[GTO]
  1 0
 s 1 1.00
  0.250000 1.000000
[MO]
 Sym= 1s+
 Ene= -0.1
 Spin= Alpha
 Occup= 1.0
   1 1.0
"""
    e_path = tmp_path / "mol_hf_electrons.molden"
    p_path = tmp_path / "mol_hf_positrons.molden"
    e_path.write_text(h_elec, encoding="utf-8")
    p_path.write_text(h_pos, encoding="utf-8")

    v_pos = compute_molden_density_grid(str(p_path), nx=16, ny=16, nz=16)
    assert v_pos is not None
    assert "Positron" in v_pos.title

    v_ann = compute_molden_annihilation_density_grid(str(e_path), str(p_path), nx=15, ny=15, nz=15)
    assert v_ann is not None
    assert "Annihilation" in v_ann.title
    assert float(np.min(v_ann.grid_data)) >= 0.0
    assert float(np.max(v_ann.grid_data)) > 0.01


def test_custom_3atom_plane_orthonormal_frame():
    """Verify ESP2DContourDialog._compute_custom_plane_frame builds a right-handed orthonormal frame."""
    from agymolden.dialog_esp_2d import ESP2DContourDialog

    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    frame = ESP2DContourDialog._compute_custom_plane_frame(coords, "1, 2, 3")
    assert frame is not None
    origin, u_hat, v_hat, n_hat = frame
    assert np.allclose(origin, np.mean(coords, axis=0))
    assert np.isclose(np.linalg.norm(u_hat), 1.0)
    assert np.isclose(np.linalg.norm(v_hat), 1.0)
    assert np.isclose(np.linalg.norm(n_hat), 1.0)
    assert np.isclose(np.dot(u_hat, v_hat), 0.0, atol=1e-7)
    assert np.isclose(np.dot(u_hat, n_hat), 0.0, atol=1e-7)
    assert np.isclose(np.dot(v_hat, n_hat), 0.0, atol=1e-7)
    # Collinear points must return None cleanly
    collinear = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert ESP2DContourDialog._compute_custom_plane_frame(collinear, "1, 2, 3") is None






