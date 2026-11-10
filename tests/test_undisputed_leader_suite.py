"""
Comprehensive tests for the Undisputed #1 Suite upgrades:
1. Pure-NumPy Marching Tetrahedra fallback when scikit-image is absent (`_pure_numpy_marching_tetrahedra`)
2. Pure-Python Bravais crystal system & Miller slab fallback when spglib/ASE are absent
3. Instant Semiempirical HOMO/LUMO & ESP 3D surface evaluation without external Molden/QM binaries
4. Single-source-of-truth `MoleculeModel` delegation on `MainWindow`
5. Student / Essentials Workspace Mode & Universal MLIP Engine (`mlip_engine.py`)
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from PyQt6 import QtWidgets


def test_pure_numpy_marching_tetrahedra_fallback():
    """Verify 3D isosurface extraction works via pure NumPy without scikit-image."""
    from agymolden.volumetric import _pure_numpy_marching_tetrahedra

    # Create a 3D Gaussian sphere scalar field
    nx, ny, nz = 20, 20, 20
    x = np.linspace(-3.0, 3.0, nx, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    grid = np.exp(-(X**2 + Y**2 + Z**2))

    verts, faces, normals = _pure_numpy_marching_tetrahedra(grid, level=0.3)
    assert len(verts) > 20
    assert len(faces) > 10
    assert normals.shape == verts.shape
    assert np.all(np.isfinite(verts))
    assert np.all(np.isfinite(normals))


def test_pbc_pure_python_fallbacks():
    """Verify space group classification and Miller surface slab generation work in pure NumPy."""
    from agymolden.pbc import UnitCell, _classify_bravais_crystal_system, _pure_numpy_surface_slab

    cubic_cell = UnitCell(a=5.0, b=5.0, c=5.0, alpha=90.0, beta=90.0, gamma=90.0)
    intl, pg, num = _classify_bravais_crystal_system(cubic_cell)
    assert intl == "Pm-3m"
    assert num == 221

    hex_cell = UnitCell(a=4.0, b=4.0, c=6.5, alpha=90.0, beta=90.0, gamma=120.0)
    intl_h, pg_h, num_h = _classify_bravais_crystal_system(hex_cell)
    assert intl_h == "P6/mmm"
    assert num_h == 191

    # Pure-NumPy surface slab (1, 1, 1)
    syms = ["Cu", "Cu"]
    coords = np.array([[0.0, 0.0, 0.0], [2.5, 2.5, 0.0]])
    slab_syms, slab_coords, slab_cell = _pure_numpy_surface_slab(
        syms, coords, cubic_cell, hkl=(1, 0, 0), layers=2, vacuum=8.0
    )
    assert len(slab_syms) >= 2
    assert slab_coords.shape == (len(slab_syms), 3)
    assert slab_cell.c > 16.0


def test_instant_semiempirical_orbitals_and_esp():
    """Verify 1-click HOMO/LUMO and ESP surface generation work without any external Molden file."""
    from agymolden.semiempirical import (
        compute_semiempirical_properties,
        evaluate_semiempirical_orbital_grid,
        compute_semiempirical_esp_mapped_density,
    )

    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.000, 0.000, 0.117],
        [0.000, 0.757, -0.469],
        [0.000, -0.757, -0.469],
    ])
    props = compute_semiempirical_properties(symbols, coords)
    assert "eigvecs" in props
    assert "eigvals_ev" in props

    # Evaluate HOMO grid (MO #3 for water: 8 valence electrons -> 4 occupied MOs, index 3)
    vol = evaluate_semiempirical_orbital_grid(symbols, coords, orbital_index=3, nx=24, ny=24, nz=24, qm_props=props)
    assert vol is not None
    assert vol.grid_data.shape == (24, 24, 24)
    assert float(np.max(np.abs(vol.grid_data))) > 1e-3

    # Evaluate instant ESP-mapped density surface
    esp_mesh = compute_semiempirical_esp_mapped_density(symbols, coords, nx=24, ny=24, nz=24, qm_props=props)
    assert esp_mesh is not None
    assert len(esp_mesh.vertices) > 10
    assert esp_mesh.vertex_colors is not None
    assert esp_mesh.vertex_colors.shape == (len(esp_mesh.vertices), 3)


def test_mlip_engine_optimization():
    """Verify Universal MLIP engine optimizes water cleanly with SCC-TB+D3 fallback."""
    from agymolden.mlip_engine import optimize_geometry_mlip, detect_available_mlip_backends

    backends = detect_available_mlip_backends()
    assert backends["builtin_scc_tb"] is True

    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.000, 0.000, 0.000],
        [1.150, 0.000, 0.000],
        [-0.300, 1.100, 0.000],
    ])
    opt_coords, final_e, props = optimize_geometry_mlip(symbols, coords, max_iter=60)
    assert opt_coords.shape == (3, 3)
    assert np.all(np.isfinite(opt_coords))
    assert "mlip_backend" in props
    # O-H bond length should relax from 1.15 A closer to ~0.96-1.02 A
    d_oh = float(np.linalg.norm(opt_coords[1] - opt_coords[0]))
    assert 0.85 <= d_oh <= 1.08


def test_single_source_of_truth_model_and_student_mode():
    """Verify MainWindow delegates zatoms/coords to self.model and Student Mode configures instant QM."""
    from agymolden.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    win.load_preset_molecule("Water (H2O)")

    # Verify single source of truth delegation
    assert win.zatoms is win.model.zatoms
    assert np.shares_memory(win.coords, win.model.coords)
    assert len(win.model.zatoms) == 3

    # Switch to Student / Essentials mode
    for i in range(win.combo_workspace_mode.count()):
        if win.combo_workspace_mode.itemData(i) == "student":
            win.combo_workspace_mode.setCurrentIndex(i)
            break

    assert win._get_selected_opt_method() == "semiempirical"
    assert win.btn_toolbar_brorsen.isHidden() is True

    # Test 1-click HOMO without loading any external Molden file
    win._on_quick_view_homo()
    assert win._active_molden_data is not None
    assert len(win._active_molden_data.orbitals) >= 4
    assert len(win.viewer.active_isosurfaces) >= 1
