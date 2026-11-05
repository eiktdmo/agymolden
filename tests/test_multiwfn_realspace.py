"""Tests for Multiwfn & Real-Space Grid Wavefunction Analysis Engine.

Verifies:
1. Real-space scalar grid classification (ELF, LOL, RDG/NCI, ESP, Density).
2. Multiwfn/NCIPLOT standard color mapping across attractive, vdW, and steric regimes.
3. Generation of colored NCI isosurface meshes with companion sign(lambda_2)*rho scalar field.
"""

import numpy as np
import pytest

from agymolden.volumetric import VolumetricData
from agymolden.multiwfn_realspace import (
    classify_real_space_grid,
    compute_nci_color_mapping,
    create_nci_isosurface_mesh,
    RealSpaceFieldInfo,
)


def _make_dummy_grid(title: str, comment: str, min_val: float, max_val: float) -> VolumetricData:
    nx, ny, nz = 10, 10, 10
    data = np.linspace(min_val, max_val, nx * ny * nz, dtype=np.float32).reshape((nx, ny, nz))
    return VolumetricData(
        title=title,
        comment=comment,
        symbols=["O", "H", "H"],
        atom_coords=np.array([[0,0,0], [0,0.7,0], [0,-0.7,0]], dtype=float),
        origin=np.array([-2.0, -2.0, -2.0]),
        voxels_x=nx,
        voxels_y=ny,
        voxels_z=nz,
        step_x=np.array([0.4, 0, 0]),
        step_y=np.array([0, 0.4, 0]),
        step_z=np.array([0, 0, 0.4]),
        grid_data=data,
        is_orbital=False,
    )


def test_real_space_grid_classification():
    """Verify classification of ELF, LOL, NCI/RDG, and ESP cube grids."""
    # ELF
    v_elf = _make_dummy_grid("Water ELF", "Electron Localization Function grid from Multiwfn", 0.0, 1.0)
    info_elf = classify_real_space_grid(v_elf)
    assert info_elf.field_type == "ELF"
    assert info_elf.recommended_isovalue == 0.80

    # LOL
    v_lol = _make_dummy_grid("Benzene LOL", "Localized Orbital Locator", 0.0, 0.95)
    info_lol = classify_real_space_grid(v_lol)
    assert info_lol.field_type == "LOL"
    assert info_lol.recommended_isovalue == 0.55

    # NCI / RDG
    v_rdg = _make_dummy_grid("NCI analysis", "Reduced density gradient s(r)", 0.0, 2.5)
    info_rdg = classify_real_space_grid(v_rdg)
    assert info_rdg.field_type == "NCI_RDG"
    assert info_rdg.recommended_isovalue == 0.50

    # ESP
    v_esp = _make_dummy_grid("Water ESP", "Electrostatic Potential", -0.05, 0.05)
    info_esp = classify_real_space_grid(v_esp)
    assert info_esp.field_type == "ESP"


def test_nci_color_mapping_regimes():
    """Verify that attractive regions are blue, vdW are green, and steric clashes are red."""
    test_vals = np.array([
        -0.03,  # Strong attractive H-bond -> Blue
        -0.01,  # Weak attractive -> Cyan
         0.00,  # Pure vdW dispersion -> Green
         0.01,  # Weak repulsion -> Yellow/Orange
         0.03,  # Strong steric repulsion -> Red
    ], dtype=np.float32)

    colors = compute_nci_color_mapping(test_vals)
    assert colors.shape == (5, 3)

    # Index 0: Blue dominant (B > R)
    assert colors[0, 2] > colors[0, 0]

    # Index 2: Green dominant (G > R and G > B)
    assert colors[2, 1] > colors[2, 0]
    assert colors[2, 1] > colors[2, 2]

    # Index 4: Red dominant (R > B)
    assert colors[4, 0] > colors[4, 2]


def test_create_nci_isosurface_mesh():
    """Verify NCI mesh generation and property interpolation."""
    nx, ny, nz = 16, 16, 16
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    z = np.linspace(-2, 2, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    # Synthetic sphere with s = r
    rdg_data = np.sqrt(X**2 + Y**2 + Z**2).astype(np.float32)
    rho_data = (-0.02 + 0.04 * (X / 2.0)).astype(np.float32)

    v_rdg = VolumetricData(
        title="RDG", comment="", symbols=["O"], atom_coords=np.array([[0,0,0]], dtype=float),
        origin=np.array([-2.0, -2.0, -2.0]), voxels_x=nx, voxels_y=ny, voxels_z=nz,
        step_x=np.array([4.0/nx, 0, 0]), step_y=np.array([0, 4.0/ny, 0]), step_z=np.array([0, 0, 4.0/nz]),
        grid_data=rdg_data, is_orbital=False
    )
    v_rho = VolumetricData(
        title="SignLambda2Rho", comment="", symbols=["O"], atom_coords=np.array([[0,0,0]], dtype=float),
        origin=np.array([-2.0, -2.0, -2.0]), voxels_x=nx, voxels_y=ny, voxels_z=nz,
        step_x=np.array([4.0/nx, 0, 0]), step_y=np.array([0, 4.0/ny, 0]), step_z=np.array([0, 0, 4.0/nz]),
        grid_data=rho_data, is_orbital=False
    )

    mesh = create_nci_isosurface_mesh(v_rdg, v_rho, rdg_isovalue=1.2)
    assert mesh is not None
    assert len(mesh.vertices) > 0
    assert mesh.vertex_colors is not None
    assert mesh.vertex_colors.shape == (len(mesh.vertices), 3)
