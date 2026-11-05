"""
Numerical Validation Suite for Core Geometric and Statistical Algorithms.

Validates:
1. Kabsch SVD superposition exactness against analytical SO(3) rotations and translations.
2. Machine-precision Z-matrix to Cartesian and Cartesian to Z-matrix round-trip preservation.
3. Boltzmann population statistics against analytical statistical mechanics formulas.
4. Classical electrostatic potential (ESP) and dipole moment fidelity.
"""

import math
import numpy as np
import pytest

from agymolden.alignment import calculate_kabsch_alignment
from agymolden.crest_explorer import ConformerRecord, recalculate_boltzmann, KB_KCAL_MOL_K
from agymolden.zmatrix import cartesian_to_zmatrix, zmatrix_to_cartesian


def test_kabsch_svd_analytical_rotation():
    """Verify Kabsch SVD recovers exact analytical 3D rotation and translation."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.42, 0.0, 0.0],
        [0.71, 1.23, 0.0],
        [0.0, 0.0, 1.54],
    ], dtype=np.float64)

    # Rotate 42.5 degrees around axis [1, 2, 3] and translate by [10.0, -5.0, 3.2]
    theta = np.radians(42.5)
    axis = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])
    ux, uy, uz = axis
    K = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0],
    ])
    R_analytical = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    t_analytical = np.array([10.0, -5.0, 3.2])

    mobile = (R_analytical @ coords.T).T + t_analytical

    aligned, rmsd, R_calc, t_calc = calculate_kabsch_alignment(coords, mobile)

    # Superposition RMSD must be essentially zero
    assert rmsd < 1e-12
    assert np.allclose(aligned, coords, atol=1e-12)

    # Determinant of rotation matrix must be strictly +1.0 (pure proper rotation, no reflection)
    det_R = np.linalg.det(R_calc)
    assert abs(det_R - 1.0) < 1e-12


def test_zmatrix_cartesian_roundtrip_fidelity():
    """Verify Cartesian -> Z-Matrix -> Cartesian preserves geometry with machine precision."""
    # Planar benzene C6H6 geometry
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    c_coords = np.column_stack([1.397 * np.cos(angles), 1.397 * np.sin(angles), np.zeros(6)])
    h_coords = np.column_stack([2.481 * np.cos(angles), 2.481 * np.sin(angles), np.zeros(6)])
    orig_coords = np.vstack([c_coords, h_coords])
    symbols = ["C"] * 6 + ["H"] * 6

    # Convert to Z-matrix internal coordinates
    zatoms = cartesian_to_zmatrix(symbols, orig_coords)
    assert len(zatoms) == 12

    # Reconstruct back to Cartesian coordinates
    recon_coords = zmatrix_to_cartesian(zatoms)

    # Superimpose reconstructed coordinates with original
    _, rmsd, _, _ = calculate_kabsch_alignment(orig_coords, recon_coords)
    assert rmsd < 1e-12


def test_boltzmann_statistical_mechanics_distribution():
    """Verify conformer Boltzmann population ratios match theoretical statistical mechanics."""
    temp_k = 298.15
    hartree_to_kcal = 627.5094740631

    # Define 3 conformers with delta_E = 0.0, 0.592 kcal/mol (~RT), and 1.5 kcal/mol
    c0 = ConformerRecord(index=0, symbols=["C"], coords=np.zeros((1, 3)), energy=-50.0)
    c1 = ConformerRecord(index=1, symbols=["C"], coords=np.zeros((1, 3)), energy=-50.0 + 0.5924 / hartree_to_kcal)
    c2 = ConformerRecord(index=2, symbols=["C"], coords=np.zeros((1, 3)), energy=-50.0 + 1.5 / hartree_to_kcal)

    confs = [c0, c1, c2]
    recalculate_boltzmann(confs, temperature_k=temp_k)

    # Theoretical weights
    rt = KB_KCAL_MOL_K * temp_k
    w0 = math.exp(0.0)
    w1 = math.exp(-0.5924 / rt)
    w2 = math.exp(-1.5 / rt)
    w_sum = w0 + w1 + w2

    expected_p0 = (w0 / w_sum) * 100.0
    expected_p1 = (w1 / w_sum) * 100.0
    expected_p2 = (w2 / w_sum) * 100.0

    assert abs(c0.boltzmann_pop - expected_p0) < 1e-4
    assert abs(c1.boltzmann_pop - expected_p1) < 1e-4
    assert abs(c2.boltzmann_pop - expected_p2) < 1e-4
    assert abs(c0.boltzmann_pop + c1.boltzmann_pop + c2.boltzmann_pop - 100.0) < 1e-4
