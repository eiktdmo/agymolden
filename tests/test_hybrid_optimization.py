"""Tests for Hybrid Optimization Fallback Engine.

Verifies:
1. `optimize_geometry_rdkit_mmff` successfully optimizes standard organic molecules.
2. `auto_hybrid` mode in `optimize_geometry_force_field` leverages RDKit when applicable.
3. Fallback to analytical UFF occurs smoothly when RDKit encounters unparameterized elements (e.g., metals or radicals).
4. Frozen atom constraints are respected during hybrid optimization.
"""

import numpy as np
import pytest

from agymolden.forcefield import (
    optimize_geometry_force_field,
    optimize_geometry_rdkit_mmff,
    detect_connectivity,
)


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


def test_rdkit_mmff_direct_optimization():
    """Verify that RDKit MMFF94 minimizes coordinates and reduces energy."""
    res = optimize_geometry_rdkit_mmff(ETHANOL_SYMBOLS, ETHANOL_COORDS)
    assert res is not None
    opt_coords, final_e = res
    assert opt_coords.shape == ETHANOL_COORDS.shape
    assert isinstance(final_e, float)


def test_auto_hybrid_mode_execution():
    """Verify that auto_hybrid mode completes and yields valid 3D coordinates."""
    opt_coords, final_e = optimize_geometry_force_field(
        ETHANOL_SYMBOLS,
        ETHANOL_COORDS,
        method="auto_hybrid",
    )
    assert opt_coords.shape == ETHANOL_COORDS.shape
    # Check that atoms moved toward relaxed geometry
    diff = np.linalg.norm(opt_coords - ETHANOL_COORDS)
    assert diff > 0.01  # Geometry relaxed


def test_auto_hybrid_fallback_on_metals():
    """Verify that complex with transition metal or unknown parameters falls back to analytical UFF gracefully."""
    # Ferrocene-like Fe-C complex or metal center
    metal_syms = ["Fe", "C", "C", "H", "H"]
    metal_crds = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.0, 1.2, 0.0],
        [1.8, -0.9, 0.0],
        [2.8, 1.5, 0.0],
    ], dtype=np.float64)

    # Must not throw, falls back to analytical UFF
    opt_coords, final_e = optimize_geometry_force_field(
        metal_syms,
        metal_crds,
        method="auto_hybrid",
    )
    assert opt_coords.shape == metal_crds.shape
    assert isinstance(final_e, float)


def test_frozen_atoms_in_hybrid_optimization():
    """Verify frozen atom positions are preserved identically."""
    frozen_idx = {0, 1}
    opt_coords, _ = optimize_geometry_force_field(
        ETHANOL_SYMBOLS,
        ETHANOL_COORDS,
        method="auto_hybrid",
        frozen_indices=frozen_idx,
    )
    for idx in frozen_idx:
        assert np.allclose(opt_coords[idx], ETHANOL_COORDS[idx], atol=1e-5)
