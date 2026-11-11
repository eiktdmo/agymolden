"""Unit and regression tests for Catalysis Steric Descriptors (%V_bur, Sterimol)
and Marcus Electron Transfer kinetics workbench.
"""

import numpy as np
import pytest

from agymolden.sterics import (
    compute_buried_volume,
    compute_sterimol,
)
from agymolden.dialog_marcus import compute_marcus_kinetics


def test_buried_volume_water_dimer():
    """Test buried volume computation around oxygen in water dimer or coordination toy."""
    # Oxygen at origin, two H's attached
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.95, 0.0, 0.0],
        [-0.24, 0.92, 0.0],
    ])

    # Buried volume around atom 0 (O) with radius 3.5 Å, excluding O
    res = compute_buried_volume(symbols, coords, center_idx=0, sphere_radius=3.5, mesh_spacing=0.15)
    assert 0.0 < res.percent_v_bur < 100.0
    # Water hydrogens should occupy a small but non-zero volume within 3.5 A
    assert 1.0 < res.percent_v_bur < 25.0
    assert res.steric_map_grid is not None
    assert res.steric_map_grid.ndim == 2
    # Check quadrant structure
    assert "Q1_NE" in res.quadrants
    assert "Q2_NW" in res.quadrants
    assert "Q3_SW" in res.quadrants
    assert "Q4_SE" in res.quadrants


def test_sterimol_methane_c_h():
    """Test Verloop Sterimol (L, B1, B5) along C-H bond."""
    symbols = ["C", "H", "H", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.09, 0.0, 0.0],
        [-0.36, 1.03, 0.0],
        [-0.36, -0.51, 0.89],
        [-0.36, -0.51, -0.89],
    ])

    res = compute_sterimol(symbols, coords, atom_base_idx=0, atom_substituent_idx=1)
    # L length along primary axis must be >= 1.09 + vdw(H)
    assert res.l_value > 1.5
    # B1 <= B5 by definition
    assert res.b1_value <= res.b5_value
    assert res.b1_value > 0.0


def test_marcus_theory_kinetics():
    """Test Marcus electron transfer calculations in normal and inverted regimes."""
    # Test 1: Normal activation regime: DeltaG0 = -0.3 eV, lambda_total = 0.8 eV
    res_norm = compute_marcus_kinetics(
        delta_g0_ev=-0.3,
        lambda_in_ev=0.3,
        lambda_out_ev=0.5,
        v_da_cm1=50.0,
        temp_k=298.15,
    )
    assert res_norm["regime"] == "Normal Activation (|ΔG°| < λ)"
    assert res_norm["lambda_total_ev"] == 0.8
    assert res_norm["delta_g_act_ev"] > 0.0
    assert res_norm["k_et_s1"] > 0.0

    # Test 2: Activationless regime: DeltaG0 = -0.8 eV, lambda_total = 0.8 eV
    res_actless = compute_marcus_kinetics(
        delta_g0_ev=-0.8,
        lambda_in_ev=0.4,
        lambda_out_ev=0.4,
        v_da_cm1=50.0,
        temp_k=298.15,
    )
    assert "Activationless" in res_actless["regime"]
    assert pytest.approx(res_actless["delta_g_act_ev"], abs=1e-5) == 0.0

    # Test 3: Inverted regime: DeltaG0 = -1.5 eV, lambda_total = 0.8 eV
    res_inv = compute_marcus_kinetics(
        delta_g0_ev=-1.5,
        lambda_in_ev=0.4,
        lambda_out_ev=0.4,
        v_da_cm1=50.0,
        temp_k=298.15,
    )
    assert "Inverted" in res_inv["regime"]
    assert res_inv["delta_g_act_ev"] > 0.0
