"""Comprehensive unit and regression tests for Quantum Chemistry Quintet:
1. Open-shell spin contamination & Yamaguchi projection (spin_tools.py)
2. Electrochemical redox potentials & Born-Haber cycle (electrochemistry.py)
3. Minimum energy crossing point (MECP) interpolation & ORCA generator (mecp_tools.py)
4. Energy decomposition analysis (EDA) & SAPT parser (eda_tools.py)
5. 2D Relaxed PES scan grid generator (pes_2d_tools.py)
"""

import numpy as np
import pytest

from agymolden.spin_tools import evaluate_spin_contamination, compute_yamaguchi_projection
from agymolden.electrochemistry import compute_redox_potential, REFERENCE_ELECTRODES
from agymolden.mecp_tools import interpolate_mecp_seam, generate_orca_mecp_input
from agymolden.eda_tools import parse_eda_text
from agymolden.pes_2d_tools import (
    ScanCoordinateDef,
    generate_orca_2d_scan_input,
    generate_gaussian_2d_scan_input,
)


def test_spin_contamination_evaluation():
    # Doublet (S = 0.5, exact <S^2> = 0.75)
    res_clean = evaluate_spin_contamination(multiplicity=2, s2_calc=0.76)
    assert not res_clean.is_contaminated
    assert res_clean.s2_exact == 0.75
    assert res_clean.unpaired_electrons == 1
    assert res_clean.contamination_percent < 5.0

    # Severe contamination (e.g. <S^2> = 1.15 for a doublet)
    res_contam = evaluate_spin_contamination(multiplicity=2, s2_calc=1.15)
    assert res_contam.is_contaminated
    assert res_contam.contamination_percent > 50.0


def test_yamaguchi_projection_and_j_coupling():
    # Broken-symmetry diradical: E(BS) = -100.005 Ha (<S^2> = 1.05), E(HS) = -100.000 Ha (<S^2> = 2.05)
    res = compute_yamaguchi_projection(
        e_bs_ha=-100.005,
        e_hs_ha=-100.000,
        s2_bs=1.05,
        s2_hs=2.05,
    )
    # E_AP must be lower than E_BS for an antiferromagnetically coupled singlet
    assert res.e_ap_ha < -100.005
    assert res.e_ap_delta_kcal < 0.0
    # J must be negative (antiferromagnetic)
    assert res.j_yamaguchi_cm1 < 0.0
    assert "Antiferromagnetic" in res.coupling_type


def test_electrochemical_redox_potential():
    # 1e- reduction: G(Ox) = -150.0 Ha, G(Red) = -150.15 Ha
    # Solvation free energies: dG_solv(Ox) = -10 kcal/mol, dG_solv(Red) = -60 kcal/mol
    res = compute_redox_potential(
        g_ox_gas_ha=-150.0,
        g_red_gas_ha=-150.15,
        dg_solv_ox_kcal=-10.0,
        dg_solv_red_kcal=-60.0,
        electrons_n=1,
        she_absolute_v=4.281,
    )
    assert res.dg_gas_kcal < 0.0
    assert res.ddg_solv_kcal == -50.0
    assert res.dg_total_solv_kcal < res.dg_gas_kcal
    assert isinstance(res.e0_she_v, float)
    assert "SCE (Sat. Calomel, +0.241 V vs SHE)" in res.potentials_vs_ref
    assert "Fc/Fc+ (Ferrocene in MeCN, +0.641 V vs SHE)" in res.potentials_vs_ref


def test_mecp_seam_interpolation():
    c1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    c2 = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])

    res = interpolate_mecp_seam(c1, c2, e_state1_ref_ev=0.0, e_state2_ref_ev=0.3, n_points=51)
    assert len(res.coordinates) == 51
    assert 0 <= res.crossing_index < 51
    assert 0.0 <= res.crossing_lambda <= 1.0
    assert res.energy_diff_ev[res.crossing_index] < 0.25
    assert pytest.approx(res.crossing_lambda, abs=0.05) == 0.525

    # Check ORCA deck generation
    deck = generate_orca_mecp_input(["Fe", "O"], c1, mult_1=1, mult_2=3)
    assert "SurfCross" in deck
    assert "State1 0 1" in deck
    assert "State2 0 3" in deck


def test_eda_sapt_parser():
    sample_text = """
    -------------------------------------------------------------------
    SAPT Results
    -------------------------------------------------------------------
    Electrostatics:   -8.4500 kcal/mol
    Exchange:         12.3000 kcal/mol
    Induction:        -3.1500 kcal/mol
    Dispersion:       -4.6000 kcal/mol
    Total SAPT0:      -3.9000 kcal/mol
    -------------------------------------------------------------------
    """
    res = parse_eda_text(sample_text)
    assert res is not None
    assert res.method == "Psi4 SAPT"
    assert res.elstat_kcal == -8.45
    assert res.pauli_kcal == 12.30
    assert res.orb_kcal == -3.15
    assert res.disp_kcal == -4.60
    assert res.total_int_kcal == -3.90


def test_pes_2d_scan_generator():
    symbols = ["C", "C", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.34, 0.0, 0.0],
        [-0.5, 0.9, 0.0],
        [1.8, 0.9, 0.0],
    ])
    scan1 = ScanCoordinateDef("distance", (0, 1), 1.2, 1.8, 6)
    scan2 = ScanCoordinateDef("dihedral", (2, 0, 1, 3), 0.0, 180.0, 12)

    # ORCA test
    orca_deck = generate_orca_2d_scan_input(symbols, coords, scan1, scan2)
    assert "%geom" in orca_deck
    assert "Scan B 0 1 = 1.200, 1.800, 6" in orca_deck
    assert "Scan D 2 0 1 3 = 0.000, 180.000, 12" in orca_deck

    # Gaussian test
    gau_deck = generate_gaussian_2d_scan_input(symbols, coords, scan1, scan2)
    assert "Opt=ModRedundant" in gau_deck
    # Gaussian uses 1-based indexing
    assert "B 1 2 S 6" in gau_deck
    assert "D 3 1 2 4 S 12" in gau_deck
