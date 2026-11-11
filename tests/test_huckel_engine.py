"""Comprehensive unit tests for Hückel Molecular Orbital (HMO) Theory Engine.
Tests:
- Ethylene (analytical: E = alpha +/- beta)
- Butadiene (analytical: E = alpha +/- 1.618 beta, alpha +/- 0.618 beta)
- Benzene (Frost circle: E = alpha + 2 beta, alpha + beta (x2), alpha - beta (x2), alpha - 2 beta)
- Cyclobutadiene (antiaromatic 4pi diradical with non-bonding SOMOs)
- Delocalization energy & Coulson bond orders
"""

import math
import numpy as np
import pytest

from agymolden.huckel_engine import detect_pi_system, solve_huckel_system


def test_huckel_ethylene():
    symbols = ["C", "C"]
    pi_idx = [0, 1]
    pi_bonds = [(0, 1)]

    res = solve_huckel_system(symbols, pi_idx, pi_bonds, custom_electrons=2)
    assert len(res.orbitals) == 2
    # m_beta sorted descending: bonding m = +1.0, antibonding m = -1.0
    assert pytest.approx(res.orbitals[0].m_beta, abs=1e-4) == 1.0
    assert pytest.approx(res.orbitals[1].m_beta, abs=1e-4) == -1.0
    # Occupancy
    assert res.orbitals[0].occupancy == 2
    assert res.orbitals[1].occupancy == 0
    # Total pi energy: 2*(alpha + 1*beta) = 2 alpha + 2 beta
    assert pytest.approx(res.total_pi_energy_beta, abs=1e-4) == 2.0
    # Delocalization energy for ethylene is 0
    assert pytest.approx(res.delocalization_energy_beta, abs=1e-4) == 0.0
    # Coulson pi-bond order for double bond = 1.0
    assert pytest.approx(res.coulson_bond_orders[(0, 1)], abs=1e-4) == 1.0


def test_huckel_13_butadiene():
    # Butadiene: 4 carbons, 4 pi electrons
    # Golden ratio roots: m = (sqrt(5)+1)/2 = 1.618, (sqrt(5)-1)/2 = 0.618, -0.618, -1.618
    symbols = ["C", "C", "C", "C"]
    pi_idx = [0, 1, 2, 3]
    pi_bonds = [(0, 1), (1, 2), (2, 3)]

    res = solve_huckel_system(symbols, pi_idx, pi_bonds, custom_electrons=4)
    assert len(res.orbitals) == 4
    assert pytest.approx(res.orbitals[0].m_beta, abs=1e-3) == 1.618
    assert pytest.approx(res.orbitals[1].m_beta, abs=1e-3) == 0.618
    assert pytest.approx(res.orbitals[2].m_beta, abs=1e-3) == -0.618
    assert pytest.approx(res.orbitals[3].m_beta, abs=1e-3) == -1.618

    # Occupancies: 2, 2, 0, 0
    assert res.orbitals[0].occupancy == 2
    assert res.orbitals[1].occupancy == 2
    assert res.orbitals[2].occupancy == 0
    assert res.orbitals[3].occupancy == 0

    # Total Pi Energy = 2*(1.618) + 2*(0.618) = 4.472 beta
    assert pytest.approx(res.total_pi_energy_beta, abs=1e-3) == 4.472
    # Delocalization Energy DE = 4.472 - 4.000 = 0.472 beta
    assert pytest.approx(res.delocalization_energy_beta, abs=1e-3) == 0.472

    # Coulson bond orders: central C2-C3 has partial double bond character (0.447)
    assert pytest.approx(res.coulson_bond_orders[(0, 1)], abs=1e-3) == 0.894
    assert pytest.approx(res.coulson_bond_orders[(1, 2)], abs=1e-3) == 0.447
    assert pytest.approx(res.coulson_bond_orders[(2, 3)], abs=1e-3) == 0.894


def test_huckel_benzene_aromaticity():
    # Benzene: 6 carbons, 6 pi electrons
    # Roots: m = 2 (x1), 1 (x2 degenerate), -1 (x2 degenerate), -2 (x1)
    symbols = ["C"] * 6
    pi_idx = list(range(6))
    pi_bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

    res = solve_huckel_system(symbols, pi_idx, pi_bonds, custom_electrons=6)
    assert "Aromatic" in res.aromaticity_status
    assert res.is_cyclic is True
    assert res.frost_polygon_n == 6

    # Verify roots
    m_vals = [mo.m_beta for mo in res.orbitals]
    assert pytest.approx(m_vals[0], abs=1e-4) == 2.0
    assert pytest.approx(m_vals[1], abs=1e-4) == 1.0
    assert pytest.approx(m_vals[2], abs=1e-4) == 1.0
    assert pytest.approx(m_vals[3], abs=1e-4) == -1.0
    assert pytest.approx(m_vals[4], abs=1e-4) == -1.0
    assert pytest.approx(m_vals[5], abs=1e-4) == -2.0

    # Total energy = 2*(2.0) + 4*(1.0) = 8.0 beta
    assert pytest.approx(res.total_pi_energy_beta, abs=1e-4) == 8.0
    # DE = 8.0 - 6.0 = 2.0 beta (famous Hückel resonance energy of benzene)
    assert pytest.approx(res.delocalization_energy_beta, abs=1e-4) == 2.0

    # All C-C bond orders are exactly equal to 2/3 ≈ 0.6667
    for b in pi_bonds:
        assert pytest.approx(res.coulson_bond_orders[b], abs=1e-3) == 0.667


def test_huckel_cyclobutadiene_antiaromaticity():
    # Cyclobutadiene: 4 carbons, 4 pi electrons
    # Roots: m = 2, 0 (x2 degenerate non-bonding), -2
    symbols = ["C"] * 4
    pi_idx = list(range(4))
    pi_bonds = [(0, 1), (1, 2), (2, 3), (3, 0)]

    res = solve_huckel_system(symbols, pi_idx, pi_bonds, custom_electrons=4)
    assert "Antiaromatic" in res.aromaticity_status
    # Verify non-bonding SOMOs
    m_vals = [mo.m_beta for mo in res.orbitals]
    assert pytest.approx(m_vals[0], abs=1e-4) == 2.0
    assert pytest.approx(m_vals[1], abs=1e-4) == 0.0
    assert pytest.approx(m_vals[2], abs=1e-4) == 0.0
    assert pytest.approx(m_vals[3], abs=1e-4) == -2.0

    # Total energy = 2*(2.0) + 1*(0.0) + 1*(0.0) = 4.0 beta
    # Delocalization Energy DE = 4.0 - 4.0 = 0.0 beta (Zero stabilization!)
    assert pytest.approx(res.delocalization_energy_beta, abs=1e-4) == 0.0
