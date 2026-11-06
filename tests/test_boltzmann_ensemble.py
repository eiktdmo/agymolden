"""
Unit tests for Boltzmann Conformer Ensemble & Property Averager.
"""

import pytest
import math
import numpy as np

from agymolden.crest_explorer import ConformerRecord
from agymolden.ensemble_averager import (
    calculate_radius_of_gyration,
    calculate_conformational_entropy,
    calculate_ensemble_statistics,
    R_CAL_MOL_K,
)


def test_radius_of_gyration():
    # Linear diatomic C-C: distance 1.54 Å
    symbols = ["C", "C"]
    coords = np.array([[-0.77, 0.0, 0.0], [0.77, 0.0, 0.0]])
    rg = calculate_radius_of_gyration(symbols, coords)
    assert pytest.approx(rg, 0.01) == 0.77

    # Empty
    assert calculate_radius_of_gyration([], np.empty((0, 3))) == 0.0


def test_conformational_entropy():
    # 1 state -> 0 entropy
    s, dG, neff = calculate_conformational_entropy(np.array([1.0]), 298.15)
    assert pytest.approx(s, 1e-4) == 0.0
    assert pytest.approx(dG, 1e-4) == 0.0
    assert pytest.approx(neff, 1e-4) == 1.0

    # 2 equal states -> S = R * ln(2), N_eff = 2.0
    s2, dG2, neff2 = calculate_conformational_entropy(np.array([0.5, 0.5]), 298.15)
    expected_s = R_CAL_MOL_K * math.log(2.0)
    assert pytest.approx(s2, 1e-3) == expected_s
    assert pytest.approx(neff2, 1e-3) == 2.0
    assert dG2 < 0.0  # -T*S must be negative (stabilizing free energy term)

    # 4 equal states -> N_eff = 4.0
    s4, dG4, neff4 = calculate_conformational_entropy(np.array([0.25, 0.25, 0.25, 0.25]), 298.15)
    assert pytest.approx(neff4, 1e-3) == 4.0


def test_ensemble_statistics():
    c1 = ConformerRecord(
        index=1,
        symbols=["C", "C"],
        coords=np.array([[-0.77, 0.0, 0.0], [0.77, 0.0, 0.0]]),
        energy=-78.0,
        delta_e_kcal=0.0,
        boltzmann_pop=70.0,
        cumulative_pop=70.0,
    )
    c2 = ConformerRecord(
        index=2,
        symbols=["C", "C"],
        coords=np.array([[-0.80, 0.0, 0.0], [0.80, 0.0, 0.0]]),
        energy=-77.998,
        delta_e_kcal=1.2,
        boltzmann_pop=30.0,
        cumulative_pop=100.0,
    )

    stats = calculate_ensemble_statistics([c1, c2], temperature_k=298.15)
    assert stats["n_total"] == 2
    assert stats["n_within_3kcal"] == 2
    assert stats["n_eff"] > 1.0
    assert stats["s_conf_cal"] > 0.0
    assert stats["minus_t_s_kcal"] < 0.0
    assert pytest.approx(stats["avg_rg"], 0.01) == 0.779
    assert "Ensemble: 2 conformers" in stats["summary_text"]
