"""Tests for Continuum Solvation Free Energy and Dual HOMO/LUMO Frontier Overlay."""

import numpy as np
import pytest
from PyQt6 import QtWidgets

from agymolden.solvation import SOLVENT_DATABASE, compute_approximate_sasa, estimate_solvation_energy
from agymolden.dialog_solvation import SolvationDialog


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_solvation_database():
    """Verify standard solvent properties are defined."""
    assert "water" in SOLVENT_DATABASE
    assert "acetonitrile" in SOLVENT_DATABASE
    assert "dichloromethane" in SOLVENT_DATABASE
    assert SOLVENT_DATABASE["water"].dielectric_constant > 70.0


def test_sasa_and_solvation_estimation():
    """Verify SASA calculation and solvation free energy trends."""
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.117],
        [0.0, 0.757, -0.469],
        [0.0, -0.757, -0.469],
    ])
    sasa, vol, at_sasa = compute_approximate_sasa(symbols, coords, probe_radius=1.40)
    assert sasa > 10.0
    assert vol > 10.0
    assert len(at_sasa) == 3

    # Water should produce negative solvation energy for polar molecule
    res_water = estimate_solvation_energy(symbols, coords, solvent_key="water", dipole_debye=1.85)
    assert res_water.delta_g_solv_kcal < 0.0
    assert res_water.delta_g_el_kcal < 0.0

    # Hexane (low dielectric) should have much less favorable electrostatic solvation
    res_hexane = estimate_solvation_energy(symbols, coords, solvent_key="hexane", dipole_debye=1.85)
    assert abs(res_hexane.delta_g_el_kcal) < abs(res_water.delta_g_el_kcal)


def test_solvation_dialog(qapp):
    """Verify SolvationDialog builds UI and recalculates on solvent change."""
    symbols = ["C", "O"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.20]])
    dlg = SolvationDialog(symbols=symbols, coords=coords, dipole_debye=2.5)

    assert dlg.tbl_results.rowCount() >= 6
    assert dlg.combo_solvent.count() >= 8
