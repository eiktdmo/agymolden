"""Tests for Arrhenius temperature sensitivity and 2D Ramachandran PES features."""

import numpy as np
import pytest
from PyQt6 import QtWidgets

from agymolden.dialog_reaction_thermo import ReactionThermoDialog, SpeciesThermo
from agymolden.dialog_ramachandran_2d import Ramachandran2DScanDialog


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_arrhenius_kinetics_dialog(qapp):
    """Verify Arrhenius & Eyring temperature sensitivity calculation and plotting."""
    dlg = ReactionThermoDialog()
    r = SpeciesThermo(name="Reactant", role="reactant", coefficient=1.0, gibbs_ha=-100.0, enthalpy_ha=-100.0)
    ts = SpeciesThermo(name="TS", role="ts", coefficient=1.0, gibbs_ha=-99.98, enthalpy_ha=-99.98)
    p = SpeciesThermo(name="Product", role="product", coefficient=1.0, gibbs_ha=-100.02, enthalpy_ha=-100.02)
    dlg.species = [r, ts, p]

    # Verify that recalculate works
    dlg._recalculate()
    assert "ΔG‡" in dlg.lbl_delta_g_act.text()


def test_ramachandran_2d_dialog(qapp):
    """Verify 2D Ramachandran PES surface generator builds and computes."""
    symbols = ["C", "C", "C", "C"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 1.5, 0.0],
        [0.0, 1.5, 1.2],
    ])
    dlg = Ramachandran2DScanDialog(symbols=symbols, coords=coords)
    assert dlg.combo_phi.count() >= 1
    assert dlg.combo_psi.count() >= 1
    assert "Computed 2D energy surface" in dlg.lbl_status.text()
