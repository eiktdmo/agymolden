"""Tests for Conceptual DFT (CDFT) and Fukui reactivity module."""

import numpy as np
import pytest
from PyQt6 import QtWidgets

from agymolden.cdft import compute_global_cdft, compute_condensed_fukui
from agymolden.dialog_cdft import CDFTWorkbenchDialog


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_global_cdft_indices():
    """Verify calculation of chemical potential, hardness, electrophilicity, and softness."""
    homo = -6.50
    lumo = -1.50
    g = compute_global_cdft(homo, lumo)

    # I = 6.5, A = 1.5
    assert pytest.approx(g.ip_ev, 1e-4) == 6.50
    assert pytest.approx(g.ea_ev, 1e-4) == 1.50

    # Hardness η = I - A = 5.0 eV
    assert pytest.approx(g.hardness_eta, 1e-4) == 5.00

    # Softness S = 1 / (2η) = 0.1 eV^-1
    assert pytest.approx(g.softness_s, 1e-4) == 0.10

    # Chemical potential μ = -(I + A) / 2 = -4.00 eV
    assert pytest.approx(g.chemical_potential_mu, 1e-4) == -4.00

    # Mulliken electronegativity χ = -μ = +4.00 eV
    assert pytest.approx(g.electronegativity_chi, 1e-4) == 4.00

    # Electrophilicity ω = μ² / (2η) = 16.0 / 10.0 = 1.60 eV
    assert pytest.approx(g.electrophilicity_omega, 1e-4) == 1.60


def test_condensed_fukui_and_dual_descriptor():
    """Verify calculation of f+, f-, and dual descriptor (Δf)."""
    symbols = ["C", "O"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.20]])
    loc = compute_condensed_fukui(symbols, coords)

    assert len(loc.f_plus) == 2
    assert len(loc.f_minus) == 2
    assert len(loc.dual_descriptor) == 2

    # In carbonyl C=O, carbon is electrophilic (Δf > 0) and oxygen is nucleophilic (Δf < 0)
    assert loc.dual_descriptor[0] > 0.0  # Carbon
    assert loc.dual_descriptor[1] < 0.0  # Oxygen


def test_cdft_dialog(qapp):
    """Verify CDFT dialog builds UI, populates tables, and emits dual descriptor signal."""
    symbols = ["C", "C", "O"]
    coords = np.zeros((3, 3))
    dlg = CDFTWorkbenchDialog(symbols=symbols, coords=coords, homo_ev=-6.0, lumo_ev=-1.0)
    dlg._suppress_msg = True

    assert dlg.tbl_global.rowCount() >= 10
    assert dlg.tbl_local.rowCount() == 3

    emitted = []
    dlg.apply_dual_descriptor_to_viewer.connect(lambda s, d: emitted.append((s, d)))
    dlg._on_apply_to_viewer()
    assert len(emitted) == 1
    syms, duals = emitted[0]
    assert len(syms) == 3
    assert len(duals) == 3
