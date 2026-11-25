"""
Unit and Integration Tests for Apex Triad Features:
1. NMR Chemical Shift Referencing & Spectrum Plotter (nmr_engine.py, dialog_nmr.py)
2. Microkinetics & Catalytic Cycle Simulator (microkinetics.py, dialog_microkinetics.py)
3. Intrinsic Bond Orbitals & Lewis Localization (ibo_tools.py, dialog_ibo.py)
"""

import os
import pytest
import numpy as np
from PyQt6 import QtWidgets

from agymolden.nmr_engine import (
    NMR_REFERENCE_STANDARDS,
    compute_chemical_shifts,
    synthesize_nmr_spectrum,
)
from agymolden.microkinetics import (
    ElementaryStep,
    eyring_rate_constant,
    solve_microkinetics,
)
from agymolden.ibo_tools import (
    compute_ibos,
    LocalizedOrbital,
    IBOResult,
)


def test_nmr_chemical_shift_referencing():
    """Verify linear referencing delta = (intercept - sigma_iso) / (-slope) for 1H and 13C."""
    # Test shieldings mock
    mock_shieldings = [
        {"atom_index": 0, "symbol": "C", "sigma_iso": 60.0, "anisotropy": 12.0},
        {"atom_index": 1, "symbol": "C", "sigma_iso": 160.0, "anisotropy": 5.0},
        {"atom_index": 2, "symbol": "H", "sigma_iso": 29.5, "anisotropy": 3.0},
        {"atom_index": 3, "symbol": "H", "sigma_iso": 30.5, "anisotropy": 1.5},
    ]

    # Direct TMS model for 1H (B3LYP/6-31G(d): intercept 31.88, slope -1.0)
    h_peaks = compute_chemical_shifts(mock_shieldings, nucleus="1H", standard_key="B3LYP/6-31G(d) [Direct TMS]")
    assert len(h_peaks) == 2
    # delta = 31.88 - sigma_iso
    p1 = [p for p in h_peaks if p.atom_index == 2][0]
    p2 = [p for p in h_peaks if p.atom_index == 3][0]
    assert pytest.approx(p1.chemical_shift_ppm, 0.01) == (31.88 - 29.5)
    assert pytest.approx(p2.chemical_shift_ppm, 0.01) == (31.88 - 30.5)

    # Tantillo scaled model for 13C: intercept 181.24, slope -0.96
    c_peaks = compute_chemical_shifts(mock_shieldings, nucleus="13C", standard_key="Empirical Scaled (Tantillo Model)")
    assert len(c_peaks) == 2
    for p in c_peaks:
        expected = (181.24 - p.sigma_iso) / 0.96
        assert pytest.approx(p.chemical_shift_ppm, 0.01) == expected


def test_nmr_spectrum_synthesis():
    """Verify continuous Lorentzian line broadening synthesis."""
    mock_shieldings = [
        {"atom_index": 0, "symbol": "H", "sigma_iso": 24.6, "anisotropy": 0.0},
        {"atom_index": 1, "symbol": "H", "sigma_iso": 30.8, "anisotropy": 0.0},
    ]
    peaks = compute_chemical_shifts(mock_shieldings, nucleus="1H")
    spectrum = synthesize_nmr_spectrum(peaks, nucleus="1H", fwhm_ppm=0.05, num_points=500)

    assert len(spectrum.ppm_grid) == 500
    assert len(spectrum.intensity_grid) == 500
    assert np.max(spectrum.intensity_grid) == pytest.approx(100.0, 0.1)
    # Check that peaks correspond to local maxima
    for p in peaks:
        closest_idx = int(np.argmin(np.abs(spectrum.ppm_grid - p.chemical_shift_ppm)))
        assert spectrum.intensity_grid[closest_idx] > 50.0


def test_eyring_rate_constant():
    """Verify Eyring transition state theory rate calculation."""
    # At 298.15 K, barrier of 0 kcal/mol -> ~6.21e12 s^-1
    k0 = eyring_rate_constant(0.0, 298.15)
    assert 6.0e12 < k0 < 6.5e12

    # Barrier of 20 kcal/mol at 298.15 K -> ~1.3e-2 s^-1
    k20 = eyring_rate_constant(20.0, 298.15)
    assert 1.0e-3 < k20 < 1.0e-1

    # Barrier of 30 kcal/mol -> very slow (~6e-10 s^-1)
    k30 = eyring_rate_constant(30.0, 298.15)
    assert 1e-12 < k30 < 1e-8


def test_microkinetics_catalytic_cycle():
    """Verify catalytic cycle ODE integration and TOF calculation."""
    # A -> B -> P catalytic cycle
    # Step 1: Cat + A -> CatA (barrier 15.0)
    # Step 2: CatA -> Cat + P (barrier 18.0)
    steps = [
        ElementaryStep("Binding", {"Cat": 1.0, "A": 1.0}, {"CatA": 1.0}, delta_g_act_fwd=15.0),
        ElementaryStep("Reaction", {"CatA": 1.0}, {"Cat": 1.0, "P": 1.0}, delta_g_act_fwd=18.0),
    ]
    concs = {"Cat": 0.01, "A": 1.0, "CatA": 0.0, "P": 0.0}

    res = solve_microkinetics(
        steps=steps,
        initial_concentrations=concs,
        time_span=(0.0, 3600.0),
        temperature_k=298.15,
        catalyst_name="Cat",
        product_name="P",
    )

    assert len(res.species_names) == 4
    assert res.catalyst_species == "Cat"
    assert res.product_species == "P"
    # Product P must accumulate over time
    p_idx = res.species_names.index("P")
    assert res.concentrations[p_idx, -1] > res.concentrations[p_idx, 0]
    # TOF and TON must be positive
    assert res.turnover_frequency_tof > 0.0
    assert res.turnover_number_ton > 0.0


def test_ibo_localization_pyscf():
    """Verify PySCF IAO/IBO localization on a sample molden file."""
    pytest.importorskip("pyscf")
    sample_molden = "examples/h2o_orca.molden"
    if not os.path.exists(sample_molden):
        pytest.skip(f"{sample_molden} not found")

    res = compute_ibos(sample_molden, exponent=4)
    assert isinstance(res, IBOResult)
    # H2O has 10 electrons = 5 occupied pairs (1 core, 2 O-H bonds, 2 lone pairs)
    assert len(res.orbitals) == 5
    assert res.ibo_coeff_matrix.shape[1] == 5

    # Check classifications
    types = [o.orbital_type for o in res.orbitals]
    assert "core" in types
    assert "bond" in types
    assert "lone_pair" in types


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_apex_triad_dialog_smoke(app):
    """Verify GUI instantiation and basic lifecycle for all three Apex Triad dialogs."""
    from agymolden.dialog_nmr import NMRDialog
    from agymolden.dialog_microkinetics import MicrokineticsDialog
    from agymolden.dialog_ibo import IBODialog

    # 1. NMR Dialog
    mock_shieldings = [
        {"atom_index": 0, "symbol": "H", "sigma_iso": 29.5, "anisotropy": 1.0},
        {"atom_index": 1, "symbol": "C", "sigma_iso": 120.0, "anisotropy": 10.0},
    ]
    nmr_dlg = NMRDialog(shieldings=mock_shieldings)
    assert nmr_dlg.tbl_peaks.rowCount() == 1  # 1H active initially
    nmr_dlg.close()

    # 2. Microkinetics Dialog
    micro_dlg = MicrokineticsDialog()
    assert micro_dlg.tbl_steps.rowCount() >= 3
    assert micro_dlg.result is not None
    micro_dlg.close()

    # 3. IBO Dialog (requires PySCF; skip on non-PySCF environments to avoid modal warning)
    import importlib.util
    if importlib.util.find_spec("pyscf") is not None:
        ibo_dlg = IBODialog(molden_path="examples/h2o_orca.molden")
        assert ibo_dlg.tbl_ibos.rowCount() == 5
        ibo_dlg.close()
