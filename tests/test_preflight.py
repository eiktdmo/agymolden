"""
Tests for Pre-flight calculation validation engine.
"""

import numpy as np
import pytest

from agymolden.qm_input_generator import QMCalculationParameters
from agymolden.calc_runner.preflight import validate_qm_calculation, ValidationIssue, PreflightReport


def test_preflight_valid_water_singlet():
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ])
    params = QMCalculationParameters(
        job_type="Opt + Freq",
        method="B3LYP",
        basis="def2-SVP",
        charge=0,
        multiplicity=1,
        memory_gb=4,
        n_procs=2,
    )
    report = validate_qm_calculation("gaussian", symbols, coords, params)
    assert report.is_valid is True
    assert report.has_errors is False
    assert report.total_electrons == 10
    assert report.num_alpha == 5
    assert report.num_beta == 5


def test_preflight_detects_spin_multiplicity_parity_mismatch():
    # Water has 10 electrons. Multiplicity 2 (Doublet) is impossible for neutral closed shell!
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ])
    params = QMCalculationParameters(
        job_type="SP",
        method="B3LYP",
        basis="def2-SVP",
        charge=0,
        multiplicity=2,  # Parity error!
    )
    report = validate_qm_calculation("gaussian", symbols, coords, params)
    assert report.is_valid is False
    assert report.has_errors is True
    assert any("Spin Multiplicity Parity" in iss.title for iss in report.issues)


def test_preflight_detects_nuclear_clash():
    # Two hydrogens placed at 0.1 Angstrom (nuclear clash)
    symbols = ["H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1],  # Severe clash!
    ])
    params = QMCalculationParameters(job_type="SP", method="HF", basis="STO-3G", charge=0, multiplicity=1)
    report = validate_qm_calculation("gaussian", symbols, coords, params)
    assert report.is_valid is False
    assert any("Severe Nuclear Clash" in iss.title for iss in report.issues)


def test_preflight_detects_excessive_memory_allocation():
    symbols = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    params = QMCalculationParameters(
        job_type="SP", method="HF", basis="STO-3G", charge=0, multiplicity=1,
        memory_gb=999999,  # 1 Million GB RAM!
    )
    report = validate_qm_calculation("gaussian", symbols, coords, params)
    assert report.is_valid is False
    assert any("Memory Exceeds Physical RAM" in iss.title for iss in report.issues)


def test_preflight_gaussian_specific_basis_and_dispersion_normalization_notes():
    symbols = ["O", "H", "H"]
    coords = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
    params = QMCalculationParameters(
        job_type="SP",
        method="M06-2X",
        basis="def2-SVP",
        dispersion="D3BJ",
        charge=0,
        multiplicity=1,
        memory_gb=2,
    )
    report = validate_qm_calculation("gaussian", symbols, coords, params)
    assert report.is_valid is True  # Errors = 0, Infos generated
    info_titles = [iss.title for iss in report.issues]
    assert any("Ahlrichs Basis Hyphen Auto-Normalized" in t for t in info_titles)
    assert any("Becke-Johnson Damping Adjusted" in t for t in info_titles)


def test_preflight_dialog_gui_smoke(qapp):
    symbols = ["O", "H", "H"]
    coords = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
    params = QMCalculationParameters(
        job_type="SP",
        method="B3LYP",
        basis="def2-SVP",
        charge=0,
        multiplicity=2,  # Error
    )
    report = validate_qm_calculation("gaussian", symbols, coords, params)
    from agymolden.calc_runner.dialog_preflight import PreflightValidationDialog
    dlg = PreflightValidationDialog(None, report, qm_name="Gaussian")
    assert dlg.windowTitle().startswith("Pre-Flight Inspection")
    assert hasattr(dlg, "btn_action")  # Error mode displays "Fix Calculation Setup" button
