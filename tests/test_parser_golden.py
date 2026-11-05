"""
Golden-File Validation Test Suite for Quantum Chemistry Log Parsers.

Verifies parser extraction accuracy against real computational outputs from:
- ORCA (4.x / 5.x)
- Gaussian (G09 / G16)
- newBRORSEN (electronic & positronic multi-component)
- PySCF & openLOWDIN
"""

import os
import pytest
import numpy as np

from agymolden.qm_log_parser import parse_qm_logfile, QMLogParseResult


def test_golden_orca_parser():
    """Verify golden extraction on real ORCA geometry optimization output."""
    path = "orca_calc.out"
    if not os.path.isfile(path):
        pytest.skip(f"Reference log {path} not found")

    res = parse_qm_logfile(path)
    assert isinstance(res, QMLogParseResult)
    assert res.program_name.lower() == "orca"
    assert res.converged is True
    assert res.symbols == ["O", "H", "H"]
    assert len(res.trajectory_coords) == 5
    assert len(res.trajectory_energies) == 5

    # Check energetic monotonicity during optimization (final energy lower than initial)
    assert res.trajectory_energies[-1] < res.trajectory_energies[0]
    # Check dipole moment is physical for water (~1.9-2.1 Debye in cc-pVDZ)
    assert res.dipole_debye is not None
    assert 1.8 < res.dipole_debye < 2.2
    # Check HOMO / LUMO physical gap
    assert res.homo_ev is not None and res.lumo_ev is not None
    gap = res.lumo_ev - res.homo_ev
    assert 8.0 < gap < 11.0


def test_golden_gaussian_parser():
    """Verify golden extraction on real Gaussian optimization log."""
    path = "examples/AgyMolden_generated_calculation.log"
    if not os.path.isfile(path):
        pytest.skip(f"Reference log {path} not found")

    res = parse_qm_logfile(path)
    assert isinstance(res, QMLogParseResult)
    assert res.program_name.lower() == "gaussian"
    assert res.converged is True
    assert res.symbols[0] == "C" and len(res.symbols) == 5
    assert len(res.trajectory_coords) >= 2
    assert len(res.trajectory_energies) >= 2
    assert res.homo_ev is not None and res.lumo_ev is not None
    assert res.lumo_ev > res.homo_ev


def test_golden_brorsen_multistage_parser():
    """Verify golden extraction on multi-stage newBRORSEN electronic/positronic calculations."""
    for path in ["HLi_electronic_positronic.out", "2HLi_SP_HF_cc-pvdz.out"]:
        if not os.path.isfile(path):
            continue
        res = parse_qm_logfile(path)
        assert isinstance(res, QMLogParseResult)
        assert res.program_name == "newBRORSEN"
        assert res.symbols == ["Li", "H"]
        assert len(res.trajectory_coords) >= 1
        assert len(res.trajectory_energies) >= 1
        # HLi bond length in cc-pVDZ should be ~1.59-1.62 Angstroms
        d = np.linalg.norm(res.trajectory_coords[0][1] - res.trajectory_coords[0][0])
        assert 1.58 < d < 1.63


def test_golden_brorsen_opt_trajectory():
    """Verify golden extraction of full optimization trajectory from real output."""
    path = "C2H4_OPT_HF_cc-pvdz.out"
    if not os.path.isfile(path):
        pytest.skip(f"Reference log {path} not found")

    res = parse_qm_logfile(path)
    assert res.program_name == "newBRORSEN"
    assert res.symbols == ["C", "C", "H", "H", "H", "H"]
    assert len(res.trajectory_coords) == 4
    assert res.converged is True
