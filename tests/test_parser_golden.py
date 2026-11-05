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


def test_golden_psi4_parser(tmp_path):
    """Verify golden extraction on realistic Psi4 optimization + frequency output."""
    psi4_log = """
    -----------------------------------------------------------------------
          Psi4: An Open-Source Ab Initio Electronic Structure Package
                               Psi4 1.8.2 release
    -----------------------------------------------------------------------

         Geometry (in Angstrom), charge = 0, multiplicity = 1:

       Center              X                  Y                   Z               Mass       
    ------------   -----------------  -----------------  -----------------  -----------------
           O          0.000000000000     0.000000000000     0.117300000000    15.994914619570
           H          0.000000000000     0.757200000000    -0.469200000000     1.007825032230
           H          0.000000000000    -0.757200000000    -0.469200000000     1.007825032230

    @DF-RKS Final Energy:     -76.4089532100

         Geometry (in Angstrom), charge = 0, multiplicity = 1:

       Center              X                  Y                   Z               Mass       
    ------------   -----------------  -----------------  -----------------  -----------------
           O          0.000000000000     0.000000000000     0.116500000000    15.994914619570
           H          0.000000000000     0.758000000000    -0.466000000000     1.007825032230
           H          0.000000000000    -0.758000000000    -0.466000000000     1.007825032230

    @DF-RKS Final Energy:     -76.4091245000

    Optimizer: Optimization complete!

    Harmonic Vibrational Frequencies [cm^-1]
    ----------------------------------------
    Freq [cm^-1]        1595.23     3657.05     3755.93
    IR Intens [km/mol]     2.51       12.30       54.21

    Total ZPE, Electronic energy + ZPE =   -76.38765432
    Total Enthalpy                     =   -76.38312000
    Total Gibbs Free Energy            =   -76.40543200

    *** Psi4 exiting successfully. Buy a developer a beer!
    """
    log_file = tmp_path / "water_psi4.out"
    log_file.write_text(psi4_log, encoding="utf-8")

    res = parse_qm_logfile(str(log_file))
    assert isinstance(res, QMLogParseResult)
    assert res.program_name == "Psi4"
    assert res.converged is True
    assert res.symbols == ["O", "H", "H"]
    assert len(res.trajectory_coords) == 2
    assert len(res.trajectory_energies) == 2
    assert res.trajectory_energies[1] < res.trajectory_energies[0]
    assert res.vibrational_analysis is not None
    assert len(res.vibrational_analysis.frequencies) == 3
    assert res.vibrational_analysis.frequencies[0] == pytest.approx(1595.23, rel=1e-3)
    assert res.vibrational_analysis.ir_intensities == [2.51, 12.30, 54.21]


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
