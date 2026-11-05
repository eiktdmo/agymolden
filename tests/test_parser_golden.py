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
    assert len(res.trajectory_coords) >= 5
    assert len(res.trajectory_energies) >= 5

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


def test_golden_transition_state_imaginary_freq(tmp_path):
    """Verify golden extraction of TS candidate with exactly 1 imaginary frequency."""
    ts_log = """Gaussian, Inc.
 Entering Gaussian System
 #P B3LYP/6-31G* opt=(ts,calcfc,noeigen) freq

 Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          6           0        0.000000    0.000000    0.000000
      2          1           0        0.000000    0.000000    1.350000
      3          9           0        0.000000    0.000000   -1.450000
 ---------------------------------------------------------------------
 Harmonic frequencies (cm**-1), IR intensities (KM/Mole)
 Frequencies --   -485.3200              1120.5000              2980.2000
 IR Inten    --    240.5000                18.2000                95.0000
 Atom  AN      X      Y      Z        X      Y      Z        X      Y      Z
   1   6      0.00   0.00   0.45     0.00   0.00   0.00     0.00   0.00   0.00
   2   1      0.00   0.00  -0.78     0.00   0.50   0.00     0.50   0.00   0.00
   3   9      0.00   0.00   0.12     0.00  -0.25  -0.43    -0.25  -0.43   0.00
 Normal termination of Gaussian 16.
"""
    f = tmp_path / "ts_reaction.log"
    f.write_text(ts_log, encoding="utf-8")

    res = parse_qm_logfile(str(f))
    assert res.symbols == ["C", "H", "F"]
    assert res.vibrational_analysis is not None
    # Verify the imaginary mode is present
    freqs = res.vibrational_analysis.frequencies
    assert len(freqs) == 3
    # First frequency is negative (imaginary)
    assert freqs[0] < 0.0 or pytest.approx(freqs[0], abs=1e-1) == -485.3


def test_golden_linear_diatomic_and_polyatomic():
    """Verify parsing real linear molecule calculations (CO2 and 2HLi)."""
    co2_path = "CO2_SP_HF_cc-pvdz.out"
    if os.path.isfile(co2_path):
        res_co2 = parse_qm_logfile(co2_path)
        assert res_co2.symbols == ["C", "O", "O"]
        assert len(res_co2.trajectory_coords) >= 1
        # Check linear symmetry: C at center, O atoms symmetric
        coords = res_co2.trajectory_coords[-1]
        d1 = np.linalg.norm(coords[0] - coords[1])
        d2 = np.linalg.norm(coords[0] - coords[2])
        assert pytest.approx(d1, rel=1e-2) == d2


def test_qc_doctor_multi_engine_diagnostics():
    """Verify QC Doctor accurately detects and diagnoses common QM error patterns."""
    from agymolden.calc_runner.diagnostics import diagnose_qm_error

    # 1. Gaussian Link 502 SCF convergence failure
    g_log = "Error in l502.exe: Convergence failure -- run terminated."
    diag_g = diagnose_qm_error(g_log, engine="gaussian")
    assert diag_g is not None
    assert "Link 502" in diag_g.title or "SCF" in diag_g.title
    assert "scf=(xqc" in diag_g.remedy

    # 2. ORCA MaxCore memory fault
    o_log = "An error has occurred: Out of memory in cholesky decomposition. Allocate memory."
    diag_o = diagnose_qm_error(o_log, engine="orca")
    assert diag_o is not None
    assert "Memory" in diag_o.title
    assert "%maxcore" in diag_o.remedy

    # 3. Psi4 Optking optimization failure
    p_log = "Optimization failed to converge in 50 iterations in optking."
    diag_p = diagnose_qm_error(p_log, engine="psi4")
    assert diag_p is not None
    assert "Optking" in diag_p.title or "Optimization" in diag_p.title
    assert "geom_maxiter" in diag_p.remedy

