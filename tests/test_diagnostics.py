import pytest
from agymolden.calc_runner.diagnostics import diagnose_orca_error, QMDiagnosticResult


def test_diagnose_orca_mpi_error():
    log = "ORCA finished by error in ORCA_Init\norterun failed to launch child process"
    diag = diagnose_orca_error(log)
    assert diag is not None
    assert "MPI" in diag.title
    assert "nprocs 1" in diag.remedy


def test_diagnose_orca_scf_error():
    log = "SCF NOT CONVERGED AFTER 125 CYCLES. Giving up."
    diag = diagnose_orca_error(log)
    assert diag is not None
    assert "SCF Did Not Converge" in diag.title
    assert "SlowConv" in diag.remedy


def test_diagnose_orca_basis_error():
    log = "Basis set def2-FOO not found on disk"
    diag = diagnose_orca_error(log)
    assert diag is not None
    assert "Basis Set" in diag.title


def test_diagnose_orca_memory_error():
    log = "Out of memory in matrix diagonalization: cannot allocate memory"
    diag = diagnose_orca_error(log)
    assert diag is not None
    assert "Insufficient Memory" in diag.title


def test_diagnose_gaussian_scf_and_opt_errors():
    from agymolden.calc_runner.diagnostics import diagnose_qm_error

    log_scf = "Error termination via Lnk1e in /usr/local/g16/l502.exe\nConvergence failure -- run terminated."
    diag1 = diagnose_qm_error(log_scf, engine="gaussian")
    assert diag1 is not None
    assert "Link 502" in diag1.title or "SCF" in diag1.title
    assert "scf=(qc" in diag1.remedy or "xqc" in diag1.remedy

    log_opt = "Maximum number of optimization cycles reached.\nError termination via Lnk1e in /usr/local/g16/l9999.exe"
    diag2 = diagnose_qm_error(log_opt, engine="gaussian")
    assert diag2 is not None
    assert "Link 9999" in diag2.title
    assert "calcfc" in diag2.remedy

    log_qperr = "QPErr --- A syntax error was detected in the input line.\nError termination via Lnk1e in /g16/l1.exe"
    diag3 = diagnose_qm_error(log_qperr, engine="gaussian")
    assert diag3 is not None
    assert "Link 1" in diag3.title
    assert "def2SVP" in diag3.remedy

    log_geom = "Wanted a floating point number as input.\nFound an integer as input.\nError termination via Lnk1e in /g16/l101.exe"
    diag4 = diagnose_qm_error(log_geom, engine="gaussian")
    assert diag4 is not None
    assert "Link 101" in diag4.title


def test_diagnose_pyscf_errors():
    from agymolden.calc_runner.diagnostics import diagnose_qm_error

    log_pyscf = "WARN: SCF not converged after 50 cycles.\nCalculation failed."
    diag = diagnose_qm_error(log_pyscf, engine="pyscf")
    assert diag is not None
    assert "PySCF" in diag.title
    assert "newton" in diag.remedy
