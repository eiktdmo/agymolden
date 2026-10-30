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
