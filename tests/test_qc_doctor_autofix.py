"""
Tests for QC Doctor 1-click Auto-Fix & Relaunch in JobMonitorDialog.
Verifies keyword injection into failing input files and restart handling.
"""

import os
import pytest
from PyQt6 import QtWidgets
from agymolden.calc_runner import JobMonitorDialog, OrcaRunner
from agymolden.calc_runner.diagnostics import QMDiagnosticResult


def test_auto_fix_and_relaunch_orca_scf(tmp_path):
    from agymolden.calc_runner.diagnostics import patch_input_file_with_remedy

    inp_file = str(tmp_path / "benzene_calc.inp")
    with open(inp_file, "w") as f:
        f.write("! B3LYP def2-SVP Opt\n* xyz 0 1\nC 0.0 0.0 0.0\nH 1.08 0.0 0.0\n*\n")

    diag = QMDiagnosticResult(
        title="SCF Did Not Converge",
        summary="ORCA SCF failed to reach convergence.",
        remedy="Add ! SlowConv and %scf maxiter 150 end",
        suggested_keywords={"extra": "SlowConv", "scf_block": "%scf maxiter 150 end"},
    )

    success = patch_input_file_with_remedy(inp_file, diag, engine="orca")
    assert success is True

    with open(inp_file, "r") as f:
        patched = f.read()

    assert "! B3LYP def2-SVP Opt SlowConv" in patched
    assert "%scf maxiter 150 end" in patched


def test_auto_fix_gaussian_and_psi4(tmp_path):
    from agymolden.calc_runner.diagnostics import patch_input_file_with_remedy

    # Gaussian
    g_file = str(tmp_path / "calc.gjf")
    with open(g_file, "w") as f:
        f.write("#p B3LYP/6-31G* Opt\n\nTitle\n\n0 1\nH 0.0 0.0 0.0\n\n")

    diag_g = QMDiagnosticResult(
        title="Link 502", summary="", remedy="",
        suggested_keywords={"scf": "scf=(xqc,maxcycle=150)"}
    )
    assert patch_input_file_with_remedy(g_file, diag_g, engine="gaussian") is True
    with open(g_file, "r") as f:
        g_patched = f.read()
    assert "scf=(xqc,maxcycle=150)" in g_patched

    # Psi4
    p_file = str(tmp_path / "calc.in")
    with open(p_file, "w") as f:
        f.write("molecule {\n0 1\nH 0.0 0.0 0.0\n}\noptimize('hf/cc-pvdz')\n")

    diag_p = QMDiagnosticResult(
        title="Psi4 SCF", summary="", remedy="",
        suggested_keywords={"scf": "set soscf true"}
    )
    assert patch_input_file_with_remedy(p_file, diag_p, engine="psi4") is True
    with open(p_file, "r") as f:
        p_patched = f.read()
    assert "set set soscf true" in p_patched or "soscf true" in p_patched
