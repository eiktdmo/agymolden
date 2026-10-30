"""Unit tests for the modular calculation runner subsystem."""

import os
import sys
import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets

from agymolden.calc_runner import (
    BaseCalcRunner,
    JobStatus,
    OptimizationStep,
    BrorsenRunner,
    OrcaRunner,
    GenericQMRunner,
    JobMonitorDialog,
    detect_conda_environments,
    find_brorsen_wrappers,
)


def test_detect_conda_environments():
    envs = detect_conda_environments()
    assert len(envs) > 0
    # Must contain at least the active python
    names = [e[0] for e in envs]
    assert any("Current" in n for n in names)


def test_find_brorsen_wrappers():
    wrappers = find_brorsen_wrappers()
    assert isinstance(wrappers, list)
    # newBRORSEN.v04.09.py is in the workspace
    basenames = [os.path.basename(w) for w in wrappers]
    assert any("newBRORSEN" in b for b in basenames)


def test_brorsen_runner_parsing(qapp):
    runner = BrorsenRunner()
    captured_steps = []
    runner.step_updated.connect(captured_steps.append)

    # Feed typical output lines
    sample_output = [
        "Starting calculation stage 1",
        "--- Cycle 1 ---",
        "Total Energy = -76.12345678",
        "Cartesian coordinates:",
        "O  0.000000  0.000000  0.117000",
        "H  0.000000  0.757000 -0.468000",
        "H  0.000000 -0.757000 -0.468000",
        "--- Gradient norm: 0.0123 ---",
        "Optimization converged successfully",
    ]

    for line in sample_output:
        runner._parse_line(line)

    # 1 emission on geometry parse + 1 update emission on convergence
    assert len(captured_steps) >= 1
    step = captured_steps[-1]
    assert step.cycle == 1
    assert pytest.approx(step.energy, rel=1e-5) == -76.12345678
    assert step.symbols == ["O", "H", "H"]
    assert step.coords.shape == (3, 3)
    assert runner.status == JobStatus.CONVERGED
    assert step.is_converged is True


def test_orca_runner_parsing(qapp):
    runner = OrcaRunner()
    captured_steps = []
    runner.step_updated.connect(captured_steps.append)

    sample_output = [
        "OPTIMIZATION CYCLE   2",
        "FINAL SINGLE POINT ENERGY      -154.23456789",
        "CARTESIAN COORDINATES (ANGSTROEM)",
        "---------------------------------",
        "C   0.000000   0.000000   0.000000",
        "O   0.000000   0.000000   1.200000",
        "",
        "*** THE OPTIMIZATION HAS CONVERGED ***",
    ]

    for line in sample_output:
        runner._parse_line(line)

    assert len(captured_steps) >= 1
    step = captured_steps[-1]
    assert step.cycle == 2
    assert pytest.approx(step.energy, rel=1e-5) == -154.23456789
    assert step.symbols == ["C", "O"]
    assert step.coords.shape == (2, 3)
    assert runner.status == JobStatus.CONVERGED
    assert step.is_converged is True


def test_job_monitor_dialog_init(qapp):
    dlg = JobMonitorDialog()
    assert dlg.combo_conda.count() > 0
    assert dlg.chk_live_3d.isChecked()
    assert not dlg.btn_stop.isEnabled()
    dlg.close()


def test_generic_qm_runner_parsing(qapp):
    # Test Gaussian output parsing
    g_runner = GenericQMRunner(code_key="gaussian")
    steps = []
    g_runner.step_updated.connect(steps.append)

    gaussian_lines = [
        " SCF Done:  E(RB3LYP) =  -76.408953210     A.U. after    9 cycles",
        "                         Standard orientation:                         ",
        " ---------------------------------------------------------------------",
        " Center     Atomic      Atomic             Coordinates (Angstroms)",
        " Number     Number       Type             X           Y           Z",
        " ---------------------------------------------------------------------",
        "      1          8           0        0.000000    0.000000    0.117300",
        "      2          1           0        0.000000    0.757200   -0.469200",
        "      3          1           0        0.000000   -0.757200   -0.469200",
        " ---------------------------------------------------------------------",
        " Optimization completed.",
    ]
    for line in gaussian_lines:
        g_runner._parse_line(line)

    assert len(steps) >= 1
    last = steps[-1]
    assert pytest.approx(last.energy, rel=1e-6) == -76.408953210
    assert last.symbols == ["O", "H", "H"]
    assert last.coords.shape == (3, 3)
    assert g_runner.status == JobStatus.CONVERGED

    # Test PySCF output parsing
    p_runner = GenericQMRunner(code_key="pyscf")
    p_steps = []
    p_runner.step_updated.connect(p_steps.append)
    p_runner._parse_line("converged SCF energy = -76.0267656731")
    assert len(p_steps) == 1
    assert pytest.approx(p_steps[0].energy, rel=1e-6) == -76.0267656731



def test_remote_ssh_password_wrapping():
    from agymolden.calc_runner.remote_ssh import _wrap_ssh_command, RemoteHostConfig

    cmd = ["ssh", "user@host", "echo READY"]
    wrapped_cmd, env = _wrap_ssh_command(cmd, password="secretpassword")
    assert "SSHPASS" in env
    assert env["SSHPASS"] == "secretpassword"
    assert "sshpass" in wrapped_cmd[0]


def test_dialog_launch_remote_actions(qapp):
    from agymolden.calc_runner.dialog_launch import BrorsenLaunchDialog

    dlg = BrorsenLaunchDialog(None, input_file="tests/test.inp")
    assert hasattr(dlg, "btn_check_remote_env")
    assert hasattr(dlg, "btn_upload_wrapper")
    assert hasattr(dlg, "txt_ssh_password")
