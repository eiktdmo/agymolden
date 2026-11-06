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


def test_generic_qm_runner_gaussian_live(qapp, tmp_path):
    from agymolden.qm_detect import detect_qm_executable
    info = detect_qm_executable("gaussian")
    if not info.found:
        pytest.skip("Gaussian executable not available")

    inp = tmp_path / "h2.com"
    inp.write_text("""%mem=2GB
%nprocshared=2
#P HF/STO-3G

H2 test

0 1
H 0.0 0.0 0.0
H 0.0 0.0 0.74

""")

    runner = GenericQMRunner(code_key="gaussian")
    finished_box = []
    runner.finished.connect(lambda ok, msg: finished_box.append((ok, msg)))

    ok = runner.start_job(str(inp), work_dir=str(tmp_path))
    assert ok is True

    # Process events until finished
    for _ in range(120):
        if finished_box:
            break
        QtCore.QCoreApplication.processEvents()
        QtCore.QThread.msleep(100)

    assert finished_box, "Gaussian job timed out"
    success, msg = finished_box[0]
    assert success is True
    assert (tmp_path / "h2.log").exists()
    log_text = (tmp_path / "h2.log").read_text()
    assert "Normal termination" in log_text
    assert len(runner.steps) >= 1
    assert runner.steps[-1].energy is not None


def test_job_monitor_auto_load_results_and_companion_files(qapp, tmp_path, monkeypatch):
    from agymolden.calc_runner.dialog_job_monitor import JobMonitorDialog
    from agymolden.calc_runner.base import BaseCalcRunner, JobStatus

    # Create dummy calculation results in tmp_path
    inp_file = tmp_path / "water.inp"
    inp_file.write_text("! HF def2-SVP\n* xyz 0 1\nO 0.0 0.0 0.0\nH 0.0 0.7 0.5\nH 0.0 -0.7 0.5\n*\n")
    out_file = tmp_path / "water.out"
    out_file.write_text("FINAL SINGLE POINT ENERGY -76.02676\n****ORCA TERMINATED NORMALLY****\n")
    molden_file = tmp_path / "water.molden"
    molden_file.write_text(
        "[Molden Format]\n[Atoms] AU\nO 1 8 0.0 0.0 0.0\nH 2 1 0.0 1.3 0.9\nH 3 1 0.0 -1.3 0.9\n"
        "[GTO]\n1 0\ns 1 1.00\n 1.0 1.0\n\n[MO]\nSym=1a\nEne=-20.5\nSpin=Alpha\nOccup=2.0\n1 1.0\n"
    )
    cube_file = tmp_path / "water_esp.cube"
    cube_file.write_text("ESP Cube file\nDummy\n   1   0.0 0.0 0.0\n   10  0.1 0.0 0.0\n   10  0.0 0.1 0.0\n   10  0.0 0.0 0.1\n   8 8.0 0.0 0.0 0.0\n 0.1\n")

    class DummyRunner(BaseCalcRunner):
        def start_job(self, input_path: str, work_dir=None):
            self.input_file = input_path
            self.work_dir = work_dir or str(tmp_path)
            self.log_file = str(out_file)
            return True
        def stop_job(self):
            pass

    runner = DummyRunner()
    runner.start_job(str(inp_file), work_dir=str(tmp_path))

    # Mock MainWindow
    class MockMainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self._suppress_companion_prompt = False
            self._job_monitor_dlg = None
            self.opened_paths = []
            self.viewer = type("MockViewer", (), {"active_isosurfaces": [], "surface_opacity": 1.0, "update": lambda: None})()
            self.table_orbitals = QtWidgets.QTableWidget()
            self._active_molden_path = None
            self._active_molden_data = None
            self._last_calculation_results = {}

        def open_path(self, path):
            self.opened_paths.append(path)

        def statusBar(self):
            return type("MockStatusBar", (), {"showMessage": lambda *args: None})()

    mw = MockMainWindow()
    dlg = JobMonitorDialog(parent=mw, runner=runner)
    mw._job_monitor_dlg = dlg

    # Verify auto-load checkbox exists and is checked by default
    assert hasattr(dlg, "chk_auto_load")
    assert dlg.chk_auto_load.isChecked() is True

    # Connect monitor completion signal to mw._on_live_calc_completed
    from agymolden.mw_calculations import CalculationsMixin
    mw._on_live_calc_completed = CalculationsMixin._on_live_calc_completed.__get__(mw, MockMainWindow)
    mw.load_orbitals_from_molden = CalculationsMixin.load_orbitals_from_molden.__get__(mw, MockMainWindow)
    mw.set_orbitals_from_data = CalculationsMixin.set_orbitals_from_data.__get__(mw, MockMainWindow)
    dlg.calculation_completed.connect(mw._on_live_calc_completed)

    # Trigger job completion (mocking QMessageBox.information so it does not block in headless test)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Ok)
    dlg._on_finished(success=True, message="Completed successfully")

    # Assert results are loaded automatically without manual prompt
    assert mw._active_molden_path == str(molden_file)
    assert mw._last_calculation_results["molden_file"] == str(molden_file)
    assert str(cube_file) in mw._last_calculation_results["cube_files"]
    assert str(out_file) in mw.opened_paths


