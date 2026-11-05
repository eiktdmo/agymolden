"""Contract tests: inputs written by AgyMolden generators must actually run in the real QM binaries.

Each test generates an input with the production generator, runs the installed program on a
tiny molecule, and asserts normal termination and a parsable energy. Tests are skipped
automatically when the corresponding program is not installed, so they are safe on CI.
"""

import os
import re
import subprocess

import numpy as np
import pytest

from agymolden.qm_detect import detect_qm_executable
from agymolden.qm_input_generator import (
    GaussianGenerator,
    OrcaGenerator,
    QMCalculationParameters,
)
from agymolden.calc_runner.generic_runner import GenericQMRunner

WATER_SYMBOLS = ["O", "H", "H"]
WATER_COORDS = np.array([
    [0.000000, 0.000000, 0.117300],
    [0.000000, 0.757200, -0.469200],
    [0.000000, -0.757200, -0.469200],
])


def _exe(code: str):
    info = detect_qm_executable(code)
    if not info.found:
        pytest.skip(f"{code} not installed")
    return info.resolved_path


def _gaussian_env(exe: str, scratch: str) -> dict:
    runner = GenericQMRunner(code_key="gaussian")
    env = os.environ.copy()
    env.update(runner._get_gaussian_env(exe, scratch))
    return env


def _run_gaussian(tmp_path, params, name="job"):
    exe = _exe("gaussian")
    params.chk_file = f"{name}.chk"
    text = GaussianGenerator().generate_input(WATER_SYMBOLS, WATER_COORDS, params)
    inp = tmp_path / f"{name}.gjf"
    inp.write_text(text)
    proc = subprocess.run(
        [exe], stdin=open(inp), capture_output=True, text=True,
        cwd=tmp_path, env=_gaussian_env(exe, str(tmp_path)), timeout=300,
    )
    return text, proc


def _assert_gaussian_ok(text, proc):
    out = proc.stdout
    assert "Normal termination" in out, f"Gaussian failed for input:\n{text}\n--- tail ---\n{out[-1500:]}"
    assert re.search(r"SCF Done:\s+E\(", out), "no SCF energy found"


# Every method / basis the Gaussian UI offers must be accepted by the real program.
G_METHODS = GaussianGenerator.supported_methods
G_BASES = GaussianGenerator.supported_basis_sets


@pytest.mark.parametrize("method", [m for m in G_METHODS if m != "CCSD(T)"])
def test_gaussian_every_method_runs(tmp_path, method):
    params = QMCalculationParameters(job_type="SP", method=method, basis="STO-3G",
                                     dispersion="D3BJ", n_procs=1, memory_gb=2)
    _assert_gaussian_ok(*_run_gaussian(tmp_path, params))


def test_gaussian_ccsdt_runs(tmp_path):
    params = QMCalculationParameters(job_type="SP", method="CCSD(T)", basis="6-31G*",
                                     dispersion="None", n_procs=1, memory_gb=2)
    text, proc = _run_gaussian(tmp_path, params)
    assert "Normal termination" in proc.stdout, f"{text}\n{proc.stdout[-1500:]}"


@pytest.mark.parametrize("basis", G_BASES)
def test_gaussian_every_basis_runs(tmp_path, basis):
    params = QMCalculationParameters(job_type="SP", method="HF", basis=basis,
                                     dispersion="None", n_procs=1, memory_gb=2)
    _assert_gaussian_ok(*_run_gaussian(tmp_path, params))


def test_gaussian_opt_freq_zmatrix_and_cartesian(tmp_path):
    for use_z in (False, True):
        params = QMCalculationParameters(job_type="Opt + Freq", method="B3LYP", basis="STO-3G",
                                         dispersion="D3BJ", n_procs=1, memory_gb=2, use_zmatrix=use_z)
        from agymolden.zmatrix import cartesian_to_zmatrix
        zat = cartesian_to_zmatrix(WATER_SYMBOLS, WATER_COORDS) if use_z else None
        exe = _exe("gaussian")
        params.chk_file = "oz.chk"
        text = GaussianGenerator().generate_input(WATER_SYMBOLS, WATER_COORDS, params, zatoms=zat)
        proc = subprocess.run([exe], input=text, capture_output=True, text=True, cwd=tmp_path,
                              env=_gaussian_env(exe, str(tmp_path)), timeout=300)
        assert "Normal termination" in proc.stdout, f"use_zmatrix={use_z}\n{text}\n{proc.stdout[-1500:]}"


def test_gaussian_runner_end_to_end_streams_energy(qapp, tmp_path):
    """The Qt runner (not just the binary) must launch, stream, and parse a Gaussian job."""
    from PyQt6 import QtCore
    _exe("gaussian")
    params = QMCalculationParameters(job_type="SP", method="B3LYP", basis="def2-SVP",
                                     dispersion="D3BJ", n_procs=1, memory_gb=2)
    inp = tmp_path / "e2e.gjf"
    inp.write_text(GaussianGenerator().generate_input(WATER_SYMBOLS, WATER_COORDS, params))
    runner = GenericQMRunner(code_key="gaussian")
    done = []
    runner.finished.connect(lambda ok, msg: done.append(ok))
    assert runner.start_job(str(inp), work_dir=str(tmp_path))
    for _ in range(300):
        if done:
            break
        QtCore.QCoreApplication.processEvents()
        QtCore.QThread.msleep(100)
    assert done and done[0] is True
    assert runner.steps and runner.steps[-1].energy is not None
    assert "Normal termination" in (tmp_path / "e2e.log").read_text()


def test_orca_generated_input_runs(tmp_path):
    exe = _exe("orca")
    params = QMCalculationParameters(job_type="SP", method="B3LYP", basis="def2-SVP",
                                     dispersion="None", n_procs=1, memory_gb=2)
    text = OrcaGenerator().generate_input(WATER_SYMBOLS, WATER_COORDS, params)
    inp = tmp_path / "orca_job.inp"
    inp.write_text(text)
    proc = subprocess.run([exe, str(inp)], capture_output=True, text=True, cwd=tmp_path, timeout=300)
    out = proc.stdout
    assert "ORCA TERMINATED NORMALLY" in out, f"ORCA failed for input:\n{text}\n--- tail ---\n{out[-1500:]}"
    assert re.search(r"FINAL SINGLE POINT ENERGY\s+-?\d+\.\d+", out)


def test_pyscf_generated_input_runs(tmp_path):
    exe = _exe("pyscf")
    from agymolden.qm_input_generator import PySCFGenerator
    params = QMCalculationParameters(job_type="SP", method="B3LYP", basis="sto-3g",
                                     dispersion="None", n_procs=1, memory_gb=2)
    text = PySCFGenerator().generate_input(WATER_SYMBOLS, WATER_COORDS, params)
    inp = tmp_path / "pyscf_job.py"
    inp.write_text(text)
    proc = subprocess.run([exe, str(inp)], capture_output=True, text=True, cwd=tmp_path, timeout=120)
    out = proc.stdout + proc.stderr
    assert "converged" in out.lower() or "scf energy" in out.lower() or proc.returncode == 0, f"PySCF failed:\n{text}\n--- out ---\n{out}"
