"""Tests for classroom-proof ORCA MPI detection, MCA defaults, and automatic 1-core fallback."""

from __future__ import annotations

import os
import stat
from PyQt6 import QtCore

from agymolden.calc_runner.orca_runner import OrcaRunner


def test_extract_nprocs_variants():
    assert OrcaRunner._extract_nprocs("! B3LYP def2-SVP\n* xyz 0 1\nH 0 0 0\n*") == 1
    assert OrcaRunner._extract_nprocs("! B3LYP def2-SVP PAL4\n* xyz 0 1\nH 0 0 0\n*") == 4
    assert OrcaRunner._extract_nprocs("! B3LYP def2-SVP\n%pal nprocs 6 end\n* xyz 0 1\nH 0 0 0\n*") == 6
    assert OrcaRunner._extract_nprocs("! HF\n%pal\n  nprocs 8\nend\n") == 8


def test_rewrite_input_for_serial_preserves_deck():
    deck = (
        "! B3LYP def2-SVP OPT PAL4\n"
        "%maxcore 2000\n"
        "%pal\n"
        "  nprocs 4\n"
        "end\n"
        "* xyz 0 1\n"
        "H 0.0 0.0 0.0\n"
        "H 0.0 0.0 0.74\n"
        "*\n"
    )
    serial = OrcaRunner._rewrite_input_for_serial(deck)
    assert "PAL4" not in serial
    assert "%pal nprocs 1 end" in serial
    assert "! B3LYP def2-SVP OPT" in serial
    assert "%maxcore 2000" in serial
    assert "H 0.0 0.0 0.74" in serial
    assert OrcaRunner._extract_nprocs(serial) == 1


def test_is_mpi_failure_detection():
    orca_134 = (
        "BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES\n"
        "EXIT CODE: 134\n"
        "ORCA finished by error termination in Startup\n"
        "Calling Command: mpirun -np 4 /opt/orca/orca_startup_mpi\n"
    )
    assert OrcaRunner._is_mpi_failure(orca_134) is True
    assert OrcaRunner._is_mpi_failure("There are not enough slots available in the system") is True
    assert OrcaRunner._is_mpi_failure("error while loading shared libraries: libmpi.so.40") is True
    assert OrcaRunner._is_mpi_failure("SCF NOT CONVERGED AFTER 125 CYCLES") is False


def test_resolve_orca_mpi_env_prefers_openmpi41_strips_conda_and_sets_mca(tmp_path, monkeypatch):
    conda_bin = tmp_path / "anaconda3" / "bin"
    ompi5_bin = tmp_path / "ompi5" / "bin"
    ompi41_bin = tmp_path / "ompi41" / "bin"
    ompi41_lib = tmp_path / "ompi41" / "lib"
    for d in (conda_bin, ompi5_bin, ompi41_bin, ompi41_lib):
        d.mkdir(parents=True)

    (ompi41_lib / "libmpi.so.40").write_text("")

    for path, msg in (
        (conda_bin / "mpirun", "HYDRA version 3.4"),
        (ompi5_bin / "mpirun", "mpirun (Open MPI) 5.0.3"),
        (ompi41_bin / "mpirun", "mpirun (Open MPI) 4.1.8"),
    ):
        path.write_text(f"#!/bin/sh\necho '{msg}'\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setenv("PATH", f"{conda_bin}{os.pathsep}{ompi5_bin}{os.pathsep}{ompi41_bin}")
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{tmp_path / 'anaconda3' / 'lib'}")
    for key in ("OMPI_MCA_rmaps_base_oversubscribe", "OMPI_MCA_btl_vader_single_copy_mechanism", "PMIX_MCA_gds"):
        monkeypatch.delenv(key, raising=False)

    env = OrcaRunner._resolve_orca_mpi_env()
    assert env is not None
    assert "anaconda3" not in env["PATH"].lower()
    assert env["PATH"].split(os.pathsep)[0] == str(ompi41_bin)
    assert str(ompi41_lib) in env["LD_LIBRARY_PATH"].split(os.pathsep)[0]
    assert env["OMPI_MCA_rmaps_base_oversubscribe"] == "1"
    assert env["OMPI_MCA_btl_vader_single_copy_mechanism"] == "none"
    assert env["PMIX_MCA_gds"] == "hash"


def test_preflight_fallback_rewrites_input_to_serial(tmp_path, monkeypatch, qapp):
    fake_orca = tmp_path / "orca"
    fake_orca.write_text("#!/bin/sh\nexit 0\n")
    fake_orca.chmod(fake_orca.stat().st_mode | stat.S_IXUSR)

    inp = tmp_path / "h2.inp"
    inp.write_text("! HF def2-SVP\n%pal nprocs 4 end\n* xyz 0 1\nH 0 0 0\nH 0 0 0.74\n*\n")

    runner = OrcaRunner()
    runner.configure(str(fake_orca))
    launched = []
    monkeypatch.setattr(OrcaRunner, "check_mpi_health", classmethod(lambda cls, orca_bin=None, force=False: (False, "no mpirun")))
    monkeypatch.setattr(runner, "_setup_process", lambda prog, args, work_dir, extra_env=None: launched.append((prog, args)))

    lines = []
    runner.line_received.connect(lambda msg, _err: lines.append(msg))

    assert runner.start_job(str(inp), str(tmp_path)) is True
    assert len(launched) == 1
    assert (tmp_path / "h2.parallel.inp").is_file()
    assert "%pal nprocs 4 end" in (tmp_path / "h2.parallel.inp").read_text()
    assert "%pal nprocs 1 end" in inp.read_text()
    assert any("Automatically switched to 1-core serial mode" in ln for ln in lines)


def test_runtime_mpi_failure_retries_once_in_serial(tmp_path, monkeypatch, qapp):
    fake_orca = tmp_path / "orca"
    fake_orca.write_text("#!/bin/sh\nexit 0\n")
    fake_orca.chmod(fake_orca.stat().st_mode | stat.S_IXUSR)

    inp = tmp_path / "h2.inp"
    inp.write_text("! HF def2-SVP PAL4\n* xyz 0 1\nH 0 0 0\nH 0 0 0.74\n*\n")

    runner = OrcaRunner()
    runner.configure(str(fake_orca))
    calls = []
    monkeypatch.setattr(OrcaRunner, "check_mpi_health", classmethod(lambda cls, orca_bin=None, force=False: (True, "OpenMPI 4.1.8 ready")))
    monkeypatch.setattr(runner, "_setup_process", lambda prog, args, work_dir, extra_env=None: calls.append((prog, args)))

    finished_events = []
    runner.finished.connect(lambda ok, msg: finished_events.append((ok, msg)))

    assert runner.start_job(str(inp), str(tmp_path)) is True
    assert len(calls) == 1
    assert runner._requested_nprocs == 4

    # Simulate ORCA crashing during MPI startup with exit code 134
    runner._raw_log_buffer.extend([
        "BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES",
        "EXIT CODE: 134",
        "ORCA finished by error termination in Startup",
        "Calling Command: mpirun -np 4 orca_startup_mpi",
    ])
    runner._on_process_finished(134, QtCore.QProcess.ExitStatus.NormalExit)

    # Should have backed up, rewritten to serial, and triggered 1 retry without emitting finished yet
    assert len(calls) == 2
    assert finished_events == []
    assert "PAL4" not in inp.read_text()
    assert (tmp_path / "h2.parallel.inp").is_file()
    assert runner._serial_fallback_used is True

    # If the serial retry itself fails, it must NOT loop infinitely
    runner._on_process_finished(1, QtCore.QProcess.ExitStatus.NormalExit)
    assert len(calls) == 2
    assert len(finished_events) == 1
    assert finished_events[0][0] is False
