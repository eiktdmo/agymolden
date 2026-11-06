"""
Tests for automatic session recovery (.agyproj) and batch calculation queue in JobMonitorDialog.
"""

import os
import pytest
from PyQt6 import QtWidgets

from agymolden.zmatrix import ZAtom
from agymolden.project_bundle import save_project_bundle, load_project_bundle
from agymolden.calc_runner import JobMonitorDialog, OrcaRunner


def test_autosave_recovery_project_format(tmp_path):
    recovery_file = str(tmp_path / "autosave_recovery.agyproj")

    zatoms = [
        ZAtom(symbol="O", ref1=None, dist=0.0, ref2=None, angle=0.0, ref3=None, dihedral=0.0),
        ZAtom(symbol="H", ref1=0, dist=0.96, ref2=None, angle=0.0, ref3=None, dihedral=0.0),
        ZAtom(symbol="H", ref1=0, dist=0.96, ref2=1, angle=104.5, ref3=None, dihedral=0.0),
    ]
    coords = [[0.0, 0.0, 0.0], [0.0, 0.75, 0.58], [0.0, -0.75, 0.58]]

    assert save_project_bundle(
        recovery_file,
        zatoms=zatoms,
        coords=coords,
        metadata={"title": "Autosave Recovery Test"},
    ) is True
    assert os.path.isfile(recovery_file)

    loaded = load_project_bundle(recovery_file)
    assert loaded is not None
    assert len(loaded["zatoms"]) == 3
    assert loaded["zatoms"][0].symbol == "O"


def test_job_monitor_batch_queue(tmp_path, qapp):
    f1 = str(tmp_path / "job1.inp")
    f2 = str(tmp_path / "job2.gjf")
    with open(f1, "w") as f:
        f.write("! HF def2-SVP SP\n")
    with open(f2, "w") as f:
        f.write("#p HF/STO-3G SP\n")

    runner = OrcaRunner()
    dlg = JobMonitorDialog(runner=runner)

    # Populate batch queue manually
    dlg.batch_queue = [
        {"path": f1, "engine": "orca", "status": "Pending"},
        {"path": f2, "engine": "gaussian", "status": "Pending"},
    ]
    dlg._refresh_queue_table()
    assert dlg.tbl_queue.rowCount() == 2

    # Mock execution sequence
    dispatched = []
    dlg.run_calculation = lambda p, runner_type=None: dispatched.append((p, runner_type))

    dlg._on_start_queue()
    assert len(dispatched) == 1
    assert dispatched[0] == (f1, "orca")
    assert dlg.batch_queue[0]["status"] == "Running"

    # Simulate completion of job 1
    dlg._advance_queue(completed_success=True)
    assert dlg.batch_queue[0]["status"] == "Done"
    assert dlg.current_queue_idx == 1
