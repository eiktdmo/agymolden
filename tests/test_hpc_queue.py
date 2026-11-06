"""
Unit tests for Remote HPC Cluster & Slurm Queue Manager.
"""

import pytest
from PyQt6 import QtWidgets

from agymolden.calc_runner.hpc_queue import parse_squeue_output, HpcJobRecord, HpcQueueWidget
from agymolden.dock_panels import WorkspaceDockManager, HpcQueueDock


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_parse_squeue_pipe_delimited():
    raw = (
        "JOBID|NAME|USER|STATE|TIME|TIME_LIMIT|PARTITION|NODELIST(REASON)\n"
        "1001|opt_ts|andres|RUNNING|01:15:30|24:00:00|gpu|node04\n"
        "1002|crest_scan|andres|PENDING|00:00:00|12:00:00|batch|Priority\n"
    )
    records = parse_squeue_output(raw)
    assert len(records) == 2

    assert records[0].job_id == "1001"
    assert records[0].name == "opt_ts"
    assert records[0].state == "RUNNING"
    assert records[0].time_elapsed == "01:15:30"
    assert records[0].partition == "gpu"
    assert records[0].reason_or_nodes == "node04"

    assert records[1].job_id == "1002"
    assert records[1].state == "PENDING"


def test_parse_squeue_whitespace_fallback():
    raw = (
        "JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)\n"
        "5542 batch opt_b3 andres R 02:40 1 node02\n"
        "5543 batch freq_calc andres PD 00:00 1 (Resources)\n"
    )
    records = parse_squeue_output(raw)
    assert len(records) == 2
    assert records[0].job_id == "5542"
    assert records[0].state == "R"
    assert records[1].job_id == "5543"
    assert records[1].state == "PD"


def test_parse_squeue_empty():
    assert parse_squeue_output("") == []
    assert parse_squeue_output("JOBID|NAME|USER|STATE\n") == []


def test_hpc_queue_widget(qapp):
    widget = HpcQueueWidget()
    fake_jobs = [
        HpcJobRecord("9901", "opt_job", "andres", "RUNNING", "00:10:00", "24:00:00", "gpu", "node01"),
        HpcJobRecord("9902", "freq_job", "andres", "PENDING", "00:00:00", "12:00:00", "batch", "Resources"),
    ]
    widget.jobs = fake_jobs
    widget._populate_table()

    assert widget.table.rowCount() == 2
    assert widget.table.item(0, 0).text() == "9901"
    assert widget.table.item(1, 0).text() == "9902"


def test_hpc_queue_dock(qapp):
    win = QtWidgets.QMainWindow()
    mgr = WorkspaceDockManager(win)

    dock = mgr.get_or_create_hpc_queue()
    assert isinstance(dock, HpcQueueDock)
    assert "hpc_queue" in mgr.docks
    assert mgr.get_or_create_hpc_queue() is dock
