"""
Unit tests for AgyMolden Unified Dockable Workspace System (Phase C).
"""

import os
import pytest
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from agymolden.dock_panels import (
    WorkspaceDockWidget,
    JobMonitorDock,
    QMOutputDock,
    DihedralScanDock,
    WorkspaceDockManager,
)


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_workspace_dock_widget_basics(qapp):
    win = QtWidgets.QMainWindow()
    dock = WorkspaceDockWidget("Test Dock", win)
    win.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
    assert dock.windowTitle() == "Test Dock"
    assert dock.isFloating() is False
    dock.toggle_floating()
    assert dock.isFloating() is True
    dock.toggle_floating()
    assert dock.isFloating() is False


def test_dock_manager_creation_and_panels(qapp):
    win = QtWidgets.QMainWindow()
    mgr = WorkspaceDockManager(win)

    # Test Job Monitor dock
    job_dock = mgr.get_or_create_job_monitor()
    assert isinstance(job_dock, JobMonitorDock)
    assert "job_monitor" in mgr.docks

    # Getting again returns the same instance
    assert mgr.get_or_create_job_monitor() is job_dock

    # Test Output Viewer dock
    out_dock = mgr.get_or_create_output_viewer()
    assert isinstance(out_dock, QMOutputDock)
    assert "output_viewer" in mgr.docks

    # Float all and dock all
    mgr.float_all_docks()
    assert job_dock.isFloating() is True
    assert out_dock.isFloating() is True

    mgr.dock_all_panels()
    assert job_dock.isFloating() is False
    assert out_dock.isFloating() is False


def test_dihedral_scan_dock(qapp):
    win = QtWidgets.QMainWindow()
    mgr = WorkspaceDockManager(win)

    symbols = ["C", "C", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.5, 1.0, 0.0]
    ], dtype=float)
    bonds = [(0, 1), (0, 2), (1, 3)]
    atom_indices = (2, 0, 1, 3)

    scan_dock = mgr.open_dihedral_scan(
        symbols=symbols,
        coords=coords,
        bonds=bonds,
        atom_indices=atom_indices,
    )
    assert isinstance(scan_dock, DihedralScanDock)
    assert "dihedral_scan" in mgr.docks


def test_workspace_layout_save_reset(qapp):
    win = QtWidgets.QMainWindow()
    mgr = WorkspaceDockManager(win)
    mgr.get_or_create_job_monitor()
    mgr.get_or_create_output_viewer()

    # Layout save & reset should run without exceptions
    mgr.save_layout()
    mgr.reset_layout()
    mgr.restore_layout()
