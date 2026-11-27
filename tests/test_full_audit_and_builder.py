"""Comprehensive audit tests for 3D-first diatomic building, anti-crowding layout, and all Agemacs invocation paths."""

import json
import os
import subprocess
import numpy as np
import pytest
from PyQt6 import QtCore, QtNetwork, QtWidgets


@pytest.fixture
def main_win(qapp):
    from agymolden.main_window import MainWindow

    win = MainWindow()
    yield qapp, win
    win.close()


def test_diatomic_build_without_zmatrix_and_persisted_toggle(main_win, tmp_path, monkeypatch):
    """Build CO triple-bond diatomic via 1-click element pills & bond-order strip with Z-Matrix hidden, plus Option D persistence."""
    from agymolden.zmatrix import ZAtom

    app, win = main_win
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.NativeFormat,
        QtCore.QSettings.Scope.UserScope,
        str(tmp_path),
    )
    monkeypatch.setenv("AGYMOLDEN_TEST_PERSIST_UI", "1")

    # Hide right pane (3D-first mode) and disable auto-H for raw diatomic building
    win._on_toggle_right_pane(False)
    win.chk_auto_h.setChecked(False)
    assert win.right_stack.isHidden()
    assert not win.btn_toggle_right_pane.isChecked()

    # 1-click C pill places C at origin
    win.btn_elem_pills["C"].click()
    assert len(win.zatoms) == 1
    assert win.zatoms[0].symbol == "C"

    # Add bonded N (simulating 3D click/drag) -> N2 diatomic
    n2_coords = np.array([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]])
    win.set_molecule_from_zatoms([ZAtom("N"), ZAtom("N", 0, 1.10)], exact_coords=n2_coords)

    # 1-click triple bond button applies bond order 3 to N2 diatomic without opening Z-Matrix
    win.btn_bo_3.click()
    assert win.viewer.bond_orders.get((0, 1)) == 3
    assert win.right_stack.isHidden()

    # H pill on empty canvas
    win.zatoms = []
    win.coords = np.empty((0, 3))
    win.btn_elem_pills["H"].click()
    assert len(win.zatoms) == 1
    assert win.zatoms[0].symbol == "H"

    # Toggle 2D | 3D split on & off
    win._on_toggle_2d_3d_split(True)
    assert not win.right_stack.isHidden()
    assert win.right_stack.currentWidget() is win.sketcher_2d_pane
    win._on_toggle_2d_3d_split(False)
    assert win.right_stack.currentWidget() is win.right_container

    # Toggle Z-Matrix off & on persists ui/show_right_pane
    win._on_toggle_right_pane(False)
    from agymolden.session_state import get_settings

    assert get_settings("Preferences").value("ui/show_right_pane", type=bool) is False
    win._on_toggle_right_pane(True)
    assert get_settings("Preferences").value("ui/show_right_pane", type=bool) is True


def test_anti_crowding_dock_tabification_and_viewport_width(main_win):
    """Opening multiple docks and toggling right_stack must tabify docks and never crush the 3D viewport."""
    app, win = main_win
    win.resize(1280, 720)
    win.show()
    app.processEvents()

    # Open 3 bottom docks -> must be tabified
    job_dock = win.dock_manager.get_or_create_job_monitor()
    con_dock = win.dock_manager.get_or_create_console()
    hpc_dock = win.dock_manager.get_or_create_hpc_queue()
    app.processEvents()
    assert con_dock in win.tabifiedDockWidgets(job_dock)
    assert hpc_dock in win.tabifiedDockWidgets(job_dock)

    # Open 2 right docks -> must be tabified
    out_dock = win.dock_manager.get_or_create_output_viewer()
    doc_dock = win.dock_manager.open_qc_doctor("SCF converged")
    app.processEvents()
    assert doc_dock in win.tabifiedDockWidgets(out_dock)

    # Opening Z-Matrix / right_stack hides docked right docks so they never sit side-by-side
    win._on_toggle_right_pane(True)
    app.processEvents()
    assert out_dock.isHidden()
    assert doc_dock.isHidden()
    assert win.viewer.width() >= 300


def test_all_agemacs_invocation_paths_and_real_ipc(main_win, tmp_path, monkeypatch):
    """Verify real QLocalServer IPC delivery, Popen fallback, and all 7 UI Agemacs entry points."""
    from agymolden import companion
    from agymolden.calc_runner.dialog_job_monitor import JobMonitorDialog
    from agymolden.dialog_output_viewer import QMOutputViewerDialog
    from agymolden.dialog_qm_input import QMCalculationDialog
    from agymolden.dialog_quick_orca import QuickOrcaDialog
    from agymolden.zmatrix import ZAtom

    app, win = main_win

    # 1. Real QLocalServer IPC test
    received_msgs = []
    srv = QtNetwork.QLocalServer()
    QtNetwork.QLocalServer.removeServer("agemacs_companion_ipc")
    assert srv.listen("agemacs_companion_ipc")

    def _on_conn():
        while srv.hasPendingConnections():
            s = srv.nextPendingConnection()
            if s is not None:
                s.waitForReadyRead(300)
                raw = bytes(s.readAll()).decode("utf-8", errors="ignore").strip()
                s.disconnectFromServer()
                if raw:
                    received_msgs.append(json.loads(raw))

    srv.newConnection.connect(_on_conn)

    sample_file = tmp_path / "water.xyz"
    sample_file.write_text("3\nwater\nO 0 0 0\nH 0 0.75 0.5\nH 0 -0.75 0.5\n", encoding="utf-8")

    # Pass a relative path to verify abspath normalization
    monkeypatch.chdir(tmp_path)
    assert companion.send_to_agemacs(["water.xyz"]) is True
    for _ in range(20):
        app.processEvents()
        if received_msgs:
            break
        QtCore.QThread.msleep(10)
    srv.close()
    assert received_msgs
    assert received_msgs[0]["action"] == "open"
    assert received_msgs[0]["files"] == [str(sample_file)]

    # 2. Popen fallback when no IPC server is listening
    popen_calls = []
    with monkeypatch.context() as m_ctx:
        m_ctx.setattr(companion, "find_agemacs_command", lambda: (["/usr/local/bin/agemacs"], None))
        m_ctx.setattr(subprocess, "Popen", lambda cmd, **kw: popen_calls.append((cmd, kw)))
        assert companion.send_to_agemacs([str(sample_file)]) is True
    assert popen_calls
    assert popen_calls[0][0] == ["/usr/local/bin/agemacs", str(sample_file)]
    assert popen_calls[0][1]["stdin"] is subprocess.DEVNULL
    if os.name != "nt":
        assert popen_calls[0][1]["start_new_session"] is True

    # 3. All 7 UI entry points
    sent = []
    monkeypatch.setattr(companion, "send_to_agemacs", lambda files, parent=None: (sent.extend(files), True)[1])

    out_log = tmp_path / "calc.out"
    out_log.write_text("ORCA TERMINATED NORMALLY\n", encoding="utf-8")
    molden_file = tmp_path / "calc.molden"
    molden_file.write_text("[Molden Format]\n", encoding="utf-8")

    win.current_file_path = str(sample_file)
    win.zatoms = [ZAtom("O"), ZAtom("H", 0, 0.96), ZAtom("H", 0, 0.96, 1, 104.5)]
    win.coords = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    win._last_loaded_qm_log_path = None
    win._active_molden_path = str(molden_file)

    # (a) Toolbar btn_agemacs_menu (first 4 non-browse actions)
    for act in win.btn_agemacs_menu.menu().actions()[:4]:
        act.trigger()
    app.processEvents()
    assert len(sent) >= 4
    assert str(molden_file) in sent  # _active_molden_path picked up!

    # (b) File -> Send / Export to Agemacs submenu
    sent.clear()
    ag_menus = [a.menu() for a in win.menuBar().actions()[0].menu().actions() if a.menu() and "Agemacs" in a.text()]
    assert ag_menus
    ag_menus[0].actions()[0].trigger()
    app.processEvents()
    assert str(sample_file) in sent

    # (c) QM Summary banner button
    sent.clear()
    win.btn_qm_sum_agemacs.click()
    assert str(molden_file) in sent

    # (d) Output Viewer (disk file + unsaved text)
    sent.clear()
    ov = QMOutputViewerDialog(win, file_path=str(out_log))
    ov.btn_open_agemacs.click()
    assert str(out_log) in sent
    ov.file_path = ""
    ov.txt_viewer.setPlainText("raw output")
    ov.btn_open_agemacs.click()
    assert any(p.endswith("agymolden_output.log") for p in sent)
    ov.close()

    # (e) QuickOrcaDialog
    sent.clear()
    d_orca = QuickOrcaDialog(win, zatoms=win.zatoms)
    d_orca.btn_open_agemacs.click()
    assert any(p.endswith(".inp") for p in sent)
    d_orca.close()

    # (f) QMCalculationDialog
    sent.clear()
    d_qm = QMCalculationDialog(win, zatoms=win.zatoms, symbols=["O", "H", "H"], coords=win.coords)
    d_qm.btn_open_agemacs.click()
    assert any("agymolden_input" in p for p in sent)
    d_qm.close()

    # (g) JobMonitorDialog
    sent.clear()
    jm = JobMonitorDialog(win)
    jm.runner = type("R", (), {"input_file": str(sample_file), "log_file": str(out_log), "work_dir": str(tmp_path)})()
    jm._on_open_in_agemacs()
    assert str(sample_file) in sent and str(out_log) in sent
    jm.close()
