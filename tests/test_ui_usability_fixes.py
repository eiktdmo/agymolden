"""Regression tests for Windows maximize, onboarding theme contrast, output viewer dock, Agemacs export, Wayland safety, and trajectory parsing."""

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets


@pytest.fixture
def main_win(qapp):
    from agymolden.main_window import MainWindow

    win = MainWindow()
    yield qapp, win
    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_mainwindow_maximizable_without_finite_cap(main_win):
    """MainWindow must not cap maximumSize to screen geometry (which disables Maximize on Windows)."""
    _, win = main_win
    assert win.maximumWidth() >= 16777215
    assert win.maximumHeight() >= 16777215
    flags = win.windowFlags()
    assert flags & QtCore.Qt.WindowType.WindowMaximizeButtonHint
    assert flags & QtCore.Qt.WindowType.WindowMinimizeButtonHint


def test_onboarding_dialog_theme_contrast_and_maximize(qapp):
    """OnboardingTourDialog cards must use readable contrast in both light and dark themes and allow maximize."""
    from agymolden.dialog_onboarding import OnboardingTourDialog

    class _FakeParent(QtWidgets.QWidget):
        def __init__(self, mode):
            super().__init__()
            self.viewer = type("V", (), {"bg_color_mode": mode})()

    p_light = _FakeParent("white")
    dlg_light = OnboardingTourDialog(p_light)
    assert dlg_light.windowFlags() & QtCore.Qt.WindowType.WindowMaximizeButtonHint
    cards_light = dlg_light.pages_stack.widget(0).findChildren(QtWidgets.QFrame)
    assert cards_light
    qss_light = cards_light[0].styleSheet()
    assert "#f8fafc" in qss_light
    assert "color: #0f172a" in qss_light
    dlg_light.close()

    p_dark = _FakeParent("dark")
    dlg_dark = OnboardingTourDialog(p_dark)
    cards_dark = dlg_dark.pages_stack.widget(0).findChildren(QtWidgets.QFrame)
    assert cards_dark
    qss_dark = cards_dark[0].styleSheet()
    assert "#0f172a" in qss_dark
    assert "color: #f1f5f9" in qss_dark
    dlg_dark.close()


def test_output_viewer_dock_close_reopen_and_path_resolution(main_win, tmp_path):
    """_on_view_qm_output_file opens standalone non-modal Output Viewer; dock Close hides dock without hiding inner widget."""
    _, win = main_win
    log_a = tmp_path / "mol.out"
    log_a.write_text("TOTAL SCF ENERGY\nFINAL SINGLE POINT ENERGY -76.0\n", encoding="utf-8")
    mol_file = tmp_path / "mol.molden"
    mol_file.write_text("[Molden Format]\n", encoding="utf-8")

    # Sibling .out resolution from current_file_path (.molden) opens standalone non-modal window
    win.current_file_path = str(mol_file)
    win._last_loaded_qm_log_path = None
    win._on_view_qm_output_file()

    ov_win = getattr(win, "_output_viewer_win", None)
    assert ov_win is not None
    assert not ov_win.isModal()
    assert ov_win.file_path == str(log_a)
    assert "mol.out" in ov_win.lbl_path.text()
    assert ov_win._find_main_window() is win
    ov_win.close()

    # Dockable Output Viewer: clicking Close while docked hides the dock, leaving viewer_dlg visible when reopened
    dock = win.dock_manager.get_or_create_output_viewer(str(log_a))
    assert dock is not None
    assert dock.viewer_dlg.file_path == str(log_a)
    dock.viewer_dlg._on_close_clicked()
    assert dock.isHidden()

    log_b = tmp_path / "second.out"
    log_b.write_text("FINAL SINGLE POINT ENERGY -100.0\n", encoding="utf-8")
    win.dock_manager.get_or_create_output_viewer(str(log_b))
    assert not dock.isHidden()
    assert not dock.viewer_dlg.isHidden()
    assert dock.viewer_dlg.file_path == str(log_b)
    assert "second.out" in dock.viewer_dlg.lbl_path.text()


def test_agemacs_export_visibility_and_attr_lookup(main_win, tmp_path, monkeypatch):
    """btn_agemacs_menu stays visible in Simple Mode and exports current_file_path / _last_loaded_qm_log_path / zatoms."""
    from agymolden import companion
    from agymolden.zmatrix import ZAtom

    app, win = main_win
    sent = []
    monkeypatch.setattr(companion, "send_to_agemacs", lambda files, parent=None: (sent.extend(files), True)[1])

    # Primary toolbar remains visible when Simple Mode hides toolbar_qm
    win._on_toggle_simple_mode(True)
    assert not win.toolbar_qm.isVisible()
    assert win.toolbar_qm is not win.btn_agemacs_menu.parentWidget()
    assert not win.btn_agemacs_menu.parentWidget().isHidden()
    win._on_toggle_simple_mode(False)

    # Triggering menu action uses deferred QTimer.singleShot(0, ...)
    xyz_path = tmp_path / "sample.xyz"
    xyz_path.write_text("1\nH\nH 0 0 0\n", encoding="utf-8")
    win.current_file_path = str(xyz_path)
    win.btn_agemacs_menu.menu().actions()[0].trigger()
    app.processEvents()
    assert str(xyz_path) in sent

    # _on_send_qm_output_to_agemacs finds _last_loaded_qm_log_path
    sent.clear()
    out_path = tmp_path / "calc.out"
    out_path.write_text("ORCA TERMINATED NORMALLY\n", encoding="utf-8")
    win._last_loaded_qm_log_path = str(out_path)
    win._on_send_qm_output_to_agemacs()
    assert str(out_path) in sent

    # _on_send_zmat_to_agemacs falls back to zatoms when text_raw_zmat is empty
    sent.clear()
    if hasattr(win, "text_raw_zmat"):
        win.text_raw_zmat.clear()
    win.zatoms = [ZAtom("O"), ZAtom("H", 0, 0.96)]
    win._on_send_zmat_to_agemacs()
    assert any(p.endswith(".zmat") for p in sent)


def test_orca_runner_prefers_openmpi_over_hydra(tmp_path, monkeypatch):
    """OrcaRunner._resolve_orca_mpi_env must skip MPICH/Hydra mpirun and prepend OpenMPI."""
    import sys
    from agymolden.calc_runner.orca_runner import OrcaRunner

    if sys.platform != "linux":
        pytest.skip("Linux-specific MPI PATH resolution")

    conda_bin = tmp_path / "conda" / "bin"
    conda_bin.mkdir(parents=True)
    hydra_mpirun = conda_bin / "mpirun"
    hydra_mpirun.write_text("#!/bin/sh\necho 'HYDRA build details: Version 4.1.1'\n", encoding="utf-8")
    hydra_mpirun.chmod(0o755)

    ompi_prefix = tmp_path / "openmpi"
    ompi_bin = ompi_prefix / "bin"
    ompi_lib = ompi_prefix / "lib"
    ompi_bin.mkdir(parents=True)
    ompi_lib.mkdir(parents=True)
    ompi_mpirun = ompi_bin / "mpirun"
    ompi_mpirun.write_text("#!/bin/sh\necho 'mpirun (Open MPI) 5.0.9'\n", encoding="utf-8")
    ompi_mpirun.chmod(0o755)
    (ompi_lib / "libmpi.so.40").write_text("", encoding="utf-8")

    monkeypatch.setenv("PATH", f"{conda_bin}:{ompi_bin}:/usr/bin:/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{tmp_path / 'anaconda3' / 'lib'}")

    resolved = OrcaRunner._resolve_orca_mpi_env()
    assert resolved is not None
    assert resolved["PATH"].split(":")[0] == str(ompi_bin)
    assert resolved["LD_LIBRARY_PATH"].split(":")[0] == str(ompi_lib)
    assert "conda" not in resolved["PATH"].lower()
    assert "conda" not in resolved["LD_LIBRARY_PATH"].lower()


def test_output_viewer_dock_dialog_parent_and_widget_flags(main_win, tmp_path):
    """Docked panels must use Qt.WindowType.Widget, _dialog_parent() must return MainWindow, and Gaussian steps index."""
    _, win = main_win
    g_log = tmp_path / "gauss_opt.log"
    g_log.write_text(
        "Step number   1 out of a maximum of  20\n"
        "SCF Done:  E(RHF) = -76.020000\n"
        "Step number   2 out of a maximum of  20\n"
        "SCF Done:  E(RHF) = -76.021100\n",
        encoding="utf-8",
    )
    dock = win.dock_manager.get_or_create_output_viewer(str(g_log))
    assert not dock.viewer_dlg.isWindow()
    assert dock.viewer_dlg._dialog_parent() is win
    labels = [dock.viewer_dlg.list_sections.item(i).text() for i in range(dock.viewer_dlg.list_sections.count())]
    assert any("Optimization Step 1" in lbl for lbl in labels)
    assert any("Optimization Step 2" in lbl for lbl in labels)

    job_dock = win.dock_manager.get_or_create_job_monitor()
    assert not job_dock.monitor.isWindow()


def test_ui_safeguards_skips_messagebox_and_filedialog(qapp, monkeypatch):
    """LowResolutionSafeguardFilter must not clamp QMessageBox or QFileDialog."""
    from agymolden import ui_safeguards

    called = []
    monkeypatch.setattr(ui_safeguards, "clamp_window_to_screen", lambda obj: (called.append(obj), True)[1])
    flt = ui_safeguards.LowResolutionSafeguardFilter()
    msg = QtWidgets.QMessageBox()
    fdlg = QtWidgets.QFileDialog()
    ev = QtCore.QEvent(QtCore.QEvent.Type.Show)
    flt.eventFilter(msg, ev)
    flt.eventFilter(fdlg, ev)
    assert called == []


def test_mainwindow_event_filter_returns_bool_on_copy_paste(main_win, monkeypatch):
    """MainWindow.eventFilter must return a bool (never None) on Ctrl+C and Ctrl+V."""
    from PyQt6 import QtGui

    _, win = main_win
    monkeypatch.setattr(win, "_copy_selected_fragment", lambda: None)
    monkeypatch.setattr(win, "_on_paste", lambda: None)
    win.viewer.setFocus()
    ev_c = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_C, QtCore.Qt.KeyboardModifier.ControlModifier)
    ev_v = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_V, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert isinstance(win.eventFilter(win, ev_c), bool)
    assert isinstance(win.eventFilter(win, ev_v), bool)


def test_orca_crlf_multistep_and_cclib_fallback(tmp_path, monkeypatch):
    """CRLF ORCA multi-step optimizations must parse all steps even if cclib returns only 1 frame."""
    from agymolden import qm_log_parser

    orca_out = tmp_path / "orca_crlf.out"
    content = (
        "                                 *****************\r\n"
        "                                 * O   R   C   A *\r\n"
        "                                 *****************\r\n"
        "CARTESIAN COORDINATES (ANGSTROEM)\r\n"
        "---------------------------------\r\n"
        "  H      0.000000    0.000000    0.000000\r\n"
        "  H      0.000000    0.000000    0.700000\r\n\r\n"
        "FINAL SINGLE POINT ENERGY      -1.10000000\r\n"
        "CARTESIAN COORDINATES (ANGSTROEM)\r\n"
        "---------------------------------\r\n"
        "  H      0.000000    0.000000    0.000000\r\n"
        "  H      0.000000    0.000000    0.740000\r\n\r\n"
        "FINAL SINGLE POINT ENERGY      -1.11700000\r\n"
        "                             ****ORCA TERMINATED NORMALLY****\r\n"
    )
    orca_out.write_bytes(content.encode("utf-8"))

    # Simulate a cclib parser returning only 1 coordinate frame
    class _StubData:
        atomnos = [1, 1]
        atomcoords = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]])
        scfenergies = np.array([-30.39])

    class _ORCAparser:
        def parse(self):
            return _StubData()

    monkeypatch.setattr(qm_log_parser.cclib.io, "ccopen", lambda _p: _ORCAparser())
    res = qm_log_parser.parse_qm_logfile(str(orca_out))
    assert len(res.trajectory_coords) == 2
    assert len(res.trajectory_energies) == 2


def test_trajectory_slider_min_width_and_tab_restore(main_win):
    """slider_trajectory must have >=90px minWidth and tab switching must preserve trajectory steps."""
    from agymolden.zmatrix import ZAtom

    _, win = main_win
    assert win.slider_trajectory.minimumWidth() >= 90

    win.zatoms = [ZAtom("H"), ZAtom("H", 0, 0.74)]
    win.coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    win.trajectory_coords = [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.70]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
    ]
    win.trajectory_energies = [-29.9, -30.4]
    win.trajectory_symbols = ["H", "H"]
    win.grp_trajectory.setVisible(True)
    win.slider_trajectory.setRange(0, 1)
    win.slider_trajectory.setValue(0)

    snap = win._snapshot_current_mol_tab()
    win._clear_stale_surfaces_and_qm_state()
    assert win.grp_trajectory.isHidden()

    win._restore_mol_tab(snap)
    assert len(win.trajectory_coords) == 2
    assert not win.grp_trajectory.isHidden()
    assert win.slider_trajectory.maximum() == 1
    assert win.slider_trajectory.value() == 0


def test_brorsen_env_detection_excludes_conda():
    """detect_conda_environments must never return Conda/Anaconda/Mamba interpreters."""
    from agymolden.calc_runner.brorsen_runner import detect_conda_environments

    envs = detect_conda_environments(force_refresh=True)
    for label, path in envs:
        low = f"{label} {path}".lower()
        assert "conda" not in low
        assert "anaconda" not in low
        assert "mambaforge" not in low
        assert "miniforge" not in low


def test_default_cores_is_one_and_persists_last_calc(main_win, tmp_path, monkeypatch):
    """Default core count must be 1 everywhere, and QuickOrcaDialog / QMCalculationDialog must recall last calc."""
    from PyQt6 import QtCore
    from agymolden.dialog_qm_input import QMCalculationDialog
    from agymolden.dialog_quick_orca import QuickOrcaDialog
    from agymolden.qm_input_generator import QMCalculationParameters

    assert QMCalculationParameters().n_procs == 1

    _, win = main_win
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.NativeFormat,
        QtCore.QSettings.Scope.UserScope,
        str(tmp_path),
    )
    monkeypatch.setenv("AGYMOLDEN_TEST_RESTORE_CALC", "1")

    # 1. QuickOrcaDialog: default cores == 1, save & restore
    d1 = QuickOrcaDialog(win, zatoms=[])
    assert d1.spn_cores.value() == 1
    d1.spn_cores.setValue(3)
    d1.spn_mem.setValue(4000)
    d1.combo_basis.setCurrentText("def2-TZVP")
    d1._save_last_calc_settings()
    d1.close()

    d2 = QuickOrcaDialog(win, zatoms=[])
    assert d2.spn_cores.value() == 3
    assert d2.spn_mem.value() == 4000
    assert d2.combo_basis.currentText() == "def2-TZVP"
    d2.close()

    # 2. QMCalculationDialog: default cores == 1, save & restore
    q1 = QMCalculationDialog(win, zatoms=[], symbols=["O", "H", "H"], coords=np.zeros((3, 3)))
    assert q1.spin_procs.value() == 1
    q1.spin_procs.setValue(2)
    q1.spin_mem.setValue(8)
    q1.combo_basis.setCurrentText("cc-pVTZ")
    q1._save_last_calc_settings()
    q1.close()

    q2 = QMCalculationDialog(win, zatoms=[], symbols=["O", "H", "H"], coords=np.zeros((3, 3)))
    assert q2.spin_procs.value() == 2
    assert q2.spin_mem.value() == 8
    assert q2.combo_basis.currentText() == "cc-pVTZ"
    q2.close()


def test_output_viewer_section_scroll_and_nonmodal_workers_and_companion_env(qapp, tmp_path, monkeypatch):
    """Section clicks/selection scroll Output Viewer, run_in_background is NonModal/delayed, and companion env is clean."""
    from PyQt6 import QtCore, QtWidgets
    from agymolden import companion, workers
    from agymolden.dialog_output_viewer import QMOutputViewerDialog

    # 1. Output Viewer section jump scrolls verticalScrollBar and moves cursor
    long_log = tmp_path / "long.out"
    filler = "\n".join(f"SCF iter {i}" for i in range(120))
    long_log.write_text(f"{filler}\nFINAL SINGLE POINT ENERGY -76.123456\n", encoding="utf-8")
    ov = QMOutputViewerDialog(None, file_path=str(long_log))
    ov.show()
    qapp.processEvents()
    ov.list_sections.setCurrentRow(0)
    qapp.processEvents()
    assert ov.txt_viewer.textCursor().blockNumber() >= 100
    assert ov.txt_viewer.verticalScrollBar().value() > 0
    ov.close()

    # 2. workers.run_in_background creates a NonModal QProgressDialog and does not show immediately
    created_dlgs = []
    orig_qpd = QtWidgets.QProgressDialog

    class _TrackedQPD(orig_qpd):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created_dlgs.append(self)

    monkeypatch.setattr(QtWidgets, "QProgressDialog", _TrackedQPD)
    monkeypatch.setattr(workers, "SYNCHRONOUS", False)
    done_vals = []
    workers.run_in_background(None, "Busy", "Working...", lambda: 42, lambda r: done_vals.append(r), lambda e: None)
    assert created_dlgs
    assert created_dlgs[0].windowModality() == QtCore.Qt.WindowModality.NonModal
    assert not created_dlgs[0].isVisible()
    qapp.processEvents()
    # Wait briefly for worker thread to finish
    for _ in range(40):
        if done_vals:
            break
        QtCore.QThread.msleep(15)
        qapp.processEvents()
    assert done_vals == [42]

    # 3. companion._clean_subprocess_env strips wrapper vars and restores _AGYMOLDEN_ORIG_LD_PRELOAD
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/fake_venv")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    monkeypatch.setenv("_AGYMOLDEN_ORIG_LD_PRELOAD", "")
    clean = companion._clean_subprocess_env()
    assert "VIRTUAL_ENV" not in clean
    assert "QT_QPA_PLATFORM" not in clean
    assert "LD_PRELOAD" not in clean


