"""Regression tests for Windows maximize, onboarding theme contrast, output viewer dock, and Agemacs export."""

import pytest
from PyQt6 import QtCore, QtWidgets


@pytest.fixture
def main_win(qapp):
    from agymolden.main_window import MainWindow

    win = MainWindow()
    yield qapp, win
    win.close()


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
    """Closing docked Output Viewer must hide the dock (not inner widget), and _on_view_qm_output_file resolves logs."""
    _, win = main_win
    log_a = tmp_path / "mol.out"
    log_a.write_text("TOTAL SCF ENERGY\nFINAL SINGLE POINT ENERGY -76.0\n", encoding="utf-8")
    mol_file = tmp_path / "mol.molden"
    mol_file.write_text("[Molden Format]\n", encoding="utf-8")

    # Sibling .out resolution from current_file_path (.molden)
    win.current_file_path = str(mol_file)
    win._last_loaded_qm_log_path = None
    win._on_view_qm_output_file()

    dock = win.dock_manager.docks.get("output_viewer")
    assert dock is not None
    assert dock.viewer_dlg.file_path == str(log_a)
    assert "mol.out" in dock.viewer_dlg.lbl_path.text()

    # Clicking Close while docked hides the dock, leaving viewer_dlg visible when reopened
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

    _, win = main_win
    sent = []
    monkeypatch.setattr(companion, "send_to_agemacs", lambda files, parent=None: (sent.extend(files), True)[1])

    # Primary toolbar remains visible when Simple Mode hides toolbar_qm
    win._on_toggle_simple_mode(True)
    assert not win.toolbar_qm.isVisible()
    assert not win.btn_agemacs_menu.isHidden()
    win._on_toggle_simple_mode(False)

    # _on_open_in_agemacs finds current_file_path
    xyz_path = tmp_path / "sample.xyz"
    xyz_path.write_text("1\nH\nH 0 0 0\n", encoding="utf-8")
    win.current_file_path = str(xyz_path)
    win._on_open_in_agemacs()
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
