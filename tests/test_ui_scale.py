"""Tests for built-in UI scaling, live font-size controls, and stylesheet scaling."""

from __future__ import annotations

import os
import pytest
from PyQt6 import QtCore, QtWidgets

from agymolden.ui_scale import (
    DEFAULT_FONT_PT,
    _scale_stylesheet_fonts,
    apply_live_font_size,
    apply_startup_scale_env,
    get_current_font_size,
    get_saved_font_size,
    get_saved_ui_scale,
    install_ui_scale_filter,
    save_font_size,
    save_ui_scale,
)


@pytest.fixture(autouse=True)
def _isolate_settings_and_font(qapp, tmp_path, monkeypatch):
    import agymolden.session_state as session_state

    def _tmp_settings(app: str = "AgyMolden"):
        return QtCore.QSettings(str(tmp_path / f"{app}.ini"), QtCore.QSettings.Format.IniFormat)

    monkeypatch.setattr(session_state, "get_settings", _tmp_settings)
    yield
    apply_live_font_size(DEFAULT_FONT_PT, app=qapp, persist=False)


def test_scale_stylesheet_fonts_px_and_pt():
    qss = "QPushButton { background-color: #2563eb; font-size: 11px; padding: 4px; } QLabel { font-size: 10pt; }"
    scaled = _scale_stylesheet_fonts(qss, 1.4)
    assert "font-size: 15px;" in scaled
    assert "font-size: 14pt;" in scaled
    assert "padding: 4px;" in scaled

    # Minimum clamp of 7
    tiny = _scale_stylesheet_fonts("QLabel { font-size: 8px; }", 0.5)
    assert "font-size: 7px;" in tiny


def test_live_font_size_no_compound_scaling_and_restores_exact(qapp):
    btn = QtWidgets.QPushButton("Test")
    orig_qss = "QPushButton { font-size: 11px; font-weight: bold; padding: 2px 8px; }"
    btn.setStyleSheet(orig_qss)
    btn.show()
    qapp.processEvents()

    apply_live_font_size(14, app=qapp, persist=False)
    assert qapp.font().pointSize() == 14
    assert "font-size: 15px;" in btn.styleSheet()

    # Applying 16pt should scale from base (11px * 1.6 = 18px), not from 15px
    apply_live_font_size(16, app=qapp, persist=False)
    assert qapp.font().pointSize() == 16
    assert "font-size: 18px;" in btn.styleSheet()

    # Resetting to 10pt restores exact original stylesheet
    apply_live_font_size(10, app=qapp, persist=False)
    assert qapp.font().pointSize() == 10
    assert btn.styleSheet() == orig_qss
    btn.close()
    btn.deleteLater()


def test_external_stylesheet_change_rebaselines_cleanly(qapp):
    lbl = QtWidgets.QLabel("Status")
    lbl.setStyleSheet("color: #f8fafc; font-size: 11px;")
    lbl.show()
    qapp.processEvents()

    apply_live_font_size(15, app=qapp, persist=False)
    assert "font-size: 16px;" in lbl.styleSheet() or "font-size: 17px;" in lbl.styleSheet()

    # Simulate theme toggle updating stylesheet externally
    lbl.setStyleSheet("color: #0f172a; font-size: 11px;")
    qapp.processEvents()
    # Event filter should re-baseline to new color and re-apply 1.5x scale
    assert "#0f172a" in lbl.styleSheet()
    assert "font-size: 11px;" not in lbl.styleSheet()

    lbl.close()
    lbl.deleteLater()


def test_newly_polished_dialog_inherits_live_font_scale(qapp):
    install_ui_scale_filter(qapp)
    apply_live_font_size(16, app=qapp, persist=False)

    dlg = QtWidgets.QDialog()
    btn = QtWidgets.QPushButton("Action", dlg)
    btn.setStyleSheet("font-size: 11px; padding: 4px;")
    layout = QtWidgets.QVBoxLayout(dlg)
    layout.addWidget(btn)
    dlg.show()
    qapp.processEvents()

    assert "font-size: 18px;" in btn.styleSheet()
    dlg.close()
    dlg.deleteLater()


def test_startup_scale_env_and_persistence(monkeypatch):
    monkeypatch.delenv("QT_SCALE_FACTOR", raising=False)
    save_ui_scale(1.4)
    assert get_saved_ui_scale() == pytest.approx(1.4, abs=0.01)

    applied = apply_startup_scale_env(None)
    assert applied == pytest.approx(1.4, abs=0.01)
    assert os.environ.get("QT_SCALE_FACTOR") == "1.40"

    # CLI --scale overrides QSettings and env var
    applied_cli = apply_startup_scale_env(1.75)
    assert applied_cli == pytest.approx(1.75, abs=0.01)
    assert os.environ.get("QT_SCALE_FACTOR") == "1.75"

    # Reset saved scale back to 1.0
    save_ui_scale(1.0)
    monkeypatch.delenv("QT_SCALE_FACTOR", raising=False)


def test_mainwindow_actions_and_preferences_dialog(qapp):
    from agymolden.main_window import MainWindow
    from agymolden.dialog_preferences import PreferencesDialog

    win = MainWindow()
    assert hasattr(win, "act_font_inc")
    assert hasattr(win, "act_font_dec")
    assert hasattr(win, "act_font_reset")

    win.act_font_inc.trigger()
    assert get_current_font_size() == 12
    assert qapp.font().pointSize() == 12

    win.act_font_reset.trigger()
    assert get_current_font_size() == 10

    save_font_size(14)
    save_ui_scale(1.25)
    dlg = PreferencesDialog(win)
    assert dlg.spin_cores.value() == 1
    assert dlg.spin_font_size.value() == 14
    assert float(dlg.combo_ui_scale.currentData()) == pytest.approx(1.25, abs=0.01)

    dlg.spin_font_size.setValue(16)
    dlg._save_settings()
    assert get_current_font_size() == 16
    assert get_saved_font_size() == 16

    # Reset persisted settings
    save_font_size(10)
    save_ui_scale(1.0)
    dlg.close()
    dlg.deleteLater()
    win.close()
    win.deleteLater()
    qapp.processEvents()
