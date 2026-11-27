"""Regression tests for P1-P5 UI ergonomics, clickability, contrast, and safeguards."""

from __future__ import annotations

from PyQt6 import QtWidgets


def test_mainwindow_shrinks_for_1024x600_and_buttons_are_clickable(qapp):
    from agymolden.main_window import MainWindow

    win = MainWindow()
    win.resize(1024, 600)
    win.show()
    qapp.processEvents()

    assert win.minimumSize().height() <= 600
    assert win.height() <= 600

    for btn in (
        win.btn_mode_toggle,
        win.btn_ribbon_tweezers,
        win.btn_auto_relax,
        win.btn_2d_3d_split,
        win.btn_toggle_right_pane,
        win.btn_clean_selection,
        win.btn_elem_pills["C"],
        win.btn_bo_1,
    ):
        assert btn.height() >= 28, f"Button {btn.text()!r} too small: {btn.height()}px"

    # Redundant brush combo is hidden from ribbon (1-click pills replace it)
    assert win.combo_brush_element.isVisible() is False

    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_help_menu_organized_and_no_duplicate_manual(qapp):
    from agymolden.main_window import MainWindow

    win = MainWindow()
    help_menu = None
    for act in win.menuBar().actions():
        if "Help" in act.text():
            help_menu = act.menu()
            break
    assert help_menu is not None

    non_sep = [a for a in help_menu.actions() if not a.isSeparator()]
    assert len(non_sep) <= 12
    manual_acts = [a for a in non_sep if "User Manual" in a.text()]
    assert len(manual_acts) == 1

    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_sample_library_white_theme_contrast_and_no_conda_labels(qapp, tmp_path):
    from agymolden.main_window import MainWindow
    from agymolden.dialog_samples import SampleGalleryDialog
    from agymolden.calc_runner.dialog_launch import BrorsenLaunchDialog

    win = MainWindow()
    win.set_color_theme("white")
    dlg = SampleGalleryDialog(win)
    assert "#0f172a" in dlg.lbl_title.styleSheet()
    assert "#1e293b" in dlg.lbl_description.styleSheet()
    dlg.close()
    dlg.deleteLater()

    inp = tmp_path / "h2.inp"
    inp.write_text("# test\n")
    launch = BrorsenLaunchDialog(win, str(inp))
    assert "conda" not in launch.combo_conda.toolTip().lower()
    assert "conda" not in launch.txt_remote_py.placeholderText().lower()
    launch.close()
    launch.deleteLater()

    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_qm_input_fits_small_screen_and_quick_orca_buttons_clickable(qapp):
    import inspect
    import agymolden.dialog_qm_input as dqi
    from agymolden.main_window import MainWindow
    from agymolden.dialog_quick_orca import QuickOrcaDialog

    QMCls = [c for _, c in inspect.getmembers(dqi, inspect.isclass) if issubclass(c, QtWidgets.QDialog)][0]
    win = MainWindow()

    qi = QMCls(win, symbols=["H", "H"], coords=[[0, 0, 0], [0, 0, 0.74]], zatoms=win.zatoms)
    qi.show()
    qapp.processEvents()
    assert qi.minimumSize().width() <= 850
    assert qi.btn_open_agemacs.height() >= 28
    assert qi.btn_chain_stage.height() >= 28
    qi.close()
    qi.deleteLater()

    qo = QuickOrcaDialog(win, zatoms=win.zatoms)
    qo.show()
    qapp.processEvents()
    assert qo.btn_use_all_cores.height() >= 28
    qo.close()
    qo.deleteLater()

    win.show()
    qapp.processEvents()
    assert win.welcome_banner.isVisible() is True
    win.btn_dismiss_welcome.click()
    qapp.processEvents()
    assert win.welcome_banner.isVisible() is False

    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_job_monitor_nonmodal_viewers_and_no_tiny_10px_fonts(qapp):
    import glob
    import re
    from agymolden.calc_runner.dialog_job_monitor import JobMonitorDialog

    assert hasattr(JobMonitorDialog, "_open_viewer_nonmodal")
    jm_src = open("agymolden/calc_runner/dialog_job_monitor.py", encoding="utf-8").read()
    assert "QMOutputViewerDialog" not in jm_src or ".exec()" not in jm_src.split("def _open_viewer_nonmodal")[1].split("def _on_filter_changed")[0]

    tiny_matches = []
    for path in glob.glob("agymolden/**/*.py", recursive=True):
        txt = open(path, encoding="utf-8", errors="ignore").read()
        if re.search(r"font-size\s*:\s*(?:8|9|10)px", txt):
            tiny_matches.append(path)
    assert tiny_matches == [], f"Found tiny <=10px font-size rules in: {tiny_matches}"

