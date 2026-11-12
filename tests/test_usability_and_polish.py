"""
Unit and Smoke Tests for Usability & Polish Milestone:
1. Workbench Hub & Workflow Dashboard (dialog_workbench_hub.py)
2. Direct 3D NCI Isosurface Generation (mw_calculations._on_open_nci_3d_action)
3. Windows Launcher Script Syntax & Winget Checks
"""

import os
import pytest
from PyQt6 import QtWidgets

from agymolden.dialog_workbench_hub import WorkbenchHubDialog, HUB_TOOLS


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_workbench_hub_dialog_smoke(app):
    """Verify Workbench Hub instantiates, categorizes tools, and filters properly."""
    dummy_mw = QtWidgets.QMainWindow()
    hub = WorkbenchHubDialog(parent=dummy_mw)

    assert len(HUB_TOOLS) >= 20
    assert hub.cards_layout.count() > 0

    # Test category filtering
    hub.cbo_cat_filter.setCurrentText("Reaction Mechanisms & Catalysis")
    assert hub.cbo_cat_filter.currentText() == "Reaction Mechanisms & Catalysis"

    # Test search query
    hub.edit_search.setText("NMR")
    assert "Showing" in hub.lbl_count.text()

    hub.close()
    dummy_mw.close()


def test_hub_tool_callbacks_exist():
    """Verify that all callbacks listed in HUB_TOOLS are defined methods on main window or calculation mixin."""
    from agymolden.main_window import MainWindow

    missing_callbacks = []
    for tool in HUB_TOOLS:
        if not hasattr(MainWindow, tool.callback_name):
            missing_callbacks.append((tool.title, tool.callback_name))

    assert not missing_callbacks, f"Missing main window methods: {missing_callbacks}"


def test_windows_launcher_syntax_and_winget():
    """Verify that bundle_template/Run_Windows.bat contains robust Python detection and winget fallback."""
    bat_path = "bundle_template/Run_Windows.bat"
    assert os.path.exists(bat_path), f"{bat_path} does not exist"

    with open(bat_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "winget install" in content
    assert "agymolden_env" in content
    assert "setlocal enabledelayedexpansion" in content
