"""
Unit tests for AgyMolden Interactive Embedded Python Console.
"""

import pytest
import numpy as np
from PyQt6 import QtCore, QtWidgets

from agymolden.dock_console import AgyScriptingAPI, PythonConsoleWidget
from agymolden.dock_panels import WorkspaceDockManager, PythonConsoleDock


class DummyZAtom:
    def __init__(self, sym: str):
        self.symbol = sym


class DummyMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.zatoms = [DummyZAtom("C"), DummyZAtom("O")]
        self.coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=float)
        self.viewer = None


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_scripting_api_geometry(qapp):
    win = DummyMainWindow()
    api = AgyScriptingAPI(win)

    assert api.symbols == ["C", "O"]
    assert np.allclose(api.coords, [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    assert pytest.approx(api.distance(0, 1), 0.001) == 1.2

    # Translation
    api.translate(1.0, 2.0, 3.0)
    assert np.allclose(win.coords, [[1.0, 2.0, 3.0], [2.2, 2.0, 3.0]])

    # Center
    api.center()
    centroid = np.mean(win.coords, axis=0)
    assert np.allclose(centroid, [0.0, 0.0, 0.0])

    # Rotation (90 deg z)
    c_before = np.copy(win.coords)
    api.rotate(90.0, "z")
    # Distance between atoms should remain 1.2 Å
    assert pytest.approx(api.distance(0, 1), 0.001) == 1.2


def test_console_widget_execution(qapp):
    win = DummyMainWindow()
    console = PythonConsoleWidget(None, main_window=win)

    # Test arithmetic expression
    console.txt_input.setText("1 + 1")
    console._on_execute()
    assert "2" in console.txt_output.toPlainText()

    # Test API command execution
    console.txt_input.setText("agy.translate(5.0, 0.0, 0.0)")
    console._on_execute()
    assert np.allclose(win.coords[0, 0], 5.0)
    assert np.allclose(win.coords[1, 0], 6.2)

    # Test print command
    console.txt_input.setText("print('Chemistry is cool')")
    console._on_execute()
    assert "Chemistry is cool" in console.txt_output.toPlainText()

    # Test command history
    assert len(console.history) >= 3


def test_dock_manager_integration(qapp):
    win = QtWidgets.QMainWindow()
    mgr = WorkspaceDockManager(win)

    dock = mgr.get_or_create_console()
    assert isinstance(dock, PythonConsoleDock)
    assert "python_console" in mgr.docks
    assert mgr.get_or_create_console() is dock
