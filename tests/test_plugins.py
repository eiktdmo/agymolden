"""Tests for User Scripting & Plugin Runner."""

import numpy as np
import pytest
from agymolden.plugins import execute_plugin_script, PythonScriptRunnerDialog
from agymolden.zmatrix import ZAtom


class DummyWindow:
    def __init__(self):
        self.zatoms = [
            ZAtom(symbol="O", dist=0.0, angle=0.0, dihedral=0.0),
            ZAtom(symbol="H", dist=0.96, angle=0.0, dihedral=0.0, ref1=0),
            ZAtom(symbol="H", dist=0.96, angle=104.5, dihedral=0.0, ref1=0, ref2=1),
        ]
        self.coords = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.96],
            [0.93, 0.0, -0.24],
        ])
        self.undo_called = False

    def _push_undo_state(self):
        self.undo_called = True


def test_execute_plugin_script_modify_coords():
    win = DummyWindow()
    script = """
new_crds = np.copy(coords)
new_crds[:, 2] += 2.5
set_coords(new_crds)
result_val = 42
"""
    ns = execute_plugin_script(script, win)
    assert win.undo_called is True
    assert np.isclose(win.coords[0, 2], 2.5)
    assert np.isclose(win.coords[1, 2], 3.46)
    assert ns["result_val"] == 42


def test_execute_plugin_script_modify_molecule():
    win = DummyWindow()
    script = """
new_symbols = ['C', 'C']
new_coords = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
set_molecule(new_symbols, new_coords)
"""
    execute_plugin_script(script, win)
    assert win.undo_called is True
    assert len(win.zatoms) == 2
    assert [a.symbol for a in win.zatoms] == ["C", "C"]
    assert np.allclose(win.coords, [[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])


def test_plugin_runner_dialog(qapp):
    win = DummyWindow()
    dlg = PythonScriptRunnerDialog(win)
    assert dlg.combo_plugins is not None
    assert dlg.editor is not None
    dlg.editor.setPlainText("x = 100")
    # Verify no exceptions
    dlg.close()
