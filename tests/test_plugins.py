"""Tests for User Scripting & Plugin API v1."""

import numpy as np
import pytest
from agymolden.plugins import (
    execute_plugin_script,
    PythonScriptRunnerDialog,
    PluginBase,
    PluginMetadata,
    PluginContext,
    PluginManager,
)
from agymolden.builtin_plugins.radius_of_gyration import GyrationShapeDescriptorPlugin
from agymolden.builtin_plugins.align_principal_axes import PrincipalAxesAlignerPlugin
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
        self.status_log = []

    def _push_undo_state(self):
        self.undo_called = True

    def statusBar(self):
        class DummyStatusBar:
            def __init__(self, parent):
                self.parent = parent
            def showMessage(self, msg, timeout=0):
                self.parent.status_log.append(msg)
        return DummyStatusBar(self)


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


def test_plugin_api_manager_discovery():
    win = DummyWindow()
    manager = PluginManager(win)
    discovered = manager.discover_and_load_all()
    assert len(discovered) >= 2
    assert "Gyration Radius & Shape Descriptors" in discovered
    assert "Align Molecule to Principal Inertial Axes" in discovered


def test_plugin_radius_of_gyration():
    win = DummyWindow()
    ctx = PluginContext(win)
    plugin = GyrationShapeDescriptorPlugin()
    res = plugin.run(ctx)

    assert "rg" in res
    assert res["rg"] > 0.2 and res["rg"] < 1.2
    assert "moments" in res
    assert len(res["moments"]) == 3
    I1, I2, I3 = res["moments"]
    assert I1 <= I2 <= I3
    assert "rotor_type" in res
    assert len(win.status_log) > 0


def test_plugin_align_principal_axes():
    win = DummyWindow()
    ctx = PluginContext(win)
    plugin = PrincipalAxesAlignerPlugin()
    new_crds = plugin.run(ctx)

    assert win.undo_called is True
    assert new_crds.shape == (3, 3)
    # The center of mass of new_crds should be approximately at origin
    masses = np.array([15.999, 1.008, 1.008])
    com_new = np.sum(new_crds * masses[:, np.newaxis], axis=0) / np.sum(masses)
    assert np.allclose(com_new, [0.0, 0.0, 0.0], atol=1e-5)


def test_plugin_lifecycle_hooks():
    class TestHookPlugin(PluginBase):
        metadata = PluginMetadata(name="Test Hook Plugin")
        def __init__(self):
            super().__init__()
            self.loaded = False
            self.geom_changed_count = 0

        def on_load(self, context):
            self.loaded = True

        def on_geometry_change(self, context, symbols, coords):
            self.geom_changed_count += 1

        def on_calculation_completed(self, context, result):
            self.calc_completed_count = getattr(self, "calc_completed_count", 0) + 1

        def on_file_loaded(self, context, filepath):
            self.file_loaded_count = getattr(self, "file_loaded_count", 0) + 1

        def run(self, context):
            return "executed"

    win = DummyWindow()
    manager = PluginManager(win)
    test_plugin = TestHookPlugin()
    manager.register_plugin(test_plugin)

    assert test_plugin.loaded is True
    assert manager.execute_plugin("Test Hook Plugin") == "executed"

    manager.notify_geometry_changed(["H", "H"], np.zeros((2, 3)))
    assert test_plugin.geom_changed_count == 1

    manager.notify_calculation_completed({"energy": -40.12})
    assert test_plugin.calc_completed_count == 1

    manager.notify_file_loaded("benzene.xyz")
    assert test_plugin.file_loaded_count == 1


def test_plugin_runner_dialog(qapp):
    win = DummyWindow()
    dlg = PythonScriptRunnerDialog(win)
    assert dlg.combo_plugins is not None
    assert dlg.combo_plugins.count() >= 3  # Placeholder + at least 2 built-in plugins
    assert dlg.editor is not None
    dlg.editor.setPlainText("x = 100")
    dlg.close()
