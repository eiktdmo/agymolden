"""Unit tests for Ctrl + Drag and Molecular Tweezers interactive mechanics."""

import os
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets, QtCore, QtGui
from agymolden.main_window import MainWindow


@pytest.fixture
def win():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    return window


def test_tweezers_drag_hydrogen(win):
    """Verify that dragging a hydrogen atom with Tweezers pins it at mouse target and relaxes heavy atoms."""
    win.load_preset_molecule("Water (H2O)")
    win._on_toggle_tweezers(True)
    assert win.live_tweezers_enabled is True
    assert win.viewer.tweezers_active is True

    orig_h = np.copy(win.coords[1])
    target_h = orig_h + [0.4, 0.2, 0.1]
    win.viewer.atom_move_drag_idx = 1
    win.viewer.atom_move_drag_indices = [1]

    win._on_atom_moved_by_drag(1, target_h[0], target_h[1], target_h[2])
    # The dragged hydrogen must remain pinned at the target coordinates
    assert np.allclose(win.coords[1], target_h, atol=1e-4)
    assert np.allclose(win.viewer.coords[1], target_h, atol=1e-4)
    # Oxygen should have relaxed
    assert np.all(np.isfinite(win.coords))

    # Mouse release
    win._on_atom_drag_finished(1)
    assert np.allclose(win.coords[1], target_h, atol=1e-3)
    assert len(win.zatoms) == 3


def test_tweezers_drag_heavy_atom(win):
    """Verify that dragging a heavy atom (Carbon in Ethanol) pins it and relaxes adjacent atoms."""
    win.load_preset_molecule("Ethanol (C2H5OH)")
    win._on_toggle_tweezers(True)

    orig_c0 = np.copy(win.coords[0])
    target_c0 = orig_c0 + [0.5, 0.0, 0.0]
    win.viewer.atom_move_drag_idx = 0
    win.viewer.atom_move_drag_indices = [0]

    win._on_atom_moved_by_drag(0, target_c0[0], target_c0[1], target_c0[2])
    assert np.allclose(win.coords[0], target_c0, atol=1e-4)
    assert np.all(np.isfinite(win.coords))

    win._on_atom_drag_finished(0)
    assert np.allclose(win.coords[0], target_c0, atol=1e-3)


def test_multi_atom_drag_sync_without_tweezers(win):
    """Verify that translating multiple selected atoms keeps them synchronized and intact."""
    win.load_preset_molecule("Ethanol (C2H5OH)")
    win._on_toggle_tweezers(False)

    # Select atoms 0 and 1 (both carbons)
    win.viewer.multi_selected = [0, 1]
    win.viewer.atom_move_drag_idx = 0
    win.viewer.atom_move_drag_indices = [0, 1]

    orig_c0 = np.copy(win.coords[0])
    orig_c1 = np.copy(win.coords[1])
    orig_dist = float(np.linalg.norm(orig_c0 - orig_c1))
    delta = np.array([1.0, -0.5, 0.3])

    win.viewer.coords[0] += delta
    win.viewer.coords[1] += delta
    win._on_atom_moved_by_drag(0, win.viewer.coords[0][0], win.viewer.coords[0][1], win.viewer.coords[0][2])

    # Both atoms must have moved in win.coords
    assert np.allclose(win.coords[0], orig_c0 + delta, atol=1e-4)
    assert np.allclose(win.coords[1], orig_c1 + delta, atol=1e-4)

    win._on_atom_drag_finished(0)
    assert np.allclose(win.coords[0], orig_c0 + delta, atol=1e-3)
    assert np.allclose(win.coords[1], orig_c1 + delta, atol=1e-3)
    new_dist = float(np.linalg.norm(win.coords[0] - win.coords[1]))
    assert np.isclose(orig_dist, new_dist, atol=1e-3)


def test_live_table_and_adjust_panel_updates(win):
    """Verify that table cell distance and adjust panel sliders update live during drag."""
    win.load_preset_molecule("Water (H2O)")
    win._on_toggle_tweezers(False)

    win.viewer.selected_atom_idx = 1
    win.viewer.multi_selected = [0, 1]
    win.viewer.smartslide_selection_active = True
    win.viewer.atom_move_drag_idx = 1
    win.viewer.atom_move_drag_indices = [1]

    orig_pos = np.copy(win.coords[1])
    new_pos = orig_pos + [0.5, 0.0, 0.0]
    win.viewer.coords[1] = new_pos

    win._on_atom_moved_by_drag(1, new_pos[0], new_pos[1], new_pos[2])

    # SmartSlide spin_bond should reflect the live distance (2 decimal places)
    expected_d = float(np.linalg.norm(win.coords[0] - new_pos))
    assert np.isclose(win.spin_bond.value(), expected_d, atol=0.02)
