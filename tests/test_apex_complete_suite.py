"""Tests for apex-level feature additions:
1. Add-on & Scientific Backend Manager (ExtensionManagerDialog)
2. Chemistry Sample & Benchmark Gallery (SampleGalleryDialog)
3. Reaction Path NEB Trajectory & Sparkline synchronization
4. OS Desktop & App Menu shortcut integration
"""

import os
import tempfile
import numpy as np
import pytest
from PyQt6 import QtWidgets, QtCore

from agymolden.dialog_addons import ExtensionManagerDialog, check_module_status, BACKEND_CATALOG
from agymolden.data_samples import SAMPLE_CATALOG
from agymolden.dialog_samples import SampleGalleryDialog
from agymolden.dialog_reaction_path import ReactionPathDialog


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_addons_manager_dialog(qapp):
    """Verify Add-ons manager dialog populates and correctly checks environment backends."""
    dlg = ExtensionManagerDialog()
    assert dlg.table.rowCount() == len(BACKEND_CATALOG)
    # Check that numpy or scipy can be detected
    installed, ver = check_module_status("numpy")
    assert installed is True
    assert len(ver) > 0


def test_chemistry_samples_catalog(qapp):
    """Verify that sample gallery has valid benchmark systems and parses properly."""
    assert len(SAMPLE_CATALOG) >= 4
    for sample in SAMPLE_CATALOG:
        lines = sample.xyz_data.strip().splitlines()
        assert int(lines[0]) > 0
        coords_lines = lines[2:]
        assert len(coords_lines) == int(lines[0])

    dlg = SampleGalleryDialog()
    assert dlg.list_samples.count() == len(SAMPLE_CATALOG)


def test_reaction_path_energy_profile_emission(qapp):
    """Verify reaction path preview emits trajectory coordinates alongside energy profile."""
    dlg = ReactionPathDialog()
    r_syms = ["H", "H"]
    r_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    p_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.50]])
    dlg.r_symbols = r_syms
    dlg.r_coords = r_coords
    dlg.p_symbols = r_syms
    dlg.p_coords = p_coords

    received = []
    dlg.trajectory_preview.connect(lambda s, f, e: received.append((s, f, e)))

    dlg.combo_method.setCurrentIndex(1)  # Linear transit
    dlg.spin_images.setValue(5)
    dlg._on_compute_path()
    assert dlg.path_images is not None
    assert len(dlg.path_images) == 5

    dlg._on_preview_trajectory()
    assert len(received) == 1
    syms, frames, energies = received[0]
    assert len(frames) == 5
    assert len(energies) == 5
    # Middle frame should be the transition state summit
    assert energies[2] > energies[0]
    assert energies[2] > energies[4]
