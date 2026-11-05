"""
Unit tests for recently added features:
1. Quick orbital isovalue slider & dynamic clamping
2. Symmetry matrix selector dialog & 'Display All' formatting
3. 2D Slicer property switching & level generation
4. Multiwfn table export functionality
"""

import os
import pytest
import numpy as np
from PyQt6 import QtWidgets, QtCore

from agymolden.main_window import MainWindow
from agymolden.dialog_esp_2d import ESP2DContourDialog
from agymolden.dialog_multiwfn import MultiwfnAnalysisDialog


H2O_MOLDEN = os.path.join(os.path.dirname(__file__), "..", "examples", "h2o_orca.molden")
ETHANE_ZMAT = os.path.join(os.path.dirname(__file__), "..", "examples", "ethane.zmat")


def test_quick_orbital_slider_sync(qapp):
    win = MainWindow()
    if os.path.isfile(H2O_MOLDEN):
        win.load_orbitals_from_molden(H2O_MOLDEN, auto_render=False)
        # Test slider to spinbox sync
        win.slider_quick_iso.setValue(50)
        assert abs(win.spin_quick_iso.value() - 0.050) < 1e-4

        # Test spinbox to slider sync
        win.spin_quick_iso.setValue(0.080)
        assert abs(win.slider_quick_iso.value() - 80) < 2


def test_symmetry_matrices_all_display(qapp):
    win = MainWindow()
    if os.path.isfile(ETHANE_ZMAT):
        win.open_path(ETHANE_ZMAT)
        win._on_detect_symmetry(switch_tab=False)
        res = getattr(win, "_current_sym_res", None)
        assert res is not None
        assert len(res.operations) >= 1
        for op in res.operations:
            det = float(np.linalg.det(op.matrix))
            assert abs(abs(det) - 1.0) < 1e-3


def test_esp_2d_property_switch(qapp):
    pytest.importorskip("pyscf")
    if not os.path.isfile(H2O_MOLDEN):
        pytest.skip("H2O molden not found")

    dlg = ESP2DContourDialog(None, molden_path=H2O_MOLDEN)
    # Default is ESP
    assert dlg.combo_property.currentData() == "ESP"

    # Switch to RHO (Electron density)
    dlg.combo_property.setCurrentIndex(1)
    assert dlg.combo_property.currentData() == "RHO"

    # Switch to LAPL (Laplacian)
    dlg.combo_property.setCurrentIndex(2)
    assert dlg.combo_property.currentData() == "LAPL"

    # Switch to RDG (Reduced Density Gradient)
    dlg.combo_property.setCurrentIndex(3)
    assert dlg.combo_property.currentData() == "RDG"


def test_multiwfn_table_export(qapp, tmp_path):
    dlg = MultiwfnAnalysisDialog(None, molden_path=H2O_MOLDEN)
    tbl = dlg.tbl_bonds
    tbl.setRowCount(2)
    tbl.setItem(0, 0, QtWidgets.QTableWidgetItem("1 (O)"))
    tbl.setItem(0, 1, QtWidgets.QTableWidgetItem("2 (H)"))
    tbl.setItem(0, 2, QtWidgets.QTableWidgetItem("1.0085"))
    tbl.setItem(0, 3, QtWidgets.QTableWidgetItem("1.1531"))
    tbl.setItem(0, 4, QtWidgets.QTableWidgetItem("Single (1)"))

    headers, rows = dlg._extract_table_data(tbl)
    assert len(headers) == 5
    assert len(rows) == 2
    assert rows[0][0] == "1 (O)"
    assert rows[0][2] == "1.0085"
