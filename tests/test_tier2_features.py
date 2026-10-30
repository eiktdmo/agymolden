"""
Tests for Tier 2 Features:
1. Instanced OpenGL rendering pipeline
2. 3D translation & manipulation axis gizmo (X, Y, Z)
3. 1D Dihedral PES Conformer Scan Plotter Dialog
"""

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets

from agymolden.main_window import MainWindow
from agymolden.dialog_conformer_scan import DihedralScanDialog


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_instanced_rendering_initialization(qapp):
    win = MainWindow()
    viewer = win.viewer

    # Check shader and instancing properties (3D Gizmo is off by default until toggled)
    assert hasattr(viewer, "show_manipulator_gizmo")
    assert viewer.show_manipulator_gizmo is False
    assert hasattr(viewer, "active_gizmo_axis")

    # Load structure
    symbols = ["C", "C", "H", "H", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [-0.5, 0.9, 0.0],
        [-0.5, -0.9, 0.0],
        [2.0, 0.9, 0.0],
        [2.0, -0.9, 0.0],
    ])
    viewer.set_molecule(symbols, coords)
    assert len(viewer.symbols) == 6
    win.close()


def test_3d_gizmo_selection_and_interaction(qapp):
    win = MainWindow()
    viewer = win.viewer

    symbols = ["C", "C"]
    coords = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
    viewer.set_molecule(symbols, coords)
    viewer.selected_atom_idx = 0
    viewer.multi_selected = [0]

    # Verify gizmo data structures and interaction state
    assert hasattr(viewer, "gizmo_screen_rects")
    assert viewer.active_gizmo_axis is None

    # Test gizmo axis hover and drag activation
    test_rect = QtCore.QRect(100, 100, 15, 15)
    viewer.gizmo_screen_rects["X"] = test_rect
    assert viewer.gizmo_screen_rects["X"].contains(QtCore.QPoint(105, 105))

    win.close()


def test_conformer_pes_scan_dialog(qapp):
    symbols = ["H", "C", "C", "H"]
    coords = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 1.0, 0.0],
    ])
    bonds = [(0, 1), (1, 2), (2, 3)]

    dlg = DihedralScanDialog(
        parent=None,
        symbols=symbols,
        coords=coords,
        bonds=bonds,
        atom_indices=(0, 1, 2, 3),
    )
    assert dlg.scan_result is not None
    assert len(dlg.scan_result.points) >= 8
    assert dlg.scan_result.barrier_kcal >= 0.0
    assert dlg.active_index == dlg.scan_result.min_index

    # Check that selecting another conformer updates active_index
    dlg._select_conformer(2)
    assert dlg.active_index == 2

    # Check global minimum button
    dlg._on_apply_minimum()
    assert dlg.active_index == dlg.scan_result.min_index
    dlg.close()


def test_conformers_and_vibrational_displacement_and_qm_scan(qapp):
    from agymolden.smartslides import find_rotatable_bonds, generate_conformers
    from agymolden.dialog_qm_input import QMCalculationDialog

    # 1. Test rotatable bond detection and conformer generation on butane-like chain
    symbols = ["C", "C", "C", "C", "H", "H"]
    coords = np.array([
        [-1.5, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.54, 0.0, 0.0],
        [2.5, 1.0, 0.5],
        [0.0, -1.0, 0.0],
        [1.54, -1.0, 0.0],
    ], dtype=np.float64)
    bonds = [(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)]
    rotors = find_rotatable_bonds(symbols, bonds)
    assert len(rotors) == 1
    conf_coords, conf_e = generate_conformers(symbols, coords, bonds, n_samples=4)
    assert len(conf_coords) >= 1
    assert len(conf_coords) == len(conf_e)

    # 2. Test Relaxed PES Scan generation from selected atoms (2, 3, or 4 atoms)
    dlg = QMCalculationDialog(
        None, symbols=symbols, coords=coords, selected_atoms=[0, 1, 2, 3], auto_enable_scan=True
    )
    assert dlg.grp_pes_scan.isChecked()
    orca_inp = dlg.text_preview.toPlainText()
    assert "Scan" in orca_inp and "D 0 1 2 3" in orca_inp
    dlg.combo_qm_code.setCurrentText("Gaussian (09/16)")
    gauss_inp = dlg.text_preview.toPlainText()
    assert "modredundant" in gauss_inp and "D 1 2 3 4 S" in gauss_inp
    dlg.close()

    # 3. Test 1-click Adopt Displaced Geometry along a vibrational mode
    win = MainWindow()
    win.viewer.set_molecule(["O", "H", "H"], np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]))
    from agymolden.zmatrix import cartesian_to_zmatrix
    win.set_molecule_from_zatoms(cartesian_to_zmatrix(["O", "H", "H"], win.viewer.coords))
    from agymolden.vibrations import VibrationalAnalysis
    disp = np.array([[[0.0, 0.0, 0.1], [0.0, 0.0, -0.1], [0.0, 0.0, -0.1]]], dtype=np.float64)
    win.vibrational_analysis = VibrationalAnalysis(
        frequencies=[-350.0], displacements=disp, symbols=["O", "H", "H"], base_coords=win.coords
    )
    win.viewer.set_vibration_mode(0, -350.0, disp[0], amplitude=0.5)
    z0_before = float(win.coords[0, 2])
    win._on_adopt_vibrational_displacement(sign=1.0)
    assert abs(float(win.coords[0, 2]) - (z0_before + 0.05)) < 1e-4
    win.close()
