"""Tests for Surfaces & ESP Menu, Small Screen UI, and Quick ORCA Studio."""

import os
import pytest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets, QtCore
from agymolden.main_window import MainWindow
from agymolden.dialog_quick_orca import QuickOrcaDialog
from agymolden.zmatrix import ZAtom


@pytest.fixture
def main_win():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    yield app, win
    win.close()
    win.deleteLater()
    app.processEvents()


def test_surfaces_menu_and_toolbar(main_win):
    app, win = main_win

    # 1. Verify Surfaces Menu exists
    assert hasattr(win, "menu_surfaces")
    actions = [a.text() for a in win.menu_surfaces.actions() if not a.isSeparator()]
    
    assert any("Electrostatic Potential (ESP)" in a for a in actions)
    assert any("2D Electrostatic Potential" in a for a in actions)
    assert any("Molecular Orbitals & Isosurfaces Studio" in a for a in actions)
    assert any("HOMO" in a for a in actions)
    assert any("LUMO" in a for a in actions)
    assert any("Clear" in a for a in actions)

    # 2. Verify Toolbar ESP Button exists
    assert hasattr(win, "btn_toolbar_esp")
    assert win.btn_toolbar_esp is not None
    assert "ESP" in win.btn_toolbar_esp.text()

    # 3. Verify Small-Screen Responsive Features & Scroll Areas
    assert hasattr(win, "scroll_tab_table")
    assert isinstance(win, MainWindow)
    assert isinstance(win.scroll_tab_table, QtWidgets.QScrollArea)
    assert win.scroll_tab_table.widgetResizable()
    assert isinstance(win.tab_vibrations, QtWidgets.QScrollArea)
    assert isinstance(win.tab_symmetry, QtWidgets.QScrollArea)
    assert isinstance(win.tab_orbitals, QtWidgets.QScrollArea)
    assert win.grp_adjust.isCheckable()
    assert win.grp_adjust.isChecked()

    # Test toggling grp_adjust collapsed / expanded
    win.grp_adjust.setChecked(False)
    assert not win.grp_adjust.isChecked()
    win.grp_adjust.setChecked(True)
    assert win.grp_adjust.isChecked()

    # 4. Verify Z-Matrix action buttons under small screen
    assert win.btn_add_atom.minimumWidth() >= 65
    assert win.btn_add_hydrogens.minimumWidth() >= 65
    assert win.btn_del_atom.minimumWidth() >= 65
    assert win.btn_close_ring.minimumWidth() >= 80
    assert win.btn_reorder_zmat.minimumWidth() >= 80
    assert win.btn_optimize_ff.minimumWidth() >= 75
    assert win.btn_conformer_search.minimumWidth() >= 80

    # Test resizing down to small screen
    win.resize(750, 480)
    app.processEvents()
    assert win.width() <= 800
    assert win.height() <= 550


def test_quick_orca_dialog_options_and_esp(main_win):
    app, win = main_win

    zatoms = [
        ZAtom("C"),
        ZAtom("H", 0, 1.09),
        ZAtom("H", 0, 1.09, 1, 109.5),
        ZAtom("H", 0, 1.09, 1, 109.5, 2, 120.0),
        ZAtom("H", 0, 1.09, 1, 109.5, 2, -120.0),
    ]

    dlg = QuickOrcaDialog(parent=win, zatoms=zatoms)

    # Verify method & basis selectors
    assert dlg.combo_method.count() >= 10
    assert dlg.combo_basis.count() >= 8

    # Default check
    assert dlg.combo_method.currentData() == "B3LYP"
    assert dlg.combo_basis.currentData() == "def2-SVP"

    # Verify Molden checkbox exists (preferred over ORCA %plots esp crash)
    assert hasattr(dlg, "chk_molden")
    assert dlg.chk_molden.isChecked()

    # Verify MO cube export exists
    assert hasattr(dlg, "chk_mo_cubes")
    dlg.chk_mo_cubes.setChecked(True)
    deck = dlg.get_orca_input_text()
    assert "homo.cube" in deck
    assert "%plots" in deck

    # Verify changing functional & basis
    idx_wb = [i for i in range(dlg.combo_method.count()) if "wB97X-D3" in dlg.combo_method.itemText(i)][0]
    dlg.combo_method.setCurrentIndex(idx_wb)

    idx_tz = [i for i in range(dlg.combo_basis.count()) if "def2-TZVP" in dlg.combo_basis.itemText(i)][0]
    dlg.combo_basis.setCurrentIndex(idx_tz)

    # Verify solvation
    dlg.combo_solvent.setCurrentText("Water (CPCM)")
    # Verify properties
    dlg.chk_nmr.setChecked(True)
    dlg.chk_polar.setChecked(True)

    deck2 = dlg.get_orca_input_text()
    assert "wB97X-D3" in deck2
    assert "def2-TZVP" in deck2
    assert "CPCM(Water)" in deck2
    assert "NMR" in deck2
    assert "Polar" in deck2

    # Check live preview tab updates
    assert dlg.txt_deck_preview.toPlainText() == deck2

    # Verify composite method handling
    idx_r2 = [i for i in range(dlg.combo_method.count()) if "r2SCAN-3c" in dlg.combo_method.itemText(i)][0]
    dlg.combo_method.setCurrentIndex(idx_r2)
    assert not dlg.combo_basis.isEnabled()
    assert not dlg.lbl_compat_info.isHidden()
    assert "r2SCAN-3c" in dlg.lbl_compat_info.text()
    deck3 = dlg.get_orca_input_text()
    assert "! r2SCAN-3c" in deck3
    assert "def2-TZVP" not in deck3

    dlg.close()
    dlg.deleteLater()
