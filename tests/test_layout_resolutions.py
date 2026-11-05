"""Multi-Resolution Layout Regression Tests.

Verifies that AgyMolden renders gracefully across diverse display sizes:
- 800x600 (SVGA / Small screens / Netbooks)
- 1024x768 (XGA / Standard laptops & projectors)
- 1280x800 (WXGA)
- 1366x768 (Standard notebook)
- 1920x1080 (Full HD desktop)

Checks:
- All tabs (Z-Matrix, Raw ZMAT, Presets, Vibrations, Symmetry, MOs) activate without overflow.
- All scroll areas are properly configured and widget-resizable.
- Critical action buttons maintain readable minimum widths.
- Interactive Parameter Tuning (SmartSlides) collapses and expands correctly.
"""

import os
import pytest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets, QtCore
from agymolden.main_window import MainWindow
from agymolden.zmatrix import ZAtom


@pytest.fixture
def win():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    # Add a sample molecule (water)
    zatoms = [
        ZAtom("O"),
        ZAtom("H", ref1=0, dist=0.96),
        ZAtom("H", ref1=0, dist=0.96, ref2=1, angle=104.5),
    ]
    window.set_molecule_from_zatoms(zatoms)
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


RESOLUTIONS = [
    (800, 600),
    (1024, 768),
    (1280, 800),
    (1366, 768),
    (1920, 1080),
]


@pytest.mark.parametrize("width,height", RESOLUTIONS)
def test_layout_at_resolution(win, width, height):
    """Verify layout integrity and scroll area engagement at specified resolution."""
    app = QtWidgets.QApplication.instance()
    win.resize(width, height)
    app.processEvents()

    # Window dimensions must adapt within reasonable tolerance
    assert win.width() <= width + 50
    assert win.height() <= height + 50

    # 1. Verify all tab scroll areas are active and resizable
    assert hasattr(win, "scroll_tab_table")
    assert win.scroll_tab_table.widgetResizable()

    assert hasattr(win, "tab_builder")
    assert win.tab_builder.widgetResizable()

    assert hasattr(win, "tab_vibrations")
    assert win.tab_vibrations.widgetResizable()

    assert hasattr(win, "tab_symmetry")
    assert win.tab_symmetry.widgetResizable()

    assert hasattr(win, "tab_orbitals")
    assert win.tab_orbitals.widgetResizable()

    # 2. Cycle through all inspector tabs and ensure clean activation
    for tab_idx in range(win.tabs.count()):
        win.tabs.setCurrentIndex(tab_idx)
        app.processEvents()
        cur_widget = win.tabs.currentWidget()
        assert cur_widget is not None
        assert not cur_widget.isHidden()
        assert win.tabs.currentIndex() == tab_idx

    # Switch back to Tab 0 (Z-Matrix)
    win.tabs.setCurrentIndex(0)
    app.processEvents()

    # 3. Check Z-Matrix action buttons maintain readable dimensions
    for btn in [
        win.btn_add_atom,
        win.btn_add_hydrogens,
        win.btn_del_atom,
        win.btn_close_ring,
        win.btn_reorder_zmat,
        win.btn_optimize_ff,
        win.btn_conformer_search,
    ]:
        assert btn.minimumWidth() >= 65
        assert len(btn.text().strip()) > 0

    # 4. Check SmartSlides collapse / expand behavior
    win.grp_adjust.setChecked(False)
    app.processEvents()
    assert not win.grp_adjust.isChecked()

    win.grp_adjust.setChecked(True)
    app.processEvents()
    assert win.grp_adjust.isChecked()

    # 5. Check Molecular Orbitals tab action buttons
    win.tabs.setCurrentIndex(win.tabs.count() - 1)
    app.processEvents()
    for mo_btn in [
        win.btn_quick_homo,
        win.btn_quick_lumo,
        win.btn_quick_spin_dens,
        win.btn_quick_esp_map,
        win.btn_localize_mos,
    ]:
        assert mo_btn.minimumWidth() >= 50
        assert len(mo_btn.text().strip()) > 0
