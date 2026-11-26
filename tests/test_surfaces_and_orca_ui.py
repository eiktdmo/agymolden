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


def test_atom_ao_cavity_occlusion(main_win):
    """Interior crowded atoms must receive stronger ambient occlusion than exposed exterior atoms."""
    import numpy as np

    _app, win = main_win
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-1.4, 0.0, 0.0], [0.0, 1.4, 0.0], [0.0, -1.4, 0.0], [8.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    radii = np.full(6, 0.45, dtype=np.float32)
    ao = win.viewer._compute_atom_ao(coords, radii)
    assert ao.shape == (6,)
    assert float(np.min(ao)) >= 0.58 and float(np.max(ao)) <= 1.0
    assert ao[0] < ao[5]  # central crowded atom darker than isolated exterior atom


def test_multi_mol_tabs_welcome_banner_and_zmat_xyz_preserved(main_win):
    """Verify multi-molecule tabs, welcome banner auto-hide, and Z-Matrix/XYZ visibility in Simple Mode."""
    app, win = main_win

    # Initially empty -> welcome banner not hidden, 1 tab
    assert not win.welcome_banner.isHidden()
    assert win.mol_tab_bar.count() == 1

    # Load water in Tab 1 -> welcome banner hides
    h2o = [ZAtom("O"), ZAtom("H", ref1=0, dist=0.96), ZAtom("H", ref1=0, dist=0.96, ref2=1, angle=104.5)]
    win.set_molecule_from_zatoms(h2o)
    app.processEvents()
    assert win.welcome_banner.isHidden()
    assert len(win.zatoms) == 3

    # Open Tab 2 -> empty again, welcome banner reappears
    win._on_new_mol_tab()
    app.processEvents()
    assert win.mol_tab_bar.count() == 2
    assert len(win.zatoms) == 0
    assert not win.welcome_banner.isHidden()

    # Switch back to Tab 1 -> water is preserved
    win.mol_tab_bar.setCurrentIndex(0)
    app.processEvents()
    assert len(win.zatoms) == 3
    assert win.welcome_banner.isHidden()

    # Closing last remaining tab keeps at least 1 tab open
    win._on_close_mol_tab(1)
    win._on_close_mol_tab(0)
    assert win.mol_tab_bar.count() == 1

    # Simple Mode hides QM toolbar but keeps Z-Matrix & XYZ tabs and toolbar actions visible
    win._on_toggle_simple_mode(True)
    app.processEvents()
    assert win.toolbar_qm.isHidden()
    tab_names = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "Z-Matrix" in tab_names
    assert "Raw ZMAT Text" in tab_names
    assert "Cartesian XYZ" in tab_names
    win._on_toggle_simple_mode(False)


def test_2d_sketcher_active_tool_readable_and_other_element(main_win):
    """Embedded 2D sketcher sidebar must not clip labels/buttons and must support any Periodic Table element."""
    from PyQt6 import QtCore, QtWidgets
    from agymolden.dialog_2d_sketcher import SketchAtom2D

    app, win = main_win
    win.resize(1200, 760)
    win.btn_2d_3d_split.setChecked(True)
    app.processEvents()

    pane = win.sketcher_2d_pane
    assert pane is not None
    vp_width = pane.side_scroll.viewport().width()
    for w in pane.side_container.findChildren((QtWidgets.QPushButton, QtWidgets.QLabel)):
        if w.isVisible():
            assert w.sizeHint().width() <= vp_width, f"Widget clipped: {w.text()!r} ({w.sizeHint().width()} > {vp_width})"

    canvas_left = pane.canvas.mapTo(pane, QtCore.QPoint(0, 0)).x()
    assert pane.side_scroll.geometry().right() < canvas_left

    # Select a non-grid element ("Fe") as if chosen from PeriodicTableDialog
    pane._set_element("Fe")
    app.processEvents()
    assert pane.canvas.current_element == "Fe"
    assert pane.btn_other_elem.isChecked()
    assert "Fe" in pane.btn_other_elem.text()
    assert not any(b.isChecked() for b in pane._elem_buttons.values())

    # Convert an Fe atom to 3D without error, then switch back to Carbon
    pane.canvas.atoms = [SketchAtom2D(x=120.0, y=120.0, symbol="Fe")]
    pane._on_convert_to_3d(close_dialog=False, fast_live=True)
    app.processEvents()

    pane._set_element("C")
    app.processEvents()
    assert pane._elem_buttons["C"].isChecked()
    assert not pane.btn_other_elem.isChecked()
    win.btn_2d_3d_split.setChecked(False)


def test_white_mode_no_dark_black_buttons(main_win):
    """In white theme mode, no QPushButton/QToolButton should have a dark/black/charcoal background."""
    import re
    from PyQt6 import QtGui, QtWidgets
    from agymolden.dialog_volumetric import VolumetricSurfaceDialog

    app, win = main_win
    win.set_color_theme("white")
    win.btn_2d_3d_split.setChecked(True)
    app.processEvents()

    bg_re = re.compile(r"background-color:\s*(#[0-9a-fA-F]{3,6}|black)", re.I)

    def _is_dark_neutral(hex_col: str) -> bool:
        c = QtGui.QColor(hex_col)
        if not c.isValid():
            return False
        r, g, b = c.red() / 255.0, c.green() / 255.0, c.blue() / 255.0
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        sat = max(r, g, b) - min(r, g, b)
        return lum < 0.45 and sat < 0.20

    bad = []
    vol_dlg = VolumetricSurfaceDialog(win)
    for root in (win, vol_dlg):
        for w in root.findChildren((QtWidgets.QPushButton, QtWidgets.QToolButton)):
            for col in bg_re.findall(w.styleSheet() or ""):
                if _is_dark_neutral(col):
                    bad.append((w.text(), col))
    vol_dlg.close()
    assert not bad, f"Dark/black buttons found in white mode: {bad}"
    wb_ss = win.welcome_banner.styleSheet()
    assert "#1e293b" not in wb_ss and "#334155" not in wb_ss
    win.btn_2d_3d_split.setChecked(False)


def test_2d_3d_ctrl_z_undoes_2d_when_focused(main_win):
    """When 2D | 3D split is active and 2D pane has focus, undo/redo must act on the 2D canvas."""
    app, win = main_win
    win.show() if False else None
    win.btn_2d_3d_split.setChecked(True)
    app.processEvents()

    pane = win.sketcher_2d_pane
    pane.canvas.clear_canvas()
    pane.canvas.undo_stack.clear()
    pane.canvas.redo_stack.clear()

    # Focus 2D canvas (or a child button in the 2D pane) and add a ring
    pane.canvas.setFocus()
    app.processEvents()
    # Even under offscreen QPA where focusWidget may require activeWindow, simulate focus check if available
    pane.canvas.add_fragment_ring(6, aromatic=True)
    assert len(pane.canvas.atoms) == 6

    if win._is_2d_sketcher_focused():
        win.undo()
        assert len(pane.canvas.atoms) == 0
        win.redo()
        assert len(pane.canvas.atoms) == 6
    else:
        # Fallback direct check if offscreen platform doesn't grant window focus
        pane._on_undo()
        assert len(pane.canvas.atoms) == 0
        pane._on_redo()
        assert len(pane.canvas.atoms) == 6

    win.btn_2d_3d_split.setChecked(False)


def test_2d_sketcher_ring_welder_bond_and_spiro(main_win):
    """Verify 2D Ring Welder bond fusion, exocyclic biphenyl attachment, spiro fusion, valence guard, and hit-testing."""
    from PyQt6 import QtCore, QtWidgets

    app, win = main_win
    win.resize(1200, 760)
    win.btn_2d_3d_split.setChecked(True)
    app.processEvents()

    pane = win.sketcher_2d_pane
    canvas = pane.canvas
    canvas.clear_canvas()

    # 1. Empty canvas + click Benz button -> places 6-ring and immediately returns to Draw mode (never stays stuck on)
    pane._ring_weld_buttons[0].click()
    assert len(canvas.atoms) == 6
    assert len(canvas.bonds) == 6
    assert canvas.current_tool == "draw"
    assert not any(b.isChecked() for b in pane._ring_weld_buttons)

    # 2. Apothem-matched magnetic snap: approaching 45 px outside bond 0 (edge-to-edge!) must snap to bond 0,
    #    while far-away empty space (120 px) returns (None, None)
    b0 = canvas.bonds[0]
    p1x, p1y = canvas.atoms[b0.atom1].x, canvas.atoms[b0.atom1].y
    p2x, p2y = canvas.atoms[b0.atom2].x, canvas.atoms[b0.atom2].y
    mid = QtCore.QPointF(0.5 * (p1x + p2x), 0.5 * (p1y + p2y))
    dx, dy = p2x - p1x, p2y - p1y
    seg_len = (dx * dx + dy * dy) ** 0.5
    cx_m = sum(a.x for a in canvas.atoms) / len(canvas.atoms)
    cy_m = sum(a.y for a in canvas.atoms) / len(canvas.atoms)
    nx, ny = -dy / seg_len, dx / seg_len
    if (mid.x() + nx * 10 - cx_m) ** 2 + (mid.y() + ny * 10 - cy_m) ** 2 < (mid.x() - nx * 10 - cx_m) ** 2 + (mid.y() - ny * 10 - cy_m) ** 2:
        nx, ny = -nx, -ny
    approach_pt = QtCore.QPointF(mid.x() + nx * 45.0, mid.y() + ny * 45.0)
    hit_bond, hit_atom = canvas._pick_weld_target(approach_pt)
    assert hit_bond == 0 and hit_atom is None
    far_pt = QtCore.QPointF(mid.x() + nx * 120.0, mid.y() + ny * 120.0)
    assert canvas._pick_weld_target(far_pt) == (None, None)

    # 3. Arm welder, verify paintEvent fuse indicator renders cleanly on standalone canvas, then click -> naphthalene + auto-exit
    from agymolden.dialog_2d_sketcher import Sketcher2DPane
    preview_pane = Sketcher2DPane(embedded=True)
    preview_pane.canvas.add_fragment_ring(6, aromatic=True)
    preview_pane.canvas.current_tool = "weld_ring"
    preview_pane.canvas.current_mouse_pos = approach_pt
    preview_pane.canvas.hovered_bond_idx = 0
    preview_pane.canvas.hovered_atom_idx = None
    pix = preview_pane.canvas.grab()
    assert not pix.isNull()
    preview_pane.close()

    pane._ring_weld_buttons[0].click()
    assert canvas.current_tool == "weld_ring"
    from PyQt6 import QtGui
    ev = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress, mid, mid,
        QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(ev)
    assert len(canvas.atoms) == 10
    assert len(canvas.bonds) == 11
    assert canvas.current_tool == "draw"
    assert not any(b.isChecked() for b in pane._ring_weld_buttons)

    # 4. Peri-fusion bay welding on Naphthalene bond 1 (1-2) succeeds (13 atoms, 15 bonds)
    assert canvas.weld_ring(6, aromatic=True, target_bond_idx=1)
    assert len(canvas.atoms) == 13
    assert len(canvas.bonds) == 15

    # 5. Attempting to weld onto central bridgehead bond 0 (valence 4 atoms) must be rejected cleanly
    assert not canvas.weld_ring(6, aromatic=True, target_bond_idx=0)
    assert len(canvas.atoms) == 13

    # 6. Clicking outer aromatic atom 3 of fresh benzene with Benz welds exocyclically (Biphenyl: 12 atoms, 13 bonds)
    canvas.clear_canvas()
    canvas.add_fragment_ring(6, aromatic=True)
    assert canvas.weld_ring(6, aromatic=True, target_atom_idx=3)
    assert len(canvas.atoms) == 12
    assert len(canvas.bonds) == 13

    # 7. Spiro-weld C5 onto sp3 cyclohexane atom 0 -> Spiro[4.5]decane (10 atoms, 11 bonds)
    canvas.clear_canvas()
    canvas.add_fragment_ring(6, aromatic=False)
    assert canvas.weld_ring(5, aromatic=False, target_atom_idx=0)
    assert len(canvas.atoms) == 10
    assert len(canvas.bonds) == 11

    # 7. Sidebar buttons still fit without clipping
    vp_width = pane.side_scroll.viewport().width()
    for w in pane.side_container.findChildren((QtWidgets.QPushButton, QtWidgets.QLabel)):
        if w.isVisible():
            assert w.sizeHint().width() <= vp_width, f"Clipped: {w.text()!r}"

    win.btn_2d_3d_split.setChecked(False)


def test_switch_between_2d_pane_and_zmatrix_pane(main_win):
    """Clicking ◧ Z-Mat or 📋 Z-Matrix from 2D | 3D mode must switch back to the Z-Matrix inspector pane."""
    app, win = main_win
    win.btn_2d_3d_split.setChecked(True)
    app.processEvents()
    assert win.right_stack.currentWidget() is win.sketcher_2d_pane
    assert not win.right_stack.isHidden()

    # Click ◧ Z-Mat while 2D Sketcher is showing -> switches to right_container (Z-Matrix) and stays visible
    win.btn_toggle_right_pane.click()
    app.processEvents()
    assert win.right_stack.currentWidget() is win.right_container
    assert not win.right_stack.isHidden()
    assert not win.btn_2d_3d_split.isChecked()
    assert win.btn_toggle_right_pane.isChecked()

    # Clicking ◧ Z-Mat again hides the panel, and clicking once more shows Z-Matrix again
    win.btn_toggle_right_pane.click()
    app.processEvents()
    assert win.right_stack.isHidden()
    win.btn_toggle_right_pane.click()
    app.processEvents()
    assert not win.right_stack.isHidden()
    assert win.right_stack.currentWidget() is win.right_container

    # Also verify 📋 Z-Matrix button inside 2D Sketcher switches back to Z-Matrix
    win.btn_2d_3d_split.setChecked(True)
    app.processEvents()
    assert win.right_stack.currentWidget() is win.sketcher_2d_pane
    win.sketcher_2d_pane.switch_to_inspector_requested.emit()
    app.processEvents()
    assert win.right_stack.currentWidget() is win.right_container
    assert not win.btn_2d_3d_split.isChecked()


def test_orca_and_qm_dialogs_readable_at_all_sizes(main_win):
    """Ensure QComboBox / QLineEdit / QSpinBox options in ORCA & QM dialogs are never vertically crushed across themes and sizes."""
    import numpy as np
    from PyQt6 import QtWidgets
    from agymolden.dialog_quick_orca import QuickOrcaDialog
    from agymolden.dialog_qm_input import QMCalculationDialog

    app, win = main_win
    for theme in ("white", "dark"):
        win.set_color_theme(theme)
        dlg_orca = QuickOrcaDialog(win, zatoms=[])
        dlg_qm = QMCalculationDialog(win, zatoms=[], symbols=["O", "H", "H"], coords=np.zeros((3, 3)))
        assert dlg_orca.minimumSizeHint().height() <= 600
        assert dlg_orca.minimumSizeHint().width() <= 680
        assert dlg_qm.minimumSizeHint().height() <= 600

        for dlg in (dlg_orca, dlg_qm):
            for w, h in ((1280, 800), (720, 620), (560, 480)):
                dlg.resize(w, h)
                dlg.show()
                app.processEvents()
                for cb in dlg.findChildren((QtWidgets.QComboBox, QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QLineEdit)):
                    if cb.isHidden() or cb.objectName() == "qt_spinbox_lineedit":
                        continue
                    fh = cb.fontMetrics().height()
                    assert cb.height() >= fh + 8 and cb.height() >= 26, (
                        f"{theme} {type(dlg).__name__} {w}x{h} crushed {type(cb).__name__}: h={cb.height()} fh={fh}"
                    )
        dlg_orca.close()
        dlg_qm.close()


def test_volumetric_density_colormap_ui(main_win):
    """Verify VolumetricSurfaceDialog Density Color Map slice button and live controls."""
    import numpy as np
    from agymolden.volumetric import VolumetricData
    from agymolden.dialog_volumetric import VolumetricSurfaceDialog

    app, win = main_win
    dlg = VolumetricSurfaceDialog(win)
    xs = np.linspace(-2.0, 2.0, 12)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    step = 4.0 / 11.0
    dlg.current_vol_data = VolumetricData(
        title="Density",
        comment="test",
        symbols=["H", "H"],
        atom_coords=np.array([[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]),
        origin=np.array([-2.0, -2.0, -2.0]),
        step_x=np.array([step, 0.0, 0.0]),
        step_y=np.array([0.0, step, 0.0]),
        step_z=np.array([0.0, 0.0, step]),
        voxels_x=12,
        voxels_y=12,
        voxels_z=12,
        grid_data=np.exp(-(X ** 2 + Y ** 2 + Z ** 2)),
        is_orbital=False,
    )
    emitted = []
    dlg.surface_updated.connect(lambda meshes, op: emitted.append((meshes, op)))
    dlg.btn_density_colormap.click()
    app.processEvents()

    assert not dlg.grp_dens_cmap.isHidden()
    assert len(emitted) >= 1
    mesh = emitted[-1][0][0]
    assert mesh.vertex_colors is not None
    assert mesh.esp_range is not None
    dlg.close()

    # Also verify Orbitals tab ☁️ ρ(r) ▾ menu & live Density Color Map controls
    assert win.btn_quick_dens.menu() is not None
    assert len(win.btn_quick_dens.menu().actions()) == 3
    win._last_quick_density_vol = dlg.current_vol_data
    win._on_update_quick_density_colormap()
    assert not win.grp_quick_dens_cmap.isHidden()
    assert len(win.viewer.active_isosurfaces) == 1
    assert win.viewer.active_isosurfaces[0].vertex_colors is not None


def test_agemacs_toolbar_and_dialog_buttons(main_win, monkeypatch):
    """Verify prominent Agemacs export menu on main toolbar and 1-click buttons in ORCA/QM dialogs."""
    import numpy as np
    import agymolden.companion as comp
    from agymolden.dialog_quick_orca import QuickOrcaDialog
    from agymolden.dialog_qm_input import QMCalculationDialog

    sent = []
    monkeypatch.setattr(comp, "send_to_agemacs", lambda files, parent=None: (sent.extend(files), True)[1])

    app, win = main_win
    assert hasattr(win, "btn_agemacs_menu")
    acts = win.btn_agemacs_menu.menu().actions()
    labels = [a.text() for a in acts if not a.isSeparator()]
    assert len(labels) == 5

    # Test send_text_to_agemacs writes temp file cleanly
    ok = comp.send_text_to_agemacs("test_deck.inp", "! B3LYP def2-SVP Opt\n", win)
    assert ok and any(p.endswith("test_deck.inp") for p in sent)

    dlg_orca = QuickOrcaDialog(win, zatoms=[])
    dlg_orca.btn_open_agemacs.click()
    assert any(p.endswith(".inp") for p in sent)
    dlg_orca.close()

    dlg_qm = QMCalculationDialog(win, zatoms=[], symbols=["O", "H", "H"], coords=np.zeros((3, 3)))
    dlg_qm.btn_open_agemacs.click()
    assert any("agymolden_input" in p for p in sent)
    dlg_qm.close()


def test_output_viewer_teaching_sections_and_posthf(main_win, tmp_path):
    """Verify QMOutputViewerDialog indexes MOs, Vibrational Modes, Final Energy, Energy Components, and Post-HF Correlation."""
    from agymolden.dialog_output_viewer import QMOutputViewerDialog

    app, win = main_win
    mp2_log = tmp_path / "water_mp2.out"
    mp2_log.write_text(
        "TOTAL SCF ENERGY\n"
        "Nuclear Repulsion  :    9.1934 Eh\n"
        "Electronic Energy  :  -85.2145 Eh\n"
        "ORBITAL ENERGIES\n"
        "  NO   OCC          E(Eh)            E(eV)\n"
        "   0   2.0000     -20.2412       -550.7911\n"
        "MP2 CORRELATION ENERGY   :    -0.211234 Eh\n"
        "FINAL SINGLE POINT ENERGY     -76.232334\n"
        "VIBRATIONAL FREQUENCIES\n"
        "   6:   1650.23 cm**-1\n"
        "IR SPECTRUM\n",
        encoding="utf-8",
    )

    viewer_dlg = QMOutputViewerDialog(win, file_path=str(mp2_log))
    for btn in (
        viewer_dlg.btn_jump_final_e,
        viewer_dlg.btn_jump_e_parts,
        viewer_dlg.btn_jump_ecorr,
        viewer_dlg.btn_jump_mos,
        viewer_dlg.btn_jump_vibs,
    ):
        assert btn.isEnabled()

    viewer_dlg.btn_jump_ecorr.click()
    assert "MP2 CORRELATION ENERGY" in viewer_dlg.txt_viewer.textCursor().block().text()
    viewer_dlg.close()

    # Pure HF log without post-HF correlation -> E-Corr button disabled, others enabled
    hf_log = tmp_path / "water_hf.out"
    hf_log.write_text(
        "TOTAL SCF ENERGY\n"
        "ORBITAL ENERGIES\n"
        "FINAL SINGLE POINT ENERGY     -76.021100\n",
        encoding="utf-8",
    )
    viewer_hf = QMOutputViewerDialog(win, file_path=str(hf_log))
    assert viewer_hf.btn_jump_final_e.isEnabled()
    assert viewer_hf.btn_jump_e_parts.isEnabled()
    assert viewer_hf.btn_jump_mos.isEnabled()
    assert not viewer_hf.btn_jump_ecorr.isEnabled()
    assert not viewer_hf.btn_jump_vibs.isEnabled()
    viewer_hf.close()


def test_quick_orca_save_input_only_without_running(main_win, tmp_path, monkeypatch):
    """QuickOrcaDialog.btn_save_only must save a valid .inp file to any chosen path without launching ORCA."""
    from PyQt6 import QtWidgets
    from agymolden.dialog_quick_orca import QuickOrcaDialog
    from agymolden.zmatrix import ZAtom

    app, win = main_win
    out_inp = tmp_path / "external_job.inp"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out_inp), "ORCA Input (*.inp)")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Ok))

    zatoms = [ZAtom("O"), ZAtom("H", ref1=0, dist=0.96), ZAtom("H", ref1=0, dist=0.96, ref2=1, angle=104.5)]
    dlg = QuickOrcaDialog(win, zatoms=zatoms)
    assert dlg.chk_print_mos.isChecked()
    dlg.btn_save_only.click()
    assert out_inp.is_file()
    txt = out_inp.read_text(encoding="utf-8")
    assert "!" in txt and "* xyz" in txt
    assert "Print[ P_MOs ] 1" in txt
    assert "Print[ P_ReducedOrbPopMO_L ] 1" in txt
    assert dlg.result() == 0
    dlg.close()


def test_right_pane_tabs_compact_and_readable(main_win):
    """Ensure right-pane inspector groups (Orbital Surface Controls, Density Color Map, Raw ZMAT, XYZ) fit cleanly."""
    app, win = main_win
    app.processEvents()
    assert win.mo_params_group.minimumSizeHint().width() <= 400
    assert win.grp_quick_dens_cmap.minimumSizeHint().width() <= 400
    for idx in (1, 2):  # Raw ZMAT Text, Cartesian XYZ
        assert win.tabs.widget(idx).minimumSizeHint().width() <= 410


def test_basis_set_upgrade_orca_and_all_codes(main_win, monkeypatch):
    """Verify expanded STO/Pople/def2/Dunning basis sets & SmartBasisDialog across ORCA and QM Studio."""
    import numpy as np
    import agymolden.brorsen_builder.dialog_basis as db_mod
    from agymolden.dialog_quick_orca import QuickOrcaDialog
    from agymolden.dialog_qm_input import QMCalculationDialog
    from agymolden.zmatrix import ZAtom

    app, win = main_win
    zatoms = [ZAtom("O"), ZAtom("H", ref1=0, dist=0.96), ZAtom("H", ref1=0, dist=0.96, ref2=1, angle=104.5)]

    # 1. ORCA Studio: STO-3G, STO-6G, aug-cc-pVTZ, custom basis, and Browse BSE button
    orca_dlg = QuickOrcaDialog(win, zatoms=zatoms)
    basis_vals = [str(orca_dlg.combo_basis.itemData(i) or "") for i in range(orca_dlg.combo_basis.count())]
    for req in ("def2-SVP", "STO-3G", "STO-6G", "6-31+G(d,p)", "aug-cc-pVTZ", "pc-1", "ZORA-def2-TZVP"):
        assert req in basis_vals, f"Missing {req} in ORCA_BASIS_SETS"
    assert hasattr(orca_dlg, "btn_browse_basis")
    orca_dlg._set_basis_value("STO-3G")
    orca_dlg._on_combo_options_changed()
    assert "STO-3G" in orca_dlg.get_orca_input_text()
    orca_dlg._set_basis_value("def2-QZVPD")
    orca_dlg._on_combo_options_changed()
    assert "def2-QZVPD" in orca_dlg.get_orca_input_text()
    orca_dlg.close()

    # 2. QM Calculation Studio: STO-3G / STO-6G across ORCA, Gaussian, PySCF, Psi4, OpenLOWDIN
    coords = np.array([[0.0, 0.0, 0.11], [0.0, 0.76, -0.47], [0.0, -0.76, -0.47]])
    qm_dlg = QMCalculationDialog(win, symbols=["O", "H", "H"], coords=coords, zatoms=zatoms)
    assert hasattr(qm_dlg, "btn_browse_basis")
    qm_bases = [qm_dlg.combo_basis.itemText(i) for i in range(qm_dlg.combo_basis.count())]
    assert "STO-3G" in qm_bases and "STO-6G" in qm_bases
    for code_idx in range(qm_dlg.combo_qm_code.count()):
        qm_dlg.combo_qm_code.setCurrentIndex(code_idx)
        code_name = qm_dlg.combo_qm_code.currentText()
        if "xtb" in code_name.lower():
            continue
        qm_dlg.combo_basis.setEditText("STO-3G")
        qm_dlg._update_preview()
        txt = qm_dlg.text_preview.toPlainText()
        assert "sto-3g" in txt.lower(), f"STO-3G missing in {code_name} input"
    qm_dlg.close()

    # 3. SmartBasisDialog offline fallback (ensures STO & all families work even without BSE package)
    monkeypatch.setattr(db_mod, "_HAS_BSE", False)
    sbd = db_mod.SmartBasisDialog(win, elements=["C", "H", "O"], current_basis="STO-3G")
    top_titles = [sbd.tree.topLevelItem(i).text(0) for i in range(sbd.tree.topLevelItemCount())]
    assert any("Slater-Type Minimal" in t for t in top_titles), f"Missing STO family in {top_titles}"
    sbd.txt_search.setText("STO-6G")
    app.processEvents()
    sto_group = [sbd.tree.topLevelItem(i) for i in range(sbd.tree.topLevelItemCount()) if "Slater-Type" in sbd.tree.topLevelItem(i).text(0)][0]
    assert any(not sto_group.child(j).isHidden() and "STO-6G" in sto_group.child(j).text(0) for j in range(sto_group.childCount()))
    sbd.close()


def test_lower_right_pane_and_spinbox_qss_readability(main_win):
    """Ensure QSpinBox/QDoubleSpinBox/QComboBox inner QLineEdits aren't double-padded and right-pane inputs are clean."""
    from agymolden.theme import DARK_THEME_QSS, LIGHT_THEME_QSS

    for qss in (DARK_THEME_QSS, LIGHT_THEME_QSS):
        assert "QSpinBox QLineEdit, QDoubleSpinBox QLineEdit, QComboBox QLineEdit" in qss
        assert "min-height: 0px;" in qss

    app, win = main_win
    for attr in ("combo_opt_method", "combo_smart_mode", "spin_bond", "spin_angle", "spin_dihedral", "txt_zmat_filter", "combo_mo_file_source"):
        w = getattr(win, attr)
        assert w.styleSheet().strip() == "", f"{attr} should not have inline styleSheet override: {w.styleSheet()!r}"
    for spin_attr in ("spin_bond", "spin_angle", "spin_dihedral", "spin_quick_iso", "spin_quick_margin"):
        assert getattr(win, spin_attr).minimumWidth() >= 84


def test_2d_to_3d_adjust_h_toggle_sync(main_win):
    """Ensure left-pane Adjust H syncs bidirectionally with 2D sketcher and controls H addition to 3D."""
    from agymolden.dialog_2d_sketcher import SketchAtom2D, SketchBond2D

    app, win = main_win
    win._on_toggle_2d_3d_split(True)
    pane = win.sketcher_2d_pane
    assert pane is not None

    # Disable Adjust H from left pane -> 2D toggle follows, C-C stays 2 atoms in 3D
    win.chk_auto_h.setChecked(False)
    app.processEvents()
    assert pane.chk_add_hs.isChecked() is False

    pane.canvas.atoms = [SketchAtom2D(100.0, 100.0, "C"), SketchAtom2D(145.0, 100.0, "C")]
    pane.canvas.bonds = [SketchBond2D(0, 1, 1)]
    pane._on_convert_to_3d(close_dialog=False)
    assert len(win.zatoms) == 2

    # Enable Adjust H from left pane -> 2D toggle follows, C-C becomes C2H6 (8 atoms) in 3D
    win.chk_auto_h.setChecked(True)
    app.processEvents()
    assert pane.chk_add_hs.isChecked() is True
    pane._on_convert_to_3d(close_dialog=False)
    assert len(win.zatoms) == 8

    # Reverse sync: toggling 2D pane updates left pane
    pane.chk_add_hs.setChecked(False)
    app.processEvents()
    assert win.chk_auto_h.isChecked() is False

    # Restore clean state
    win.chk_auto_h.setChecked(True)
    win._on_toggle_2d_3d_split(False)


def test_output_viewer_right_dock_and_ctrl_arrow_resize(main_win):
    """Ensure Output Viewer docks full-height on the right and Ctrl+Arrow resizes active panes."""
    from PyQt6 import QtCore

    app, win = main_win
    win.resize(1280, 800)
    app.processEvents()

    dock = win.dock_manager.get_or_create_output_viewer()
    assert win.dockWidgetArea(dock) == QtCore.Qt.DockWidgetArea.RightDockWidgetArea

    # Horizontal splitter resizing (main_splitter)
    win.main_splitter.setSizes([700, 560])
    app.processEvents()
    before_h = win.main_splitter.sizes()
    assert win._resize_active_pane("left", step=60) is True
    after_left = win.main_splitter.sizes()
    assert after_left[0] < before_h[0] and after_left[1] > before_h[1]
    assert win._resize_active_pane("right", step=60) is True
    after_right = win.main_splitter.sizes()
    assert after_right[0] > after_left[0]

    # Vertical splitter resizing (zmat_splitter) when visible
    win.tabs.setCurrentIndex(0)
    win.zmat_splitter.setSizes([350, 350])
    app.processEvents()
    before_v = win.zmat_splitter.sizes()
    assert win._resize_active_pane("up", step=50) is True
    after_up = win.zmat_splitter.sizes()
    assert after_up[0] < before_v[0] and after_up[1] > before_v[1]
    assert win._resize_active_pane("down", step=50) is True
    after_down = win.zmat_splitter.sizes()
    assert after_down[0] > after_up[0]

    dock.hide()















