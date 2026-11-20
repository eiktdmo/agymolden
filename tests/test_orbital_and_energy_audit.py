"""Audit regression tests for Molden orbital plotting and energy diagram rendering."""

import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets
import agymolden.workers as workers
from agymolden.main_window import MainWindow
from agymolden.volumetric import parse_molden_file


@pytest.fixture
def main_win(monkeypatch):
    monkeypatch.setattr(workers, "SYNCHRONOUS", True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    yield app, win
    win.close()
    win.deleteLater()
    app.processEvents()


def test_parse_molden_mo_variants(tmp_path):
    """Ensure parse_molden_file handles spaces around '=', Fortran 'D' exponents, missing Sym=, and Ene=0.0."""
    m_path = tmp_path / "variants.molden"
    m_path.write_text(
        "[Molden Format]\n"
        "[Atoms] AU\n"
        "H 1 1 0.0 0.0 0.0\n"
        "H 2 1 0.0 0.0 1.4\n"
        "[MO]\n"
        " Sym = 1sigma_g\n"
        " Ene = -5.000D-01\n"
        " Spin = Alpha\n"
        " Occup = 2.0\n"
        "  1  7.071D-01\n"
        "  2  7.071D-01\n"
        " Ene = 0.0\n"
        " Spin = Alpha\n"
        " Occup = 0.0\n"
        "  1  7.071D-01\n"
        "  2 -7.071D-01\n"
        " Ene = 2.500D-01\n"
        " Sym = 2sigma_u\n"
        " Occup = 0.0\n"
        "  1  5.000D-01\n"
        "  2 -5.000D-01\n",
        encoding="utf-8",
    )
    data = parse_molden_file(str(m_path))
    assert len(data.orbitals) == 3
    assert data.orbitals[0].sym == "1sigma_g"
    assert data.orbitals[0].energy == pytest.approx(-0.5)
    assert data.orbitals[0].occupancy == pytest.approx(2.0)
    assert data.orbitals[1].energy == pytest.approx(0.0)
    assert data.orbitals[2].sym == "2sigma_u"
    assert data.orbitals[2].energy == pytest.approx(0.25)


def test_evaluate_molden_falls_back_when_pyscf_fails(monkeypatch):
    """If PySCF molden.load raises, evaluate_molden_orbital_grid must fall back to _load_fallback_molden."""
    import agymolden.volumetric as vol_mod

    vol_mod._MO_CTX_CACHE.clear()
    monkeypatch.setattr(vol_mod, "_load_pyscf_molden", lambda _p: (_ for _ in ()).throw(RuntimeError("boom")))
    vol = vol_mod.evaluate_molden_orbital_grid("examples/h2o_orca.molden", orbital_index=4, nx=20, ny=20, nz=20)
    assert vol is not None
    assert float(abs(vol.grid_data).max()) > 0.05


def test_empty_companion_molden_does_not_block_log_orbitals(main_win, tmp_path):
    """An empty .molden (no [MO]) alongside an ORCA/Gaussian log must not block loading MOs & energy diagram."""
    app, win = main_win
    # Create a minimal log with MOs or copy h2o_orca.molden as _electrons.molden alongside an empty .molden
    log_path = tmp_path / "calc.out"
    log_path.write_text(
        open("examples/AgyMolden_generated_calculation.log", "r", encoding="utf-8").read(),
        encoding="utf-8",
    )
    empty_molden = tmp_path / "calc.molden"
    empty_molden.write_text(
        "[Molden Format]\n[Atoms] AU\nLi 1 3 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    assert win.load_orbitals_from_molden(str(empty_molden), auto_render=False) is False

    win._suppress_companion_prompt = True
    win.open_path(str(log_path))
    app.processEvents()

    assert win.table_orbitals.rowCount() > 0
    assert len(win.mo_ladder_widget._orbitals) > 0
    assert win.mo_ladder_widget._homo_idx is not None


def test_clicking_preselected_homo_and_ladder_renders_orbital(main_win):
    """Clicking pre-selected HOMO in table_orbitals or mo_ladder_widget must render 3D orbital surfaces."""
    app, win = main_win
    assert win.load_orbitals_from_molden("examples/h2o_orca.molden", auto_render=False) is True
    assert len(win.viewer.active_isosurfaces) == 0

    homo_row = win.mo_ladder_widget._homo_idx
    assert homo_row is not None
    # Clicking already-selected HOMO row in table
    win._on_mo_row_clicked(homo_row)
    app.processEvents()
    assert len(win.viewer.active_isosurfaces) == 2

    # Clear and click via MO Energy Ladder
    win._on_clear_orbital_surface()
    assert len(win.viewer.active_isosurfaces) == 0
    win._on_mo_ladder_clicked(homo_row)
    app.processEvents()
    assert len(win.viewer.active_isosurfaces) == 2


def test_command_palette_esp_target_tab_and_clear_state(main_win):
    """Ensure _on_open_volumetric_dialog(target_tab='esp') and state clearing work cleanly."""
    app, win = main_win
    win.load_orbitals_from_molden("examples/h2o_orca.molden", auto_render=False)
    win._on_open_volumetric_dialog(target_tab="esp")
    app.processEvents()

    win._clear_stale_surfaces_and_qm_state(clear_molden=True)
    assert len(win.mo_ladder_widget._orbitals) == 0
    assert win.table_orbitals.rowCount() == 0


def test_molden_without_mo_section_returns_none_cleanly(tmp_path):
    """evaluate_molden_orbital_grid on a .molden without [MO] must return None instead of crashing."""
    import agymolden.volumetric as vol_mod

    m_path = tmp_path / "no_mo.molden"
    m_path.write_text(
        "[Molden Format]\n"
        "[Atoms] AU\n"
        "H 1 1 0.0 0.0 0.0\n"
        "[GTO]\n"
        "  1 0\n"
        "s 1 1.0\n"
        "  1.0 1.0\n\n",
        encoding="utf-8",
    )
    vol_mod._MO_CTX_CACHE.clear()
    assert vol_mod.evaluate_molden_orbital_grid(str(m_path), orbital_index=0, nx=12, ny=12, nz=12) is None


def test_quick_orca_multiplicity_prediction_and_core_detection(main_win):
    """QuickOrcaDialog must predict multiplicity from electron parity and detect CPU cores."""
    from agymolden.dialog_quick_orca import QuickOrcaDialog
    from agymolden.zmatrix import ZAtom

    _app, win = main_win
    # Odd-electron CH3 radical (9 electrons -> doublet mult=2)
    ch3 = [ZAtom("C"), ZAtom("H", 0, 1.09), ZAtom("H", 0, 1.09, 1, 120.0), ZAtom("H", 0, 1.09, 1, 120.0, 2, 180.0)]
    dlg = QuickOrcaDialog(parent=win, zatoms=ch3)
    assert dlg.spn_mult.value() == 2

    # Changing charge to +1 gives 8 electrons -> singlet mult=1
    dlg.spn_charge.setValue(1)
    assert dlg.spn_mult.value() == 1

    n_cpus = max(1, os.cpu_count() or 1)
    assert dlg.spn_cores.maximum() >= n_cpus
    assert str(n_cpus) in dlg.lbl_detected_cores.text()
    dlg.btn_use_all_cores.click()
    assert dlg.spn_cores.value() == n_cpus
    assert f"%pal nprocs {n_cpus} end" in dlg.txt_deck_preview.toPlainText()
    dlg.close()

