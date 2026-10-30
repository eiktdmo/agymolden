"""Offscreen GUI smoke tests: load a Molden file, evaluate orbitals/ESP, and move the isovalue."""

import os
import time

import pytest

pytest.importorskip("pyscf")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets  # noqa: E402

EXAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "h2o_orca.molden")


def _pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()


@pytest.fixture
def dialog(monkeypatch):
    import agymolden.workers as workers
    from agymolden.main_window import MainWindow
    from agymolden.dialog_volumetric import VolumetricSurfaceDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(workers, "SYNCHRONOUS", True)
    win = MainWindow()
    dlg = VolumetricSurfaceDialog(win, viewer=win.viewer)
    yield app, dlg
    dlg.close()
    win.close()


def test_molden_load_orbital_esp_and_isovalue(dialog):
    app, dlg = dialog
    dlg.load_molden_path(EXAMPLE)
    assert dlg.combo_mo.count() > 0
    assert dlg.current_vol_data is not None  # HOMO evaluated on load

    dlg._on_compute_esp()
    assert "Electrostatic" in dlg.current_vol_data.title

    for v in (0.01, 0.02, 0.05, 0.1):  # rapid isovalue changes are debounced, must not hang
        dlg.spin_iso.setValue(v)
    _pump(app, 1.5)
    assert "Marching cubes error" not in dlg.lbl_stats.text()


def test_orbital_grid_cache_returns_same_shape():
    from agymolden.volumetric import evaluate_molden_orbital_grid
    a = evaluate_molden_orbital_grid(EXAMPLE, orbital_index=0, nx=24, ny=24, nz=24)
    b = evaluate_molden_orbital_grid(EXAMPLE, orbital_index=3, nx=24, ny=24, nz=24)
    assert a is not None and b is not None
    assert a.grid_data.shape == b.grid_data.shape == (24, 24, 24)


def test_grid_resolution_selector_and_orca_raman_uvvis(dialog, tmp_path):
    app, dlg = dialog
    # 1. Verify Low (30^3), Med (40^3), High (60^3) grid selector in VolumetricSurfaceDialog
    assert hasattr(dlg, "combo_grid_points")
    dlg.combo_grid_points.setCurrentIndex(0)  # Low (30^3)
    assert dlg._get_grid_pts() == 30
    dlg.load_molden_path(EXAMPLE)
    assert dlg.current_vol_data.grid_data.shape == (30, 30, 30)

    # 2. Verify ORCA Raman + UV-Vis fallback parser
    from agymolden.qm_log_parser import _parse_orca_logfile_fallback
    sample_orca = tmp_path / "water_raman_uv.out"
    sample_orca.write_text(
        """
                                 *****************
                                 * O   R   C   A *
                                 *****************
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  O      0.000000    0.000000    0.117300
  H      0.000000    0.757200   -0.469200
  H      0.000000   -0.757200   -0.469200

FINAL SINGLE POINT ENERGY       -76.408953210

VIBRATIONAL FREQUENCIES
-----------------------
   6:      1648.50 cm**-1
   7:      3657.10 cm**-1
   8:      3755.90 cm**-1

NORMAL MODES
------------
                  6          7          8
      0       0.000000   0.000000   0.000000
      1       0.000000   0.000000   0.070000
      2       0.070000   0.050000   0.000000
      3       0.000000   0.000000   0.000000
      4      -0.430000   0.580000  -0.550000
      5      -0.560000  -0.400000  -0.430000
      6       0.000000   0.000000   0.000000
      7       0.430000  -0.580000  -0.550000
      8      -0.560000  -0.400000   0.430000

IR SPECTRUM
-----------
 Mode   freq       eps      Int      T**2         TX        TY        TZ
   6:  1648.50   0.01234   72.40   0.002700  ( 0.000000  0.000000  0.051962)
   7:  3657.10   0.00120    8.50   0.000143  ( 0.000000  0.000000  0.011958)
   8:  3755.90   0.00850   45.20   0.000740  ( 0.000000  0.027203  0.000000)

RAMAN SPECTRUM
--------------
 Mode    freq (cm**-1)   Activity   Depolarization
   6:       1648.50       12.3400      0.720000
   7:       3657.10      108.5000      0.140000
   8:       3755.90       31.2000      0.750000

ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
----------------------------------------------------------
State   Energy (cm-1)   Wavelength (nm)   fosc         T2
   1      60120.0          166.3        0.042100     0.02300
   2      78500.0          127.4        0.115000     0.04800

****ORCA TERMINATED NORMALLY****
""",
        encoding="utf-8",
    )
    res = _parse_orca_logfile_fallback(str(sample_orca))
    assert res is not None
    assert res.vibrational_analysis is not None
    assert res.vibrational_analysis.ir_intensities == [72.40, 8.50, 45.20]
    assert res.vibrational_analysis.raman_activities == [12.34, 108.5, 31.2]
    assert res.et_energies == [166.3, 127.4]
    assert res.et_oscs == [0.0421, 0.115]

    # 3. Verify MO Energy Ladder Widget populates when loading orbitals into MainWindow
    win = dlg.parent()
    win.load_orbitals_from_molden(EXAMPLE)
    assert hasattr(win, "mo_ladder_widget")
    assert len(win.mo_ladder_widget._orbitals) > 0
    assert win.mo_ladder_widget._homo_idx is not None

    # 4. Verify Brorsen pre-flight validator detects charge/multiplicity parity mismatch
    from agymolden.brorsen_builder import BrorsenInputBuilderDialog
    import numpy as np
    b_dlg = BrorsenInputBuilderDialog(win, symbols=["O", "H", "H"], coords=np.zeros((3, 3)))
    assert b_dlg._validate_stages_preflight() == []
    b_dlg.generator.stages[0].charge = 0
    b_dlg.generator.stages[0].mult = 2  # 10 electrons cannot be doublet
    warnings = b_dlg._validate_stages_preflight()
    assert len(warnings) == 1
    assert "incompatible with Multiplicity" in warnings[0]
    b_dlg.close()


