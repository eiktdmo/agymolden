"""Tests for symmetry, vibrations, semiempirical, XYZ I/O and background workers."""
import os
import time

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("OpenGL")

from agymolden.zmatrix import (  # noqa: E402
    ZAtom, zmatrix_to_cartesian, parse_xyz, write_xyz,
    parse_zmatrix_text, format_zmatrix_text,
)
from agymolden.symmetry import find_symmetry_operations  # noqa: E402
from agymolden.vibrations import compute_force_field_vibrations  # noqa: E402
from agymolden.forcefield import optimize_geometry_force_field  # noqa: E402
from agymolden.semiempirical import optimize_geometry_semiempirical  # noqa: E402

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")


def _water():
    zat = [ZAtom(symbol="O"),
           ZAtom(symbol="H", ref1=0, dist=0.96),
           ZAtom(symbol="H", ref1=0, dist=0.96, ref2=1, angle=104.5)]
    return ["O", "H", "H"], zmatrix_to_cartesian(zat)


def test_symmetry_water_is_c2v():
    symbols, coords = _water()
    assert find_symmetry_operations(symbols, coords).symbol == "C2v"


def test_symmetry_methane_is_td():
    d = 1.09 / np.sqrt(3)
    coords = np.array([[0, 0, 0], [d, d, d], [d, -d, -d], [-d, d, -d], [-d, -d, d]], float)
    assert find_symmetry_operations(["C", "H", "H", "H", "H"], coords).symbol == "Td"


def test_vibrations_water_force_field():
    symbols, coords = _water()
    eq, analysis = compute_force_field_vibrations(symbols, coords, optimize_first=True)
    assert analysis.n_modes == 9
    assert analysis.displacements.shape == (9, 3, 3)
    assert np.all(np.isfinite(analysis.frequencies))
    genuine = sorted(f for f in analysis.frequencies if f > 200)
    assert len(genuine) == 3  # bend + 2 stretches


def test_optimize_force_field_lowers_or_keeps_energy():
    symbols, coords = _water()
    distorted = coords * 1.15
    opt, e = optimize_geometry_force_field(symbols, distorted)
    assert np.isfinite(e)
    assert 0.8 < np.linalg.norm(opt[1] - opt[0]) < 1.1


def test_semiempirical_optimize_returns_properties():
    symbols, coords = _water()
    opt, e, props = optimize_geometry_semiempirical(symbols, coords, max_iter=50)
    assert opt.shape == (3, 3) and np.isfinite(e)
    assert "gap_ev" in props


def test_xyz_roundtrip(tmp_path):
    symbols, coords = _water()
    path = str(tmp_path / "w.xyz")
    write_xyz(path, symbols, coords)
    s2, c2 = parse_xyz(path)
    assert s2 == symbols
    assert np.allclose(c2, coords, atol=1e-4)


def test_zmatrix_text_roundtrip_example_files():
    for name in ("ethane.zmat", "toluene.zmat"):
        with open(os.path.join(EXAMPLES, name)) as f:
            zat, variables = parse_zmatrix_text(f.read())
        again, vars2 = parse_zmatrix_text(format_zmatrix_text(zat, variables))
        assert [a.symbol for a in again] == [a.symbol for a in zat]
        assert np.allclose(zmatrix_to_cartesian(again), zmatrix_to_cartesian(zat), atol=1e-3)


# ---------------------------------------------------------------- GUI / workers
@pytest.fixture(scope="module")
def qapp():
    from PyQt6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait(qapp, cond, timeout=60):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        qapp.processEvents()
        time.sleep(0.01)
    return cond()


def test_background_worker_success_and_error(qapp, threaded_jobs):
    from PyQt6 import QtWidgets
    from agymolden.workers import run_in_background
    parent = QtWidgets.QWidget()
    got = {}
    run_in_background(parent, "t", "m", lambda: 41 + 1,
                      lambda r: got.setdefault("ok", r), lambda e: got.setdefault("err", e))
    assert _wait(qapp, lambda: "ok" in got) and got["ok"] == 42

    def boom():
        raise ValueError("bad")
    run_in_background(parent, "t", "m", boom,
                      lambda r: got.setdefault("ok2", r), lambda e: got.setdefault("err", e))
    assert _wait(qapp, lambda: "err" in got) and isinstance(got["err"], ValueError)


def test_background_worker_cancel_discards_result(qapp, threaded_jobs):
    from PyQt6 import QtWidgets
    from agymolden.workers import run_in_background
    parent = QtWidgets.QWidget()
    got = []
    run_in_background(parent, "t", "m", lambda: (time.sleep(0.3), 1)[1],
                      got.append, got.append)
    for child in parent.findChildren(QtWidgets.QProgressDialog):
        child.canceled.emit()
    _wait(qapp, lambda: not parent._active_workers)
    assert got == []


def test_mainwindow_optimize_and_vibrations_end_to_end(qapp, threaded_jobs):
    from agymolden.main_window import MainWindow
    w = MainWindow()
    w.open_path(os.path.join(EXAMPLES, "ethane.zmat"))
    n = len(w.zatoms)
    assert n > 2

    w._on_optimize_geometry()
    assert _wait(qapp, lambda: "converged" in w.statusBar().currentMessage()), w.statusBar().currentMessage()
    assert len(w.zatoms) == n

    w._on_calculate_vibrations()
    assert _wait(qapp, lambda: w.vibrational_analysis is not None)
    assert w.vibrational_analysis.n_modes == 3 * n
    assert w.table_vibrations.rowCount() == 3 * n


def test_dipole_charges_and_png_export(qapp, tmp_path):
    from agymolden.main_window import MainWindow
    from agymolden.zmatrix import ZAtom
    w = MainWindow()
    # Water molecule: polar (O negative, H positive, non-zero dipole)
    w.set_molecule_from_zatoms([
        ZAtom(symbol="O"),
        ZAtom(symbol="H", ref1=0, dist=0.96),
        ZAtom(symbol="H", ref1=0, dist=0.96, ref2=1, angle=104.5),
    ])
    q, mu_vec, mu_mag = w.viewer._charges_and_dipole()
    assert len(q) == 3
    assert q[0] < -0.2 and q[1] > 0.1 and q[2] > 0.1
    assert abs(float(q.sum())) < 1e-6
    assert mu_mag > 1.0

    # Color mapping: negative -> red-dominant, positive -> blue-dominant
    r_neg, _, b_neg = w.viewer._charge_to_rgb(q[0])
    r_pos, _, b_pos = w.viewer._charge_to_rgb(q[1])
    assert r_neg > b_neg and b_pos > r_pos

    # Toggle UI checkboxes
    w.chk_dipole.setChecked(True)
    w.chk_charges.setChecked(True)
    assert w.viewer.show_dipole is True
    assert w.viewer.show_charges is True

    # Test PNG image export
    png_path = str(tmp_path / "water_test.png")
    w.export_image(png_path, transparent=True, scale=2)
    assert os.path.isfile(png_path)
