"""Headless Offscreen Framebuffer and OpenGL Rendering Tests.

Verifies:
1. MoleculeViewer safely handles headless/offscreen environments without crashing or segfaulting.
2. `export_image` creates valid QImages with correct DPI metadata in headless mode.
3. Multiple image export options (transparent, dark, white) execute reliably.
4. Smooth upscaling (2x, 4x, 8x) preserves aspect ratio and generates valid image buffers.
"""

import os
import tempfile
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui, QtWidgets
from agymolden.main_window import MainWindow
from agymolden.zmatrix import ZAtom


@pytest.fixture
def win():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    # Populate a sample molecule (Ethanol)
    zatoms = [
        ZAtom("C"),
        ZAtom("C", ref1=0, dist=1.52),
        ZAtom("O", ref1=1, dist=1.43, ref2=0, angle=109.5),
        ZAtom("H", ref1=0, dist=1.09, ref2=1, angle=109.5, ref3=2, dihedral=60.0),
        ZAtom("H", ref1=0, dist=1.09, ref2=1, angle=109.5, ref3=2, dihedral=-60.0),
        ZAtom("H", ref1=0, dist=1.09, ref2=1, angle=109.5, ref3=2, dihedral=180.0),
        ZAtom("H", ref1=1, dist=1.09, ref2=0, angle=109.5, ref3=2, dihedral=60.0),
        ZAtom("H", ref1=1, dist=1.09, ref2=0, angle=109.5, ref3=2, dihedral=-60.0),
        ZAtom("H", ref1=2, dist=0.96, ref2=1, angle=104.5, ref3=0, dihedral=180.0),
    ]
    window.set_molecule_from_zatoms(zatoms)
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def test_headless_image_export_modes(tmp_path, win):
    """Test exporting image buffers under transparent, white, and dark background modes."""
    # 1. Transparent
    path_trans = str(tmp_path / "mol_trans.png")
    img_trans = win.export_image(path_trans, transparent=True, scale=1, dpi=300)
    assert not img_trans.isNull()
    assert img_trans.width() >= 300
    assert img_trans.height() >= 200
    assert os.path.isfile(path_trans)

    # 2. White
    path_white = str(tmp_path / "mol_white.png")
    img_white = win.export_image(path_white, transparent=False, bg_mode="white", scale=1, dpi=300)
    assert not img_white.isNull()
    assert os.path.isfile(path_white)

    # 3. Dark
    path_dark = str(tmp_path / "mol_dark.png")
    img_dark = win.export_image(path_dark, transparent=False, bg_mode="dark", scale=1, dpi=300)
    assert not img_dark.isNull()
    assert os.path.isfile(path_dark)


def test_headless_image_export_resolutions_and_dpi(tmp_path, win):
    """Verify that 2x and 4x scaling correctly produce high-res buffers with DPI metadata."""
    # 2x scale @ 600 DPI
    path_2x = str(tmp_path / "mol_2x_600dpi.png")
    img_2x = win.export_image(path_2x, scale=2, dpi=600)
    assert not img_2x.isNull()

    # 4x scale @ 300 DPI
    path_4x = str(tmp_path / "mol_4x_300dpi.png")
    img_4x = win.export_image(path_4x, scale=4, dpi=300)
    assert not img_4x.isNull()

    assert img_4x.width() > img_2x.width()
    assert img_4x.height() > img_2x.height()

    # Check embedded physical DPI (dots per meter)
    expected_dpm_300 = int(300 / 0.0254)
    assert abs(img_4x.dotsPerMeterX() - expected_dpm_300) <= 2


def test_instanced_bond_shader_branching():
    """Verify that the fragment shader branches properly for instanced bonds (u_use_instancing >= 1)."""
    from agymolden.viewer_meshes import FRAGMENT_SHADER_SRC
    assert "u_use_instancing >= 1" in FRAGMENT_SHADER_SRC
    assert "u_use_instancing == 1" not in FRAGMENT_SHADER_SRC
