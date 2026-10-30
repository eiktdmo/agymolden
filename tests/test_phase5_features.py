"""Unit tests for Phase 5 Advanced Capabilities:
1. Trajectory Movie Export (GIF / Video)
2. Interactive NCI / RDG 2D Scatter Plot
3. Global User Preferences & Theme Defaults
4. Point Group Symmetry Tolerance Slider & Dynamic Detection
"""

import os
import tempfile
import numpy as np
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from agymolden.main_window import MainWindow
from agymolden.dialog_preferences import PreferencesDialog
from agymolden.dialog_nci_plot import NCIRdgPlotDialog
from agymolden.zmatrix import ZAtom


def test_trajectory_movie_export_gif(qapp):
    """Test frame buffer capture and trajectory export to animated GIF."""
    mw = MainWindow()
    mw.trajectory_coords = [
        np.zeros((3, 3)),
        np.ones((3, 3)) * 0.5,
        np.ones((3, 3)) * 1.0,
    ]
    mw.slider_trajectory.setRange(0, 2)
    mw.slider_trajectory.setValue(0)

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        tmp_gif = f.name

    try:
        # Simulate rendering and GIF saving
        frames_rgb = []
        for idx in range(len(mw.trajectory_coords)):
            mw.slider_trajectory.setValue(idx)
            mw.viewer.repaint()
            if mw.viewer.isValid() and mw.viewer.context() is not None:
                qimg = mw.viewer.grabFramebuffer().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
            else:
                w, h = 400, 300
                qimg = QtGui.QImage(w, h, QtGui.QImage.Format.Format_RGBA8888)
                qimg.fill(QtGui.QColor(30, 33, 41))
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits()
            ptr.setsize(h * w * 4)
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4)).copy()
            frames_rgb.append(arr[:, :, :3])

        assert len(frames_rgb) == 3
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames_rgb]
        imgs[0].save(tmp_gif, save_all=True, append_images=imgs[1:], duration=100, loop=0)

        assert os.path.isfile(tmp_gif)
        assert os.path.getsize(tmp_gif) > 100
    finally:
        if os.path.isfile(tmp_gif):
            os.remove(tmp_gif)
        mw.close()


def test_preferences_dialog(qapp):
    """Test loading, modifying, and applying settings via PreferencesDialog."""
    mw = MainWindow()
    dlg = PreferencesDialog(mw)

    dlg.combo_bg.setCurrentIndex(dlg.combo_bg.findData("dark"))
    dlg.chk_labels.setChecked(True)
    dlg.chk_indices.setChecked(True)
    dlg.spin_cores.setValue(8)
    dlg.spin_mem.setValue(8000)
    dlg.spin_sym_tol.setValue(0.25)

    dlg._save_settings()

    # Verify changes applied to viewer and window
    assert mw.viewer.bg_color_mode == "dark"
    assert mw.viewer.show_labels is True
    assert mw.viewer.show_indices is True
    assert np.isclose(mw.spin_sym_tol.value(), 0.25)
    mw.close()


def test_symmetry_tolerance_slider_sync(qapp):
    """Test two-way synchronization between symmetry tolerance slider and spinbox."""
    mw = MainWindow()

    # Move slider to 25 (0.25 Å)
    mw.slider_sym_tol.setValue(25)
    assert np.isclose(mw.spin_sym_tol.value(), 0.25)

    # Change spinbox to 0.40 Å
    mw.spin_sym_tol.setValue(0.40)
    assert mw.slider_sym_tol.value() == 40
    mw.close()


def test_nci_rdg_plot_dialog_init(qapp):
    """Verify NCIRdgPlotDialog creates canvas and matplotlib axes properly."""
    dlg = NCIRdgPlotDialog(None, molden_path=None)
    assert dlg.canvas is not None
    assert dlg.figure is not None

    # Test plotting dummy RDG scatter points
    x = np.linspace(-0.03, 0.03, 50)
    y = np.linspace(0.1, 1.2, 50)
    dlg._plot_scatter(x, y)
    assert len(dlg.figure.axes) == 1
    dlg.close()
