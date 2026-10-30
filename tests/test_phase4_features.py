"""Unit tests for Phase 4 features:
1. Kabsch 3D Alignment & RMSD Calculation
2. Structure Overlay Dialog
3. Vibrational Frequency Filtering & Vector Scale Tuning
4. Quick ORCA Spin Parity Validation
"""

import numpy as np
import pytest
from PyQt6 import QtWidgets

from agymolden.alignment import calculate_kabsch_alignment
from agymolden.dialog_quick_orca import QuickOrcaDialog
from agymolden.vibrations import VibrationalAnalysis
from agymolden.zmatrix import ZAtom


def test_kabsch_alignment_identical():
    """Identical structures should give 0.0 RMSD and identity rotation."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    aligned, rmsd, R, t = calculate_kabsch_alignment(coords, coords)
    assert np.isclose(rmsd, 0.0, atol=1e-6)
    assert np.allclose(aligned, coords, atol=1e-5)
    assert np.allclose(R, np.eye(3), atol=1e-5)
    assert np.allclose(t, np.zeros(3), atol=1e-5)


def test_kabsch_alignment_translated_rotated():
    """Translated and rotated structures should be aligned back with 0.0 RMSD."""
    ref = np.array([
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [0.0, 1.2, 0.0],
        [0.5, 0.5, 1.1]
    ], dtype=float)

    # Apply 90-degree Z-rotation and (5, -3, 2) translation
    theta = np.pi / 2.0
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    trans = np.array([5.0, -3.0, 2.0])
    mobile = np.dot(ref, rot_z.T) + trans

    aligned, rmsd, R, t = calculate_kabsch_alignment(ref, mobile)
    assert np.isclose(rmsd, 0.0, atol=1e-5)
    assert np.allclose(aligned, ref, atol=1e-4)


def test_orca_parity_validation(qapp):
    """Test electron/multiplicity parity validation in QuickOrcaDialog."""
    # Water: O (8) + 2*H (1+1) = 10 electrons
    zatoms = [
        ZAtom(symbol="O"),
        ZAtom(symbol="H", ref1=0, dist=0.96),
        ZAtom(symbol="H", ref1=0, dist=0.96, ref2=1, angle=104.5),
    ]

    dlg = QuickOrcaDialog(None, zatoms=zatoms)

    # Neutral singlet (charge=0, mult=1) -> 10 e- -> valid
    dlg.spn_charge.setValue(0)
    dlg.spn_mult.setValue(1)
    dlg._validate_charge_multiplicity()
    assert "Valid state" in dlg.lbl_parity_warning.text()

    # Neutral doublet (charge=0, mult=2) -> 10 e- -> invalid!
    dlg.spn_mult.setValue(2)
    assert "Spin mismatch" in dlg.lbl_parity_warning.text()

    # Cation doublet (charge=+1, mult=2) -> 9 e- -> valid
    dlg.spn_charge.setValue(1)
    dlg.spn_mult.setValue(2)
    assert "Valid state" in dlg.lbl_parity_warning.text()

    # Cation singlet (charge=+1, mult=1) -> 9 e- -> invalid
    dlg.spn_charge.setValue(1)
    dlg.spn_mult.setValue(1)
    assert "Spin mismatch" in dlg.lbl_parity_warning.text()
    dlg.close()


def test_vibrational_filter_and_vector_scale(qapp):
    """Test UI vibration filter combobox and vector scale slider."""
    from agymolden.main_window import MainWindow

    mw = MainWindow()

    # Mock VibrationalAnalysis with frequencies: negative (TS), low, fingerprint, functional group
    freqs = [-150.0, 12.0, 1000.0, 1720.0, 3600.0]
    disps = np.zeros((5, 3, 3))
    symbols = ["C", "H", "H"]
    coords = np.zeros((3, 3))
    analysis = VibrationalAnalysis(
        frequencies=freqs,
        displacements=disps,
        symbols=symbols,
        base_coords=coords,
        source="Test Mock",
    )
    mw.coords = coords
    mw.zatoms = [ZAtom("C"), ZAtom("H"), ZAtom("H")]
    mw._apply_vibrations((coords, analysis))

    assert mw.table_vibrations.rowCount() == 5

    # Filter to Imaginary Only
    idx_ts = mw.combo_vib_filter.findData("imag")
    mw.combo_vib_filter.setCurrentIndex(idx_ts)
    mw._on_filter_vibrations(idx_ts)
    assert mw.table_vibrations.isRowHidden(0) is False  # -150 cm-1 shown
    assert mw.table_vibrations.isRowHidden(1) is True   # 12 cm-1 hidden
    assert mw.table_vibrations.isRowHidden(2) is True   # 1000 cm-1 hidden

    # Filter to Functional Groups (1500 - 4000 cm-1)
    idx_fg = mw.combo_vib_filter.findData("functional")
    mw.combo_vib_filter.setCurrentIndex(idx_fg)
    mw._on_filter_vibrations(idx_fg)
    assert mw.table_vibrations.isRowHidden(0) is True   # -150 cm-1 hidden
    assert mw.table_vibrations.isRowHidden(2) is True   # 1000 cm-1 hidden
    assert mw.table_vibrations.isRowHidden(3) is False  # 1720 cm-1 shown
    assert mw.table_vibrations.isRowHidden(4) is False  # 3600 cm-1 shown

    # Test Vector Scale Slider (range is 20 to 300, meaning 0.2x to 3.0x; 200 is 2.0x)
    mw.slider_vib_vec_scale.setValue(200)  # 2.0x
    assert np.isclose(mw.viewer.vib_vector_scale, 2.0)
    mw.close()
